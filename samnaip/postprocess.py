"""Masks -> polygons, shape/NDVI filters, de-duplication and building regularization."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.strtree import STRtree

# Default filter rules (map units are metres in UTM).
# Building rules were tuned on the residential AOI only (see README); the mixed AOI is held out.
# NDVI is computed from NAIP digital numbers, so roofs sit well below zero in this scene.
BUILDING_RULES = dict(min_area=40.0, max_area=6000.0, max_ndvi=-0.15, min_brightness=90.0, max_brightness=165.0,
                      min_rectangularity=0.60, max_elongation=3.0, min_solidity=0.75, min_score=0.90)
FIELD_RULES = dict(min_area=4000.0, max_area=1e7, min_solidity=0.85, min_rectangularity=0.55,
                   max_elongation=8.0)


def mask_to_polygons(mask: np.ndarray, transform, min_pixels: int = 20) -> list[Polygon]:
    """Vectorize a boolean mask into polygons (map coordinates)."""
    from rasterio import features

    out = []
    m = mask.astype(np.uint8)
    for geom, val in features.shapes(m, mask=m.astype(bool), transform=transform, connectivity=4):
        if val != 1:
            continue
        p = shape(geom)
        if not p.is_valid:
            p = p.buffer(0)
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
        if p.area >= min_pixels * abs(transform.a * transform.e):
            out.append(Polygon(p.exterior))  # drop holes: roofs/fields are treated as solid
    return out


def shape_metrics(p: Polygon) -> dict:
    """Area, rectangularity (area / min. rotated rectangle), elongation and solidity."""
    mrr = p.minimum_rotated_rectangle
    xs, ys = mrr.exterior.coords.xy
    sides = sorted({round(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]), 6) for i in range(4)})
    short, long_ = (sides[0], sides[-1]) if sides else (0.0, 0.0)
    hull = p.convex_hull.area
    return {
        "area": p.area,
        "rectangularity": p.area / mrr.area if mrr.area > 0 else 0.0,
        "elongation": long_ / short if short > 0 else np.inf,
        "solidity": p.area / hull if hull > 0 else 0.0,
    }


def zonal_mean(poly: Polygon, raster: np.ndarray, transform) -> float:
    """Mean of `raster` inside `poly` (pixel centres)."""
    from rasterio import features
    from rasterio.windows import from_bounds

    win = from_bounds(*poly.bounds, transform=transform).round_offsets().round_lengths()
    r0, c0 = max(int(win.row_off), 0), max(int(win.col_off), 0)
    r1 = min(r0 + int(win.height) + 1, raster.shape[0])
    c1 = min(c0 + int(win.width) + 1, raster.shape[1])
    if r1 <= r0 or c1 <= c0:
        return float("nan")
    from rasterio.transform import Affine

    t = transform @ Affine.translation(c0, r0)
    m = features.rasterize([(poly, 1)], out_shape=(r1 - r0, c1 - c0), transform=t, dtype=np.uint8).astype(bool)
    vals = raster[r0:r1, c0:c1][m]
    return float(vals.mean()) if vals.size else float("nan")


def passes(metrics: dict, rules: dict) -> bool:
    """Rules are 'min_<metric>' / 'max_<metric>' bounds on any metric in `metrics`."""
    for key, bound in rules.items():
        kind, name = key.split("_", 1)
        v = metrics.get(name)
        if v is None or not np.isfinite(v):
            return False
        if (kind == "min" and v < bound) or (kind == "max" and v > bound):
            return False
    return True


def filter_candidates(cands: list[dict], rasters: dict, transform, rules: dict) -> list[dict]:
    """Keep candidates ({'geometry', 'score'}) that pass the rules. `rasters` maps a metric name
    (e.g. 'ndvi', 'brightness') to a per-pixel array whose polygon mean is tested."""
    kept = []
    for c in cands:
        m = shape_metrics(c["geometry"])
        if m["area"] < rules.get("min_area", 0) or m["area"] > rules.get("max_area", np.inf):
            continue
        m["score"] = c.get("score", 1.0)
        if not passes({k: v for k, v in m.items()}, {k: v for k, v in rules.items() if k.split("_", 1)[1] in m}):
            continue
        for name, arr in rasters.items():
            if f"min_{name}" in rules or f"max_{name}" in rules:
                m[name] = zonal_mean(c["geometry"], arr, transform)
        if passes(m, rules):
            kept.append({**c, **m})
    return kept


def dedupe(cands: list[dict], overlap: float = 0.5) -> list[dict]:
    """Greedy NMS: highest score first; suppress any lower-scored polygon whose
    intersection / (smaller area) exceeds `overlap`."""
    if not cands:
        return []
    order = sorted(cands, key=lambda c: c.get("score", 0.0), reverse=True)
    geoms = [c["geometry"] for c in order]
    tree = STRtree(geoms)
    suppressed = np.zeros(len(order), dtype=bool)
    kept = []
    for i, g in enumerate(geoms):
        if suppressed[i]:
            continue
        kept.append(order[i])
        for j in tree.query(g):
            j = int(j)
            if j <= i or suppressed[j]:
                continue
            k = geoms[j]
            if g.intersection(k).area / max(min(g.area, k.area), 1e-9) > overlap:
                suppressed[j] = True
    return kept


def regularize_building(p: Polygon, rect_threshold: float = 0.85, tolerance: float = 0.9) -> Polygon:
    """Near-rectangular roofs -> minimum rotated rectangle; others -> Douglas-Peucker simplification."""
    m = shape_metrics(p)
    if m["rectangularity"] >= rect_threshold:
        return p.minimum_rotated_rectangle
    s = p.simplify(tolerance, preserve_topology=True)
    return s if s.is_valid and s.area > 0 else p


def to_gdf(cands: list[dict], crs: str):
    import geopandas as gpd

    if not cands:
        return gpd.GeoDataFrame({"score": []}, geometry=[], crs=crs)
    cols = {k: [c.get(k) for c in cands] for k in cands[0] if k != "geometry"}
    return gpd.GeoDataFrame(cols, geometry=[c["geometry"] for c in cands], crs=crs)
