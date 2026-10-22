"""Extract buildings and fields from NAIP with Segment Anything and score them against reference data.

Examples:
    python run.py                                   # all three AOIs, all methods
    python run.py --aoi residential --method auto
    python run.py --bbox -96.305 30.580 -96.292 30.590 --name mytest
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import time
from pathlib import Path

# Windows: pip PyTorch ships Intel OpenMP while conda-forge MKL/numpy load LLVM OpenMP. Both are
# compatible implementations; without this flag the process aborts when the second one loads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from shapely.geometry import box

from samnaip import evaluate as ev
from samnaip import imagery, plots
from samnaip import postprocess as pp
from samnaip import segment as sg

ROOT = Path(__file__).resolve().parent
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"
BUILDING_PROMPT = "building. house. roof."
FIELD_PROMPT = "rectangular field."
MIN_REF = 50  # minimum reference objects for object-level scores


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        timer = self

        class _T:
            def __enter__(self):
                _sync()
                self.s = time.perf_counter()

            def __exit__(self, *a):
                _sync()
                timer.t[key] = round(timer.t.get(key, 0) + time.perf_counter() - self.s, 2)

        return _T()


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def cached(path: Path, reuse: bool, fn):
    """Raw candidates are slow (GPU) and filters are fast: cache raw output so filters can be re-run."""
    if reuse and path.exists():
        return pickle.loads(path.read_bytes())
    out = fn()
    path.write_bytes(pickle.dumps(out))
    return out


def save_geojson(cands, crs, path, keep=("score",), simplify_m=0.0):
    gdf = pp.to_gdf(cands, crs)
    if simplify_m:  # output size only; metrics are computed on the unsimplified polygons
        gdf["geometry"] = gdf.geometry.simplify(simplify_m, preserve_topology=True)
    cols = [c for c in keep if c in gdf.columns]
    gdf = gdf[cols + ["geometry"]].to_crs("EPSG:4326")
    for c in cols:
        gdf[c] = gdf[c].astype(float).round(3)
    gdf.to_file(path, driver="GeoJSON", COORDINATE_PRECISION=7)


def run_aoi(name, grid, label, methods, sam, text_model, timer, args):
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    with timer(f"{name}/download"):
        naip_meta = imagery.fetch_naip(grid, d / "naip.tif")
        ms = imagery.fetch_ms_buildings(grid, d / "ms_buildings.geojson")
        osm = imagery.fetch_osm(grid, d / "osm_buildings.geojson", "building")
        osm_fields = imagery.fetch_osm(grid, d / "osm_farmland.geojson", "farmland")
        try:
            cdl = imagery.fetch_cdl_fields(grid, d / "cdl_fields.npy")
        except Exception as e:  # CDL is a nice-to-have check
            print("CDL unavailable:", e)
            cdl = None
    rgb, ndvi, _ = imagery.read_naip(d / "naip.tif")
    T, shape, crs = grid.transform, rgb.shape[:2], grid.crs
    R = {"ndvi": ndvi, "brightness": rgb.astype(np.float32).mean(-1)}
    refs = {"ms_buildings": list(ms.geometry), "osm_buildings": list(osm.geometry)}
    out = {"label": label, "bounds_utm": [round(v, 1) for v in grid.bounds], "crs": crs,
           "lonlat_bbox": [round(v, 5) for v in grid.lonlat_bbox()], "naip": naip_meta,
           "role": imagery.AOIS.get(name, {}).get("role", "custom"),
           "n_reference": {k: len(v) for k, v in refs.items()},
           "n_ms_reference_crossing_edge": int(sum(not g.within(box(*grid.bounds)) for g in refs["ms_buildings"])), "buildings": {}, "fields": {}}
    print(f"[{name}] NAIP {naip_meta['year']} | MS {len(ms)} | OSM {len(osm)} buildings")

    building_res, raw_auto = {}, []
    # ---- buildings -------------------------------------------------------------------------
    bmethods = methods if imagery.AOIS.get(name, {}).get("buildings", True) else set()
    if "auto" in bmethods:
        with timer(f"{name}/sam_auto_buildings"):
            raw_auto = cached(d / "raw_sam_auto_buildings.pkl", args.reuse,
                              lambda: sg.sam_automatic(sam, rgb, T, tile=args.tile, overlap=args.overlap))
            cands = pp.dedupe(pp.filter_candidates(raw_auto, R, T, pp.BUILDING_RULES), 0.5)
        building_res["sam_auto"] = (cands, len(raw_auto))
    if "text" in bmethods and text_model is not None:
        with timer(f"{name}/sam_text_buildings"):
            raw = cached(d / "raw_sam_text_buildings.pkl", args.reuse,
                         lambda: text_model(rgb, T, BUILDING_PROMPT, tile=args.tile, overlap=args.overlap))
            rules = {k: v for k, v in pp.BUILDING_RULES.items() if k != "min_score"}  # DINO scores differ
            cands = pp.dedupe(pp.filter_candidates(raw, R, T, rules), 0.5)
        building_res["sam_text"] = (cands, len(raw))
    if "baseline" in bmethods:
        with timer(f"{name}/baseline_buildings"):
            raw = sg.threshold_baseline(rgb, ndvi, T)
            rules = {k: v for k, v in pp.BUILDING_RULES.items() if k != "min_score"}
            cands = pp.filter_candidates(raw, R, T, rules)
        building_res["baseline"] = (cands, len(raw))

    panel = {}
    for m, (cands, n_raw) in building_res.items():
        # same edge rule as the references: keep objects whose centroid is inside the square
        keep_ids = {id(g) for g in ev.centroid_inside([c["geometry"] for c in cands], grid.bounds)}
        cands = [c for c in cands if id(c["geometry"]) in keep_ids]
        unreg = [c["geometry"] for c in cands]
        for c in cands:
            c["geometry"] = pp.regularize_building(c["geometry"])
        geoms = [c["geometry"] for c in cands]
        out["buildings"][m] = {"n_raw_masks": n_raw}
        if len(refs["ms_buildings"]) >= MIN_REF:
            u = ev.evaluate(unreg, refs["ms_buildings"], shape, T)
            out["buildings"][m]["ms_buildings_before_regularization"] = {k: u[k] for k in ("f1", "mean_matched_iou", "pixel_iou")}
        for rk, ref in refs.items():
            if len(ref) >= MIN_REF:  # too few reference objects make P/R/F1 meaningless
                out["buildings"][m][rk] = ev.evaluate(geoms, ref, shape, T)
                out["buildings"][m][rk].update(ev.block_bootstrap_f1(geoms, ref, grid.bounds))
        save_geojson(cands, crs, RESULTS / f"{name}_buildings_{m}.geojson")
        ms_scores = out["buildings"][m].get("ms_buildings", {})
        out["buildings"][m]["n_pred"] = len(geoms)
        panel[m] = {"geoms": geoms, "f1": ms_scores.get("f1", float("nan"))}
        print(f"  buildings/{m}: n={len(geoms)} " + json.dumps({k: ms_scores.get(k) for k in ("precision", "recall", "f1", "pixel_iou")}))
    if panel and len(refs["ms_buildings"]) >= MIN_REF:
        plots.building_panel(FIGS / f"{name}_buildings.png", rgb, T, [c["geometry"] for c in raw_auto], panel,
                             refs["ms_buildings"], "Microsoft", f"{label}: buildings (NAIP {naip_meta['year']})")

    # ---- fields ------------------------------------------------------------------------------
    field_res, raw_scene = {}, []
    do_fields = imagery.AOIS.get(name, {}).get("fields", True)
    if "auto" in methods and do_fields:
        with timer(f"{name}/sam_auto_fields"):
            raw_scene = cached(d / "raw_sam_auto_fields.pkl", args.reuse, lambda: sg.sam_automatic(
                sam, rgb, T, tile=max(shape), overlap=0, points_per_side=32, pred_iou_thresh=0.84,
                stability_score_thresh=0.88, min_mask_region_area=500))
            field_res["sam_auto"] = pp.dedupe(pp.filter_candidates(raw_scene, R, T, pp.FIELD_RULES), 0.3)
    if "text" in methods and text_model is not None and do_fields:
        with timer(f"{name}/sam_text_fields"):
            raw = cached(d / "raw_sam_text_fields.pkl", args.reuse,
                         lambda: text_model(rgb, T, FIELD_PROMPT, tile=max(shape), overlap=0, max_box_frac=0.6,
                                            box_threshold=0.15, text_threshold=0.15))
            field_res["sam_text"] = pp.dedupe(pp.filter_candidates(raw, R, T, pp.FIELD_RULES), 0.3)
    fpanel = {}
    for m, cands in field_res.items():
        geoms = [c["geometry"] for c in cands]
        r = {"n_fields": len(geoms), "field_area_ha": round(sum(g.area for g in geoms) / 1e4, 2)}
        pred_mask = ev.rasterize(geoms, shape, T)
        if cdl is not None:
            r["vs_cdl_cropland"] = ev.pixel_scores(pred_mask, cdl)
            r["cdl_cropland_ha"] = round(float(cdl.sum()) * grid.res ** 2 / 1e4, 2)
        if len(osm_fields):
            r["vs_osm_farmland"] = ev.evaluate(geoms, list(osm_fields.geometry), shape, T)
        r = {k: ({kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} if isinstance(v, dict) else v)
             for k, v in r.items()}
        out["fields"][m] = r
        save_geojson(cands, crs, RESULTS / f"{name}_fields_{m}.geojson", simplify_m=1.0)
        fpanel[m] = {"geoms": geoms}
        print(f"  fields/{m}: {json.dumps(r)}")
    if fpanel:
        plots.field_panel(FIGS / f"{name}_fields.png", rgb, T, [c["geometry"] for c in raw_scene], fpanel, cdl,
                          list(osm_fields.geometry), f"{label}: fields (NAIP {naip_meta['year']})")
    return out, {"rgb": rgb, "T": T, "ref": refs["ms_buildings"],
                 "text": panel.get("sam_text", {}).get("geoms", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", nargs="*", default=list(imagery.AOIS), help="named AOIs: " + ", ".join(imagery.AOIS))
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="custom lon/lat bbox instead of --aoi")
    ap.add_argument("--name", default="custom", help="name for a --bbox run")
    ap.add_argument("--method", default="all", choices=["all", "auto", "text", "baseline"])
    ap.add_argument("--model-type", default="vit_h", choices=list(sg.SAM_URLS))
    ap.add_argument("--checkpoint", default=None, help="SAM .pth (default data/weights/<official name>)")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--reuse", action="store_true", help="reuse cached raw masks in data/<aoi>/ (re-run filters only)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    methods = {"auto", "text", "baseline"} if args.method == "all" else {args.method}
    ckpt = Path(args.checkpoint or DATA / "weights" / sg.SAM_URLS[args.model_type].rsplit("/", 1)[1])
    if not ckpt.exists():
        raise SystemExit(f"Missing {ckpt}; download it from {sg.SAM_URLS[args.model_type]}")
    for p in (RESULTS, FIGS):
        p.mkdir(exist_ok=True)

    timer = Timer()
    sam, text_model = None, None
    with timer("load_models"):
        if methods & {"auto", "text"}:
            sam = sg.load_sam(str(ckpt), args.model_type, device)
        if "text" in methods:
            try:
                text_model = sg.TextPromptedSAM(sam, device)
            except Exception as e:
                print("Text-prompted SAM unavailable:", e)

    if args.bbox:
        jobs = [(args.name, imagery.grid_from_bbox(*args.bbox), args.name)]
    else:
        jobs = [(a, imagery.grid_from_center(*imagery.AOIS[a]["center"]), imagery.AOIS[a]["label"]) for a in args.aoi]

    metrics_path = RESULTS / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    aois = metrics.setdefault("aois", {})
    keep = {}
    for name, grid, label in jobs:
        aois[name], keep[name] = run_aoi(name, grid, label, methods, sam, text_model, timer, args)

    prev_run = metrics.get("run", {})
    metrics["run"] = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "torch": torch.__version__, "python": platform.python_version(),
        "sam_model": f"SAM {args.model_type} ({ckpt.name})",
        "text_detector": sg.DINO_MODEL if text_model is not None else None,
        "building_prompt": BUILDING_PROMPT, "field_prompt": FIELD_PROMPT,
        "tile_px": args.tile, "overlap_px": args.overlap, "iou_threshold": 0.5,
        "building_rules": pp.BUILDING_RULES, "field_rules": pp.FIELD_RULES,
        "runtime_s": timer.t,
        "runtime_note": "timings of the last full GPU run",
    }
    if args.reuse and prev_run.get("runtime_s"):
        # cached masks: keep the GPU timings of the last full run instead of near-zero reuse timings
        metrics["run"]["runtime_s"] = prev_run["runtime_s"]
        metrics["run"]["runtime_note"] = ("timings of the last full GPU run; metrics were then recomputed "
                                          "from cached masks with --reuse")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    names = [a for a in imagery.AOIS if a in aois and aois[a]["buildings"]]
    if names:
        plots.metrics_chart(FIGS / "metrics.png", aois, names)
    held = aois.get("mixed", {}).get("buildings", {})
    if "mixed" in keep and keep["mixed"]["text"] and {"sam_text", "baseline"} <= set(held):
        # Hero: the held-out area, best held-out method, zoomed so roof-level alignment is visible.
        k, t = keep["mixed"], held["sam_text"]["ms_buildings"]
        lines = [f"Held-out area, Bryan TX: building F1 {t['f1']:.2f} (text-prompted SAM) vs "
                 f"{held['baseline']['ms_buildings']['f1']:.2f} (NDVI baseline)",
                 f"{t['n_pred']} buildings extracted in 1 km² | precision {t['precision']:.2f}, "
                 f"recall {t['recall']:.2f} at IoU ≥ 0.5"]
        plots.hero(FIGS / "hero.png", k["rgb"], k["T"], k["text"], k["ref"], lines,
                   "Buildings and fields from 4-band aerial imagery, no training labels",
                   "Text-prompted SAM + filters", window=(600, 330, 960))
    print(json.dumps(timer.t, indent=2))


if __name__ == "__main__":
    main()
