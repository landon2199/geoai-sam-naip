"""Object- and pixel-level accuracy of extracted polygons against reference polygons."""
from __future__ import annotations

import numpy as np
from shapely.strtree import STRtree


def polygon_iou(a, b) -> float:
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def match_objects(pred: list, ref: list, iou_threshold: float = 0.5):
    """Greedy one-to-one matching by descending IoU. Returns (pairs [(i, j, iou)], unmatched_pred, unmatched_ref)."""
    if not pred or not ref:
        return [], list(range(len(pred))), list(range(len(ref)))
    tree = STRtree(ref)
    cand = []
    for i, p in enumerate(pred):
        for j in tree.query(p):
            iou = polygon_iou(p, ref[int(j)])
            if iou >= iou_threshold:
                cand.append((iou, i, int(j)))
    cand.sort(reverse=True)
    used_p, used_r, pairs = set(), set(), []
    for iou, i, j in cand:
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        pairs.append((i, j, iou))
    return pairs, [i for i in range(len(pred)) if i not in used_p], [j for j in range(len(ref)) if j not in used_r]


def rasterize(geoms: list, shape: tuple[int, int], transform) -> np.ndarray:
    from rasterio import features

    if not geoms:
        return np.zeros(shape, bool)
    return features.rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, dtype=np.uint8).astype(bool)


def pixel_scores(pred_mask: np.ndarray, ref_mask: np.ndarray) -> dict:
    tp = np.logical_and(pred_mask, ref_mask).sum()
    fp = np.logical_and(pred_mask, ~ref_mask).sum()
    fn = np.logical_and(~pred_mask, ref_mask).sum()
    return {
        "pixel_iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0,
        "pixel_precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "pixel_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
    }


def object_scores(pred: list, ref: list, iou_threshold: float = 0.5) -> dict:
    pairs, fp, fn = match_objects(pred, ref, iou_threshold)
    tp = len(pairs)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(ref) if ref else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "n_pred": len(pred), "n_ref": len(ref), "tp": tp, "fp": len(fp), "fn": len(fn),
        "precision": p, "recall": r, "f1": f1,
        "mean_matched_iou": float(np.mean([x[2] for x in pairs])) if pairs else 0.0,
    }


def evaluate(pred: list, ref: list, shape, transform, iou_threshold: float = 0.5) -> dict:
    """Object P/R/F1 at IoU >= threshold, pixel IoU, and total-area agreement (pred / ref)."""
    out = object_scores(pred, ref, iou_threshold)
    out.update(pixel_scores(rasterize(pred, shape, transform), rasterize(ref, shape, transform)))
    pa, ra = sum(g.area for g in pred), sum(g.area for g in ref)
    out.update({"pred_area_m2": pa, "ref_area_m2": ra, "area_ratio": pa / ra if ra else float("nan")})
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def centroid_inside(geoms: list, bounds) -> list:
    """Keep geometries whose centroid lies inside `bounds` (minx, miny, maxx, maxy).
    Applied to both predictions and references so edge objects are treated the same way."""
    minx, miny, maxx, maxy = bounds
    out = []
    for g in geoms:
        c = g.centroid
        if minx <= c.x < maxx and miny <= c.y < maxy:
            out.append(g)
    return out


def block_bootstrap_f1(pred: list, ref: list, bounds, n_blocks: int = 5, n_boot: int = 2000,
                       iou_threshold: float = 0.5, seed: int = 0) -> dict:
    """95% CI for object F1 by spatial block bootstrap.

    The square is cut into n_blocks x n_blocks sub-blocks (200 m for a 1 km square). Matching is done
    once on the full set; each true positive is assigned to its predicted polygon's block, each
    prediction and reference to the block holding its centroid. Blocks are resampled with replacement
    and F1 = 2 TP / (n_pred + n_ref) is recomputed. Resampling blocks rather than single buildings keeps
    neighbouring buildings (which share errors: canopy, roof type, reference vintage) together."""
    minx, miny, maxx, maxy = bounds
    bw, bh = (maxx - minx) / n_blocks, (maxy - miny) / n_blocks

    def block(g):
        c = g.centroid
        i = min(max(int((c.x - minx) // bw), 0), n_blocks - 1)
        j = min(max(int((c.y - miny) // bh), 0), n_blocks - 1)
        return j * n_blocks + i

    nb = n_blocks * n_blocks
    tp, npred, nref = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    pairs, _, _ = match_objects(pred, ref, iou_threshold)
    for i, _, _ in pairs:
        tp[block(pred[i])] += 1
    for g in pred:
        npred[block(g)] += 1
    for g in ref:
        nref[block(g)] += 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(n_boot, nb))
    t, p, r = tp[idx].sum(1), npred[idx].sum(1), nref[idx].sum(1)
    f1 = np.where(p + r > 0, 2 * t / np.maximum(p + r, 1), 0.0)
    lo, hi = np.percentile(f1, [2.5, 97.5])
    return {"f1_ci95": [round(float(lo), 4), round(float(hi), 4)], "method": f"spatial block bootstrap, {n_blocks}x{n_blocks} blocks, {n_boot} resamples"}
