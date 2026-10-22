"""Figures: per-AOI before/after panels, metrics chart, and the 16:9 hero image."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon

# Categorical slots 1-3 of the reference palette (fixed order: one colour per method).
METHOD_COLORS = {"sam_auto": "#2a78d6", "sam_text": "#eb6834", "baseline": "#1baf7a"}
METHOD_NAMES = {"sam_auto": "SAM automatic + filters", "sam_text": "Text-prompted SAM", "baseline": "NDVI/brightness baseline"}
TP, FP, FN = "#22c55e", "#ef4444", "#facc15"
INK, MUTED = "#0b0b0b", "#52514e"


def stretch(rgb: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    out = np.zeros(rgb.shape, np.float32)
    for b in range(3):
        a, z = np.percentile(rgb[..., b], [lo, hi])
        out[..., b] = np.clip((rgb[..., b] - a) / max(z - a, 1), 0, 1)
    return out


def _to_px(geom, transform):
    inv = ~transform
    xs, ys = geom.exterior.coords.xy
    cols, rows = zip(*[inv * (x, y) for x, y in zip(xs, ys)])
    return np.column_stack([cols, rows])


def draw_polys(ax, geoms, transform, face="none", edge="white", lw=0.8, alpha=1.0, z=3):
    patches = [MplPolygon(_to_px(g, transform), closed=True) for g in geoms if g is not None and not g.is_empty]
    pc = PatchCollection(patches, facecolor=face, edgecolor=edge, linewidths=lw, alpha=alpha, zorder=z)
    ax.add_collection(pc, autolim=False)


def mask_overlay(shape, geoms, transform, seed=0):
    """Random-colour RGBA image of polygons (the classic SAM 'everything' look)."""
    from rasterio import features

    rng = np.random.default_rng(seed)
    if not geoms:
        return np.zeros((*shape, 4), np.float32)
    order = sorted(range(len(geoms)), key=lambda i: -geoms[i].area)  # small masks drawn on top
    lab = features.rasterize([(geoms[i], k + 1) for k, i in enumerate(order)], out_shape=shape,
                             transform=transform, dtype=np.int32)
    cols = np.vstack([[0, 0, 0, 0], np.column_stack([rng.uniform(0.15, 1, (len(geoms), 3)), np.full(len(geoms), 0.55)])])
    return cols[lab].astype(np.float32)


def _clean(ax, title):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def match_split(pred, ref, iou=0.5):
    from .evaluate import match_objects

    pairs, fp, fn = match_objects(pred, ref, iou)
    return [pred[i] for i, _, _ in pairs], [pred[i] for i in fp], [ref[j] for j in fn]


def building_panel(path, rgb, transform, raw_masks, results: dict, ref, ref_name, title):
    """NAIP | SAM raw masks | per-method TP/FP/FN maps."""
    img = stretch(rgb)
    methods = [m for m in ("sam_auto", "sam_text", "baseline") if m in results]
    n = 2 + len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), dpi=130)
    axes[0].imshow(img)
    _clean(axes[0], "NAIP (true colour)")
    axes[1].imshow(img)
    axes[1].imshow(mask_overlay(rgb.shape[:2], raw_masks, transform))
    _clean(axes[1], f"SAM automatic masks ({len(raw_masks):,})")
    for ax, m in zip(axes[2:], methods):
        tp, fp, fn = match_split(results[m]["geoms"], ref)
        ax.imshow(img * 0.75)
        draw_polys(ax, fn, transform, edge=FN, lw=0.9)
        draw_polys(ax, fp, transform, face=FP, edge=FP, lw=0.6, alpha=0.55)
        draw_polys(ax, tp, transform, face=TP, edge=TP, lw=0.6, alpha=0.6)
        _clean(ax, f"{METHOD_NAMES[m]}  F1 {results[m]['f1']:.2f}")
    fig.legend(handles=[Patch(color=TP, label="match (IoU >= 0.5)"), Patch(color=FP, label="false positive"),
                        Patch(facecolor="none", edgecolor=FN, label=f"missed {ref_name} footprint")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def field_panel(path, rgb, transform, raw_masks, fields: dict, cdl_mask, osm_fields, title):
    img = stretch(rgb)
    methods = [m for m in ("sam_auto", "sam_text") if m in fields]
    n = 2 + len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), dpi=130)
    axes[0].imshow(img)
    _clean(axes[0], "NAIP (true colour)")
    axes[1].imshow(img)
    axes[1].imshow(mask_overlay(rgb.shape[:2], raw_masks, transform, seed=3))
    _clean(axes[1], f"SAM automatic masks, whole scene ({len(raw_masks):,})")
    for ax, m in zip(axes[2:], methods):
        ax.imshow(img * 0.8)
        if cdl_mask is not None:
            ax.contour(cdl_mask.astype(float), levels=[0.5], colors=[FN], linewidths=0.8)
        draw_polys(ax, osm_fields, transform, edge="white", lw=0.8)
        draw_polys(ax, fields[m]["geoms"], transform, face=METHOD_COLORS[m], edge=METHOD_COLORS[m], lw=1.0, alpha=0.45)
        _clean(ax, f"{METHOD_NAMES[m]}: {len(fields[m]['geoms'])} fields")
    handles = [Patch(color=METHOD_COLORS[m], alpha=0.6, label=f"extracted field ({METHOD_NAMES[m]})") for m in methods]
    handles.append(Patch(facecolor="none", edgecolor=FN, label="USDA CDL cropland edge (30 m)"))
    if osm_fields:
        handles.append(Patch(facecolor="none", edgecolor="white", label="OSM farmland/meadow"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def metrics_chart(path, metrics: dict, aois: list[str], ref_key: str = "ms_buildings"):
    """Grouped bars: object F1 (IoU >= 0.5) per method per AOI."""
    methods = ["sam_auto", "sam_text", "baseline"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    w = 0.26
    x = np.arange(len(aois))
    for k, m in enumerate(methods):
        vals = [metrics[a]["buildings"].get(m, {}).get(ref_key, {}).get("f1", np.nan) for a in aois]
        bars = ax.bar(x + (k - 1) * w, vals, w - 0.02, color=METHOD_COLORS[m], label=METHOD_NAMES[m], zorder=2)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    roles = {"dev": "filter rules tuned here", "held-out": "held out"}
    ax.set_xticks(x, [f"{a.capitalize()}\n({roles.get(metrics[a].get('role'), 'custom')})" for a in aois])
    ax.set_ylabel("Building F1 (IoU >= 0.5)", color=MUTED)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e5e5e2", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Object-level F1 against Microsoft Building Footprints", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def hero(path, rgb, transform, extracted, reference, stat_lines: list[str], subtitle: str,
         right_label: str, window: tuple[int, int, int] | None = None):
    """1600x900 split view of one window: plain NAIP left | extracted footprints (colour fill) with
    reference footprints (thin white outlines) right, a labelled divider, text on dark panels.
    `window` = (row0, col0, width_px) of a 16:9 crop; default is the centred full-width crop."""
    from matplotlib.patches import FancyBboxPatch
    from rasterio import features

    H, W = rgb.shape[:2]
    r0, c0, cw = window or ((H - int(W * 9 / 16)) // 2, 0, W)
    ch = int(round(cw * 9 / 16))
    img = stretch(rgb)[r0:r0 + ch, c0:c0 + cw]
    sub_t = transform @ Affine.translation(c0, r0)
    rng = np.random.default_rng(11)
    pal = np.array([[0.16, 0.47, 0.84], [0.92, 0.41, 0.20], [0.11, 0.69, 0.48], [0.93, 0.63, 0.0],
                    [0.91, 0.48, 0.64], [0.38, 0.31, 0.84]])
    lab = features.rasterize([(g, k + 1) for k, g in enumerate(extracted)], out_shape=(ch, cw),
                             transform=sub_t, dtype=np.int32) if extracted else np.zeros((ch, cw), np.int32)
    cols = np.vstack([[0, 0, 0], pal[rng.integers(0, len(pal), len(extracted))]])
    right = np.where(lab[..., None] > 0, 0.35 * img + 0.65 * cols[lab], img * 0.55)
    split = cw // 2
    comp = img.copy()
    comp[:, split:] = right[:, split:]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(comp, extent=(0, cw, ch, 0), interpolation="lanczos")
    # reference outlines on the right half only (clipped in map coordinates)
    from shapely.geometry import box as sbox

    x_mid = (sub_t * (split, 0))[0]
    x_max, y_top = sub_t * (cw, 0)
    y_bot = (sub_t * (0, ch))[1]
    half = sbox(x_mid, y_bot, x_max, y_top)
    clipped = [g.intersection(half) for g in reference if g.intersects(half)]
    clipped = [p for g in clipped for p in (g.geoms if hasattr(g, "geoms") else [g]) if p.geom_type == "Polygon"]
    draw_polys(ax, clipped, sub_t, edge="white", lw=1.1, z=4)
    ax.axvline(split, color="white", lw=2.5, zorder=5)
    ax.set_xlim(0, cw)
    ax.set_ylim(ch, 0)
    ax.axis("off")

    def panel(x, y, w, h):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.01",
                                          transform=fig.transFigure, facecolor=(0, 0, 0, 0.62), edgecolor="none"))

    panel(0.025, 0.775, 0.56, 0.19)
    fig.text(0.04, 0.885, "Segment Anything on NAIP", fontsize=40, color="white", weight="bold")
    fig.text(0.042, 0.815, subtitle, fontsize=17, color="#e5e5e2")
    panel(0.025, 0.04, 0.70, 0.035 + 0.05 * len(stat_lines))
    for k, s in enumerate(stat_lines):
        fig.text(0.04, 0.06 + 0.05 * (len(stat_lines) - 1 - k), s, fontsize=17, color="white")
    xs = split / cw
    for x, txt, ha in ((xs - 0.012, "NAIP 0.6 m (input)", "right"), (xs + 0.012, right_label, "left")):
        fig.text(x, 0.735, txt, fontsize=15, color="white", ha=ha, va="center", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=(0, 0, 0, 0.62), edgecolor="none"))
    fig.text(xs + 0.012, 0.69, "white outline = Microsoft footprint", fontsize=12, color="white", ha="left",
             va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor=(0, 0, 0, 0.62), edgecolor="none"))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _shadow():
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=4, foreground=(0, 0, 0, 0.75))]
