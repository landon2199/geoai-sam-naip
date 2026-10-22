"""Fast CPU tests: IoU matching, polygon filtering/regularization, raster <-> vector alignment."""
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.affinity import rotate
from shapely.geometry import Polygon, box

from samnaip.evaluate import evaluate, match_objects, object_scores, pixel_scores, polygon_iou, rasterize
from samnaip.imagery import Grid
from samnaip.postprocess import (BUILDING_RULES, dedupe, filter_candidates, mask_to_polygons,
                                 regularize_building, shape_metrics)
from samnaip.segment import tile_windows, touches_inner_edge


def test_polygon_iou_half_overlap():
    a, b = box(0, 0, 2, 1), box(1, 0, 3, 1)
    assert polygon_iou(a, b) == pytest.approx(1 / 3)
    assert polygon_iou(a, a) == pytest.approx(1.0)


def test_matching_is_one_to_one_and_thresholded():
    ref = [box(0, 0, 10, 10), box(20, 0, 30, 10)]
    pred = [box(0, 0, 10, 9), box(0, 0, 10, 8), box(20, 0, 22, 10), box(50, 50, 60, 60)]
    pairs, fp, fn = match_objects(pred, ref, 0.5)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0)]  # the better of two duplicates wins
    assert sorted(fp) == [1, 2, 3] and fn == [1]
    s = object_scores(pred, ref)
    assert s["precision"] == 0.25 and s["recall"] == 0.5
    assert s["f1"] == pytest.approx(2 * 0.25 * 0.5 / 0.75)


def test_empty_inputs():
    s = object_scores([], [box(0, 0, 1, 1)])
    assert s["recall"] == 0 and s["f1"] == 0


def test_shape_metrics_and_filter():
    house = rotate(box(0, 0, 15, 10), 30)
    road = box(0, 0, 200, 8)
    m = shape_metrics(house)
    assert m["area"] == pytest.approx(150)
    assert m["rectangularity"] == pytest.approx(1.0, abs=1e-6)
    assert m["elongation"] == pytest.approx(1.5, rel=1e-6)
    t = from_origin(-10, 30, 0.6, 0.6)
    rasters = {"ndvi": np.full((100, 400), -0.3, np.float32), "brightness": np.full((100, 400), 120, np.float32)}
    kept = filter_candidates([{"geometry": house, "score": 1}, {"geometry": road, "score": 1}],
                             rasters, t, BUILDING_RULES)
    assert len(kept) == 1 and kept[0]["geometry"].equals(house)


def test_filter_rejects_vegetated_and_low_score():
    t = from_origin(0, 30, 0.6, 0.6)
    rasters = {"ndvi": np.full((60, 60), 0.6, np.float32), "brightness": np.full((60, 60), 120, np.float32)}
    cand = [{"geometry": box(3, 3, 15, 15), "score": 1}]
    assert filter_candidates(cand, rasters, t, BUILDING_RULES) == []  # tree crown, not a roof
    rasters["ndvi"][:] = -0.3
    assert len(filter_candidates(cand, rasters, t, BUILDING_RULES)) == 1
    assert filter_candidates([{**cand[0], "score": 0.5}], rasters, t, BUILDING_RULES) == []  # low SAM score


def test_dedupe_keeps_highest_score():
    a = {"geometry": box(0, 0, 10, 10), "score": 0.9}
    b = {"geometry": box(1, 1, 9, 9), "score": 0.5}  # nested lower-score duplicate
    c = {"geometry": box(20, 20, 30, 30), "score": 0.1}
    kept = dedupe([b, a, c])
    assert [k["score"] for k in kept] == [0.9, 0.1]


def test_regularize_rectangle():
    noisy = Polygon([(0, 0), (10, 0.2), (10, 6), (5, 6.3), (0, 6)])
    r = regularize_building(noisy)
    assert len(r.exterior.coords) == 5  # 4 corners + closing point
    assert r.area == pytest.approx(noisy.area, rel=0.1)


def test_raster_vector_roundtrip_alignment():
    """mask -> polygon -> rasterize returns the same pixels in the same place."""
    t = from_origin(500000.0, 3380000.0, 0.6, 0.6)
    mask = np.zeros((50, 60), bool)
    mask[10:20, 30:45] = True
    polys = mask_to_polygons(mask, t)
    assert len(polys) == 1
    minx, miny, maxx, maxy = polys[0].bounds
    assert minx == pytest.approx(500000.0 + 30 * 0.6)
    assert maxy == pytest.approx(3380000.0 - 10 * 0.6)
    assert polys[0].area == pytest.approx(150 * 0.36)
    back = rasterize(polys, mask.shape, t)
    assert (back == mask).all()
    assert pixel_scores(back, mask)["pixel_iou"] == 1.0


def test_grid_geometry():
    g = Grid(x0=750000.0, y1=3390000.0, width=100, height=50)
    assert g.bounds == (750000.0, 3390000.0 - 30.0, 750060.0, 3390000.0)
    w, s, e, n = g.lonlat_bbox()
    assert -97.0 < w < e < -95.0 and 30.0 < s < n < 31.0


def test_evaluate_perfect():
    t = from_origin(0, 100, 1, 1)
    ref = [box(10, 10, 20, 20), box(40, 40, 60, 50)]
    r = evaluate(ref, ref, (100, 100), t)
    assert r["f1"] == 1.0 and r["pixel_iou"] == 1.0 and r["area_ratio"] == 1.0


def test_tiles_cover_image_and_edge_rule():
    wins = tile_windows(1000, 700, 512, 128)
    cover = np.zeros((1000, 700), bool)
    for r0, c0, r1, c1 in wins:
        assert r1 - r0 <= 512 and c1 - c0 <= 512
        cover[r0:r1, c0:c1] = True
    assert cover.all()
    m = np.zeros((512, 512), bool)
    m[0:5, 100:110] = True
    assert touches_inner_edge(m, (100, 0, 612, 512), (1000, 700))  # top edge is interior
    assert not touches_inner_edge(m, (0, 0, 512, 512), (1000, 700))  # top edge is the image edge


def test_centroid_inside_and_bootstrap():
    from samnaip.evaluate import block_bootstrap_f1, centroid_inside

    bounds = (0, 0, 100, 100)
    geoms = [box(1, 1, 5, 5), box(98, 50, 104, 54), box(-3, 10, 1, 12)]  # centroids x=3, 101, -1
    assert centroid_inside(geoms, bounds) == [geoms[0]]
    ref = [box(x, y, x + 5, y + 5) for x in range(5, 95, 20) for y in range(5, 95, 20)]
    ci = block_bootstrap_f1(ref, ref, bounds, n_boot=200)["f1_ci95"]
    assert ci == [1.0, 1.0]
    half = ref[::2]
    ci = block_bootstrap_f1(half, ref, bounds, n_boot=500)["f1_ci95"]
    f1 = object_scores(half, ref)["f1"]
    assert ci[0] <= f1 <= ci[1] and ci[0] < ci[1]
