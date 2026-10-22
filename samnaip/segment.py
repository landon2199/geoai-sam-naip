"""Segmentation methods: SAM automatic masks, text-prompted SAM (Grounding DINO boxes), and an
NDVI/brightness threshold baseline. All return candidate dicts {'geometry': Polygon, 'score': float}
in map coordinates, ready for `postprocess.filter_candidates`."""
from __future__ import annotations

import numpy as np
from rasterio.transform import Affine

from .postprocess import mask_to_polygons

SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
DINO_MODEL = "IDEA-Research/grounding-dino-base"


def tile_windows(height: int, width: int, size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """(row0, col0, row1, col1) windows covering the image; the last row/col is snapped to the edge."""
    if size >= height and size >= width:
        return [(0, 0, height, width)]
    step = size - overlap

    def starts(n):
        if size >= n:
            return [0]
        s = list(range(0, n - size, step))
        s.append(n - size)
        return sorted(set(s))

    return [(r, c, min(r + size, height), min(c + size, width)) for r in starts(height) for c in starts(width)]


def touches_inner_edge(mask: np.ndarray, win, shape, margin: int = 2) -> bool:
    """True if a tile mask touches a tile edge that is not also the image edge (likely cut off)."""
    r0, c0, r1, c1 = win
    H, W = shape
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if r0 > 0 and rows[:margin].any():
        return True
    if r1 < H and rows[-margin:].any():
        return True
    if c0 > 0 and cols[:margin].any():
        return True
    if c1 < W and cols[-margin:].any():
        return True
    return False


def load_sam(checkpoint: str, model_type: str = "vit_h", device: str = "cuda"):
    from segment_anything import sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return sam.to(device).eval()


def _masks_to_cands(masks, scores, win, transform, shape, drop_cut: bool) -> list[dict]:
    r0, c0 = win[0], win[1]
    t = transform @ Affine.translation(c0, r0)
    out = []
    for m, s in zip(masks, scores):
        if drop_cut and touches_inner_edge(m, win, shape):
            continue
        for p in mask_to_polygons(m, t):
            out.append({"geometry": p, "score": float(s)})
    return out


def sam_automatic(sam, rgb: np.ndarray, transform, tile: int = 512, overlap: int = 128, **kwargs) -> list[dict]:
    """SAM automatic mask generation, tiled. Small tiles are upsampled by SAM to 1024 px, which
    is what lets it see 10-20 m houses in 0.6 m imagery; use tile >= image size for fields."""
    from segment_anything import SamAutomaticMaskGenerator

    params = dict(points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.90,
                  crop_n_layers=0, min_mask_region_area=40, points_per_batch=128)
    params.update(kwargs)
    gen = SamAutomaticMaskGenerator(sam, **params)
    H, W = rgb.shape[:2]
    cands = []
    for win in tile_windows(H, W, tile, overlap):
        r0, c0, r1, c1 = win
        chip = np.ascontiguousarray(rgb[r0:r1, c0:c1])
        res = gen.generate(chip)
        masks = [r["segmentation"] for r in res]
        scores = [r["predicted_iou"] * r["stability_score"] for r in res]
        cands += _masks_to_cands(masks, scores, win, transform, (H, W), drop_cut=True)
    return cands


class TextPromptedSAM:
    """Grounding DINO (open-vocabulary detector, Hugging Face `transformers`) gives boxes for a text
    prompt; SAM turns each box into a mask. Same idea as samgeo's LangSAM and ArcGIS 'Text SAM'."""

    def __init__(self, sam, device: str = "cuda", model_id: str = DINO_MODEL):
        from segment_anything import SamPredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.predictor = SamPredictor(sam)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    def detect(self, chip: np.ndarray, prompt: str, box_threshold: float, text_threshold: float):
        import torch
        from PIL import Image

        img = Image.fromarray(chip)
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.dino(**inputs)
        kw = dict(text_threshold=text_threshold, target_sizes=[img.size[::-1]])
        try:
            res = self.processor.post_process_grounded_object_detection(out, inputs.input_ids, threshold=box_threshold, **kw)
        except TypeError:  # older transformers
            res = self.processor.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=box_threshold, **kw)
        r = res[0]
        return r["boxes"].detach().cpu().numpy(), r["scores"].detach().cpu().numpy()

    def __call__(self, rgb: np.ndarray, transform, prompt: str, tile: int = 512, overlap: int = 128,
                 box_threshold: float = 0.25, text_threshold: float = 0.20, max_box_frac: float = 0.5) -> list[dict]:
        import torch

        H, W = rgb.shape[:2]
        cands = []
        for win in tile_windows(H, W, tile, overlap):
            r0, c0, r1, c1 = win
            chip = np.ascontiguousarray(rgb[r0:r1, c0:c1])
            boxes, scores = self.detect(chip, prompt, box_threshold, text_threshold)
            if len(boxes) == 0:
                continue
            # Grounding DINO often returns one box for "the whole scene"; drop those.
            frac = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / float(chip.shape[0] * chip.shape[1])
            keep = frac <= max_box_frac
            boxes, scores = boxes[keep], scores[keep]
            if len(boxes) == 0:
                continue
            self.predictor.set_image(chip)
            tb = self.predictor.transform.apply_boxes_torch(torch.as_tensor(boxes, device=self.device), chip.shape[:2])
            masks = []
            for i in range(0, len(tb), 64):
                m, _, _ = self.predictor.predict_torch(point_coords=None, point_labels=None,
                                                       boxes=tb[i:i + 64], multimask_output=False)
                masks.append(m[:, 0].cpu().numpy())
            masks = np.concatenate(masks)
            cands += _masks_to_cands(masks, scores, win, transform, (H, W), drop_cut=True)
        return cands


def threshold_baseline(rgb: np.ndarray, ndvi: np.ndarray, transform, max_ndvi: float = -0.15,
                       min_brightness: float = 90.0, max_brightness: float = 165.0, open_px: int = 2) -> list[dict]:
    """Classic non-learned baseline: non-vegetated pixels in a roof-like brightness band ->
    morphological opening/closing -> connected components. Rooftops pass, but so do driveways,
    parking lots and bare soil, and touching roofs merge into one object."""
    from scipy import ndimage

    bright = rgb.astype(np.float32).mean(-1)
    m = (ndvi < max_ndvi) & (bright > min_brightness) & (bright < max_brightness)
    st = np.ones((2 * open_px + 1, 2 * open_px + 1), bool)
    m = ndimage.binary_closing(ndimage.binary_opening(m, structure=st), structure=np.ones((3, 3), bool))
    return [{"geometry": p, "score": 1.0} for p in mask_to_polygons(m, transform)]
