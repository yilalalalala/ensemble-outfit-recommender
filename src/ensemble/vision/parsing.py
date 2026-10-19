"""Garment boxes from human parsing (clothes segmentation), not from the language model.

The vision LLM (Claude on the hosted app, Qwen-VL locally) is good at *what* someone is wearing (category, colour,
a product-style description) and loose about *where*: Claude's boxes came back shrunk and shifted. Industry
pipelines split the two jobs: a VLM or tagger names the items and a segmentation model localises them. Here a
SegFormer-B2 fine-tuned for clothes parsing on ATR (``mattmdjaga/segformer_b2_clothes``; labels: hat, sunglasses,
upper-clothes, skirt, pants, dress, belt, left/right shoe, bag, scarf, plus hair/face/limbs) labels every pixel,
and each detected item takes the bounding box of its class's mask:

- the item's category maps to parsing classes (a top → upper-clothes; bottoms → pants or skirt; shoes → both
  shoes; a dress → dress, else upper-clothes + skirt/pants);
- the mask is split into connected regions and only the regions near the LLM's own (loose) box are kept, so in a
  photo with several people the box stays on the person the LLM described;
- categories the parser has no class for (jewellery, socks) keep the LLM's box.

Measured on DeepFashion2 validation photos with ground-truth boxes: see ``eval_boxes`` (D-051).
Model licence: listed as "other" on the model card; trained on the ATR research dataset. Fine for this
non-commercial portfolio; a commercial deployment needs a licensed or self-trained parser.
"""
from __future__ import annotations

import threading

import numpy as np
from PIL import Image

MODEL = "mattmdjaga/segformer_b2_clothes"
CLASSES = {"Hat": 1, "Sunglasses": 3, "Upper-clothes": 4, "Skirt": 5, "Pants": 6, "Dress": 7, "Belt": 8,
           "Left-shoe": 9, "Right-shoe": 10, "Bag": 16, "Scarf": 17}
# Detector category -> parsing classes, in order of preference (the first that has pixels wins).
FOR_CATEGORY = {
    "top": [["Upper-clothes"]], "outerwear": [["Upper-clothes"]], "knitwear": [["Upper-clothes"]],
    "bottom": [["Pants", "Skirt"]],
    "dress": [["Dress"], ["Upper-clothes", "Skirt", "Pants"]], "jumpsuit": [["Dress"], ["Upper-clothes", "Pants"]],
    "shoes": [["Left-shoe", "Right-shoe"]], "bag": [["Bag"]], "sunglasses": [["Sunglasses"]], "hat": [["Hat"]],
    "belt": [["Belt"]], "scarf": [["Scarf"]],
}
SIZE = 512                 # the parser's input resolution
GRID = 160                 # connected regions are found on a coarse grid (long side), which is plenty for boxes
MIN_SHARE = 0.0004         # a region smaller than this share of the photo is noise

_model = None
_lock = threading.Lock()   # loading and inference are not thread-safe (see vision/clip.py)


def load():
    global _model
    with _lock:
        if _model is None:
            from transformers import AutoModelForSemanticSegmentation
            _model = AutoModelForSemanticSegmentation.from_pretrained(MODEL).eval()
    return _model


def segment(img: Image.Image) -> np.ndarray:
    """Per-pixel parsing labels at the image's own size."""
    import torch
    model = load()
    x = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    with _lock, torch.no_grad():
        logits = model(pixel_values=torch.from_numpy(x.transpose(2, 0, 1)[None].copy())).logits
        up = torch.nn.functional.interpolate(logits, size=(img.height, img.width), mode="bilinear", align_corners=False)
    return up.argmax(1)[0].numpy().astype(np.uint8)


def regions(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Connected regions (4-neighbour) of a boolean mask: [(x0, y0, x1, y1, pixels)] in mask coordinates."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack, xs, ys = [(y, x)], [], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            xs.append(cx); ys.append(cy)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(xs)))
    return out


def _near(r, box, margin) -> bool:
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return r[0] < x1 + mx and r[2] > x0 - mx and r[1] < y1 + my and r[3] > y0 - my


def box_for(labels: np.ndarray, category: str, llm_box: list[int] | None) -> list[int] | None:
    """The parsed box (pixels of ``labels``) for one detected item, or None when the parser has nothing for it."""
    h, w = labels.shape
    s = GRID / max(h, w)
    gh, gw = max(1, round(h * s)), max(1, round(w * s))
    # nearest-neighbour downsample of the label map to the coarse grid
    small = labels[(np.arange(gh) / s).astype(int).clip(0, h - 1)][:, (np.arange(gw) / s).astype(int).clip(0, w - 1)]
    near = [v * s for v in llm_box] if llm_box else None
    for names in FOR_CATEGORY.get(category, []):
        mask = np.isin(small, [CLASSES[n] for n in names])
        rs = [r for r in regions(mask) if r[4] >= MIN_SHARE * gh * gw]
        if not rs:
            continue
        if near:   # keep the regions around the LLM's box (the person it described); else the biggest region
            kept = [r for r in rs if _near(r, near, 0.5)] or [max(rs, key=lambda r: r[4])]
        else:
            kept = [max(rs, key=lambda r: r[4])] if category != "shoes" else rs
        x0, y0 = min(r[0] for r in kept), min(r[1] for r in kept)
        x1, y1 = max(r[2] for r in kept), max(r[3] for r in kept)
        return [int(x0 / s), int(y0 / s), min(w, int(np.ceil(x1 / s))), min(h, int(np.ceil(y1 / s)))]
    return None


def _iou(a, b) -> float:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def refine(items: list[dict], img: Image.Image, mode: str = "replace") -> list[dict]:
    """Garment boxes from the parse (boxes in ``img`` pixels).

    ``replace``: every item the parser has a class for takes its parsed box (for an LLM that places boxes loosely).
    ``fill``: the LLM's boxes are kept and only a missing box is filled from the parse — for a grounded VLM whose
    boxes beat the parser's (replacing Qwen-VL's boxes where the two disagree lowered mean IoU from 0.88 to 0.80)."""
    if not any(it["category"] in FOR_CATEGORY for it in items):
        return items
    labels = segment(img)
    out = []
    for it in items:
        b = box_for(labels, it["category"], it.get("box"))
        own = it.get("box")
        use = b if b and (mode == "replace" or not own) else own
        out.append({**it, "box": use})
    return out
