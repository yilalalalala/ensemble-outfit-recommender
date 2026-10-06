"""Visual search and outfit-photo flows over the bundle's visual index (PyTorch, loaded lazily once).

Uploads are validated before any model sees them (``decode_upload``): media type, byte size,
pixel count (decompression-bomb guard), side lengths, and a full decode. Images are held in
memory for the request only and never written to disk or logged.

Degraded modes are explicit and deterministic:

- shop-side adapter absent -> raw FashionCLIP crop search, ``mode: "crop"`` (D-021's runner-up);
- FashionCLIP cannot load -> ``visual_model_unavailable`` (503); Complete the Look by article id
  is unaffected;
- garment detector (local VLM) unavailable or slow -> the outfit endpoint needs the client to
  send the garments it tapped; without them it answers ``detector_unavailable`` (503).

Model work runs on one worker thread with a timeout, so a slow model cannot pile up requests.
"""
from __future__ import annotations

import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout

import numpy as np

from ensemble.serving.runtime import RequestError

LIMITS = {"max_bytes": 8 * 1024 * 1024, "max_pixels": 40_000_000, "min_side": 32, "max_side": 8000,
          "media_types": ("image/jpeg", "image/png", "image/webp"), "formats": ("JPEG", "PNG", "WEBP")}

# Visual-search category -> (slot, allowed product_type_name values or None). Same table as M7a.
CATEGORIES = {
    "top": ("upper", None), "outerwear": ("upper", None), "knitwear": ("upper", None),
    "bottom": ("lower", None), "dress": ("full", None), "jumpsuit": ("full", None), "shoes": ("shoes", None),
    "bag": ("accessories", {"Bag", "Backpack", "Wallet"}),
    "jewellery": ("accessories", {"Earring", "Necklace", "Ring", "Bracelet"}),
    "hat": ("accessories", {"Hat/beanie", "Cap/peaked", "Beanie", "Hat/brim", "Cap"}),
    "scarf": ("accessories", {"Scarf"}), "belt": ("accessories", {"Belt"}),
    "sunglasses": ("accessories", {"Sunglasses"}), "socks": ("socks", None),
}
MAIN_ORDER = ["dress", "jumpsuit", "outerwear", "top", "knitwear", "bottom"]


def decode_upload(data: bytes, media_type: str | None):
    from PIL import Image, ImageOps
    if media_type not in LIMITS["media_types"]:
        raise RequestError("unsupported_media_type", f"upload must be one of {LIMITS['media_types']}", 415)
    if not data:
        raise RequestError("empty_upload", "the uploaded file is empty", 400)
    if len(data) > LIMITS["max_bytes"]:
        raise RequestError("upload_too_large", f"uploads are limited to {LIMITS['max_bytes'] // 2**20} MB", 413)
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt, (w, h) = probe.format, probe.size
            if fmt not in LIMITS["formats"]:
                raise RequestError("unsupported_media_type", f"decoded format {fmt} is not allowed", 415)
            if w * h > LIMITS["max_pixels"]:
                raise RequestError("image_too_large", f"images are limited to {LIMITS['max_pixels']:,} pixels", 413)
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except RequestError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise RequestError("malformed_image", f"the upload could not be decoded as an image ({type(e).__name__})",
                           400) from None
    w, h = img.size
    if min(w, h) < LIMITS["min_side"] or max(w, h) > LIMITS["max_side"]:
        raise RequestError("image_dimensions", f"image sides must be between {LIMITS['min_side']} and "
                                               f"{LIMITS['max_side']} pixels", 422)
    return ImageOps.exif_transpose(img).convert("RGB")


def parse_box(box: str | None, w: int, h: int) -> list[int] | None:
    """``x1,y1,x2,y2`` as fractions of the displayed image -> pixel box (None if absent or degenerate)."""
    if not box:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box.split(","))
    except ValueError:
        raise RequestError("invalid_box", "box must be four comma-separated fractions x1,y1,x2,y2") from None
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise RequestError("invalid_box", "box fractions must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    b = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
    return b if b[2] - b[0] >= 10 and b[3] - b[1] >= 10 else None


def gaps(categories: set[str]) -> list[str]:
    """Core outfit slots absent from the photo. A complete outfit yields an empty list."""
    missing = []
    if not categories & {"dress", "jumpsuit"}:
        if not categories & {"top", "outerwear", "knitwear"}:
            missing.append("upper")
        if "bottom" not in categories:
            missing.append("lower")
    if "shoes" not in categories:
        missing.append("shoes")
    if not categories & {"bag", "jewellery", "hat", "scarf", "belt", "sunglasses"}:
        missing.append("accessories")
    return missing


class VisualSearch:
    """FashionCLIP (+ DeepFashion2 user-side adapter) query encoder over the bundle's visual index."""

    def __init__(self, bundle, timeout_s: float = 20.0):
        self.bundle, self.timeout_s = bundle, timeout_s
        self.state, self.reason = "unloaded", None
        self.load_seconds = None
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visual")
        self._adapter = None
        self._mask_cache: dict = {}

    def _load(self) -> None:
        with self._lock:
            if self.state != "unloaded":
                return
            t0 = time.time()
            if self.bundle.visual is None:
                self.state, self.reason = "unavailable", "bundle has no visual index"
                return
            try:
                from ensemble.vision import clip
                clip.load()
            except Exception as e:  # noqa: BLE001 - recorded as a degraded state and reported, not swallowed
                self.state, self.reason = "unavailable", f"FashionCLIP failed to load: {type(e).__name__}: {e}"[:300]
                return
            if self.bundle.visual.get("adapted") is not None:
                try:
                    from ensemble.vision.outfit import adapter
                    self._adapter = adapter("crop")
                except Exception as e:  # noqa: BLE001 - degraded to raw crop search, reported in responses
                    self._adapter, self.reason = None, f"adapter failed to load: {type(e).__name__}"
            self.state = "ready"
            self.load_seconds = round(time.time() - t0, 2)

    @property
    def mode(self) -> str:
        return "crop_adapter" if self._adapter is not None else "crop"

    def _encode(self, img, box):
        import torch

        from ensemble.vision import clip
        crop = img.crop(box) if box else img
        v = clip.embed_images([crop])[0]
        if self._adapter is not None:
            with torch.no_grad():
                v = self._adapter.user(torch.as_tensor(v[None], dtype=torch.float32)).numpy()[0]
        return v.astype(np.float32)

    def query_vector(self, img, box):
        self._load()
        if self.state != "ready":
            raise RequestError("visual_model_unavailable", self.reason or "visual model unavailable", 503)
        fut = self._pool.submit(self._encode, img, box)
        try:
            return fut.result(timeout=self.timeout_s)
        except FutTimeout:
            raise RequestError("visual_timeout", f"image encoding exceeded {self.timeout_s:.0f} s", 504) from None

    def search(self, img, category: str, box, k: int, unavailable: set[int]) -> dict:
        if category not in CATEGORIES:
            raise RequestError("invalid_category", f"category must be one of {sorted(CATEGORIES)}")
        if not 1 <= k <= 24:
            raise RequestError("invalid_k", "k must be between 1 and 24")
        q = self.query_vector(img, box)
        vis, cat = self.bundle.visual, self.bundle.catalog
        index = vis["adapted"] if (self._adapter is not None and vis["adapted"] is not None) else vis["emb"]
        slot, types = CATEGORIES[category]
        ids = vis["ids"]
        key = (slot, tuple(sorted(types)) if types else None)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = cat.slot.reindex(ids).to_numpy() == slot
            if types:
                mask &= cat.product_type_name.reindex(ids).isin(types).to_numpy()
            self._mask_cache[key] = mask
        if unavailable:              # the index holds live articles only; drop runtime stock-outs too
            mask = mask & ~np.isin(ids, np.fromiter(unavailable, dtype=np.int64))
        idx = np.flatnonzero(mask)
        sims = index[idx] @ q                     # exact search: see D-042 for the ANN decision
        order = np.lexsort((ids[idx], -sims))[:k]
        return {"mode": self.mode, "matches": [{"article_id": int(ids[idx[j]]), "similarity": float(sims[j])}
                                               for j in order], "candidates_searched": int(len(idx))}


def detect_garments(img, timeout_s: float = 30.0) -> list[dict]:
    """Optional garment detection with the local VLM (never a paid backend from the API)."""
    import os

    from ensemble.llm.client import get_client
    from ensemble.vision.outfit import detect
    if os.environ.get("ENSEMBLE_LLM", "ollama") != "ollama":
        raise RequestError("detector_unavailable", "the API only uses the local VLM for garment detection", 503)
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(detect, img, get_client("vision"))
    try:
        garments, small = fut.result(timeout=timeout_s)
    except FutTimeout:
        raise RequestError("detector_unavailable", f"garment detection exceeded {timeout_s:.0f} s", 503) from None
    except Exception as e:  # noqa: BLE001 - the detector is optional: report it, degrade explicitly
        raise RequestError("detector_unavailable", f"garment detector failed ({type(e).__name__}); send the "
                                                   "garments you selected instead", 503) from None
    finally:
        pool.shutdown(wait=False)
    sx, sy = img.width / small.width, img.height / small.height
    for g in garments:
        if g.get("box"):
            b = g["box"]
            g["box"] = [int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)]
    return garments


def parse_garments(raw: str | None, w: int, h: int) -> list[dict] | None:
    """Client-selected garments: JSON list of {category, box?} with box as fractions."""
    if not raw:
        return None
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        raise RequestError("invalid_garments", "garments must be a JSON list") from None
    if not isinstance(items, list) or not items or len(items) > 12:
        raise RequestError("invalid_garments", "garments must be a non-empty JSON list of at most 12 items")
    out = []
    for it in items:
        if not isinstance(it, dict) or it.get("category") not in CATEGORIES:
            raise RequestError("invalid_garments", f"each garment needs a category in {sorted(CATEGORIES)}")
        box = it.get("box")
        out.append({"category": it["category"],
                    "box": parse_box(",".join(map(str, box)), w, h) if isinstance(box, list) else None})
    return out
