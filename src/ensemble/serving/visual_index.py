"""Shop-side adapted FashionCLIP vectors for the bundle's visual index (PyTorch process).

  python -m ensemble.serving.visual_index <bundle>/visual

Reads ``clip.npy`` (live articles, L2-normalised FashionCLIP image vectors) and writes
``clip_shop_adapted.npy`` through the DeepFashion2 shop-side adapter (D-020, D-021). Exits
non-zero when the adapter has not been trained; the bundle then ships raw FashionCLIP only
and the visual-search endpoint reports ``mode: crop`` instead of ``crop_adapter``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main(out: Path) -> None:
    from ensemble.vision.outfit import _adapt, adapter
    if adapter("crop") is None:
        raise SystemExit("DeepFashion2 crop adapter not trained")
    emb = np.load(out / "clip.npy").astype(np.float32)
    parts = [_adapt(emb[i:i + 8192], "shop", "crop") for i in range(0, len(emb), 8192)]
    v = np.vstack(parts).astype(np.float32)
    np.save(out / "clip_shop_adapted.npy", v.astype(np.float16))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
