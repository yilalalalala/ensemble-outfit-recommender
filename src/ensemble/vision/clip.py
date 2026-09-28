"""M7a: FashionCLIP image and text embeddings.

FashionCLIP (Chia et al., 2022) is CLIP fine-tuned on fashion product images and
descriptions. One model gives three things this project needs:
  - catalogue image embeddings (visual search, content features, cold items)
  - photo embeddings in the same space (street-to-shop matching)
  - text embeddings in the same space ("gold hoop earrings" → articles)

Product images are portrait; they are padded to a white square instead of
centre-cropped so shoes and collars are not cut off.

  python -m ensemble.vision.clip        embed every article image (~105k)
"""
from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image, ImageOps

from ensemble.config import load_config

MODEL = "patrickjohncyh/fashion-clip"
_model = _proc = None


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    global _model, _proc
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained(MODEL).to(device()).eval()
        _proc = CLIPProcessor.from_pretrained(MODEL)
    return _model, _proc


def pad_square(im: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGB")
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return canvas


def _normalise(x) -> np.ndarray:
    # transformers ≥ 5 returns a model output whose pooler_output is the projected embedding.
    x = getattr(x, "pooler_output", x)
    return torch.nn.functional.normalize(x, dim=-1).float().cpu().numpy()


@torch.no_grad()
def embed_images(images: list[Image.Image]) -> np.ndarray:
    model, proc = load()
    inputs = proc(images=[pad_square(im) for im in images], return_tensors="pt").to(device())
    return _normalise(model.get_image_features(**inputs))


@torch.no_grad()
def embed_texts(texts: list[str]) -> np.ndarray:
    model, proc = load()
    inputs = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device())
    return _normalise(model.get_text_features(**inputs))


def image_path(cfg, article_id: int):
    s = f"{article_id:010d}"
    return cfg.path("raw") / "images" / s[:3] / f"{s}.jpg"


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = pad_square(Image.open(self.paths[i]))
        im.thumbnail((336, 336))  # decode cost down; the processor resizes to 224 anyway
        return self.proc(images=im, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def embed_catalogue() -> None:
    cfg = load_config()
    import duckdb
    con = duckdb.connect(str(cfg.path("database")), read_only=True)
    ids = con.execute("SELECT article_id FROM articles ORDER BY article_id").fetchnumpy()["article_id"]
    paths = [image_path(cfg, int(a)) for a in ids]
    has = np.array([p.exists() for p in paths])
    model, proc = load()
    ds = _Images([p for p, h in zip(paths, has) if h], proc)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=6)
    out, t = [], time.time()
    for i, px in enumerate(dl):
        out.append(_normalise(model.get_image_features(pixel_values=px.to(device()))).astype(np.float16))
        if i % 40 == 0:
            print(f"  {(i + 1) * 256:,}/{len(ds):,} images ({time.time() - t:.0f}s)", flush=True)
    emb = np.zeros((len(ids), out[0].shape[1]), dtype=np.float16)
    emb[has] = np.concatenate(out)
    d = cfg.path("processed") / "clip"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "article_ids.npy", ids)
    np.save(d / "article_emb.npy", emb)
    np.save(d / "has_image.npy", has)
    print(f"embedded {has.sum():,} of {len(ids):,} articles in {time.time() - t:.0f}s")


if __name__ == "__main__":
    embed_catalogue()
