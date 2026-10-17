"""Fabric swatches: a small crop of each colourway's own product photo, shown as its colour swatch.

The catalogue's ``colour_group_name`` says "Black" for a black t-shirt and for a black-and-white striped one, so a
flat colour swatch misleads on any patterned piece. Retailers show a fabric swatch instead; here it is cut from
the product photo: estimate the studio backdrop from the photo border, keep the garment pixels, erode the mask so
the crop sits well inside the garment (not across a gap between trouser legs), and crop a square around the
garment pixel nearest the garment's centroid. Each swatch is a ~1 KB JPEG stored in its own SQLite file and
returned inline by /api/catalog/families, so a card's swatches cost no extra requests.

    python -m ensemble.serving.swatches            # all photographed articles
    python -m ensemble.serving.swatches --limit 500
"""
from __future__ import annotations

import argparse
import io
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ensemble.config import load_config

GRID = (60, 90)   # analysis resolution (the catalogue photos are 2:3)


def image_path(images: Path, article_id: int) -> Path:
    s = f"{article_id:010d}"
    return images / s[:3] / f"{s}.jpg"


def fabric_point(img: Image.Image, bg_threshold: int, erode: list[int]) -> tuple[float, float]:
    """(x, y) as fractions of the image: a point well inside the garment, nearest its centroid."""
    a = np.asarray(img.resize(GRID), dtype=np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    garment = np.abs(a - np.median(border, axis=0)).sum(-1) > bg_threshold
    if not garment.any():
        return 0.5, 0.45
    ys, xs = np.nonzero(garment)
    cy, cx = ys.mean(), xs.mean()
    for k in erode:   # the strongest erosion that still leaves garment pixels
        m = np.asarray(Image.fromarray((garment * 255).astype("uint8")).filter(ImageFilter.MinFilter(k))) > 0
        if m.any():
            ys, xs = np.nonzero(m)
            break
    i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return (xs[i] + 0.5) / GRID[0], (ys[i] + 0.5) / GRID[1]


def swatch(path: Path, size: int, crop_frac: float, bg_threshold: int, erode: list[int]) -> tuple[bytes, str]:
    """(JPEG bytes, mean colour hex) of the fabric crop."""
    img = Image.open(path)
    img.draft("RGB", (img.width // 4, img.height // 4))   # decode at 1/4 scale: fast, still sharp at 32 px
    img = img.convert("RGB")
    x, y = fabric_point(img, bg_threshold, erode)
    half = crop_frac * img.width / 2
    cx = min(max(x * img.width, half), img.width - half)
    cy = min(max(y * img.height, half), img.height - half)
    crop = img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=78, optimize=True)
    r, g, b = (int(v) for v in np.asarray(crop).reshape(-1, 3).mean(0))
    return buf.getvalue(), f"#{r:02x}{g:02x}{b:02x}"


def _work(args):
    aid, path, params = args
    try:
        return aid, *swatch(Path(path), **params)
    except Exception as e:   # a corrupt photo keeps the colour-group swatch in the UI
        return aid, None, repr(e)


def build(limit: int | None = None, log=print) -> Path:
    cfg = load_config()
    images = cfg.path("raw") / "images"
    out = cfg.path("swatch_db")
    params = dict(cfg.serving.swatch)
    serving = sqlite3.connect(cfg.path("serving_db"))
    ids = [r[0] for r in serving.execute("SELECT article_id FROM articles ORDER BY article_id")]
    jobs = [(a, str(p), params) for a in ids if (p := image_path(images, a)).exists()][:limit]
    log(f"{len(jobs):,} of {len(ids):,} articles have a photo")
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.execute("CREATE TABLE swatches (article_id INTEGER PRIMARY KEY, jpeg BLOB NOT NULL, hex TEXT NOT NULL)")
    t, ok, bad = time.time(), 0, 0
    with ProcessPoolExecutor() as ex:
        rows = []
        for aid, jpeg, hx in ex.map(_work, jobs, chunksize=64):
            if jpeg is None:
                bad += 1
                continue
            rows.append((aid, jpeg, hx)); ok += 1
            if len(rows) >= 5000:
                db.executemany("INSERT INTO swatches VALUES (?, ?, ?)", rows); rows = []
                log(f"  {ok:,} swatches · {time.time() - t:.0f}s")
        db.executemany("INSERT INTO swatches VALUES (?, ?, ?)", rows)
    db.commit(); db.execute("VACUUM"); db.close()
    tmp.replace(out)
    log(f"wrote {out} · {ok:,} swatches, {bad} unreadable photos · {time.time() - t:.0f}s · {out.stat().st_size / 1e6:.1f} MB")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    build(ap.parse_args().limit)
