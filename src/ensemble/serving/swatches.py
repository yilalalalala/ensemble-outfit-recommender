"""Colour swatches from each colourway's own product photo.

The catalogue's ``colour_group_name`` says "Black" for a black t-shirt and for a black-and-white striped one, so a
swatch coloured from the group misleads on patterned pieces, and the group's shade rarely matches the photo. Each
swatch is therefore read from the photo:

1. **Garment mask.** The studio backdrop is estimated from the photo border; the garment is what differs from it,
   eroded so nothing is sampled from an edge or a gap (between trouser legs, under a strap).
2. **Main colour.** The garment pixels are clustered in CIELAB (k-means); the largest cluster is the main colour.
3. **Clean patch.** A window slides over the garment; the chosen window lies entirely on the garment and is made
   almost only of the main colour, so buttons, lapels, seams, labels and prints are avoided wherever they sit
   (the photo's centre is often exactly where they are).
4. **Plain or pattern.** A piece is a *pattern* when the catalogue says so (stripe, check, dot, all-over pattern,
   jacquard, argyle, colour blocking, mixed) *and* the photo agrees (a second colour, clearly different, covers a
   real share of the garment). A *plain* piece is shown as one flat colour: the patch colour snapped to the nearest
   colour on a standard colour card (``CARD``, the shop's swatch palette). A *pattern* is shown as a crop of the
   window whose colour mix is closest to the whole garment's, so the swatch looks like the fabric. The snap stays
   inside the catalogue colour name's family (``FAMILIES``): people name the hue, the photo gives the shade.

Stored in its own SQLite file; /api/catalog/families returns it inline per colourway.

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

GRID = (120, 180)   # analysis resolution (the catalogue photos are 2:3)

# The standard colour card plain swatches are snapped to: the shop's colour-group palette (static/js/catalog.js SWATCH,
# without its "Other …" catch-alls) plus the standard fashion shades that palette lacks (olive, khaki, camel, taupe,
# charcoal, navy, burgundy, …), so a dark olive dress is not snapped to grey for want of a closer chip.
CARD = {
    # colour-group palette
    "Black": "#111111", "Dark Grey": "#4a4a4a", "Grey": "#8a8a8a", "Light Grey": "#cfcfcf", "White": "#ffffff",
    "Off White": "#f3efe6", "Light Beige": "#e8dcc4", "Beige": "#d6c3a1", "Dark Beige": "#b39b75", "Greyish Beige": "#bfb3a0",
    "Yellowish Brown": "#a97a3c", "Bronze/Copper": "#a8652f", "Gold": "#c9a64a", "Silver": "#c0c0c4",
    "Dark Blue": "#1f2a44", "Blue": "#3a64b0", "Light Blue": "#a9c6e8",
    "Dark Turquoise": "#19757a", "Turquoise": "#2fb3a8", "Light Turquoise": "#9edfd8",
    "Dark Green": "#234d32", "Green": "#3f8f4f", "Light Green": "#b5d9a3", "Greenish Khaki": "#7d7a4f",
    "Dark Yellow": "#c9a227", "Yellow": "#f2d03b", "Light Yellow": "#f7eaa0",
    "Dark Orange": "#d0611f", "Orange": "#ec7f2c", "Light Orange": "#f6b77f",
    "Dark Red": "#7a1c22", "Red": "#c0262d", "Light Red": "#e8777a",
    "Dark Pink": "#c2507a", "Pink": "#e58fa8", "Light Pink": "#f4c9d3",
    "Dark Purple": "#4b2a5c", "Purple": "#7a4a9a", "Light Purple": "#c9b3dd",
    # standard fashion shades
    "Charcoal": "#33343a", "Stone": "#c9c2b6", "Taupe": "#8c7f72", "Greige": "#a9a194", "Cream": "#f4ecd6",
    "Ivory": "#f7f3e8", "Camel": "#b98a55", "Tan": "#c9a47c", "Chocolate": "#4f3427", "Brown": "#6e4a33",
    "Rust": "#9d4a27", "Terracotta": "#c0623f", "Mustard": "#c99a2e", "Olive": "#5d6136", "Dark Olive": "#3c3e27",
    "Khaki": "#a39a6b", "Sage": "#9fae8f", "Forest": "#2d4734", "Mint": "#c3e4cf", "Lime": "#c8e63c",
    "Teal": "#1e6b70", "Navy": "#1b2238", "Denim": "#5a7aa3", "Light Denim": "#91a9c6", "Sky": "#86b9e3",
    "Cobalt": "#2c4fa3", "Burgundy": "#5c1a28", "Wine": "#6e2437", "Coral": "#ee7f6c", "Blush": "#e9c2bb",
    "Fuchsia": "#d1338a", "Lilac": "#b9a3d3", "Plum": "#5e2f4f", "Mauve": "#a77b8f", "Lavender": "#d7cdea",
}
# Colour families: the catalogue's colour name (set by people) decides the family, the photo picks the shade inside
# it — so a piece labelled "Dark Green" that photographs almost neutral still gets a green chip next to its name.
FAMILIES = {
    "black": ["Black", "Charcoal", "Dark Grey"],
    "white": ["White", "Off White", "Ivory", "Cream", "Light Grey", "Light Beige"],
    "grey": ["White", "Light Grey", "Grey", "Dark Grey", "Charcoal", "Silver", "Stone", "Greige", "Black"],
    "beige": ["Off White", "Ivory", "Cream", "Light Beige", "Beige", "Dark Beige", "Greyish Beige", "Stone", "Greige",
              "Taupe", "Tan", "Camel"],
    "brown": ["Yellowish Brown", "Bronze/Copper", "Brown", "Chocolate", "Camel", "Tan", "Rust", "Dark Beige", "Taupe"],
    "blue": ["Dark Blue", "Blue", "Light Blue", "Navy", "Denim", "Light Denim", "Sky", "Cobalt"],
    "turquoise": ["Dark Turquoise", "Turquoise", "Light Turquoise", "Teal", "Mint"],
    "green": ["Dark Green", "Green", "Light Green", "Greenish Khaki", "Olive", "Dark Olive", "Khaki", "Sage", "Forest",
              "Mint", "Lime"],
    "yellow": ["Dark Yellow", "Yellow", "Light Yellow", "Mustard", "Gold", "Lime", "Cream"],
    "orange": ["Dark Orange", "Orange", "Light Orange", "Rust", "Terracotta", "Coral", "Tan", "Camel"],
    "red": ["Dark Red", "Red", "Light Red", "Burgundy", "Wine", "Coral", "Rust", "Terracotta"],
    "pink": ["Dark Pink", "Pink", "Light Pink", "Blush", "Fuchsia", "Coral", "Mauve"],
    "purple": ["Dark Purple", "Purple", "Light Purple", "Lilac", "Plum", "Mauve", "Lavender"],
    "metal": ["Gold", "Silver", "Bronze/Copper", "Light Grey", "Stone"],
}
_FAMILY_WORDS = [("Turquoise", "turquoise"), ("Khaki", "green"), ("Brown", "brown"), ("Bronze", "brown"), ("Off White", "white"),
                 ("White", "white"), ("Beige", "beige"), ("Black", "black"), ("Grey", "grey"), ("Blue", "blue"),
                 ("Green", "green"), ("Yellow", "yellow"), ("Orange", "orange"), ("Red", "red"), ("Pink", "pink"),
                 ("Purple", "purple"), ("Gold", "metal"), ("Silver", "metal")]


def family_of(colour_group: str | None) -> str | None:
    """The colour family of a catalogue colour name; None (any card colour) for "Other", "Unknown", "Transparent"."""
    for word, fam in _FAMILY_WORDS:
        if colour_group and word in colour_group:
            return fam
    return None


# Catalogue appearances that can be shown as a pattern (the photo must agree); everything else is plain.
PATTERNED = {"All over pattern", "Stripe", "Check", "Dot", "Jacquard", "Argyle", "Colour blocking",
             "Mixed solid/pattern", "Other pattern"}


# ---- colour ----------------------------------------------------------------------------------------------------
def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0–255, …×3) → CIELAB (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)


CARD_NAMES = list(CARD)
CARD_LAB = to_lab(np.stack([_hex_rgb(h) for h in CARD.values()]))


def snap(lab: np.ndarray, family: str | None = None, named: str | None = None, agree: float = 8.0) -> str:
    """The standard-card colour nearest to a CIELAB colour. Lightness differences count half (as in CMC l:c = 2:1,
    the textile industry's colour-difference weighting): photo lighting shifts lightness far more than hue, and a
    dark olive should land on olive, not on a grey of the same lightness. ``named`` (the catalogue colour) wins when
    the photo is within ``agree`` of it."""
    d = CARD_LAB - lab
    dist = (d[:, 0] / 2) ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2
    # When the photo is close to the catalogue's own colour, show that colour: chip and colour name then agree.
    if named in CARD and np.sqrt(dist[CARD_NAMES.index(named)]) <= agree:
        return named
    if family in FAMILIES:
        allowed = np.isin(CARD_NAMES, FAMILIES[family])
        dist = np.where(allowed, dist, np.inf)
    return CARD_NAMES[int(np.argmin(dist))]


def kmeans(x: np.ndarray, k: int = 4, iters: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Tiny k-means (k-means++ init) on N×3 points: (labels, centres)."""
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min([((x - c) ** 2).sum(-1) for c in centres], axis=0)
        centres.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(centres)
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1)
        c = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1), c


# ---- one photo -------------------------------------------------------------------------------------------------
def analyse(img: Image.Image, appearance: str | None, colour_group: str | None = None, bg_threshold: int = 40, window: float = 0.16,
            pattern_share: float = 0.12, pattern_delta: float = 15.0, agree: float = 8.0) -> dict:
    """{kind: plain|pattern, card, hex, box (fractions of the image)} for one product photo."""
    small = img.resize(GRID)
    rgb = np.asarray(small, dtype=np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    garment = np.abs(rgb - np.median(border, axis=0)).sum(-1) > bg_threshold
    mask = np.asarray(Image.fromarray((garment * 255).astype("uint8")).filter(ImageFilter.MinFilter(5))) > 0
    if mask.sum() < 50:   # almost no garment found (white on white, tiny jewellery): fall back to the whole mask
        mask = garment if garment.sum() >= 50 else np.ones_like(garment)
    lab = to_lab(rgb)
    pts = lab[mask]
    sample = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 3000), replace=False)]
    _, centres = kmeans(sample)
    labels = np.full(mask.shape, -1)
    labels[mask] = np.argmin(((pts[:, None] - centres[None]) ** 2).sum(-1), 1)
    share = np.bincount(labels[mask], minlength=len(centres)) / mask.sum()
    main = int(np.argmax(share))
    second = [j for j in range(len(centres)) if j != main and share[j] >= pattern_share
              and np.sqrt(((centres[j] - centres[main]) ** 2).sum()) > pattern_delta]
    kind = "pattern" if appearance in PATTERNED and second else "plain"

    # Windows lying on the garment: the main colour's share (plain) or the colour-mix distance (pattern).
    w = max(6, int(window * GRID[0]))
    # Box sums over every w×w window at once (summed-area tables): garment coverage and each colour's share.
    def boxes(a):
        S = np.pad(a.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (S[w:, w:] - S[:-w, w:] - S[w:, :-w] + S[:-w, :-w]) / (w * w)
    cover = boxes(mask)
    mix = np.stack([boxes(labels == j) for j in range(len(centres))], -1)
    score = -mix[..., main] if kind == "plain" else np.abs(mix - share).sum(-1)
    y = x = None
    for need in (0.98, 0.85, 0.6):
        ok = cover >= need
        if ok.any():
            y, x = np.unravel_index(np.argmin(np.where(ok, score, np.inf)), score.shape)
            break
    if y is None:
        y, x = (GRID[1] - w) // 2, (GRID[0] - w) // 2
    win = labels[y:y + w, x:x + w]
    if kind == "plain":
        px = lab[y:y + w, x:x + w][win == main]
        px = px if len(px) else pts
        # Perceived colour follows the lit fibres, not the shadows in a texture (fur, knit, pleats): average the
        # 40th–90th lightness percentile of the patch.
        lo, hi = np.percentile(px[:, 0], [40, 90])
        lit = px[(px[:, 0] >= lo) & (px[:, 0] <= hi)]
        colour = (lit if len(lit) else px).mean(0)
        card = snap(colour, family_of(colour_group), colour_group, agree)
        hx = CARD[card]
    else:
        card, hx = None, "#{:02x}{:02x}{:02x}".format(*np.asarray(small)[y:y + w, x:x + w].reshape(-1, 3).mean(0).astype(int))
    return {"kind": kind, "card": card, "hex": hx,
            "box": (x / GRID[0], y / GRID[1], (x + w) / GRID[0], (y + w) / GRID[1])}


def swatch(path: Path, appearance: str | None, colour_group: str | None, size: int, **params) -> tuple[str, str | None, str, bytes | None]:
    """(kind, card colour name or None, hex, JPEG crop for patterns or None)."""
    img = Image.open(path)
    img.draft("RGB", (img.width // 4, img.height // 4))   # decode at 1/4 scale: fast, still sharp at 32 px
    img = img.convert("RGB")
    r = analyse(img, appearance, colour_group, **params)
    jpeg = None
    if r["kind"] == "pattern":
        x0, y0, x1, y1 = r["box"]
        # GRID has the photo's 2:3 shape, so a square window on the grid is a square on the photo.
        crop = img.crop((int(x0 * img.width), int(y0 * img.height), int(x1 * img.width), int(y1 * img.height)))
        buf = io.BytesIO()
        crop.resize((size, size), Image.LANCZOS).save(buf, format="JPEG", quality=80, optimize=True)
        jpeg = buf.getvalue()
    return r["kind"], r["card"], r["hex"], jpeg


# ---- the catalogue ---------------------------------------------------------------------------------------------
def image_path(images: Path, article_id: int) -> Path:
    s = f"{article_id:010d}"
    return images / s[:3] / f"{s}.jpg"


def _work(args):
    aid, path, appearance, colour_group, size, params = args
    try:
        return aid, *swatch(Path(path), appearance, colour_group, size, **params)
    except Exception as e:   # a corrupt photo keeps the colour-group swatch in the UI
        return aid, None, None, repr(e), None


def build(limit: int | None = None, log=print) -> Path:
    cfg = load_config()
    images = cfg.path("raw") / "images"
    out = cfg.path("swatch_db")
    p = dict(cfg.serving.swatch)
    size = p.pop("size")
    import duckdb
    con = duckdb.connect(str(cfg.path("database")), read_only=True)
    rows = con.execute("SELECT article_id, graphical_appearance_name, colour_group_name FROM articles ORDER BY article_id").fetchall()
    con.close()
    jobs = [(a, str(path), g, cg, size, p) for a, g, cg in rows if (path := image_path(images, a)).exists()][:limit]
    log(f"{len(jobs):,} of {len(rows):,} articles have a photo")
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.execute("""CREATE TABLE swatches (article_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, card TEXT,
                                         hex TEXT NOT NULL, jpeg BLOB)""")
    t, n, bad, kinds = time.time(), 0, 0, {"plain": 0, "pattern": 0}
    with ProcessPoolExecutor() as ex:
        batch = []
        for aid, kind, card, hx, jpeg in ex.map(_work, jobs, chunksize=64):
            if kind is None:
                bad += 1
                continue
            batch.append((aid, kind, card, hx, jpeg)); n += 1; kinds[kind] += 1
            if len(batch) >= 5000:
                db.executemany("INSERT INTO swatches VALUES (?, ?, ?, ?, ?)", batch); batch = []
                log(f"  {n:,} swatches · {time.time() - t:.0f}s")
        db.executemany("INSERT INTO swatches VALUES (?, ?, ?, ?, ?)", batch)
    db.commit(); db.execute("VACUUM"); db.close()
    tmp.replace(out)
    log(f"wrote {out} · {n:,} swatches ({kinds['plain']:,} plain, {kinds['pattern']:,} pattern), {bad} unreadable · "
        f"{time.time() - t:.0f}s · {out.stat().st_size / 1e6:.1f} MB")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    build(ap.parse_args().limit)
