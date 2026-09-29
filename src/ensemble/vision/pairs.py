"""Owner-supplied street↔product composites → exact-match benchmark (D-026).

Composites ("outfit breakdown" posts) put a street photo on one side and the individual product
shots, on white, on the other. Each composite yields ground-truth pairs:

  split    find the photo region and the product panel from the non-white pixel profile
           (left|right or top|bottom layouts)
  detect   Qwen2.5-VL lists garments in the photo and product images in the panel (category + box)
  pair     keep a pair only when its category occurs exactly once on both sides (unambiguous)

The benchmark injects every extracted product into the H&M live catalogue as a distractor-rich
gallery and asks, for each street garment, where its true product ranks (Recall@1/5/10).
Owner images are local only; crops are written under data/interim (gitignored).

  python -m ensemble.vision.pairs extract
  python -m ensemble.vision.pairs benchmark
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from ensemble.config import load_config
from ensemble.llm.client import OllamaClient
from ensemble.vision.outfit import CATEGORIES, _resize_bytes, detect, parse_items

cfg = load_config()
RAW = cfg.path("raw") / "outfit_pairs"
OUT = cfg.path("interim") / "outfit_pairs"
REPORT = cfg.path("reports") / "m7a"

PANEL_PROMPT = """This is the product panel of a fashion "outfit breakdown" post: separate product photos, mostly on
a white background, with brand names and prices as text. The image is {w}x{h} pixels.
List every product PHOTO (not the text labels). Return JSON only:
{{"items": [{{"category": "<one of: {cats}>", "colour": "<main colour>", "description": "<short product description>",
"box": [x1, y1, x2, y2]}}]}}
Box coordinates are integer pixels in this image and must cover only the product photo."""


def split(img: Image.Image, axis: str | None = None) -> tuple[tuple, tuple] | None:
    """Return (photo_box, panel_box) as (x1, y1, x2, y2), or None if no clear layout.
    ``axis`` ("H" left|right, "V" top|bottom) comes from the curated MANIFEST.csv when known."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    nonwhite = (a.min(axis=2) < 232) | (a.max(axis=2) - a.min(axis=2) > 18)
    h, w = nonwhite.shape
    col, row = nonwhite.mean(axis=0), nonwhite.mean(axis=1)

    def best_cut(profile, lo, hi):
        # the cut that maximises (occupancy before) − (occupancy after): a dense photo vs a sparse panel
        best, score = None, 0.0
        for c in range(int(lo * len(profile)), int(hi * len(profile)), 4):
            before, after = profile[:c].mean(), profile[c:].mean()
            if before > 0.7 and after < 0.55 and before - after > score:
                best, score = c, before - after
        return best, score

    xc, xs = best_cut(col, 0.3, 0.85)
    yc, ys = best_cut(row, 0.3, 0.9)
    if axis == "H":
        return ((0, 0, xc, h), (xc, 0, w, h)) if xc is not None else None
    if axis == "V":
        return ((0, 0, w, yc), (0, yc, w, h)) if yc is not None else None
    if xc is not None and xs >= ys:
        return (0, 0, xc, h), (xc, 0, w, h)
    if yc is not None:
        return (0, 0, w, yc), (0, yc, w, h)
    return None


def extract(limit: int | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    vlm = OllamaClient("qwen2.5vl:7b")
    path = REPORT / "pairs_extracted.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    import csv
    man = {r["file"]: r for r in csv.DictReader(open(RAW / "MANIFEST.csv"))}
    files = [RAW / f for f, r in sorted(man.items()) if r["type"] in ("composite", "jewellery_pair")][:limit]
    for p in files:
        if p.name in res:
            continue
        t = time.time()
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        layout = split(img, man[p.name]["layout"])
        if layout is None:
            res[p.name] = {"layout": None}
            print(f"  {p.name}: no composite layout (close-up without product panel)", flush=True)
            path.write_text(json.dumps(res, indent=1))
            continue
        photo, panel = img.crop(layout[0]), img.crop(layout[1])
        garments, photo_small = detect(photo, vlm, purpose="pairs")
        data, panel_small = _resize_bytes(panel)
        r = vlm.chat("Return only valid JSON.", [{"role": "user", "content": PANEL_PROMPT.format(
            w=panel_small.width, h=panel_small.height, cats=", ".join(CATEGORIES)), "images": [data]}],
            max_tokens=1500, json_mode=True, purpose="pairs")
        products = parse_items(r.text, panel_small.width, panel_small.height)
        pairs = []
        for cat in sorted({g["category"] for g in garments} & {q["category"] for q in products}):
            gs = [g for g in garments if g["category"] == cat and g["box"]]
            ps = [q for q in products if q["category"] == cat and q["box"]]
            if len(gs) == 1 and len(ps) == 1:
                k = len(pairs)
                photo_small.crop(gs[0]["box"]).save(OUT / f"{p.stem}_{k}_street.jpg", quality=92)
                panel_small.crop(ps[0]["box"]).save(OUT / f"{p.stem}_{k}_product.jpg", quality=92)
                pairs.append({"k": k, "category": cat, "street": gs[0], "product": ps[0]})
        res[p.name] = {"layout": [list(layout[0]), list(layout[1])], "garments": len(garments),
                       "products": len(products), "pairs": pairs, "seconds": round(time.time() - t, 1)}
        print(f"  {p.name}: {len(garments)} garments, {len(products)} products → {len(pairs)} pairs "
              f"{[q['category'] for q in pairs]} ({res[p.name]['seconds']}s)", flush=True)
        path.write_text(json.dumps(res, indent=1))
    n = sum(len(v.get("pairs", [])) for v in res.values())
    print(f"{n} unambiguous pairs from {sum(1 for v in res.values() if v.get('layout'))} composites")
    return res


BENCH_MODES = ["crop", "crop_nobg", "crop_adapter", "text", "text_rerank_image"]


def benchmark(gallery: str = "full") -> dict:
    """Exact-match retrieval: rank of the true product among H&M live articles + all extracted products
    of the same slot/type. No judge: the composite itself is the ground truth."""
    from ensemble.vision import clip
    from ensemble.vision import outfit as O

    res = json.loads((REPORT / "pairs_extracted.json").read_text())
    pairs = [(f, q) for f, v in res.items() for q in v.get("pairs", [])]
    live, cat_emb = O.catalogue()
    prod_imgs = {(f, q["k"]): Image.open(OUT / f"{f.rsplit('.', 1)[0]}_{q['k']}_product.jpg").convert("RGB")
                 for f, q in pairs}
    keys = list(prod_imgs)
    prod_emb = np.concatenate([clip.embed_images([prod_imgs[k] for k in keys[i:i + 64]]) for i in range(0, len(keys), 64)])
    prod_cat = [next(q["category"] for f, q in pairs if (f, q["k"]) == k) for k in keys]
    adapt = O.adapter("crop") is not None
    prod_emb_ad = O._adapt(prod_emb, "shop", "crop") if adapt else None
    cat_emb_ad = O.catalogue_shop_adapted("crop") if adapt else None

    ranks = {m: [] for m in BENCH_MODES}
    rows = []
    for f, q in pairs:
        street = Image.open(OUT / f"{f.rsplit('.', 1)[0]}_{q['k']}_street.jpg").convert("RGB")
        g = {"category": q["category"], "colour": q["street"]["colour"], "description": q["street"]["description"],
             "box": None}   # the crop is already the garment
        slot, types = CATEGORIES[q["category"]]
        hm_mask = (live.slot == slot).to_numpy(copy=True)
        if types:
            hm_mask &= live.product_type_name.isin(types).to_numpy()
        if gallery == "extracted_only":   # same-style distractors only (no H&M domain shift)
            hm_mask[:] = False
        pr_mask = np.asarray([CATEGORIES[c][0] == slot and (not types or c == q["category"]) for c in prod_cat])
        true_idx = keys.index((f, q["k"]))
        row = {"file": f, "k": q["k"], "category": q["category"], "gallery": int(hm_mask.sum() + pr_mask.sum())}
        for m in BENCH_MODES:
            if m == "text_rerank_image":
                t = O.query_vector("text", g, street)
                s_hm, s_pr = cat_emb[hm_mask] @ t, prod_emb[pr_mask] @ t
                allv = np.r_[s_hm, s_pr]
                top = np.argsort(-allv)[:50]
                img_q = O.query_vector("crop_adapter" if adapt else "crop", g, street)
                g_hm = (cat_emb_ad if adapt else cat_emb)[hm_mask]
                g_pr = (prod_emb_ad if adapt else prod_emb)[pr_mask]
                img_all = np.r_[g_hm @ img_q, g_pr @ img_q]
                score = np.full(len(allv), -np.inf)
                score[top] = 0.5 * allv[top] + 0.5 * img_all[top]
            else:
                qv = O.query_vector(m, g, street)
                use_ad = m == "crop_adapter"
                g_hm = (cat_emb_ad if use_ad else cat_emb)[hm_mask]
                g_pr = (prod_emb_ad if use_ad else prod_emb)[pr_mask]
                score = np.r_[g_hm @ qv, g_pr @ qv]
            pos_true = int(hm_mask.sum()) + int(np.flatnonzero(np.flatnonzero(pr_mask) == true_idx)[0])
            rank = int((score > score[pos_true]).sum()) + 1
            ranks[m].append(rank)
            row[m] = rank
        rows.append(row)

    def pack(r):
        a = np.asarray(r)
        return {f"recall@{k}": float(np.mean(a <= k)) for k in (1, 5, 10)} | {"median_rank": float(np.median(a)), "n": len(a)}
    out = {"gallery": gallery, "pairs": len(pairs), "overall": {m: pack(v) for m, v in ranks.items()}, "by_category": {}}
    for c in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == c]
        out["by_category"][c] = {m: pack([r[m] for r in sub]) for m in BENCH_MODES}
    (REPORT / f"pairs_benchmark{'' if gallery == 'full' else '_' + gallery}.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    print(json.dumps(out["overall"], indent=1))
    return out


EARRING_PROMPT = """The image is {w}x{h} pixels and shows a person wearing earrings. Find the single most clearly
visible earring the person is wearing. Return JSON only: {{"box": [x1, y1, x2, y2]}} with integer pixel coordinates
covering the earring with a small margin."""


def extract_jewellery_pairs() -> list[dict]:
    """Top|bottom earring posts: the bottom panel shows only the product on white, so the product is the
    non-white bounding box of the panel; the street side is a targeted earring box from the VLM."""
    import csv
    man = {r["file"]: r for r in csv.DictReader(open(RAW / "MANIFEST.csv"))}
    res = json.loads((REPORT / "pairs_extracted.json").read_text())
    vlm = OllamaClient("qwen2.5vl:7b")
    for f, r in sorted(man.items()):
        if r["type"] != "jewellery_pair":
            continue
        img = ImageOps.exif_transpose(Image.open(RAW / f)).convert("RGB")
        photo_box, panel_box = split(img, "V")
        panel = img.crop(panel_box)
        a = np.asarray(panel).astype(np.int16)
        ys, xs = np.nonzero((a.min(axis=2) < 225))
        product = panel.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        data, small = _resize_bytes(img.crop(photo_box))
        out = vlm.chat("Return only valid JSON.", [{"role": "user", "content": EARRING_PROMPT.format(w=small.width, h=small.height),
                                                   "images": [data]}], max_tokens=200, json_mode=True, purpose="pairs")
        try:
            box = [int(v) for v in json.loads(out.text)["box"]]
            pad = max(12, (box[2] - box[0]) // 2)
            box = [max(0, box[0] - pad), max(0, box[1] - pad), min(small.width, box[2] + pad), min(small.height, box[3] + pad)]
            assert box[2] - box[0] > 10 and box[3] - box[1] > 10
        except Exception:
            print(f"  {f}: no earring box", flush=True)
            continue
        stem = f.rsplit(".", 1)[0]
        small.crop(box).save(OUT / f"{stem}_0_street.jpg", quality=92)
        product.save(OUT / f"{stem}_0_product.jpg", quality=92)
        res[f]["pairs"] = [{"k": 0, "category": "jewellery", "street": {"colour": "", "description": "earrings", "box": box},
                            "product": {"box": None}}]
        print(f"  {f}: earring pair", flush=True)
    (REPORT / "pairs_extracted.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "extract":
        extract(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif stage == "jewellery":
        extract_jewellery_pairs()
    elif stage == "benchmark":
        benchmark(sys.argv[2] if len(sys.argv) > 2 else "full")
