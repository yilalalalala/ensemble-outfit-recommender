"""M7a: visual search and "snap your outfit, fill the gap" (DESIGN §7.3, D-011).

Pipeline for one photo:
  1. detect   a VLM lists garments: category, colour, description, bounding box
  2. match    each garment → catalogue articles in the right slot/type, by FashionCLIP
              similarity. Four query modes (an ablation):
                photo       whole photo embedding
                crop        garment crop embedding (falls back to photo if no valid box)
                text        embedding of the VLM's text description
                crop+text   normalised sum of crop and text embeddings (composed retrieval)
  3. gaps     core slots not present in the photo (bottoms/top or dress, shoes, bag, jewellery)
  4. complete Complete the Look for the best-matched main garment, restricted to the gaps

The catalogue is the live assortment: articles sold in the last ``live_weeks`` of data.
Runs in a PyTorch process only (never import LightGBM here; see ensemble.api.build).
"""
from __future__ import annotations

import io
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from ensemble.config import load_config
from ensemble.vision import clip

CATEGORIES = {
    # VLM category → (slot, allowed H&M product_type_name values or None for any in slot)
    "top": ("upper", None), "outerwear": ("upper", None), "knitwear": ("upper", None),
    "bottom": ("lower", None), "dress": ("full", None), "jumpsuit": ("full", None),
    "shoes": ("shoes", None),
    "bag": ("accessories", {"Bag", "Backpack", "Wallet"}),
    "jewellery": ("accessories", {"Earring", "Necklace", "Ring", "Bracelet"}),
    "hat": ("accessories", {"Hat/beanie", "Cap/peaked", "Beanie", "Hat/brim", "Cap"}),
    "scarf": ("accessories", {"Scarf"}), "belt": ("accessories", {"Belt"}),
    "sunglasses": ("accessories", {"Sunglasses"}), "socks": ("socks", None),
}
MAIN_ORDER = ["dress", "jumpsuit", "outerwear", "top", "knitwear", "bottom"]

DETECT_PROMPT = """You are a fashion image analyst. List every clothing item and accessory the main person is wearing or carrying.
The image is {w}x{h} pixels. Return JSON only, in this shape:
{{"items": [{{"category": "<one of: {cats}>", "colour": "<main colour>", "description": "<short product-style description, e.g. 'high-waisted wide-leg denim jeans'>", "box": [x1, y1, x2, y2]}}]}}
Box coordinates are integer pixels in this image. Include only items that are clearly visible. Do not describe the background."""


def _resize_bytes(img: Image.Image) -> tuple[bytes, Image.Image]:
    """Resize to what Qwen2.5-VL uses internally (area ~0.8–1.0 MP, sides multiples of 28), so the
    pixel boxes the model returns are in the coordinates of the image we hold. Without this, small
    images are upscaled inside the model and boxes overflow the stated size."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = (900_000 / (w * h)) ** 0.5
    nw, nh = max(28, round(w * scale / 28) * 28), max(28, round(h * scale / 28) * 28)
    img = img.resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), img


def parse_items(text: str, w: int, h: int) -> list[dict]:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return []
    out = []
    for it in data.get("items", []) if isinstance(data, dict) else []:
        cat = str(it.get("category", "")).lower().strip()
        if cat not in CATEGORIES:
            continue
        box = it.get("box")
        valid = (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)
                 and 0 <= box[0] < box[2] <= w * 1.02 and 0 <= box[1] < box[3] <= h * 1.02
                 and (box[2] - box[0]) * (box[3] - box[1]) >= 0.005 * w * h)
        out.append({"category": cat, "colour": str(it.get("colour", "")), "description": str(it.get("description", "")),
                    "box": [int(v) for v in box] if valid else None})
    return out


def detect(image: Image.Image, llm, purpose: str = "detect") -> tuple[list[dict], Image.Image]:
    data, small = _resize_bytes(image)
    prompt = DETECT_PROMPT.format(w=small.width, h=small.height, cats=", ".join(CATEGORIES))
    r = llm.chat("Return only valid JSON.", [{"role": "user", "content": prompt, "images": [data]}],
                 max_tokens=1500, json_mode=True, purpose=purpose)
    return parse_items(r.text, small.width, small.height), small


@lru_cache(maxsize=1)
def catalogue() -> tuple[pd.DataFrame, np.ndarray]:
    """Live assortment with FashionCLIP embeddings (L2-normalised)."""
    cfg = load_config()
    # The serving store already contains the point-in-time live catalogue and its
    # popularity snapshot.  Reading it avoids shipping the 836 MB research DuckDB
    # into the production container solely to reconstruct this small table.
    import sqlite3
    con = sqlite3.connect(f"file:{cfg.path('serving_db')}?mode=ro", uri=True)
    live = pd.read_sql_query("""
        SELECT a.article_id, a.product_code, a.prod_name, a.product_type_name,
               a.product_group_name, a.colour_group_name, a.department_name,
               a.slot, a.index_group_name, p.pop_window AS pop
        FROM articles a JOIN popularity p USING (article_id)
        WHERE a.slot IS NOT NULL AND p.pop_window > 0
        ORDER BY a.article_id""", con)
    con.close()
    d = cfg.path("processed") / "clip"
    ids, emb, has = np.load(d / "article_ids.npy"), np.load(d / "article_emb.npy"), np.load(d / "has_image.npy")
    row = pd.Series(np.arange(len(ids)), index=ids)
    live = live[live.article_id.isin(row.index)]
    live = live[has[row[live.article_id].values]].reset_index(drop=True)
    return live, emb[row[live.article_id].values].astype(np.float32)


@lru_cache(maxsize=1)
def _rembg_session():
    from rembg import new_session
    return new_session("u2net")


def remove_background(img: Image.Image) -> Image.Image:
    """Cut the garment out (U²-Net via rembg) and paste it on white, to look like a product shot."""
    from rembg import remove
    cut = remove(img.convert("RGB"), session=_rembg_session())
    bg = Image.new("RGB", cut.size, (255, 255, 255))
    bg.paste(cut, mask=cut.split()[-1])
    return bg


@lru_cache(maxsize=2)
def adapter(view: str = "crop"):
    """Street↔shop domain adapters trained on DeepFashion2 (None until trained). The plain-crop
    adapter is the default: on DeepFashion2 validation it beats the background-removed one
    (Recall@1 0.585 vs 0.545)."""
    import torch
    from ensemble.vision.deepfashion2 import DomainAdapters, paths
    path = paths()[1] / f"adapter_{view}.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu")
    m = DomainAdapters(ck["dim"], ck["hidden"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def _adapt(x: np.ndarray, domain: str, view: str = "crop") -> np.ndarray:
    import torch
    with torch.no_grad():
        return getattr(adapter(view), domain)(torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)).numpy()


@lru_cache(maxsize=2)
def catalogue_shop_adapted(view: str = "crop") -> np.ndarray:
    return _adapt(catalogue()[1], "shop", view)


def query_vector(mode: str, garment: dict, photo: Image.Image) -> np.ndarray:
    crop = photo.crop(garment["box"]) if garment.get("box") else photo
    if mode == "crop_nobg":
        return clip.embed_images([remove_background(crop)])[0]
    if mode == "crop_nobg_adapter":
        return _adapt(clip.embed_images([remove_background(crop)])[0], "user", "nobg")[0]
    if mode == "crop_adapter":
        return _adapt(clip.embed_images([crop])[0], "user", "crop")[0]
    text = f"{garment['colour']} {garment['description']}".strip()
    if mode == "photo":
        return clip.embed_images([photo])[0]
    if mode == "crop":
        return clip.embed_images([crop])[0]
    if mode == "text":
        return clip.embed_texts([text])[0]
    if mode == "text+0.3crop":   # text-led fusion; weight fixed in advance, not tuned on the eval photos
        v = 0.7 * clip.embed_texts([text])[0] + 0.3 * clip.embed_images([crop])[0]
    else:                        # "crop+text": equal weights
        v = clip.embed_images([crop])[0] + clip.embed_texts([text])[0]
    return v / np.linalg.norm(v)


def match(garment: dict, photo: Image.Image, mode: str = "crop+text", k: int = 5) -> pd.DataFrame:
    live, emb = catalogue()
    if mode.endswith("_adapter"):
        view = "nobg" if mode == "crop_nobg_adapter" else "crop"
        if adapter(view) is None:
            raise RuntimeError("DeepFashion2 adapter not trained")
        emb = catalogue_shop_adapted(view)
    slot, types = CATEGORIES[garment["category"]]
    mask = (live.slot == slot).to_numpy(copy=True)  # pandas 3: .values is read-only
    if types:
        mask &= live.product_type_name.isin(types).values
    if not mask.any():
        return live.head(0)
    idx = np.flatnonzero(mask)
    if mode == "text_rerank_image":
        # Two stages: text retrieves 50 candidates in the right category, image similarity reranks
        # them (street-to-shop adapted when available), so visual detail decides among good candidates.
        t = emb[idx] @ query_vector("text", garment, photo)
        cand = idx[np.argsort(-t)[:50]]
        if adapter("crop") is not None:
            img = catalogue_shop_adapted("crop")[cand] @ query_vector("crop_adapter", garment, photo)
        else:
            img = emb[cand] @ query_vector("crop_nobg", garment, photo)
        s = 0.5 * (emb[cand] @ query_vector("text", garment, photo)) + 0.5 * img
        order = np.argsort(-s)[:k]
        return live.iloc[cand[order]].assign(similarity=s[order])
    s = emb[idx] @ query_vector(mode, garment, photo)
    top = idx[np.argsort(-s)[:k]]
    return live.iloc[top].assign(similarity=np.sort(s)[::-1][:k])


def gaps(garments: list[dict]) -> list[str]:
    cats = {g["category"] for g in garments}
    missing = []
    if not cats & {"dress", "jumpsuit"}:
        if not cats & {"top", "outerwear", "knitwear"}:
            missing.append("upper")
        if "bottom" not in cats:
            missing.append("lower")
    if "shoes" not in cats:
        missing.append("shoes")
    if not cats & {"bag", "jewellery", "hat", "scarf", "belt", "sunglasses"}:
        missing.append("accessories")
    return missing


def snap(image: Image.Image, llm, mode: str = "text_rerank_image", k: int = 5) -> dict:
    """The full journey for one photo."""
    garments, photo = detect(image, llm)
    for g in garments:
        g["matches"] = match(g, photo, mode, k)[["article_id", "prod_name", "product_type_name",
                                                  "colour_group_name", "similarity"]].to_dict("records")
    main = next((g for c in MAIN_ORDER for g in garments if g["category"] == c and g["matches"]), None)
    missing = gaps(garments)
    ctl = {}
    if main:
        cfg = load_config()
        table = pd.read_parquet(cfg.path("processed") / "models" / "complete_the_look.parquet")
        anchor = main["matches"][0]["article_id"]
        rows = table[(table.anchor == anchor) & table.slot.isin(missing)]
        ctl = {s: g.sort_values("rank").article_id.tolist() for s, g in rows.groupby("slot")}
    return {"garments": garments, "missing_slots": missing, "anchor": main["matches"][0] if main else None,
            "complete_the_look": ctl}
