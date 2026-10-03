"""M6: FastAPI serving layer over the SQLite store (DESIGN §7).

  make serve     then open http://localhost:8010
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ensemble.config import load_config

cfg = load_config()
DB = cfg.path("serving_db")
IMAGES = cfg.path("raw") / "images"
STATIC = Path(__file__).parent / "static"
SLOT_TITLES = {"upper": "Tops", "lower": "Bottoms", "full": "Dresses & jumpsuits", "shoes": "Shoes",
               "accessories": "Accessories", "socks": "Socks & tights", "swimwear": "Swimwear"}

app = FastAPI(title="Ensemble recommender")


def q(sql: str, args=()) -> list[dict]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()


ART_COLS = "a.article_id, a.prod_name, a.product_type_name, a.colour_group_name, a.index_group_name, a.slot, a.is_jewellery"


def card(row: dict) -> dict:
    row["image"] = f"/images/{row['article_id']}.jpg"
    for key in ("reasons", "shap"):
        if isinstance(row.get(key), str):
            row[key] = json.loads(row[key])
    return row


@app.get("/api/customers")
def customers():
    return q("SELECT * FROM demo_customers")


@app.get("/api/home/{customer_idx}")
def home(customer_idx: int):
    c = q("SELECT * FROM demo_customers WHERE customer_idx = ?", (customer_idx,))
    if not c:
        raise HTTPException(404, "not a demo customer")
    for_you = [card(r) for r in q(f"""SELECT f.rank, f.score, f.reasons, {ART_COLS} FROM for_you f
                                      JOIN articles a USING (article_id) WHERE f.customer_idx = ? ORDER BY f.rank""", (customer_idx,))]
    again = [card(r) for r in q(f"""SELECT h.last_bought, h.times, {ART_COLS} FROM history h
                                    JOIN articles a USING (article_id) WHERE h.customer_idx = ?
                                    ORDER BY h.last_bought DESC, h.times DESC LIMIT 12""", (customer_idx,))]
    bin_ = q("SELECT age_bin FROM customer_age_bin WHERE customer_idx = ?", (customer_idx,))
    trending = [card(r) for r in q(f"""SELECT t.rank, {ART_COLS} FROM trending t JOIN articles a USING (article_id)
                                       WHERE t.age_bin = ? ORDER BY t.rank""", (bin_[0]["age_bin"] if bin_ else -1,))]
    return {"customer": c[0], "modules": [
        {"id": "for_you", "title": "Recommended for you", "items": for_you},
        {"id": "buy_again", "title": "Buy it again", "items": again},
        {"id": "trending", "title": "Trending this week", "items": trending},
    ]}


# Provenance labels written by ensemble.completion.serve_round3; the Round-1 table only
# ever carried "co_purchase" and "style_match", which both still resolve here.
ASSOCIATION_SOURCES = ("co_purchase", "style_co_purchase", "style_match")


def _ctl_columns() -> list[str]:
    """Columns the serving store actually has, so an older table still renders."""
    try:
        return [r["name"] for r in q("PRAGMA table_info(complete_the_look)")]
    except Exception:  # noqa: BLE001 - no serving store yet; the endpoint 404s anyway
        return []


def ctl_reasons(r: dict) -> list[dict]:
    """Reason chips for one Complete-the-Look pick, gated on evidence present in the row.

    A chip is emitted only when the column it quotes is non-null for *this* pair
    (D-034): a visually retrieved pick never claims co-purchase support, and a
    popularity fallback says so instead of borrowing a style explanation.
    """
    def num(key):
        v = r.get(key)
        return v if isinstance(v, (int, float)) and v == v else None

    out: list[dict] = []
    co, lift, s_co = num("a_co"), num("lift"), num("s_co")
    if r.get("source") in ("co_purchase", "style_match") and co and lift:
        out.append({"key": "co_purchase",
                    "text": f"Bought together {int(co)}× in past baskets, "
                            f"{lift:.1f}× more often than chance"})
    elif r.get("source") == "co_purchase" and lift:          # Round-1 table: lift only
        out.append({"key": "co_purchase", "text": f"Bought together {lift:.1f}× more often than chance"})
    elif r.get("source") == "style_co_purchase" and s_co:
        out.append({"key": "style", "text": f"Bought with this style {int(s_co)}× in past baskets"})
    elif r.get("source") == "visual_compatibility":
        out.append({"key": "visual", "text": "Visually matches this piece"})
    elif r.get("source") == "popular_in_slot":
        out.append({"key": "popular", "text": "Popular pick for this category"})
    else:
        out.append({"key": "style", "text": "Style match for this piece"})
    if num("same_colour_master") == 1:
        out.append({"key": "colour", "text": "Same colour family"})
    if num("price_tier_diff") == 0:
        out.append({"key": "price", "text": "Same price range"})
    return out[:3]


@app.get("/api/product/{article_id}")
def product(article_id: int):
    a = q("SELECT * FROM articles WHERE article_id = ?", (article_id,))
    if not a:
        raise HTTPException(404, "unknown article")
    a = card(a[0])
    have = _ctl_columns()
    extra = "".join(f", c.{c}" for c in ("a_co", "a_npmi", "s_co", "s_npmi", "backoff_level",
                                         "src_two_tower", "src_slot_pop", "tt_pair_sim", "clip_sim",
                                         "same_colour_master", "price_tier_diff", "score")
                    if c in have)
    ctl = q(f"""SELECT c.slot AS target_slot, c.rank, c.source, c.lift{extra}, {ART_COLS}
                FROM complete_the_look c JOIN articles a USING (article_id)
                WHERE c.anchor = ? ORDER BY c.slot, c.rank""", (article_id,))
    by_slot: dict[str, list] = {}
    for r in ctl:
        r = card(r)
        r["reasons"] = ctl_reasons(r)
        by_slot.setdefault(r["target_slot"], []).append(r)
    # Show a slot only when customers actually complete this item with it (at least one
    # association-backed pick); shoes and accessories are always offered. Order by evidence.
    order = ["lower", "upper", "full", "shoes", "accessories", "socks", "swimwear"]
    evidence = {s: sum(i["source"] in ASSOCIATION_SOURCES for i in items)
                for s, items in by_slot.items()}
    look = [{"slot": s, "title": SLOT_TITLES[s], "items": items} for s, items in by_slot.items()
            if evidence[s] > 0 or s in ("shoes", "accessories")]
    look.sort(key=lambda m: (-evidence[m["slot"]], order.index(m["slot"])))
    colours = [card(r) for r in q(f"""SELECT {ART_COLS} FROM articles a LEFT JOIN popularity p USING (article_id)
                                      WHERE a.product_code = ? AND a.article_id <> ?
                                      ORDER BY coalesce(p.pop_recent, 0) DESC LIMIT 12""", (a["product_code"], article_id))]
    similar = [card(r) for r in q(f"""SELECT {ART_COLS} FROM articles a JOIN popularity p USING (article_id)
                                      WHERE a.product_type_name = ? AND a.index_group_name = ?
                                        AND a.product_code <> ? ORDER BY p.pop_recent DESC LIMIT 12""",
                                  (a["product_type_name"], a["index_group_name"], a["product_code"]))]
    return {"article": a, "complete_the_look": look, "other_colours": colours, "similar": similar}


@app.get("/api/explain/{customer_idx}/{article_id}")
def explain(customer_idx: int, article_id: int):
    r = q("SELECT score, reasons, shap FROM for_you WHERE customer_idx = ? AND article_id = ?", (customer_idx, article_id))
    if not r:
        raise HTTPException(404, "not in this customer's recommendations")
    return card({**r[0], "article_id": article_id})


@app.get("/api/metrics")
def metrics():
    return {r["name"]: json.loads(r["body"]) for r in q("SELECT * FROM reports")}


class Event(BaseModel):
    customer_idx: int | None = None
    event: str                     # impression | click | add_to_cart | not_for_me
    surface: str                   # for_you | buy_again | trending | complete_the_look | other_colours | similar
    article_id: int
    anchor: int | None = None


@app.post("/api/events")
def log_event(e: Event):
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO events (customer_idx, event, surface, article_id, anchor) VALUES (?, ?, ?, ?, ?)",
                (e.customer_idx, e.event, e.surface, e.article_id, e.anchor))
    con.commit()
    con.close()
    return {"ok": True}


# --- M7a visual search and M7b assistant --------------------------------------------------------------
# These endpoints load PyTorch (FashionCLIP) lazily. This process never imports LightGBM (see api/build.py).
SESSIONS: dict[str, dict] = {}


def _cards(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    rows = {r["article_id"]: card(r) for r in q(
        f"SELECT {ART_COLS} FROM articles a WHERE a.article_id IN ({','.join('?' * len(ids))})", ids)}
    return [rows[a] for a in ids if a in rows]


def _image(upload: UploadFile):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(io.BytesIO(upload.file.read()))).convert("RGB")


@app.post("/api/visual-search")
def visual_search(photo: UploadFile = File(...), category: str = Form(...), box: str = Form("")):
    """Standalone visual search: no LLM. The user picks the category and (optionally) drags a box,
    given as fractions x1,y1,x2,y2 of the displayed image."""
    from ensemble.vision.outfit import CATEGORIES, match
    if category not in CATEGORIES:
        raise HTTPException(400, f"category must be one of {sorted(CATEGORIES)}")
    img = _image(photo)
    img.thumbnail((1024, 1024))
    g = {"category": category, "colour": "", "description": category, "box": None}
    if box:
        x1, y1, x2, y2 = (float(v) for v in box.split(","))
        g["box"] = [int(x1 * img.width), int(y1 * img.height), int(x2 * img.width), int(y2 * img.height)]
        if g["box"][2] - g["box"][0] < 10 or g["box"][3] - g["box"][1] < 10:
            g["box"] = None
    m = match(g, img, "crop_adapter", 8)  # best image-only mode (D-021)
    items = _cards(m.article_id.tolist())
    for it, sim in zip(items, m.similarity.tolist()):
        it["reasons"] = [{"key": "visual", "text": f"Visual match {sim:.2f}"}]
    return {"matches": items}


@app.post("/api/snap")
def snap_outfit(photo: UploadFile = File(...)):
    """Snap your outfit, fill the gap (D-011). Garment detection uses the configured VLM (local by default)."""
    from ensemble.llm.client import get_client
    from ensemble.vision.outfit import snap
    res = snap(_image(photo), get_client("vision"))
    garments = [{"category": g["category"], "colour": g["colour"], "description": g["description"],
                 "matches": _cards([m["article_id"] for m in g["matches"]])} for g in res["garments"]]
    look = [{"slot": s, "title": SLOT_TITLES[s], "items": _cards(ids[:6])} for s, ids in res["complete_the_look"].items()]
    return {"garments": garments, "missing_slots": res["missing_slots"],
            "anchor": res["anchor"]["article_id"] if res["anchor"] else None, "complete_the_look": look}


class SessionIn(BaseModel):
    customer_idx: int | None = None


@app.post("/api/assistant/session")
def assistant_session(body: SessionIn):
    from ensemble.assistant.agent import new_session
    from ensemble.llm.client import get_client
    s = new_session(body.customer_idx, get_client("vision"))
    s["llm"] = get_client("assistant")
    SESSIONS[s["id"]] = s
    return {"session_id": s["id"], "backend": os.environ.get("ENSEMBLE_LLM", "ollama")}


@app.post("/api/assistant/{sid}/message")
def assistant_message(sid: str, text: str = Form(...), photo: UploadFile | None = File(None)):
    from ensemble.assistant.agent import run_turn
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "unknown session")
    r = run_turn(s["llm"], s, text, _image(photo) if photo is not None and photo.filename else None)
    return {**{k: r[k] for k in ("answer", "trace", "latency_s", "cost_usd", "hallucinated")},
            "cards": _cards(r["cited"])}


# --- Human gold labels for the visual-search judge (D-019) -------------------------------------------
LABEL_MODES = ("crop_adapter", "text_rerank_image")
# Round 2 is a held-out set, stratified by category (jewellery weighted up), disjoint from round 1.
ROUND2_QUOTA = {"jewellery": 3, "bag": 2, "shoes": 1, "outerwear": 1, "bottom": 1, "top": 1, "sunglasses": 1}


def labels_file(round_: int):
    return cfg.path("reports") / "m7a" / ("human_labels.json" if round_ == 1 else f"human_labels_round{round_}.json")


def _label_tasks(round_: int = 2) -> list[dict]:
    import json as _json
    import random
    rows = _json.loads((cfg.path("reports") / "m7a" / "matches_judged.json").read_text())
    by = {}
    for r in rows:
        if r["mode"] in LABEL_MODES and r["box"]:
            by.setdefault((r["file"], r["garment"]), {})[r["mode"]] = r
    keys = sorted(k for k, v in by.items() if len(v) == len(LABEL_MODES))
    random.Random(11).shuffle(keys)
    if round_ == 1:
        chosen = keys[:10]
    else:
        used = set(keys[:10])
        cat = {k: by[k][LABEL_MODES[0]]["category"] for k in keys}
        chosen = []
        for c, n in ROUND2_QUOTA.items():
            chosen += [k for k in keys if k not in used and cat[k] == c][:n]
    tasks = []
    for f, g in chosen:
        for m in LABEL_MODES:
            r = by[(f, g)][m]
            tasks.append({"task_id": f"{f}|{g}|{m}", "file": f, "garment": g, "mode": m,
                          "category": r["category"], "candidates": r["candidates"]})
    random.Random(12).shuffle(tasks)   # modes interleaved, so the labeller cannot tell them apart
    return tasks


@app.get("/api/label/tasks")
def label_tasks(round: int = 2):
    import json as _json
    f = labels_file(round)
    done = _json.loads(f.read_text()) if f.exists() else {}
    return [{**{k: t[k] for k in ("task_id", "category")}, "crop": f"/api/label/crop/{i}.jpg?round={round}",
             "candidates": _cards(t["candidates"]), "done": t["task_id"] in done, "round": round}
            for i, t in enumerate(_label_tasks(round))]


@app.get("/api/label/crop/{i}.jpg")
def label_crop(i: int, round: int = 2):
    import json as _json
    from fastapi.responses import Response
    from PIL import Image as _Image
    from ensemble.vision.outfit import _resize_bytes
    t = _label_tasks(round)[i]
    det = _json.loads((cfg.path("reports") / "m7a" / "detections_ollama.json").read_text())
    _, small = _resize_bytes(_Image.open(cfg.path("raw") / "outfit_photos" / t["file"]))
    crop = small.crop(det[t["file"]]["items"][t["garment"]]["box"])
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")


class LabelIn(BaseModel):
    task_id: str
    relevant: list[int]
    round: int = 2


@app.post("/api/label")
def save_label(body: LabelIn):
    import json as _json
    f = labels_file(body.round)
    done = _json.loads(f.read_text()) if f.exists() else {}
    done[body.task_id] = body.relevant
    f.write_text(_json.dumps(done, indent=1))
    return {"ok": True, "labelled": len(done)}


@app.get("/images/{article_id}.jpg")
def image(article_id: int):
    s = f"{article_id:010d}"
    path = IMAGES / s[:3] / f"{s}.jpg"
    if not path.exists():
        return FileResponse(STATIC / "placeholder.svg", media_type="image/svg+xml")
    return FileResponse(path, headers={"Cache-Control": "max-age=86400"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
