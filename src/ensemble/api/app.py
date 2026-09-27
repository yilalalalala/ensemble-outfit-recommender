"""M6: FastAPI serving layer over the SQLite store (DESIGN §7).

  make serve     then open http://localhost:8010
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException
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


@app.get("/api/product/{article_id}")
def product(article_id: int):
    a = q("SELECT * FROM articles WHERE article_id = ?", (article_id,))
    if not a:
        raise HTTPException(404, "unknown article")
    a = card(a[0])
    ctl = q(f"""SELECT c.slot AS target_slot, c.rank, c.source, c.lift, {ART_COLS}
                FROM complete_the_look c JOIN articles a USING (article_id)
                WHERE c.anchor = ? ORDER BY c.slot, c.rank""", (article_id,))
    by_slot: dict[str, list] = {}
    for r in ctl:
        r = card(r)
        r["reasons"] = [{"key": "co_purchase", "text": f"Bought together {r['lift']:.1f}× more often than chance"}
                        if r["source"] == "co_purchase" else {"key": "style", "text": "Style match for this piece"}]
        by_slot.setdefault(r["target_slot"], []).append(r)
    # Show a slot only when customers actually complete this item with it (at least one
    # co-purchase-backed pick); shoes and accessories are always offered. Order by evidence.
    order = ["lower", "upper", "full", "shoes", "accessories", "socks", "swimwear"]
    evidence = {s: sum(i["source"] == "co_purchase" for i in items) for s, items in by_slot.items()}
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
