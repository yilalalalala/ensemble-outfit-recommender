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
# Below this co-purchase count a lift ratio is a small-sample artefact, so the chip
# states the count only (D-035).
LIFT_MIN_SUPPORT = 5


def _ctl_columns() -> list[str]:
    """Columns the serving store actually has, so an older table still renders."""
    try:
        return [r["name"] for r in q("PRAGMA table_info(complete_the_look)")]
    except Exception:  # noqa: BLE001 - no serving store yet; the endpoint 404s anyway
        return []


def ctl_reasons(r: dict) -> list[dict]:
    """Reason chips for one Complete-the-Look pick, gated on evidence present in the row.

    A chip is emitted only when the column it quotes is non-null for *this* pair
    (D-035): a visually retrieved pick never claims co-purchase support, and a
    popularity fallback says so instead of borrowing a style explanation.
    """
    def num(key):
        v = r.get(key)
        return v if isinstance(v, (int, float)) and v == v else None

    out: list[dict] = []
    co, lift, s_co = num("a_co"), num("lift"), num("s_co")
    if r.get("source") in ("co_purchase", "style_match") and co:
        # The multiplier is only quoted when the pair has enough support for the ratio to
        # mean something: at co = 4 a lift of 193x is a small-sample artefact, and a chip
        # that says so is the same kind of unsupported claim D-030 rules out.
        text = f"Bought together {int(co)}× in past baskets"
        if lift and co >= LIFT_MIN_SUPPORT:
            text += f", {lift:.1f}× more often than chance"
        out.append({"key": "co_purchase", "text": text})
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
                                         "src_two_tower", "src_two_tower_content", "src_slot_pop",
                                         "tt_pair_sim", "ttc_pair_sim", "clip_sim",
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


MAX_FAMILY_LOOKUP = 400


def _has_image(article_id: int) -> bool:
    s = f"{article_id:010d}"
    return (IMAGES / s[:3] / f"{s}.jpg").exists()


@app.get("/api/catalog/families")
def catalog_families(articles: str = ""):
    """Shop UI: the product family (real ``product_code``), its photographed colourways and the
    prototype USD price (``merch.py`` — demo merchandising prices, not source prices) for each
    requested article. Additive endpoint; no existing response changes."""
    from ensemble.api import merch
    try:
        ids = sorted({int(x) for x in articles.split(",") if x.strip()})
    except ValueError:
        raise HTTPException(400, "articles must be a comma-separated list of article ids")
    if len(ids) > MAX_FAMILY_LOOKUP:
        raise HTTPException(400, f"at most {MAX_FAMILY_LOOKUP} articles per request")
    if not ids:
        return {"articles": {}, "families": {}}
    own = q(f"SELECT article_id, product_code FROM articles WHERE article_id IN ({','.join('?' * len(ids))})", ids)
    code_of = {r["article_id"]: r["product_code"] for r in own}
    codes = sorted(set(code_of.values()))
    rows = q(f"""SELECT article_id, product_code, prod_name, product_type_name, product_group_name, colour_group_name,
                        department_name, index_group_name FROM articles
                 WHERE product_code IN ({','.join('?' * len(codes))}) ORDER BY article_id""", codes)
    families: dict[str, dict] = {}
    for r in rows:
        f = families.setdefault(r["product_code"], {"product_code": r["product_code"], "price_usd": merch.price_usd(r),
                                                    "variants": []})   # first row = canonical (lowest article_id)
        if r["article_id"] in code_of or _has_image(r["article_id"]):
            f["variants"].append({"article_id": r["article_id"], "prod_name": r["prod_name"],
                                  "colour_group_name": r["colour_group_name"], "image": f"/images/{r['article_id']}.jpg"})
    return {"articles": {str(a): c for a, c in code_of.items()}, "families": families}


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


# --- v2: request-time Complete the Look over a versioned serving bundle (D-041, D-042) ---------------
# Typed request/response schemas, a stable error schema with a request id, health/readiness,
# metrics and structured logs. The v1 endpoints above are unchanged. This section imports
# neither LightGBM (the ranker runs on the NumPy runtime) nor PyTorch (loaded lazily by the
# visual endpoints only).
import contextlib as _contextlib  # noqa: E402
import time as _time  # noqa: E402
import uuid as _uuid  # noqa: E402

from fastapi import Query, Request  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402

from ensemble.serving.observability import pseudonymize, request_logger  # noqa: E402
from ensemble.serving.runtime import BundleError, RequestError  # noqa: E402
from ensemble.serving.service import ServingState  # noqa: E402

STATE = ServingState()
LOG = request_logger()
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


class Reason(BaseModel):
    key: str
    evidence_type: str
    text: str


class ArticleCard(BaseModel):
    article_id: int
    prod_name: str | None = None
    product_type_name: str | None = None
    colour_group_name: str | None = None
    image: str


class CTLItem(BaseModel):
    article_id: int
    rank: int
    target_slot: str
    score: float | None
    provenance: str
    evidence_types: list[str]
    reasons: list[Reason]
    backfill: bool
    evidence: dict[str, float]
    card: ArticleCard


class CTLModule(BaseModel):
    slot: str
    title: str
    fallback_level: str
    source_anchor: int | None
    personalized: bool
    n_pool: int
    cache: str
    items: list[CTLItem]


class CTLResponse(BaseModel):
    request_id: str
    anchor: int
    anchor_slot: str
    customer_known: bool
    personalized: bool
    model_version: str
    catalog_version: str
    profile_version: str
    availability_version: str
    bundle_version: str
    modules: list[CTLModule]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request.state.error_class = code
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
                                                               "request_id": _rid(request)}},
                        headers={"X-Request-ID": _rid(request)})


@app.exception_handler(RequestError)
async def _request_error(request: Request, exc: RequestError):
    return _error(request, exc.status, exc.code, str(exc))


@app.middleware("http")
async def _observe(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or _uuid.uuid4().hex[:16]
    if len(rid) > 64 or not rid.replace("-", "").isalnum():
        rid = _uuid.uuid4().hex[:16]
    request.state.request_id = rid
    t0 = _time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001 - logged with its class and turned into the stable error schema
        request.state.error_class = type(exc).__name__
        LOG.exception("unhandled error", extra={"request_id": rid, "endpoint": request.url.path,
                                                "error_class": type(exc).__name__})
        response = JSONResponse(status_code=500, content={"error": {"code": "internal_error",
                                                                    "message": "internal error", "request_id": rid}})
    ms = (_time.perf_counter() - t0) * 1000
    path = request.scope.get("route").path if request.scope.get("route") is not None else "unmatched"
    response.headers["X-Request-ID"] = rid
    if path.startswith("/api/v2") or path in ("/healthz", "/readyz", "/metrics") or path.startswith("/admin"):
        STATE.metrics.inc("requests", endpoint=path, status=response.status_code)
        STATE.metrics.observe_ms("latency", ms, endpoint=path)
        s = request.state
        extra = {"request_id": rid, "endpoint": path, "method": request.method, "status": response.status_code,
                 "latency_ms": round(ms, 2), "error_class": getattr(s, "error_class", None),
                 "result_count": getattr(s, "result_count", None), "fallback_levels": getattr(s, "fallbacks", None),
                 "personalized": getattr(s, "personalized", None), "cache": getattr(s, "cache", None),
                 "customer_ref": getattr(s, "customer_ref", None), "degraded": getattr(s, "degraded", None)}
        rec = STATE.rec
        if rec is not None:
            extra.update(bundle_version=rec.bundle.version, model_version=rec.bundle.model_version,
                         catalog_version=rec.bundle.catalog_version)
        LOG.info("request", extra=extra)
    return response


@_contextlib.asynccontextmanager
async def _lifespan(_app):
    """Load the serving bundle at startup; readiness stays false (and says why) if it fails."""
    STATE.try_load()
    LOG.info("startup", extra={"event": "bundle_load", "degraded": None if STATE.ready else STATE.error})
    yield


app.router.lifespan_context = _lifespan


def _rec(request: Request):
    rec = STATE.rec
    if rec is None:
        raise RequestError("not_ready", f"serving bundle not loaded: {STATE.error or 'loading'}", 503)
    return rec


def _card(rec, a: int) -> dict:
    c = rec.bundle.catalog
    return {"article_id": a, "prod_name": c.at[a, "prod_name"], "product_type_name": c.at[a, "product_type_name"],
            "colour_group_name": c.at[a, "colour_group_name"], "image": f"/images/{a}.jpg"}


@app.get("/healthz")
def healthz():
    """Liveness: the process is up (it may still be loading or degraded; see /readyz)."""
    return {"status": "ok", "uptime_s": round(_time.time() - STATE.started, 1)}


@app.get("/readyz")
def readyz():
    """Readiness: true only when a validated bundle is loaded."""
    rec = STATE.rec
    if rec is None:
        return JSONResponse(status_code=503, content={"ready": False, "error": STATE.error or "loading"})
    return {"ready": True, "bundle_version": rec.bundle.version, "model_version": rec.bundle.model_version,
            "catalog_version": rec.bundle.catalog_version, "profile_version": rec.bundle.profile_version,
            "loaded_at": STATE.loaded_at}


@app.get("/api/v2/meta")
def meta(request: Request):
    rec = _rec(request)
    m = rec.bundle.manifest
    vis = STATE.visual
    return {"bundle_version": m["bundle_version"], "model_version": m["model_version"],
            "catalog_version": m["catalog_version"], "profile_version": m["profile_version"],
            "availability_version": rec.availability_version, "serving_week": m.get("serving_week"),
            "cutoff": m.get("cutoff"), "pool": m.get("pool"), "n_keys": m.get("n_keys"),
            "n_live_articles": m.get("n_live_articles"), "rankers": m.get("rankers"),
            "equivalence": m.get("equivalence"), "profiles": m.get("profiles"),
            "visual": {"index": m.get("visual"), "state": vis.state if vis else None,
                       "reason": vis.reason if vis else None},
            "personalization_enabled": rec.personalization, "load_seconds": rec.bundle.load_seconds,
            "upload_limits": {k: v for k, v in __import__("ensemble.serving.visual", fromlist=["LIMITS"]).LIMITS.items()}}


@app.get("/api/v2/complete-the-look", response_model=CTLResponse, responses={404: {"model": ErrorResponse},
                                                                           422: {"model": ErrorResponse},
                                                                           503: {"model": ErrorResponse}})
def complete_the_look_v2(request: Request, anchor: int = Query(..., ge=0), slots: str | None = None,
                         customer: int | None = Query(None, ge=0), k: int = Query(8, ge=1, le=24),
                         diversity: bool = True):
    """Complementary items for ``anchor`` in each requested (or every other) slot. With ``customer``
    the pool is re-ordered by the personalized ranker when that customer has a profile."""
    rec = _rec(request)
    request.state.customer_ref = pseudonymize(customer)
    res = rec.complete_the_look(anchor, [s.strip() for s in slots.split(",")] if slots else None, customer, k, diversity)
    for m in res["modules"]:
        for it in m["items"]:
            it["card"] = _card(rec, it["article_id"])
        STATE.metrics.inc("ctl_modules", fallback_level=m["fallback_level"], personalized=str(m["personalized"]).lower())
        STATE.metrics.inc("ctl_cache", outcome=m["cache"])
        for it in m["items"]:
            STATE.metrics.inc("ctl_items", provenance=it["provenance"])
    request.state.result_count = sum(len(m["items"]) for m in res["modules"])
    request.state.fallbacks = sorted({m["fallback_level"] for m in res["modules"]})
    request.state.personalized = res["personalized"]
    request.state.cache = sorted({m["cache"] for m in res["modules"]})
    res.pop("customer_idx", None)
    return {"request_id": _rid(request), **res}


async def _read_upload(photo: UploadFile) -> bytes:
    from ensemble.serving.visual import LIMITS
    data = await photo.read(LIMITS["max_bytes"] + 1)
    return data


@app.post("/api/v2/visual-search", responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                                              415: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def visual_search_v2(request: Request, photo: UploadFile = File(...), category: str = Form(...),
                           box: str = Form(""), k: int = Form(8)):
    """Street-to-shop search for one garment (crop + DeepFashion2 adapter; raw crop if the adapter is absent)."""
    from starlette.concurrency import run_in_threadpool

    from ensemble.serving.visual import decode_upload, parse_box
    rec = _rec(request)
    img = decode_upload(await _read_upload(photo), photo.content_type)
    res = await run_in_threadpool(STATE.visual.search, img, category, parse_box(box, img.width, img.height), k,
                                  rec.unavailable)
    request.state.result_count = len(res["matches"])
    request.state.degraded = None if res["mode"] == "crop_adapter" else "adapter_missing"
    STATE.metrics.inc("visual_searches", mode=res["mode"])
    return {"request_id": _rid(request), "mode": res["mode"], "degraded": res["mode"] != "crop_adapter",
            "candidates_searched": res["candidates_searched"], "catalog_version": rec.bundle.catalog_version,
            "matches": [{**_card(rec, m["article_id"]), "similarity": m["similarity"],
                         "reasons": [{"key": "visual", "evidence_type": "visual_similarity",
                                      "text": f"Visual match {m['similarity']:.2f}"}]} for m in res["matches"]]}


@app.post("/api/v2/outfit", responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def outfit_v2(request: Request, photo: UploadFile = File(...), garments: str = Form(""), slots: str = Form(""),
                    customer: int | None = Form(None), anchor: int | None = Form(None), k: int = Form(8)):
    """Outfit photo -> garments -> catalogue matches -> complementary slots.

    Garments come from the client (``garments`` JSON: what the user tapped) or, if absent, from
    the local VLM detector. Recommended slots are ``slots`` if given, otherwise the core slots the
    photo is missing; a complete outfit gets ``outfit_complete: true`` and no invented gap."""
    from starlette.concurrency import run_in_threadpool

    from ensemble.serving.visual import CATEGORIES, MAIN_ORDER, decode_upload, detect_garments, gaps, parse_garments
    rec = _rec(request)
    request.state.customer_ref = pseudonymize(customer)
    img = decode_upload(await _read_upload(photo), photo.content_type)
    found = parse_garments(garments, img.width, img.height)
    source = "client"
    if found is None:
        found = await run_in_threadpool(detect_garments, img)
        source = "detector"
    out_g = []
    for g in found:
        res = await run_in_threadpool(STATE.visual.search, img, g["category"], g.get("box"), 5, rec.unavailable)
        out_g.append({"category": g["category"], "slot": CATEGORIES[g["category"]][0], "box": g.get("box"),
                      "mode": res["mode"], "matches": [{**_card(rec, m["article_id"]), "similarity": m["similarity"]}
                                                       for m in res["matches"]]})
    present = sorted({g["slot"] for g in out_g})
    missing = gaps({g["category"] for g in out_g})
    requested = [s.strip() for s in slots.split(",") if s.strip()] if slots else []
    targets = requested or missing
    if anchor is None:
        main = next((g for c in MAIN_ORDER for g in out_g if g["category"] == c and g["matches"]), None)
        anchor = main["matches"][0]["article_id"] if main else None
    modules = []
    if targets and anchor is not None:
        a_slot = rec.bundle.catalog.at[anchor, "slot"] if anchor in rec.bundle.catalog.index else None
        res = rec.complete_the_look(anchor, [t for t in targets if t != a_slot] or None, customer, k)
        for m in res["modules"]:
            for it in m["items"]:
                it["card"] = _card(rec, it["article_id"])
        modules = res["modules"]
    request.state.result_count = sum(len(m["items"]) for m in modules)
    return {"request_id": _rid(request), "garment_source": source, "garments": out_g, "present_slots": present,
            "missing_slots": missing, "outfit_complete": not missing, "requested_slots": requested,
            "anchor": anchor, "modules": modules, "model_version": rec.bundle.model_version,
            "catalog_version": rec.bundle.catalog_version}


@app.get("/api/v2/metrics")
def metrics_v2():
    rec = STATE.rec
    snap = STATE.metrics.snapshot()
    if rec is not None:
        snap["cache"] = {"hits": rec.cache.hits, "misses": rec.cache.misses, "size": len(rec.cache.d),
                         "capacity": rec.cache.size}
    snap["ready"] = STATE.ready
    snap["load_history"] = STATE.load_history[-10:]
    return snap


@app.get("/metrics", response_class=PlainTextResponse)
def metrics_prometheus():
    rec = STATE.rec
    text = STATE.metrics.prometheus()
    if rec is not None:
        text += f"ensemble_cache_hits {rec.cache.hits}\nensemble_cache_misses {rec.cache.misses}\n"
    text += f"ensemble_ready {int(STATE.ready)}\n"
    return text


class ReloadIn(BaseModel):
    path: str | None = None
    verify_hashes: bool = False


def _local_only(request: Request) -> None:
    host = request.client.host if request.client else None
    if host not in LOCAL_HOSTS:
        raise RequestError("forbidden", "admin endpoints are local-only", 403)


@app.post("/admin/reload")
def admin_reload(request: Request, body: ReloadIn):
    """Load and validate a bundle, then swap it in atomically; on failure the old bundle keeps serving."""
    _local_only(request)
    try:
        return STATE.load(body.path, verify_hashes=body.verify_hashes)
    except BundleError as e:
        raise RequestError("bundle_invalid", str(e), 409) from None


class AvailabilityIn(BaseModel):
    unavailable: list[int]


@app.post("/admin/availability")
def admin_availability(request: Request, body: AvailabilityIn):
    """Runtime stock-outs on top of the bundle's availability snapshot (no inventory feed exists)."""
    _local_only(request)
    rec = _rec(request)
    return {"availability_version": rec.set_unavailable(body.unavailable), "n_unavailable": len(rec.unavailable)}


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
