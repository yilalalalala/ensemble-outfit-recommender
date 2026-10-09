"""M7b tools: the only way the assistant can see products (grounding, DESIGN §7.4).

Every tool returns real articles from the serving store or the models; the
assistant may cite only article ids that a tool returned in this session.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from ensemble.config import load_config

SLOTS = ["upper", "lower", "full", "shoes", "accessories", "socks", "swimwear"]
PRICE_TIERS = ["budget", "mid", "premium"]

TOOL_SPECS = [
    {"name": "search_catalog",
     "description": "Search the live H&M catalogue by a text description, e.g. 'gold hoop earrings' or "
                    "'black wide-leg trousers'. Use for explicit product requests.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What the customer is looking for, in product words."},
         "slot": {"type": "string", "enum": SLOTS, "description": "Optional wearable slot filter."},
         "colour": {"type": "string", "description": "Optional colour filter, e.g. 'black', 'gold'."},
         "price": {"type": "string", "enum": PRICE_TIERS, "description": "Optional price filter relative to the slot."},
         "max_results": {"type": "integer", "minimum": 1, "maximum": 8}},
         "required": ["query"]}},
    {"name": "recommend_for_customer",
     "description": "Personalised recommendations for the signed-in customer from their purchase history "
                    "(next-purchase ranker). Use for open requests like 'what should I buy'.",
     "parameters": {"type": "object", "properties": {
         "customer_idx": {"type": "integer"},
         "slot": {"type": "string", "enum": SLOTS, "description": "Only if the customer names a category (e.g. 'trousers')."},
         "max_results": {"type": "integer", "minimum": 1, "maximum": 8}},
         "required": ["customer_idx"]}},
    {"name": "complete_the_look",
     "description": "Items that complete an outfit with a given anchor article, learned from what customers "
                    "buy together. Use when the customer asks what goes with a specific item.",
     "parameters": {"type": "object", "properties": {
         "article_id": {"type": "integer"},
         "slot": {"type": "string", "enum": SLOTS, "description": "Only if the customer names the part of the outfit to fill "
                                                                  "(e.g. 'which shoes'). Never the anchor's own slot."},
         "product_type": {"type": "string", "description": "Optional, e.g. 'Earring', 'Bag', 'Boots'."},
         "colour": {"type": "string"},
         "max_results": {"type": "integer", "minimum": 1, "maximum": 8}},
         "required": ["article_id"]}},
    {"name": "analyze_outfit_photo",
     "description": "Analyse a photo the customer uploaded: detects each garment, finds the closest catalogue "
                    "articles, and lists which parts of the outfit are missing.",
     "parameters": {"type": "object", "properties": {"photo_id": {"type": "string"}}, "required": ["photo_id"]}},
    {"name": "get_article_details",
     "description": "Details for up to 8 article ids that a previous tool returned.",
     "parameters": {"type": "object", "properties": {"article_ids": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["article_ids"]}},
]


class Toolbox:
    def __init__(self, session: dict):
        cfg = load_config()
        self.db = cfg.path("serving_db")
        self.session = session          # holds photos, grounded ids, customer
        self.session.setdefault("grounded", set())

    def _q(self, sql: str, args=()) -> list[dict]:
        con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute(sql, args).fetchall()]
        finally:
            con.close()

    def _card(self, ids: list[int], extra: dict | None = None) -> list[dict]:
        if not ids:
            return []
        rows = {r["article_id"]: r for r in self._q(
            f"""SELECT article_id, prod_name, product_type_name, colour_group_name, slot, department_name
                FROM articles WHERE article_id IN ({','.join('?' * len(ids))})""", ids)}
        out = []
        for a in ids:
            if a in rows:
                r = dict(rows[a])
                r.update((extra or {}).get(a, {}))
                out.append(r)
                self.session["grounded"].add(int(a))
        return out

    # --- tools -----------------------------------------------------------------------------------
    def search_catalog(self, query: str, slot: str | None = None, colour: str | None = None,
                       price: str | None = None, max_results: int = 6) -> dict:
        from ensemble.vision import clip
        from ensemble.vision.outfit import catalogue
        live, emb = catalogue()
        mask = np.ones(len(live), dtype=bool)
        if slot in SLOTS:
            mask &= (live.slot == slot).values
        if colour:
            mask &= live.colour_group_name.str.lower().str.contains(colour.lower().split()[0], regex=False).values
        if price in PRICE_TIERS:
            tier = self._price_tier(live)
            mask &= (tier == PRICE_TIERS.index(price)).values
        if not mask.any():
            return {"results": [], "note": "No live articles match those filters."}
        idx = np.flatnonzero(mask)
        s = emb[idx] @ clip.embed_texts([query])[0]
        top = idx[np.argsort(-s)[:max_results]]
        return {"results": self._card(live.article_id.values[top].tolist())}

    def _price_tier(self, live: pd.DataFrame) -> pd.Series:
        if "price_tier" not in self.session:
            # The public source catalogue has no retail price.  Use the exact same
            # deterministic prototype merchandising policy as the shopper cards,
            # then derive relative tiers within each wearable slot.
            from ensemble.api.merch import price_usd
            m = live.copy()
            m["p"] = [price_usd(r) for r in m.to_dict("records")]
            m["tier"] = m.groupby("slot").p.transform(
                lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
            self.session["price_tier"] = m.set_index("article_id").tier.reindex(live.article_id).fillna(1).values
        return pd.Series(self.session["price_tier"])

    def recommend_for_customer(self, customer_idx: int, slot: str | None = None, max_results: int = 6) -> dict:
        rows = self._q("""SELECT f.article_id, f.reasons FROM for_you f JOIN articles a USING (article_id)
                          WHERE f.customer_idx = ? AND (? IS NULL OR a.slot = ?) ORDER BY f.rank""",
                       (customer_idx, slot, slot))
        note = None
        if not rows and slot:
            rows = self._q("""SELECT f.article_id, f.reasons FROM for_you f WHERE f.customer_idx = ? ORDER BY f.rank""",
                           (customer_idx,))
            note = f"No personalised picks in slot '{slot}'; showing the customer's top picks across categories."
        if not rows:
            return {"results": [], "note": "No personalised recommendations for this customer."}
        rows = rows[:max_results]
        reasons = {r["article_id"]: {"reasons": [x["text"] for x in json.loads(r["reasons"])]} for r in rows}
        out = {"results": self._card([r["article_id"] for r in rows], reasons)}
        if note:
            out["note"] = note
        return out

    def complete_the_look(self, article_id: int, slot: str | None = None, product_type: str | None = None,
                          colour: str | None = None, max_results: int = 6) -> dict:
        # Guard against fabricated arguments: the anchor must come from a tool result or the customer.
        known = self.session["grounded"] | self.session.get("user_ids", set())
        if int(article_id) not in known:
            return {"results": [], "error": "Unknown article_id. Only use ids returned by a tool or given by the "
                                            "customer. If the customer only described the item, call search_catalog "
                                            "first to find it, then call complete_the_look with a returned id."}
        rows = self._q("""SELECT c.article_id, c.slot, c.source, c.lift, a.product_type_name, a.colour_group_name
                          FROM complete_the_look c JOIN articles a USING (article_id)
                          WHERE c.anchor = ? ORDER BY c.slot, c.rank""", (article_id,))
        if not rows:
            return {"results": [], "note": "This article has no outfit data (not in the live wearable catalogue)."}
        # Filters are applied in order and relaxed from the last one when nothing matches
        # (graceful relaxation), so a too-specific request still gets grounded suggestions.
        filters = [("slot", slot, lambda r: r["slot"] == slot),
                   ("product_type", product_type, lambda r: product_type.lower() in r["product_type_name"].lower()),
                   ("colour", colour, lambda r: colour.lower().split()[0] in r["colour_group_name"].lower())]
        active = [f for f in filters if f[1]]
        relaxed = []
        while True:
            kept = [r for r in rows if all(fn(r) for _, _, fn in active)]
            if kept or not active:
                break
            relaxed.append(active.pop()[0])
        rows = kept[:max_results]
        why = {r["article_id"]: {"evidence": (f"bought together {r['lift']:.1f}x more often than chance"
                                              if r["source"] == "co_purchase" else "style match")} for r in rows}
        self.session["grounded"].add(int(article_id))
        out = {"anchor": article_id, "results": self._card([r["article_id"] for r in rows], why)}
        if relaxed:
            out["note"] = f"Nothing matched every filter; relaxed: {', '.join(relaxed)}."
        return out

    def analyze_outfit_photo(self, photo_id: str) -> dict:
        from ensemble.vision.outfit import snap
        img = self.session.get("photos", {}).get(photo_id)
        if img is None:
            return {"error": f"unknown photo_id {photo_id}"}
        res = snap(img, self.session["vlm"])
        self.session.setdefault("snaps", {})[photo_id] = res
        garments = []
        for g in res["garments"]:
            garments.append({"category": g["category"], "colour": g["colour"], "description": g["description"],
                             "closest_articles": self._card([m["article_id"] for m in g["matches"][:3]])})
        ctl = {s: self._card(ids[:4]) for s, ids in res["complete_the_look"].items()}
        return {"garments": garments, "missing_slots": res["missing_slots"],
                "anchor_article": res["anchor"]["article_id"] if res["anchor"] else None,
                "complete_the_look_for_missing_slots": ctl}

    def get_article_details(self, article_ids: list[int]) -> dict:
        ids = [int(a) for a in article_ids[:8] if int(a) in self.session["grounded"]]
        rows = self._q(f"""SELECT article_id, prod_name, product_type_name, colour_group_name, department_name, detail_desc
                           FROM articles WHERE article_id IN ({','.join('?' * len(ids))})""", ids) if ids else []
        return {"results": rows, "note": None if len(ids) == len(article_ids) else
                "Only articles returned by earlier tool calls can be looked up."}

    def call(self, name: str, arguments: dict) -> dict:
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_") or name not in {t["name"] for t in TOOL_SPECS}:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**arguments)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}
