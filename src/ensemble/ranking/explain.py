"""M5: per-recommendation explanations from LightGBM SHAP values (DESIGN §6).

LightGBM's ``pred_contrib=True`` returns exact TreeSHAP attributions: one
contribution per feature plus a bias term, summing to the model score.

**What a reason chip claims.** SHAP is *model attribution*: it says which
features moved this score, not why the customer will buy. A chip is shown
only when two conditions hold:

1. **attribution** — the reason's features push the score up (positive summed
   SHAP), and
2. **evidence** — the raw feature values support the sentence. For example
   "You bought this before" requires a recorded purchase of this exact article
   (``ca_n_article_life ≥ 1``); a positive SHAP value on a *missing* purchase
   date is not evidence of a purchase.

Every reason belongs to one evidence type, so the UI can tell them apart:
``personal_history`` (the customer's own purchases), ``similarity``
(co-purchase or similar customers) and ``trending`` (what sells overall).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _has(row, f) -> bool:
    v = row.get(f)
    return v is not None and v == v  # NaN (float or numpy scalar) means "no value"


def _at_least(f, x):
    return lambda r: _has(r, f) and r[f] >= x


def _any_rank(*channels):
    return lambda r: any(_has(r, f"{c}_rank") for c in channels)


# Thresholds for evidence (DS-tunable; deliberately conservative).
MIN_TYPE_PURCHASES = 2         # "You often buy this type of item"
MIN_COLOUR_SHARE = 0.2         # "In a colour you often choose"
MIN_SECTION_SHARE = 0.3        # "From the sections you shop"
PRICE_BAND = (0.75, 1.33)      # "In your usual price range": article price / customer mean
MAX_AGE_GAP = 8                # "Popular with customers your age": |age − mean buyer age|
MIN_TREND = 1.2                # "Trending this week": last week vs 4-week weekly average …
MIN_WEEKLY_SALES = 100         # … and at least this many sales last week
MAX_NEW_DAYS = 28              # "New arrival": first observed sale within 4 weeks

# (key, text, evidence type, feature prefixes, evidence predicate on raw features)
REASONS = [
    ("bought_before", "You bought this before", "personal_history",
     ("ca_n_article", "ca_days_since_article", "ca_w_article", "repeat_"), _at_least("ca_n_article_life", 1)),
    ("same_style", "A style you bought, in another colour", "personal_history",
     ("ca_n_prod", "ca_share_prod", "ca_w_prod", "ca_wshare_prod", "variant_"),
     lambda r: _has(r, "ca_n_prod_life") and r["ca_n_prod_life"] > r.get("ca_n_article_life", 0)),
    ("co_purchase", "Often bought with items you purchased", "similarity", ("cf_", "covis_"), _any_rank("cf", "covis")),
    ("similar_customers", "Popular with shoppers like you", "similarity", ("als_",), _any_rank("als")),
    ("product_type", "You often buy this type of item", "personal_history",
     ("ca_n_type", "ca_share_type", "ca_w_type", "ca_wshare_type", "ca_days_since_type"),
     _at_least("ca_n_type_life", MIN_TYPE_PURCHASES)),
    ("colour", "In a colour you often choose", "personal_history",
     ("ca_n_colour", "ca_share_colour", "ca_w_colour", "ca_wshare_colour"), _at_least("ca_share_colour_life", MIN_COLOUR_SHARE)),
    ("section", "From the sections you shop", "personal_history",
     ("ca_n_igroup", "ca_share_igroup", "ca_n_ggroup", "ca_share_ggroup", "ca_w_igroup", "ca_wshare_igroup",
      "ca_w_ggroup", "ca_wshare_ggroup", "ca_wshare_dept", "ca_days_since_dept", "dept_pop_", "section_pop_", "type_pop_"),
     lambda r: max(r.get("ca_share_igroup_life", 0) or 0, r.get("ca_share_ggroup_life", 0) or 0) >= MIN_SECTION_SHARE),
    ("price", "In your usual price range", "personal_history", ("ca_price_ratio",),
     lambda r: _has(r, "ca_price_ratio") and PRICE_BAND[0] <= r["ca_price_ratio"] <= PRICE_BAND[1]),
    ("age_group", "Popular with customers your age", "trending", ("pop_age_", "ca_age_gap", "a_buyer_age"),
     lambda r: _has(r, "pop_age_rank") or (_has(r, "ca_age_gap") and abs(r["ca_age_gap"]) <= MAX_AGE_GAP)),
    ("trending", "Trending this week", "trending", ("a_n_1w", "a_n_1d", "a_n_3d", "a_trend", "pop_", "a_n_4w"),
     lambda r: _has(r, "pop_rank") or (_has(r, "a_trend_1w_4w") and r["a_trend_1w_4w"] >= MIN_TREND
                                         and r.get("a_n_1w", 0) >= MIN_WEEKLY_SALES)),
    ("new_arrival", "New arrival", "trending", ("new_arrival_", "a_days_since_first_sale"),
     lambda r: _has(r, "a_days_since_first_sale") and r["a_days_since_first_sale"] <= MAX_NEW_DAYS),
]
REASON_TEXT = {k: t for k, t, *_ in REASONS}
REASON_TYPE = {k: e for k, _, e, *_ in REASONS}
_EVIDENCE = {k: ev for k, *_, ev in REASONS}


def reason_of(feature: str) -> str | None:
    # ``pop_age_`` must win over ``pop_``: check longer prefixes first.
    best = None
    for key, _, _, prefixes, _ in REASONS:
        for p in prefixes:
            if feature.startswith(p) and (best is None or len(p) > best[1]):
                best = (key, len(p))
    return best[0] if best else None


def supported(key: str, row: dict) -> bool:
    """Whether the raw feature values support the reason's sentence."""
    return bool(_EVIDENCE[key](row))


def explain(booster, features: list[str], df: pd.DataFrame, top_features: int = 8, top_reasons: int = 2,
            require_evidence: bool = True) -> list[dict]:
    """Return, per row of ``df``, the top supported reasons and the top feature contributions."""
    contrib = booster.predict(df[features], pred_contrib=True, num_threads=8)[:, :-1]
    keys = [reason_of(f) for f in features]
    raw = df[features].to_dict("records")
    out = []
    for row, vals in zip(contrib, raw):
        by_reason: dict[str, float] = {}
        for f_key, v in zip(keys, row):
            if f_key:
                by_reason[f_key] = by_reason.get(f_key, 0.0) + float(v)
        ranked = [k for k, v in sorted(by_reason.items(), key=lambda x: -x[1]) if v > 0]
        if require_evidence:
            ranked = [k for k in ranked if supported(k, vals)]
        reasons = ranked[:top_reasons]
        order = np.argsort(-np.abs(row))[:top_features]
        out.append({
            "reasons": [{"key": k, "text": REASON_TEXT[k], "evidence": REASON_TYPE[k]} for k in reasons],
            "shap": [{"feature": features[i], "value": round(float(row[i]), 4)} for i in order],
        })
    return out
