"""M5: per-recommendation explanations from LightGBM SHAP values (DESIGN §6).

LightGBM's ``pred_contrib=True`` returns exact TreeSHAP attributions: one
contribution per feature plus a bias term, summing to the model score.
Features are grouped into user-facing reasons; each reason is traceable to
the features that produced it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (reason key, user-facing text, feature prefixes)
REASONS = [
    ("bought_before", "You bought this before", ("ca_n_article", "ca_days_since_article", "repeat_")),
    ("same_style", "A style you bought, in another colour", ("ca_n_prod", "ca_share_prod", "variant_")),
    ("co_purchase", "Often bought with items you purchased", ("cf_",)),
    ("product_type", "You often buy this type of item", ("ca_n_type", "ca_share_type")),
    ("colour", "In a colour you often choose", ("ca_n_colour", "ca_share_colour")),
    ("section", "From the sections you shop", ("ca_n_igroup", "ca_share_igroup", "ca_n_ggroup", "ca_share_ggroup")),
    ("price", "In your usual price range", ("ca_price_ratio",)),
    ("age_group", "Popular with customers your age", ("pop_age_", "ca_age_gap", "a_buyer_age")),
    ("trending", "Trending this week", ("a_n_1w", "a_trend", "pop_", "a_n_4w")),
    ("new_arrival", "New arrival", ("new_arrival_", "a_days_since_first_sale")),
]


def reason_of(feature: str) -> str | None:
    for key, _, prefixes in REASONS:
        if any(feature.startswith(p) for p in prefixes):
            return key
    return None


REASON_TEXT = {k: t for k, t, _ in REASONS}


def explain(booster, features: list[str], df: pd.DataFrame, top_features: int = 8, top_reasons: int = 2) -> list[dict]:
    """Return, per row of ``df``, the top reasons and the top feature contributions."""
    contrib = booster.predict(df[features], pred_contrib=True, num_threads=8)[:, :-1]
    keys = [reason_of(f) for f in features]
    out = []
    for row in contrib:
        by_reason: dict[str, float] = {}
        for f_key, v in zip(keys, row):
            if f_key:
                by_reason[f_key] = by_reason.get(f_key, 0.0) + float(v)
        reasons = [k for k, v in sorted(by_reason.items(), key=lambda x: -x[1]) if v > 0][:top_reasons]
        order = np.argsort(-np.abs(row))[:top_features]
        out.append({
            "reasons": [{"key": k, "text": REASON_TEXT[k]} for k in reasons],
            "shap": [{"feature": features[i], "value": round(float(row[i]), 4)} for i in order],
        })
    return out
