"""Reason chips: SHAP attribution alone never produces a claim the raw features do not support."""
import numpy as np
import pandas as pd
import pytest

from ensemble.ranking import explain as E


class StubBooster:
    """Returns fixed SHAP contributions (last column = bias), like LightGBM's pred_contrib."""

    def __init__(self, contrib):
        self.contrib = np.asarray(contrib, dtype=float)

    def predict(self, X, pred_contrib=False, num_threads=1):
        assert pred_contrib
        return self.contrib


@pytest.mark.parametrize("feature,key", [
    ("pop_age_rank", "age_group"), ("pop_rank", "trending"), ("ca_w_article", "bought_before"),
    ("ca_days_since_article", "bought_before"), ("covis_score", "co_purchase"), ("cf_rank", "co_purchase"),
    ("dept_pop_rank", "section"), ("ca_n_prod_life", "same_style"), ("als_score", "similar_customers"),
    ("a_n_12w", None), ("c_n_tx_life", None),
])
def test_reason_mapping(feature, key):
    assert E.reason_of(feature) == key


def test_every_reason_has_text_and_evidence_type():
    for key, text, ev, prefixes, pred in E.REASONS:
        assert text and prefixes and callable(pred)
        assert ev in {"personal_history", "similarity", "trending"}


def test_bought_before_requires_a_purchase():
    assert not E.supported("bought_before", {"ca_n_article_life": 0.0, "ca_days_since_article": float("nan")})
    assert E.supported("bought_before", {"ca_n_article_life": 2.0})


def test_same_style_requires_another_colourway():
    # Bought this exact article only: the style claim ("in another colour") is not supported.
    assert not E.supported("same_style", {"ca_n_prod_life": 1.0, "ca_n_article_life": 1.0})
    assert E.supported("same_style", {"ca_n_prod_life": 2.0, "ca_n_article_life": 1.0})


def test_price_and_trending_thresholds():
    assert E.supported("price", {"ca_price_ratio": 1.0})
    assert not E.supported("price", {"ca_price_ratio": 2.5})
    assert not E.supported("trending", {"a_trend_1w_4w": 3.0, "a_n_1w": 5.0})      # too few sales
    assert E.supported("trending", {"pop_rank": 3.0})


def test_numpy_nan_is_not_evidence():
    assert not E.supported("co_purchase", {"cf_rank": np.float32("nan"), "covis_rank": np.float32("nan")})
    assert E.supported("co_purchase", {"cf_rank": np.float32(4)})


def test_explain_drops_unsupported_reasons():
    features = ["ca_days_since_article", "ca_n_article_life", "pop_rank", "a_n_1w"]
    df = pd.DataFrame({"ca_days_since_article": [np.nan, 10.0], "ca_n_article_life": [0.0, 1.0],
                       "pop_rank": [5.0, np.nan], "a_n_1w": [500.0, 20.0]}, dtype="float32")
    # Row 0: a large positive SHAP on the *missing* purchase date, and on popularity.
    contrib = [[0.9, 0.0, 0.2, 0.1, 0.0], [0.5, 0.3, 0.0, -0.1, 0.0]]
    out = E.explain(StubBooster(contrib), features, df)
    assert [r["key"] for r in out[0]["reasons"]] == ["trending"]
    assert [r["key"] for r in out[1]["reasons"]] == ["bought_before"]
    assert out[1]["reasons"][0]["evidence"] == "personal_history"
    # Without the evidence gate the unsupported claim would have been shown.
    raw = E.explain(StubBooster(contrib), features, df, require_evidence=False)
    assert raw[0]["reasons"][0]["key"] == "bought_before"
