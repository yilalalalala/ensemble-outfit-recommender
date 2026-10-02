"""Ranker training: temporal early stopping, reproducibility, deterministic tie-breaking, re-ranking rules."""
import numpy as np
import pandas as pd

from ensemble.config import Config, load_config
from ensemble.ranking.train import TrainingData, fit, top_k


def synthetic(n_weeks=3, n_cust=300, n_cand=20, seed=0) -> TrainingData:
    rng = np.random.default_rng(seed)
    data = TrainingData(["signal", "noise", "product_type_no"])
    for _ in range(n_weeks):
        n = n_cust * n_cand
        signal = rng.normal(size=n)
        X = np.c_[signal, rng.normal(size=n), rng.integers(0, 5, n)].astype(np.float32)
        y = (signal + rng.normal(scale=1.0, size=n) > 2.0).astype(np.int8)
        data.X.append(X)
        data.y.append(y)
        data.groups.append(np.full(n_cust, n_cand))
    return data


def cfg_with(**ranker):
    base = load_config()
    r = dict(base.ranker)
    r.update(ranker)
    r["params"] = {**dict(base.ranker.params), "num_threads": 2, "min_child_samples": 20}
    return Config({**base, "ranker": r})


def test_early_stopping_uses_last_training_week_and_refits():
    cfg = cfg_with(early_stopping={"max_rounds": 300, "patience": 10, "refit": True})
    booster, features, info = fit(cfg, synthetic())
    assert 1 <= info["best_iteration"] <= 300
    assert booster.current_iteration() == info["num_boost_round"] == info["best_iteration"]
    assert features == ["signal", "noise", "product_type_no"]


def test_fit_is_reproducible():
    cfg = cfg_with(early_stopping={"max_rounds": 50, "patience": 10, "refit": True})
    X = synthetic(seed=1).X[0]
    p1 = fit(cfg, synthetic())[0].predict(X)
    p2 = fit(cfg, synthetic())[0].predict(X)
    assert np.array_equal(p1, p2)


def test_top_k_breaks_ties_on_article_id():
    s = pd.DataFrame({"customer_idx": [1, 1, 1, 2], "article_id": [30, 10, 20, 5], "score": [0.5, 0.5, 0.9, 0.1]})
    assert top_k(s, k=2) == {1: [20, 10], 2: [5]}


def test_rerank_rules():
    from ensemble.ranking.rerank import apply_rules
    arts = pd.DataFrame({"product_code": ["a", "a", "a", "b", "c"], "product_type_no": [1, 1, 1, 1, 2]},
                        index=pd.Index([1, 2, 3, 4, 5], name="article_id"))
    s = pd.DataFrame({"customer_idx": [0] * 5, "article_id": [1, 2, 3, 4, 5], "score": [5, 4, 3, 2, 1],
                      "a_days_since_last_sale": [1, 40, 1, 1, 1]})
    assert apply_rules(s, arts, k=3) == {0: [1, 2, 3]}
    assert apply_rules(s, arts, k=3, max_days_since_last_sale=28) == {0: [1, 3, 4]}   # 2 looks unavailable
    assert apply_rules(s, arts, k=3, max_per_product_code=1) == {0: [1, 4, 5]}
    # Caps never shorten the list when not enough items qualify: skipped items fill the tail.
    assert apply_rules(s, arts, k=4, max_per_product_type=1) == {0: [1, 5, 2, 3]}
