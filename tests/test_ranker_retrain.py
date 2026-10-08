"""Ranker retrain: SASRec provenance features, the training-row rule, and the feature contract."""
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from ensemble.data.splits import Week
from ensemble.research import ranker_retrain as RR
from ensemble.research.ensemble import PHASE2_CHANNELS

WEEK = Week(date(2020, 9, 16), date(2020, 9, 22))


def test_provenance_ranks_percentile_and_append_flag(tmp_path):
    pd.DataFrame({"customer_idx": [1, 1, 2], "rnk": [1, 2, 1], "article_id": [10, 11, 20],
                  "score": [3.0, 2.0, 1.0]}).astype({"customer_idx": "int32", "rnk": "int16", "article_id": "int32"}) \
        .to_parquet(tmp_path / "deep_sasrec.parquet", index=False)
    df = pd.DataFrame({"customer_idx": [1, 1, 1, 2], "article_id": [10, 11, 12, 21]})
    for c in PHASE2_CHANNELS:
        df[f"{c}_rank"] = np.nan
    df.loc[0, "repeat_rank"] = 1.0          # (1, 10) was also proposed by a Phase-2 channel
    out = RR.add_provenance(duckdb.connect(), df, tmp_path, n_items=100).set_index(["customer_idx", "article_id"])
    assert out.loc[(1, 10), "sasrec_rank"] == 1 and out.loc[(1, 11), "sasrec_rank"] == 2
    assert np.isnan(out.loc[(1, 12), "sasrec_rank"]) and np.isnan(out.loc[(2, 21), "sasrec_rank"])
    assert out.loc[(1, 11), "sasrec_rank_pct"] == pytest.approx(0.02)
    assert out.loc[(1, 10), "src_sasrec_append"] == 0 and out.loc[(1, 11), "src_sasrec_append"] == 1
    assert len(out) == 4                     # one row per candidate, nothing duplicated or dropped


def _db(extra_label_rows):
    con = duckdb.connect()
    con.execute("CREATE TABLE transactions (t_dat DATE, customer_idx INT, article_id INT, price FLOAT, sales_channel_id TINYINT)")
    rows = [("2020-09-17", 1, 10), ("2020-09-18", 2, 99)] + extra_label_rows   # customer 2's purchase is not a candidate
    con.executemany("INSERT INTO transactions VALUES (?, ?, ?, 0.1, 2)", rows)
    cand = [(1, a) for a in range(10, 60)] + [(2, a) for a in range(20, 40)]
    con.execute("CREATE TEMP TABLE cand (customer_idx INT, article_id INT)")
    con.executemany("INSERT INTO cand VALUES (?, ?)", cand)
    return con


def test_training_rows_keep_every_positive_and_follow_the_shared_hash_rule():
    con = _db([])
    RR.keep_training_rows(con, WEEK, 0.5)
    kept = con.execute("SELECT customer_idx, article_id FROM cand ORDER BY 1, 2").fetchall()
    assert (1, 10) in kept                               # the positive is always kept
    assert all(c == 1 for c, _ in kept)                  # customer 2 has no retrieved purchase: dropped
    assert 10 < len(kept) < 50                           # about half of the 49 negatives remain
    # Label-week rows for articles that are not candidates cannot change the training rows.
    con2 = _db([("2020-09-19", 1, 999), ("2020-09-20", 3, 10)])
    RR.keep_training_rows(con2, WEEK, 0.5)
    assert con2.execute("SELECT customer_idx, article_id FROM cand ORDER BY 1, 2").fetchall() == kept


def test_feature_contract():
    try:
        base = RR.base_features(RR.load_config("track_a_research"))
    except Exception:  # noqa: BLE001 - the final-system models are a local, gitignored artifact
        pytest.skip("final-system ranker models not built")
    cfg = RR.load_config("track_a_research")
    assert len(base) == 91 and not set(RR.PROV) & set(base)
    assert RR.variant_features(cfg, "R1") == base
    assert RR.variant_features(cfg, "R2") == base + RR.PROV


def synthetic(feats, seed=0):
    from ensemble.ranking.train import TrainingData, _customer_groups
    rng = np.random.default_rng(seed)
    data = TrainingData(feats)
    for _ in range(3):
        cust = np.repeat(np.arange(200), 10)
        X = rng.normal(size=(len(cust), len(feats))).astype(np.float32)
        y = (X[:, 0] + rng.normal(scale=0.5, size=len(cust)) > 1.2).astype(np.int8)
        data.X.append(X), data.y.append(y), data.groups.append(_customer_groups(cust))
    return data


def test_identical_seed_fits_reproduce_and_shap_is_additive():
    # LightGBM runs in a fresh interpreter: torch, loaded by other test modules, ships another OpenMP
    # runtime and LightGBM segfaults next to it on macOS (same pattern as tests/test_ranker.py).
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = "import sys; sys.path[:0] = ['src', 'tests']; from test_ranker_retrain import *\n" + textwrap.dedent("""
        from ensemble.ranking.train import fit as lgb_fit
        cfg = RR.load_config("track_a_research")
        feats = ["f0", "f1", "f2"] + RR.PROV
        b1, _, _ = lgb_fit(RR.variant_cfg(cfg, "R2", 42), synthetic(feats))
        b2, _, _ = lgb_fit(RR.variant_cfg(cfg, "R2", 42), synthetic(feats))
        X = synthetic(feats, seed=1).X[0]
        np.testing.assert_array_equal(b1.predict(X), b2.predict(X))
        contrib = b1.predict(X, pred_contrib=True)
        np.testing.assert_allclose(contrib.sum(axis=1), b1.predict(X, raw_score=True), atol=1e-9)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, cwd=root,
                       env={**os.environ, "PYTHONPATH": str(root / "src")})
    assert r.returncode == 0, r.stderr[-2000:]


def test_provenance_features_never_produce_a_reason_chip():
    from ensemble.ranking.explain import reason_of
    assert all(reason_of(f) is None for f in RR.PROV)
