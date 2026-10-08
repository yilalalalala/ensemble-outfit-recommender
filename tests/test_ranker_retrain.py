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
