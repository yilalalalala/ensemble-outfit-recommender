"""Ranker features: point-in-time correctness, stable vocabularies, required cutoff."""
import inspect
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from ensemble.data.splits import Week
from ensemble.features import vocab as V

WEEK = Week(date(2020, 9, 16), date(2020, 9, 22))


def make_db(future_article: int):
    con = duckdb.connect()
    con.execute("""CREATE TABLE customers AS SELECT * FROM (VALUES
        (0, 30, 'ACTIVE', 'Regularly', 1.0, 1.0), (1, 60, NULL, 'NONE', NULL, NULL))
        t(customer_idx, age, club_member_status, fashion_news_frequency, FN, Active)""")
    con.execute("""CREATE TABLE articles AS SELECT * FROM (VALUES
        (1, '10', 100, 'Garment Upper body', 1, '09', 2, 11, 1, 5, 21),
        (2, '10', 100, 'Garment Upper body', 1, '10', 2, 11, 1, 5, 21),
        (3, '30', 200, 'Shoes', 2, '09', 3, 12, 2, 6, 22),
        (77, '70', 100, 'Garment Upper body', 1, '09', 2, 11, 1, 5, 21),
        (99, '90', 200, 'Shoes', 1, '09', 2, 12, 1, 6, 22))
        t(article_id, product_code, product_type_no, product_group_name, graphical_appearance_no, colour_group_code,
          perceived_colour_value_id, department_no, index_group_no, garment_group_no, section_no)""")
    con.execute("CREATE TABLE transactions (t_dat DATE, customer_idx INT, article_id INT, price FLOAT, sales_channel_id TINYINT)")
    con.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?)", [
        ("2020-07-01", 0, 3, 0.2, 1), ("2020-09-10", 0, 1, 0.1, 2), ("2020-09-12", 0, 2, 0.1, 2),
        ("2020-09-14", 1, 1, 0.1, 1), ("2020-09-15", 1, 3, 0.3, 1)])
    for c in (0, 1):
        for _ in range(3):
            con.execute("INSERT INTO transactions VALUES ('2020-09-17', ?, ?, 0.5, 1)", [c, future_article])
        con.execute("INSERT INTO transactions VALUES ('2020-09-18', ?, 1, 0.9, 2)", [c])
    con.execute("""CREATE TEMP TABLE cand AS SELECT * FROM (VALUES
        (0, 1, 1.0, 1), (0, 3, 0.5, 2), (1, 2, 1.0, 1), (1, 1, 0.2, 2))
        t(customer_idx, article_id, repeat_score, repeat_rank)""")
    return con


@pytest.fixture(autouse=True)
def _vocab_from_db(monkeypatch):
    monkeypatch.setattr(V, "load_vocab", lambda con, cfg=None: V.build_vocab(con))


def test_features_ignore_label_week():
    from ensemble.features.track_a import build_features
    a = build_features(make_db(99), WEEK, with_labels=False)
    b = build_features(make_db(77), WEEK, with_labels=False)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 4 and list(a.customer_idx) == [0, 0, 1, 1]   # deterministic order


def test_labels_come_from_label_week_only():
    from ensemble.features.track_a import build_features
    df = build_features(make_db(99), WEEK, with_labels=True)
    assert df.set_index(["customer_idx", "article_id"]).label.to_dict() == {(0, 1): 1, (0, 3): 0, (1, 1): 1, (1, 2): 0}


def test_lifetime_and_recency_features():
    from ensemble.features.track_a import build_features
    df = build_features(make_db(99), WEEK, with_labels=False).set_index(["customer_idx", "article_id"])
    assert df.loc[(0, 1), "c_n_tx_life"] == 3 and df.loc[(0, 1), "c_n_tx_1w"] == 2
    assert df.loc[(0, 1), "ca_n_article_life"] == 1
    # bought 6 days before the cutoff: weight 1 / (1 + 6/7)
    assert np.isclose(df.loc[(0, 1), "ca_w_article"], 1 / (1 + 6 / 7), atol=1e-6)
    assert df.loc[(0, 1), "ca_n_prod_life"] == 2      # articles 1 and 2 share product_code '10'


def test_vocab_codes_missing_and_unknown():
    con = make_db(99)
    v = V.build_vocab(con)
    assert v["maps"]["club_status"] == {"ACTIVE": 1}
    assert v["maps"]["news_freq"] == {"NONE": 1, "Regularly": 2}
    assert v["version"] == V.build_vocab(con)["version"]
    V.register(con, v)
    join, expr = V.encode("product_group", "a", "product_group_name")
    con.execute("CREATE TABLE a2 AS SELECT * FROM (VALUES ('Shoes'), (NULL), ('Brand new group')) a(product_group_name)")
    got = con.execute(f"SELECT {expr} FROM a2 a {join} ORDER BY a.product_group_name NULLS LAST").fetchall()
    assert got == [(None,), (v["maps"]["product_group"]["Shoes"],), (0,)]   # unknown → NULL, missing → 0


def test_feature_functions_require_cutoff():
    """CLAUDE.md: feature builders take the cutoff week as a required argument (no default)."""
    from ensemble.features import track_a
    for fn in (track_a.build_features, track_a.define_past):
        p = inspect.signature(fn).parameters["week"]
        assert p.default is inspect.Parameter.empty
