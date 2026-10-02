"""Retrieval channel contract, determinism, per-channel leakage and the budget allocator."""
from datetime import date

import duckdb
import numpy as np
import pytest

from ensemble.candidates import retrieval as R
from ensemble.config import Config, load_config
from ensemble.data.splits import Week

WEEK = Week(date(2020, 9, 16), date(2020, 9, 22))
SQL_CHANNELS = ["repeat", "pop", "pop_age", "new_arrival", "cf", "variant", "type_pop", "dept_pop",
                "section_pop", "covis", "new_arrival_pers"]


def make_db(future_article: int, future_customer_extra: bool = False):
    """Tiny catalogue and history; every row on or after WEEK.start is 'future'."""
    con = duckdb.connect()
    con.execute("CREATE TABLE customers AS SELECT * FROM (VALUES (0, 30), (1, 60), (2, 22)) t(customer_idx, age)")
    con.execute("""CREATE TABLE articles AS SELECT * FROM (VALUES
        (1, '10', 100, 1, 11, 1, 5, 1), (2, '10', 100, 1, 11, 1, 5, 2), (3, '30', 200, 2, 12, 2, 6, 3),
        (4, '40', 100, 1, 11, 1, 5, 1), (5, '50', 300, 3, 13, 1, 7, 2),
        (77, '70', 100, 1, 11, 1, 5, 1), (99, '90', 300, 3, 13, 1, 7, 1))
        t(article_id, product_code, product_type_no, department_no, section_no, index_group_no, garment_group_no,
          perceived_colour_master_id)""")
    con.execute("""CREATE TABLE transactions (t_dat DATE, customer_idx INT, article_id INT,
                   price FLOAT, sales_channel_id TINYINT)""")
    rows = [("2020-08-01", 0, 3), ("2020-09-01", 0, 1), ("2020-09-10", 0, 2), ("2020-09-12", 0, 4),
            ("2020-09-14", 1, 1), ("2020-09-15", 1, 3), ("2020-09-13", 1, 2), ("2020-09-15", 2, 5),
            ("2020-09-14", 2, 1), ("2020-09-15", 2, 4), ("2020-09-11", 1, 4), ("2020-09-12", 2, 2)]
    con.executemany("INSERT INTO transactions VALUES (?, ?, ?, 0.1, 2)", rows)
    for c in (0, 1, 2):
        for _ in range(5):
            con.execute("INSERT INTO transactions VALUES ('2020-09-17', ?, ?, 0.1, 2)", [c, future_article])
    if future_customer_extra:
        con.execute("INSERT INTO transactions VALUES ('2020-09-16', 0, 5, 0.9, 1)")
    con.execute("CREATE TABLE _users AS SELECT * FROM (VALUES (0), (1), (2)) t(customer_idx)")
    return con


def cfg_r(**over):
    base = dict(load_config().retrieval)
    base.update({"cf_min_count": 1, "covis_min_count": 1, "new_arrival_days": 30, "new_arrival_k": 5,
                 "channels": SQL_CHANNELS, **{f"{c}_k": 5 for c in SQL_CHANNELS}})
    base.update(over)
    return Config(base)


@pytest.mark.parametrize("ch", SQL_CHANNELS)
def test_channel_contract_and_no_leakage(ch):
    """Each channel writes (customer_idx, article_id, score) and ignores every row in the label week."""
    def run(future_article, extra):
        con = make_db(future_article, extra)
        r = cfg_r()
        R.build_shared(con, WEEK, r, "_users")
        R._REGISTRY[ch](con, WEEK, r, "_users")
        cols = [c[0] for c in con.execute(f"DESCRIBE _ch_{ch}").fetchall()]
        assert cols[:3] == ["customer_idx", "article_id", "score"]
        return con.execute(f"SELECT customer_idx, article_id, round(score, 9) FROM _ch_{ch} ORDER BY ALL").fetchall()

    a = run(99, False)
    assert a == run(77, True)
    assert all(row[1] not in (77, 99) for row in a)


def test_merge_caps_and_deterministic_ties():
    con = duckdb.connect()
    con.execute("CREATE TABLE _ch_a AS SELECT * FROM (VALUES (0, 5, 1.0), (0, 3, 1.0), (0, 4, 1.0), (0, 9, 2.0)) t(customer_idx, article_id, score)")
    con.execute("CREATE TABLE _ch_b AS SELECT * FROM (VALUES (0, 4, 0.5), (0, 7, 0.1)) t(customer_idx, article_id, score)")
    R.merge(con, ["a", "b"], {"a": 3, "b": 1})
    rows = con.execute("SELECT article_id, a_rank, b_rank FROM cand ORDER BY article_id").fetchall()
    # a: 9 (score 2) then ties at 1.0 broken by article_id: 3, 4; 5 is cut. b keeps only 4.
    assert rows == [(3, 2, None), (4, 3, 1), (9, 1, None)]
    assert R.channel_columns(con) == ["a", "b"]


def test_build_candidates_is_deterministic():
    def once():
        con = make_db(99)
        R.build_candidates(con, WEEK, cfg_r())
        return con.execute("SELECT * FROM cand ORDER BY customer_idx, article_id").fetchall()
    assert once() == once()


def test_unknown_channel_rejected():
    with pytest.raises(ValueError):
        R.enabled_channels(Config({"channels": ["repeat", "nope"]}))


def test_greedy_frontier_prefers_efficient_channel_and_counts_overlap_once():
    from ensemble.candidates.budget import greedy_frontier
    # Channel x: rank 1 hits uid 0; ranks 2-5 miss. Channel y: rank 1 duplicates uid 0, rank 2 hits uid 9.
    pool = {"x": (np.array([1, 2, 3, 4, 5]), np.array([0, 1, 2, 3, 4]), np.array([True, False, False, False, False])),
            "y": (np.array([1, 2]), np.array([0, 9]), np.array([True, True]))}
    f = greedy_frontier([pool], n_users=1, n_truth=2, steps=(1,))
    assert f[-1]["recall"] == 1.0
    assert f[-1]["cand"] == 6           # 7 rows, uid 0 counted once
    first_full = next(p for p in f if p["recall"] == 1.0)
    assert first_full["cand"] == 2       # both hits are taken before any of x's misses
