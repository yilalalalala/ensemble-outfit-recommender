"""Point-in-time correctness (D-003): nothing on or after week.start may be read.

Builds a tiny synthetic database, then corrupts every row inside the label week
and asserts the outputs do not change.
"""
from datetime import date

import duckdb
import pytest

from ensemble.data.splits import Week

WEEK = Week(date(2020, 9, 16), date(2020, 9, 22))


def make_db(future_article: int):
    con = duckdb.connect()
    con.execute("CREATE TABLE customers AS SELECT * FROM (VALUES (0, 30), (1, 60)) t(customer_idx, age)")
    con.execute("""CREATE TABLE transactions (t_dat DATE, customer_idx INT, article_id INT,
                   price FLOAT, sales_channel_id TINYINT)""")
    con.execute("""INSERT INTO transactions VALUES
        ('2020-09-10', 0, 1, 0.1, 2), ('2020-09-12', 0, 2, 0.1, 2),
        ('2020-09-14', 1, 1, 0.1, 1), ('2020-09-15', 1, 3, 0.1, 1)""")
    for c in (0, 1):
        for _ in range(5):
            con.execute("INSERT INTO transactions VALUES ('2020-09-17', ?, ?, 0.1, 2)", [c, future_article])
    return con


@pytest.mark.parametrize("fn", ["popular", "popular_by_age", "repeat_purchase"])
def test_baselines_ignore_label_week(fn):
    from ensemble import baselines

    def call(con):
        f = getattr(baselines, fn)
        return f(con, WEEK, [0, 1]) if fn == "repeat_purchase" else f(con, WEEK)

    assert call(make_db(99)) == call(make_db(77))


def test_retrieval_ignores_label_week():
    from ensemble.candidates.retrieval import build_candidates
    from ensemble.config import Config, load_config

    r = Config({**dict(load_config().retrieval), "cf_min_count": 1})

    def cands(future_article):
        con = make_db(future_article)
        con.execute("""CREATE TABLE articles AS SELECT * FROM (VALUES
            (1, 10, 1, 100, 11, 21), (2, 10, 1, 100, 11, 21), (3, 30, 2, 200, 12, 22),
            (77, 70, 1, 100, 11, 21), (99, 90, 1, 200, 12, 22))
            t(article_id, product_code, index_group_no, department_no, section_no, product_type_no)""")
        con.execute("CREATE TABLE _users AS SELECT * FROM (VALUES (0), (1)) t(customer_idx)")
        build_candidates(con, WEEK, r)
        return con.execute("SELECT * FROM cand ORDER BY ALL").fetchall()

    assert cands(99) == cands(77)


def test_track_b_training_pairs_precede_week():
    """Pairs mined for a week come only from baskets before its start."""
    from datetime import timedelta

    from ensemble.completion.data import mine_pairs
    from ensemble.config import Config, load_config

    base = load_config()
    cfg = Config({**base, "completion": {**dict(base.completion), "min_support": 1}})

    def pairs(future_article):
        con = make_db(future_article)
        con.execute("""CREATE TABLE articles AS SELECT * FROM (VALUES
            (1, 10, 'upper'), (2, 20, 'lower'), (3, 30, 'accessories'),
            (77, 770, 'shoes'), (99, 990, 'shoes'))
            t(article_id, product_code, slot)""")
        # Same-day basket before the week, and a cross-slot basket inside it.
        con.execute("INSERT INTO transactions VALUES ('2020-09-10', 0, 3, 0.1, 2)")
        con.execute("INSERT INTO transactions VALUES ('2020-09-17', 0, 1, 0.1, 2)")
        mine_pairs(con, WEEK, cfg)
        return con.execute("SELECT src, dst, co FROM pairs_all ORDER BY ALL").fetchall()

    assert pairs(99) == pairs(77)
    assert (1, 3, 1) in pairs(99)  # the pre-week basket is mined
