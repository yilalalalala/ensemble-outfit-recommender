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
