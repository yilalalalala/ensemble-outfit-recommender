"""Track B data: baskets, cross-slot pairs, evaluation queries (DESIGN §5).

A basket is one customer's distinct articles on one day. Everything used for
training is read from the ``window_weeks`` before the target week; evaluation
queries come from baskets inside the target week.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ensemble.data.splits import Week

SLOTS = ["upper", "lower", "full", "shoes", "accessories", "socks", "swimwear"]


def _d(x) -> str:
    return f"DATE '{x}'"


def baskets_sql(lo, hi, min_size: int, max_size: int) -> str:
    """Slotted articles in baskets of min..max distinct articles, dates in [lo, hi]."""
    return f"""
        WITH w AS (SELECT DISTINCT customer_idx, t_dat, article_id FROM transactions
                   WHERE t_dat BETWEEN {_d(lo)} AND {_d(hi)}),
        b AS (SELECT customer_idx, t_dat, count(*) AS size FROM w GROUP BY 1, 2)
        SELECT w.customer_idx, w.t_dat, w.article_id, a.slot
        FROM w JOIN b USING (customer_idx, t_dat) JOIN articles a USING (article_id)
        WHERE b.size BETWEEN {min_size} AND {max_size} AND a.slot IS NOT NULL"""


def mine_pairs(con, week: Week, cfg) -> dict:
    """Create TEMP TABLE ``pairs`` (directed a→c, cross-slot) with support and NPMI.

    Returns the filter funnel (DESIGN §5.2): how many baskets and pairs survive
    each stage.
    """
    c = cfg.completion
    lo = week.start - timedelta(weeks=int(c.window_weeks))
    hi = week.start - timedelta(days=1)
    s = cfg.sampling
    funnel = {}
    funnel["baskets_all"] = con.execute(f"""SELECT count(*) FROM (SELECT DISTINCT customer_idx, t_dat FROM transactions
                                            WHERE t_dat BETWEEN {_d(lo)} AND {_d(hi)})""").fetchone()[0]
    con.execute(f"CREATE OR REPLACE TEMP TABLE _bk AS {baskets_sql(lo, hi, int(s.min_basket_size), int(s.max_basket_size))}")
    funnel["baskets_size_2_6"] = con.execute("SELECT count(DISTINCT (customer_idx, t_dat)) FROM _bk").fetchone()[0]
    con.execute("""CREATE OR REPLACE TEMP TABLE _bk AS
                   SELECT * FROM _bk WHERE (customer_idx, t_dat) IN (
                       SELECT customer_idx, t_dat FROM _bk GROUP BY 1, 2 HAVING count(DISTINCT slot) >= 2)""")
    n = con.execute("SELECT count(DISTINCT (customer_idx, t_dat)) FROM _bk").fetchone()[0]
    funnel["baskets_multi_slot"] = n
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pairs_all AS
        WITH na AS (SELECT article_id, count(*) AS n FROM _bk GROUP BY 1),
        co AS (SELECT x.article_id AS src, y.article_id AS dst, y.slot AS dst_slot, count(*) AS co
               FROM _bk x JOIN _bk y ON x.customer_idx = y.customer_idx AND x.t_dat = y.t_dat AND x.slot <> y.slot
               GROUP BY 1, 2, 3)
        SELECT co.*, ns.n AS n_src, nd.n AS n_dst,
               co * {n}.0 / (ns.n * nd.n) AS lift,
               ln(co * {n}.0 / (ns.n * nd.n)) / -ln(co / {n}.0) AS npmi
        FROM co JOIN na ns ON ns.article_id = src JOIN na nd ON nd.article_id = dst
    """)
    funnel["pairs_cross_slot"] = con.execute("SELECT count(*) / 2 FROM pairs_all").fetchone()[0]
    # Style-level pairs (hierarchical backoff): the anchor side pooled over all colourways of a
    # product_code, counted once per basket.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE style_pairs AS
        WITH bk AS (SELECT b.*, a.product_code FROM _bk b JOIN articles a USING (article_id)),
        ns AS (SELECT product_code, count(DISTINCT (customer_idx, t_dat)) AS n FROM bk GROUP BY 1),
        nd AS (SELECT article_id, count(*) AS n FROM _bk GROUP BY 1),
        co AS (SELECT x.product_code AS src_code, y.article_id AS dst, y.slot AS dst_slot,
                      count(DISTINCT (x.customer_idx, x.t_dat)) AS co
               FROM bk x JOIN _bk y ON x.customer_idx = y.customer_idx AND x.t_dat = y.t_dat AND x.slot <> y.slot
               GROUP BY 1, 2, 3)
        SELECT co.*, co * {n}.0 / (ns.n * nd.n) AS lift,
               ln(co * {n}.0 / (ns.n * nd.n)) / -ln(co / {n}.0) AS npmi
        FROM co JOIN ns ON ns.product_code = src_code JOIN nd ON nd.article_id = dst
    """)
    funnel["style_pairs_support_ge_3"] = con.execute("SELECT count(*) FROM style_pairs WHERE co >= 3").fetchone()[0]
    for t in sorted({3, 10, int(c.min_support), int(s.min_pair_support)}):
        funnel[f"pairs_support_ge_{t}"] = con.execute(f"SELECT count(*) / 2 FROM pairs_all WHERE co >= {t}").fetchone()[0]
        funnel[f"pairs_support_ge_{t}_lift_gt_1"] = con.execute(
            f"SELECT count(*) / 2 FROM pairs_all WHERE co >= {t} AND lift > 1").fetchone()[0]
    funnel["pairs_support_ge_10_lift_ge_2"] = con.execute(
        "SELECT count(*) / 2 FROM pairs_all WHERE co >= 10 AND lift >= 2").fetchone()[0]
    return funnel


def universe(con, week: Week, cfg) -> pd.DataFrame:
    """Candidate items per slot: the live catalogue (D-013) with past popularity."""
    c = cfg.completion
    return con.execute(f"""
        WITH live AS (
            SELECT DISTINCT article_id FROM transactions
            WHERE t_dat BETWEEN {_d(week.start - timedelta(weeks=int(c.universe_weeks)))} AND {_d(week.end)}),
        pop AS (
            SELECT article_id,
                   count(*) FILTER (WHERE t_dat >= {_d(week.start - timedelta(days=int(c.pop_days)))}) AS pop_recent,
                   count(*) AS pop_window
            FROM transactions
            WHERE t_dat BETWEEN {_d(week.start - timedelta(weeks=int(c.window_weeks)))} AND {_d(week.cutoff)}
            GROUP BY 1)
        SELECT a.article_id, a.slot, a.is_jewellery,
               coalesce(p.pop_recent, 0) AS pop_recent, coalesce(p.pop_window, 0) AS pop_window
        FROM live JOIN articles a USING (article_id) LEFT JOIN pop p USING (article_id)
        WHERE a.slot IS NOT NULL
        ORDER BY a.slot, pop_recent DESC, pop_window DESC, a.article_id
    """).df()


def queries(con, week: Week, cfg) -> pd.DataFrame:
    """One query per (basket, anchor, target slot); truth = the basket's articles in that slot."""
    s = cfg.sampling
    return con.execute(f"""
        WITH b AS ({baskets_sql(week.start, week.end, int(s.min_basket_size), int(s.max_basket_size))})
        SELECT x.customer_idx, x.t_dat, x.article_id AS anchor, x.slot AS anchor_slot, y.slot AS target_slot,
               list(DISTINCT y.article_id) AS truth
        FROM b x JOIN b y ON x.customer_idx = y.customer_idx AND x.t_dat = y.t_dat AND x.slot <> y.slot
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY 1, 2, 3, 5
    """).df()


def item_table(con) -> pd.DataFrame:
    """Static article metadata for the two-tower model (known at launch, so usable for cold items)."""
    return con.execute("""
        SELECT article_id, slot, product_code, product_type_no, product_group_name, graphical_appearance_no,
               colour_group_code, perceived_colour_value_id, perceived_colour_master_id, department_no,
               index_code, index_group_no, section_no, garment_group_no
        FROM articles ORDER BY article_id""").df()


def price_tiers(con, week: Week, cfg, n_tiers: int = 5) -> dict[int, int]:
    """Per-slot price quintile of each article's mean price before the cutoff (1..5; 0 = unknown)."""
    rows = con.execute(f"""
        WITH p AS (SELECT article_id, avg(price) AS price FROM transactions
                   WHERE t_dat BETWEEN {_d(week.start - timedelta(weeks=52))} AND {_d(week.cutoff)} GROUP BY 1)
        SELECT p.article_id, ntile({n_tiers}) OVER (PARTITION BY a.slot ORDER BY p.price)
        FROM p JOIN articles a USING (article_id) WHERE a.slot IS NOT NULL""").fetchall()
    return dict(rows)


def to_lists(series) -> list[np.ndarray]:
    return [np.asarray(x, dtype=np.int64) for x in series]
