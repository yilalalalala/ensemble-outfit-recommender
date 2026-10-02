"""Track B association mining with point-in-time evidence (Round 3, Phase 4).

Everything here is derived from baskets strictly before ``week.start``. A basket
is one customer's distinct articles on one day (DESIGN §5.2); only baskets with
2..``max_basket_size`` articles spanning at least two slots are mined, because a
single-slot basket carries no cross-slot evidence.

Two temp tables are produced (names prefixed ``tb_`` so the Round-1 pipeline in
``ensemble.completion.data`` keeps its own ``pairs_all`` / ``style_pairs``):

``tb_pairs``        directed article→article cross-slot pairs
``tb_style_pairs``  the anchor side pooled over all colourways of a ``product_code``

Both carry raw co-count, time-decayed co-counts (one column per configured
half-life), marginal supports, lift, PMI and NPMI as *separate* columns: the
learned ranker chooses among them instead of the pipeline committing to one
association score (Phase 4.1).
"""
from __future__ import annotations

from datetime import timedelta

from ensemble.data.splits import Week


def _d(x) -> str:
    return f"DATE '{x}'"


def basket_sql(lo, hi, min_size: int, max_size: int, cutoff) -> str:
    """Slotted articles of multi-slot baskets with dates in [lo, hi].

    ``age_days`` is the basket's age at the cutoff, used for time decay.
    """
    return f"""
        WITH w AS (SELECT DISTINCT customer_idx, t_dat, article_id FROM transactions
                   WHERE t_dat BETWEEN {_d(lo)} AND {_d(hi)}),
        b AS (SELECT customer_idx, t_dat, count(*) AS size FROM w GROUP BY 1, 2),
        s AS (SELECT w.customer_idx, w.t_dat, w.article_id, a.slot, a.product_code,
                     date_diff('day', w.t_dat, {_d(cutoff)}) AS age_days
              FROM w JOIN b USING (customer_idx, t_dat) JOIN articles a USING (article_id)
              WHERE b.size BETWEEN {min_size} AND {max_size} AND a.slot IS NOT NULL)
        SELECT * FROM s WHERE (customer_idx, t_dat) IN (
            SELECT customer_idx, t_dat FROM s GROUP BY 1, 2 HAVING count(DISTINCT slot) >= 2)"""


def decay_cols(half_lives) -> str:
    return "".join(f", sum(pow(0.5, x.age_days / {float(h)})) AS co_d{int(h)}" for h in half_lives)


def mine(con, week: Week, cfg, basket_filter: str = "") -> dict:
    """Build ``tb_bk``, ``tb_pairs`` and ``tb_style_pairs`` for ``week``; return the funnel.

    ``week`` is required and no row dated on or after ``week.start`` is read
    (D-003). ``basket_filter`` is an optional extra SQL predicate on ``tb_bk``
    rows used by the basket-noise ablation (Phase 4.5).
    """
    c = cfg.track_b
    lo = week.start - timedelta(weeks=int(c.window_weeks))
    hi = week.cutoff
    half_lives = [int(h) for h in c.half_lives_days]
    sup = int(c.min_support)
    funnel = {}
    funnel["baskets_all"] = con.execute(
        f"SELECT count(*) FROM (SELECT DISTINCT customer_idx, t_dat FROM transactions "
        f"WHERE t_dat BETWEEN {_d(lo)} AND {_d(hi)})").fetchone()[0]
    con.execute(f"CREATE OR REPLACE TEMP TABLE tb_bk AS "
                f"{basket_sql(lo, hi, int(c.min_basket_size), int(c.max_basket_size), hi)}")
    if basket_filter:
        con.execute(f"DELETE FROM tb_bk WHERE NOT ({basket_filter})")
        con.execute("""DELETE FROM tb_bk WHERE (customer_idx, t_dat) NOT IN (
                         SELECT customer_idx, t_dat FROM tb_bk GROUP BY 1, 2
                         HAVING count(DISTINCT slot) >= 2)""")
    funnel["baskets_multi_slot"] = con.execute(
        "SELECT count(DISTINCT (customer_idx, t_dat)) FROM tb_bk").fetchone()[0]
    n = funnel["baskets_multi_slot"]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tb_pairs AS
        WITH na AS (SELECT article_id, count(*) AS n FROM tb_bk GROUP BY 1),
        co AS (SELECT x.article_id AS src, y.article_id AS dst, y.slot AS dst_slot,
                      count(*) AS co{decay_cols(half_lives)}, min(x.age_days) AS last_co_days
               FROM tb_bk x JOIN tb_bk y
                 ON x.customer_idx = y.customer_idx AND x.t_dat = y.t_dat AND x.slot <> y.slot
               GROUP BY 1, 2, 3 HAVING count(*) >= {sup})
        SELECT co.*, ns.n AS n_src, nd.n AS n_dst,
               co * {n}.0 / (ns.n * nd.n) AS lift,
               ln(co * {n}.0 / (ns.n * nd.n)) AS pmi,
               ln(co * {n}.0 / (ns.n * nd.n)) / -ln(co / {n}.0) AS npmi
        FROM co JOIN na ns ON ns.article_id = src JOIN na nd ON nd.article_id = dst""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tb_style_pairs AS
        WITH ns AS (SELECT product_code, count(DISTINCT (customer_idx, t_dat)) AS n FROM tb_bk GROUP BY 1),
        nd AS (SELECT article_id, count(*) AS n FROM tb_bk GROUP BY 1),
        co AS (SELECT x.product_code AS src_code, y.article_id AS dst, y.slot AS dst_slot,
                      count(DISTINCT (x.customer_idx, x.t_dat)) AS co{decay_cols(half_lives)},
                      min(x.age_days) AS last_co_days
               FROM tb_bk x JOIN tb_bk y
                 ON x.customer_idx = y.customer_idx AND x.t_dat = y.t_dat AND x.slot <> y.slot
               GROUP BY 1, 2, 3 HAVING count(DISTINCT (x.customer_idx, x.t_dat)) >= {sup})
        SELECT co.*, ns.n AS n_src, nd.n AS n_dst,
               co * {n}.0 / (ns.n * nd.n) AS lift,
               ln(co * {n}.0 / (ns.n * nd.n)) AS pmi,
               ln(co * {n}.0 / (ns.n * nd.n)) / -ln(co / {n}.0) AS npmi
        FROM co JOIN ns ON ns.product_code = src_code JOIN nd ON nd.article_id = dst""")
    for t in sorted({1, 2, 3, 10, sup}):
        funnel[f"pairs_support_ge_{t}"] = con.execute(
            f"SELECT count(*) FROM tb_pairs WHERE co >= {t}").fetchone()[0] if t >= sup else None
    funnel["pairs_npmi_positive"] = con.execute("SELECT count(*) FROM tb_pairs WHERE lift > 1").fetchone()[0]
    funnel["style_pairs"] = con.execute("SELECT count(*) FROM tb_style_pairs").fetchone()[0]
    funnel["training_examples"] = con.execute(
        """SELECT count(*) FROM tb_bk x JOIN tb_bk y
           ON x.customer_idx = y.customer_idx AND x.t_dat = y.t_dat AND x.slot <> y.slot""").fetchone()[0]
    return {k: v for k, v in funnel.items() if v is not None}


def basket_audit(con, week: Week, cfg) -> dict:
    """Noise proxies for the same-customer/same-day basket label (Phase 4.5).

    ``tb_bk`` must already exist. Reported, not acted on, unless the rolling
    folds show a filter helps.
    """
    lo = week.start - timedelta(weeks=int(cfg.track_b.window_weeks))
    row = con.execute(f"""
        WITH raw AS (SELECT customer_idx, t_dat, article_id, count(*) AS qty FROM transactions
                     WHERE t_dat BETWEEN {_d(lo)} AND {_d(week.cutoff)} GROUP BY 1, 2, 3),
        bk AS (SELECT customer_idx, t_dat, count(*) AS n_art, count(DISTINCT slot) AS n_slot,
                      count(DISTINCT product_code) AS n_style FROM tb_bk GROUP BY 1, 2),
        j AS (SELECT bk.*, coalesce(max(raw.qty), 1) AS max_qty, coalesce(sum(raw.qty), 0) AS units
              FROM bk LEFT JOIN raw USING (customer_idx, t_dat) GROUP BY 1, 2, 3, 4, 5)
        SELECT count(*) AS baskets,
               avg(n_art) AS mean_articles, avg(n_slot) AS mean_slots,
               avg((max_qty > 1)::INT) AS share_repeated_quantity,
               avg((n_style < n_art)::INT) AS share_multi_colourway,
               avg((n_slot >= 4)::INT) AS share_heterogeneous_4plus,
               avg(units::DOUBLE / n_art) AS mean_units_per_article
        FROM j""").df().iloc[0].to_dict()
    return {k: float(v) for k, v in row.items()}
