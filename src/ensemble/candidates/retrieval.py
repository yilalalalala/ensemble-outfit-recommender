"""M2 retrieval: independent retrieval channels merged into one candidate table.

Every channel reads only transactions with ``t_dat < week.start`` (D-003).
Output is a wide table ``cand`` with one row per (customer, article) and, per
channel, its score and rank. Those columns double as the ranker's
retrieval-source features (DESIGN §4).
"""
from __future__ import annotations

from datetime import timedelta

from ensemble.baselines import AGE_BINS
from ensemble.data.splits import Week

CHANNELS = ("repeat", "pop", "pop_age", "new_arrival", "cf", "variant")


def _d(x) -> str:
    return f"DATE '{x}'"


def build_item_pairs(con, week: Week, r) -> None:
    """Item-to-item co-purchase neighbours, normalised like cosine similarity."""
    start = _d(week.start)
    lo = _d(week.start - timedelta(weeks=int(r.cf_weeks)))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cf_hist AS
        SELECT customer_idx, article_id FROM (
            SELECT customer_idx, article_id, max(t_dat) AS last_dat
            FROM transactions WHERE t_dat >= {lo} AND t_dat < {start} GROUP BY 1, 2)
        QUALIFY row_number() OVER (PARTITION BY customer_idx ORDER BY last_dat DESC, article_id) <= {int(r.cf_recent_items)}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE item_pairs AS
        WITH n AS (SELECT article_id, count(*) AS n FROM _cf_hist GROUP BY 1),
        co AS (
            SELECT a.article_id AS src, b.article_id AS dst, count(*) AS co
            FROM _cf_hist a JOIN _cf_hist b
              ON a.customer_idx = b.customer_idx AND a.article_id <> b.article_id
            GROUP BY 1, 2 HAVING count(*) >= {int(r.cf_min_count)})
        SELECT src, dst, co, co / sqrt(ns.n * nd.n) AS sim
        FROM co JOIN n ns ON ns.article_id = src JOIN n nd ON nd.article_id = dst
        QUALIFY row_number() OVER (PARTITION BY src ORDER BY sim DESC, dst) <= {int(r.cf_neighbors)}
    """)
    con.execute("DROP TABLE _cf_hist")


def build_candidates(con, week: Week, r, users_table: str = "_users") -> None:
    """Create TEMP TABLE ``cand`` for customers in ``users_table``."""
    start = _d(week.start)
    rep_lo = _d(week.start - timedelta(weeks=int(r.repeat_weeks)))
    seed_lo = _d(week.start - timedelta(weeks=int(r.cf_seed_weeks)))
    pop_lo = _d(week.start - timedelta(days=int(r.pop_days)))

    build_item_pairs(con, week, r)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _hist AS
        SELECT t.customer_idx, t.article_id, t.t_dat
        FROM transactions t JOIN {users_table} u USING (customer_idx)
        WHERE t.t_dat >= {rep_lo} AND t.t_dat < {start}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _pop AS
        SELECT article_id, count(*) AS n FROM transactions
        WHERE t_dat >= {pop_lo} AND t_dat < {start} GROUP BY 1
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _long AS
        -- repeat purchase: recency first, then frequency
        SELECT customer_idx, article_id, 'repeat' AS ch,
               sum(1.0 / (1 + date_diff('day', t_dat, {start}) / 7.0)) AS score
        FROM _hist GROUP BY 1, 2

        UNION ALL  -- global popularity
        SELECT u.customer_idx, p.article_id, 'pop', p.n
        FROM {users_table} u CROSS JOIN (SELECT * FROM _pop ORDER BY n DESC, article_id LIMIT {int(r.pop_k)}) p

        UNION ALL  -- popularity within the customer's age band
        SELECT u.customer_idx, p.article_id, 'pop_age', p.n FROM (
            SELECT u.customer_idx, {AGE_BINS} AS age_bin FROM {users_table} u JOIN customers c USING (customer_idx)) u
        JOIN (
            SELECT {AGE_BINS} AS age_bin, t.article_id, count(*) AS n
            FROM transactions t JOIN customers c USING (customer_idx)
            WHERE t_dat >= {pop_lo} AND t_dat < {start} GROUP BY 1, 2
            QUALIFY row_number() OVER (PARTITION BY age_bin ORDER BY n DESC, article_id) <= {int(r.pop_age_k)}
        ) p USING (age_bin)

        UNION ALL  -- new arrivals: first sold within the popularity window, by sales
        SELECT u.customer_idx, p.article_id, 'new_arrival', p.n
        FROM {users_table} u CROSS JOIN (
            SELECT p.article_id, p.n FROM _pop p
            WHERE p.article_id NOT IN (SELECT article_id FROM transactions WHERE t_dat < {pop_lo})
            ORDER BY p.n DESC, p.article_id LIMIT {int(r.new_arrival_k)}) p

        UNION ALL  -- item-to-item CF from the customer's recent items
        SELECT s.customer_idx, ip.dst, 'cf', sum(ip.sim * s.w)
        FROM (SELECT customer_idx, article_id, sum(1.0 / (1 + date_diff('day', t_dat, {start}) / 7.0)) AS w
              FROM _hist WHERE t_dat >= {seed_lo} GROUP BY 1, 2) s
        JOIN item_pairs ip ON ip.src = s.article_id
        GROUP BY 1, 2

        UNION ALL  -- variants: same product_code, other colourway, ranked by recent sales
        SELECT s.customer_idx, a2.article_id, 'variant', max(coalesce(p.n, 0))
        FROM (SELECT DISTINCT customer_idx, article_id FROM _hist WHERE t_dat >= {seed_lo}) s
        JOIN articles a1 ON a1.article_id = s.article_id
        JOIN articles a2 ON a2.product_code = a1.product_code AND a2.article_id <> a1.article_id
        JOIN _pop p ON p.article_id = a2.article_id
        GROUP BY 1, 2
    """)
    caps = {"repeat": r.repeat_k, "pop": r.pop_k, "pop_age": r.pop_age_k,
            "new_arrival": r.new_arrival_k, "cf": r.cf_k, "variant": r.variant_k}
    cap_case = " ".join(f"WHEN '{c}' THEN {int(k)}" for c, k in caps.items())
    pivots = ",\n".join(
        f"max(CASE WHEN ch = '{c}' THEN score END) AS {c}_score, "
        f"min(CASE WHEN ch = '{c}' THEN rnk END) AS {c}_rank"
        for c in CHANNELS
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cand AS
        WITH ranked AS (
            SELECT *, row_number() OVER (PARTITION BY customer_idx, ch ORDER BY score DESC, article_id) AS rnk
            FROM _long)
        SELECT customer_idx, article_id, {pivots}
        FROM ranked WHERE rnk <= CASE ch {cap_case} END
        GROUP BY 1, 2
    """)
    for t in ("_long", "_hist", "_pop"):
        con.execute(f"DROP TABLE {t}")
