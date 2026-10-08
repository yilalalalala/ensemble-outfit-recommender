"""M2 retrieval: independent retrieval channels merged into one candidate table.

Every channel reads only transactions with ``t_dat < week.start`` (D-003).

**Channel contract.** A channel is a function registered with ``@channel(name)``
that creates ``TEMP TABLE _ch_<name>(customer_idx, article_id, score)`` for the
customers in ``users_table``. Higher score = better. The merge step ranks each
channel's rows per customer by ``score DESC, article_id ASC`` (deterministic
ties), keeps the top ``<name>_k`` and pivots into the wide table ``cand``: one
row per (customer, article) with ``<name>_score`` and ``<name>_rank`` per
channel (NULL when the channel did not propose it). Those columns are the
candidate's provenance and double as the ranker's retrieval-source features
(DESIGN §4).

The enabled channels and their caps come from ``retrieval.channels`` and
``retrieval.<name>_k`` in the config.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from datetime import timedelta

from ensemble.baselines import AGE_BINS
from ensemble.data.splits import Week

_REGISTRY: dict[str, Callable] = {}


def channel(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def enabled_channels(r) -> list[str]:
    chs = list(r.get("channels") or ("repeat", "pop", "pop_age", "new_arrival", "cf", "variant"))
    unknown = [c for c in chs if c not in _REGISTRY]
    if unknown:
        raise ValueError(f"unknown retrieval channels: {unknown}")
    return chs


def r_cfg():
    from ensemble.config import load_config
    return load_config()


def _d(x) -> str:
    return f"DATE '{x}'"


def _days_before(week: Week, days: int) -> str:
    return _d(week.start - timedelta(days=int(days)))


def _recency_weight(start: str, col: str = "t_dat") -> str:
    """Hyperbolic recency weight used by the history-seeded channels: 1 for yesterday, ½ a week ago."""
    return f"1.0 / (1 + date_diff('day', {col}, {start}) / 7.0)"


# ---------------------------------------------------------------------------
# Shared inputs (built once per call of build_candidates)
# ---------------------------------------------------------------------------

def _max_hist_weeks(r) -> int:
    keys = ("repeat_weeks", "cf_seed_weeks", "variant_seed_weeks", "type_pop_weeks", "dept_pop_weeks",
            "section_pop_weeks", "covis_seed_weeks", "new_arrival_pers_weeks")
    return max(int(r.get(k) or 0) for k in keys)


def build_shared(con, week: Week, r, users_table: str) -> None:
    """``_hist``: the target customers' purchases in the longest window any channel needs.
    ``_sales``: per-article sales counts over several windows and first sale date, all before the cutoff."""
    start = _d(week.start)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _hist AS
        SELECT t.customer_idx, t.article_id, t.t_dat
        FROM transactions t JOIN {users_table} u USING (customer_idx)
        WHERE t.t_dat >= {_d(week.start - timedelta(weeks=_max_hist_weeks(r)))} AND t.t_dat < {start}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _sales AS
        SELECT article_id,
               count(*) FILTER (WHERE t_dat >= {_days_before(week, r.pop_days)}) AS n,
               count(*) FILTER (WHERE t_dat >= {_days_before(week, 1)}) AS n_1d,
               count(*) FILTER (WHERE t_dat >= {_days_before(week, 3)}) AS n_3d,
               count(*) FILTER (WHERE t_dat >= {_days_before(week, 14)}) AS n_14d,
               count(*) FILTER (WHERE t_dat >= {_days_before(week, 28)}) AS n_28d,
               sum(exp(-ln(2) * date_diff('day', t_dat, {start}) / {float(r.get('pop_half_life_days', 7))})
                   ) FILTER (WHERE t_dat >= {_days_before(week, 28)}) AS n_decay,
               min(t_dat) AS first_sale
        FROM transactions WHERE t_dat < {start} GROUP BY 1
    """)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

@channel("repeat")
def ch_repeat(con, week: Week, r, users_table: str) -> None:
    """Buy it again: the customer's own articles, recency first, then frequency."""
    start = _d(week.start)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ch_repeat AS
        SELECT customer_idx, article_id, sum({_recency_weight(start)}) AS score
        FROM _hist WHERE t_dat >= {_d(week.start - timedelta(weeks=int(r.repeat_weeks)))} GROUP BY 1, 2
    """)


def _pop_metric(r) -> str:
    return {"count": "n", "decay": "n_decay", "1d": "n_1d", "3d": "n_3d", "14d": "n_14d",
            "28d": "n_28d"}[r.get("pop_metric", "count")]


@channel("pop")
def ch_pop(con, week: Week, r, users_table: str) -> None:
    """Global best sellers in the popularity window."""
    m = _pop_metric(r)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ch_pop AS
        SELECT u.customer_idx, p.article_id, p.score
        FROM {users_table} u CROSS JOIN (
            SELECT article_id, {m}::DOUBLE AS score FROM _sales WHERE {m} > 0
            ORDER BY score DESC, article_id LIMIT {int(r.pop_k)}) p
    """)


FINE_AGE_BINS = ("CASE WHEN age IS NULL THEN -1 WHEN age < 20 THEN 0 WHEN age < 23 THEN 1 WHEN age < 26 THEN 2 "
                 "WHEN age < 30 THEN 3 WHEN age < 35 THEN 4 WHEN age < 40 THEN 5 WHEN age < 45 THEN 6 "
                 "WHEN age < 50 THEN 7 WHEN age < 55 THEN 8 WHEN age < 60 THEN 9 ELSE 10 END")


def _age_bins(r, name: str) -> str:
    return FINE_AGE_BINS if r.get(f"{name}_age_bins", "coarse") == "fine" else AGE_BINS


@channel("pop_age")
def ch_pop_age(con, week: Week, r, users_table: str) -> None:
    """Best sellers within the customer's age band (``pop_age_age_bins``: coarse = the M1 baseline's 4 bands, fine = 10)."""
    AGE_BINS = _age_bins(r, "pop_age")  # noqa: N806
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ch_pop_age AS
        SELECT u.customer_idx, p.article_id, p.n::DOUBLE AS score FROM (
            SELECT u.customer_idx, {AGE_BINS} AS age_bin FROM {users_table} u JOIN customers c USING (customer_idx)) u
        JOIN (
            SELECT {AGE_BINS} AS age_bin, t.article_id, count(*) AS n
            FROM transactions t JOIN customers c USING (customer_idx)
            WHERE t_dat >= {_days_before(week, r.pop_days)} AND t_dat < {_d(week.start)} GROUP BY 1, 2
            QUALIFY row_number() OVER (PARTITION BY age_bin ORDER BY n DESC, article_id) <= {int(r.pop_age_k)}
        ) p USING (age_bin)
    """)


@channel("new_arrival")
def ch_new_arrival(con, week: Week, r, users_table: str) -> None:
    """Launch proxy: articles whose first observed sale is inside the window, by sales.
    First observed sale is not a launch date (the dataset has none)."""
    days = int(r.get("new_arrival_days", r.pop_days))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ch_new_arrival AS
        SELECT u.customer_idx, p.article_id, p.score
        FROM {users_table} u CROSS JOIN (
            SELECT article_id, n_14d::DOUBLE AS score FROM _sales
            WHERE first_sale >= {_days_before(week, days)}
            ORDER BY n_14d DESC, article_id LIMIT {int(r.new_arrival_k)}) p
    """)


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


@channel("cf")
def ch_cf(con, week: Week, r, users_table: str) -> None:
    """Item-to-item CF: cosine co-purchase neighbours of the customer's recent articles."""
    start = _d(week.start)
    build_item_pairs(con, week, r)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ch_cf AS
        SELECT s.customer_idx, ip.dst AS article_id, sum(ip.sim * s.w) AS score
        FROM (SELECT customer_idx, article_id, sum({_recency_weight(start)}) AS w
              FROM _hist WHERE t_dat >= {_d(week.start - timedelta(weeks=int(r.cf_seed_weeks)))} GROUP BY 1, 2) s
        JOIN item_pairs ip ON ip.src = s.article_id
        GROUP BY 1, 2
    """)


@channel("variant")
def ch_variant(con, week: Week, r, users_table: str) -> None:
    """Same style (product_code), other colourway, ranked by recent sales."""
    seed_lo = _d(week.start - timedelta(weeks=int(r.get("variant_seed_weeks", r.cf_seed_weeks))))
    sales = {"count": "n", "14d": "n_14d", "28d": "n_28d"}[r.get("variant_sales", "count")]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ch_variant AS
        SELECT s.customer_idx, a2.article_id, max(p.{sales})::DOUBLE AS score
        FROM (SELECT DISTINCT customer_idx, article_id FROM _hist WHERE t_dat >= {seed_lo}) s
        JOIN articles a1 ON a1.article_id = s.article_id
        JOIN articles a2 ON a2.product_code = a1.product_code AND a2.article_id <> a1.article_id
        JOIN _sales p ON p.article_id = a2.article_id AND p.{sales} > 0
        GROUP BY 1, 2
    """)


@channel("visual")
def ch_visual(con, week: Week, r, users_table: str) -> None:
    """FashionCLIP similarity to recent purchases (M7a, D-023)."""
    from ensemble.candidates.visual import build_visual_channel
    build_visual_channel(con, week, r_cfg(), users_table)
    con.execute("CREATE OR REPLACE TEMP TABLE _ch_visual AS SELECT * FROM _visual")
    con.execute("DROP TABLE _visual")


_AFFINITY_KEYS = {
    "product_type_no": "a.product_type_no", "department_no": "a.department_no", "section_no": "a.section_no",
    "garment_group_no": "a.garment_group_no",
    "type_index": "a.product_type_no * 100 + a.index_group_no",
    "dept_colour": "a.department_no * 1000 + a.perceived_colour_master_id",
}


def _affinity_pop(con, week: Week, r, users_table: str, name: str) -> None:
    """Category-conditioned popularity: best sellers within the categories the customer buys.

    Parameters are read as ``<name>_<param>``. Affinity = recency-weighted share
    of the customer's purchases in a category (``_key``, e.g. department) over
    ``_weeks``; the customer's top ``_types`` categories each contribute their
    ``_items`` best sellers in the popularity window (among the customer's age
    band when ``_by_age``); score = affinity × sales.
    """
    p = lambda k, d=None: r.get(f"{name}_{k}", d)  # noqa: E731
    AGE_BINS = _age_bins(r, name)  # noqa: N806
    start = _d(week.start)
    lo = _d(week.start - timedelta(weeks=int(p("weeks"))))
    key = _AFFINITY_KEYS[p("key")]
    m = _pop_metric(r)
    if p("by_age", False):
        best_sql = f"""
            SELECT {AGE_BINS} AS age_bin, {key} AS k, t.article_id, count(*)::DOUBLE AS n
            FROM transactions t JOIN customers c USING (customer_idx) JOIN articles a USING (article_id)
            WHERE t.t_dat >= {_days_before(week, r.pop_days)} AND t.t_dat < {start} GROUP BY 1, 2, 3
            QUALIFY row_number() OVER (PARTITION BY age_bin, k ORDER BY n DESC, t.article_id) <= {int(p("items"))}"""
        join = "JOIN best b ON b.k = t.k AND b.age_bin = t.age_bin"
    else:
        best_sql = f"""
            SELECT {key} AS k, s.article_id, s.{m}::DOUBLE AS n FROM _sales s JOIN articles a USING (article_id)
            WHERE s.{m} > 0
            QUALIFY row_number() OVER (PARTITION BY k ORDER BY s.{m} DESC, s.article_id) <= {int(p("items"))}"""
        join = "JOIN best b ON b.k = t.k"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ch_{name} AS
        WITH aff AS (
            SELECT h.customer_idx, {key} AS k, sum({_recency_weight(start, 'h.t_dat')}) AS w
            FROM _hist h JOIN articles a USING (article_id) WHERE h.t_dat >= {lo} GROUP BY 1, 2),
        top_aff AS (
            SELECT aff.customer_idx, aff.k, aff.w / sum(aff.w) OVER (PARTITION BY aff.customer_idx) AS share,
                   {AGE_BINS} AS age_bin
            FROM aff JOIN customers c USING (customer_idx)
            QUALIFY row_number() OVER (PARTITION BY aff.customer_idx ORDER BY aff.w DESC, aff.k) <= {int(p("types"))}),
        best AS ({best_sql})
        SELECT t.customer_idx, b.article_id, max(t.share * b.n) AS score
        FROM top_aff t {join} GROUP BY 1, 2
    """)


for _name in ("type_pop", "dept_pop", "section_pop"):
    channel(_name)(lambda con, week, r, users_table, _n=_name: _affinity_pop(con, week, r, users_table, _n))


@channel("covis")
def ch_covis(con, week: Week, r, users_table: str) -> None:
    """Directional, time-weighted co-visitation (``covis_*`` params).

    A pair (src → dst) counts when one customer bought dst within
    ``covis_gap_days`` after (or on the same day as) src, inside the last
    ``covis_weeks``. Each co-occurrence is weighted by the recency of the later
    purchase. Score normalisation divides by src count and dst count^beta,
    which damps best sellers (beta = 0 keeps raw conditional probability).
    """
    start = _d(week.start)
    lo = _d(week.start - timedelta(weeks=int(r.covis_weeks)))
    gap = int(r.covis_gap_days)
    beta = float(r.covis_beta)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cv_tx AS
        SELECT customer_idx, article_id, max(t_dat) AS t_dat FROM transactions
        WHERE t_dat >= {lo} AND t_dat < {start} GROUP BY customer_idx, article_id, t_dat
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _covis_pairs AS
        WITH n AS (SELECT article_id, count(*)::DOUBLE AS n FROM _cv_tx GROUP BY 1),
        co AS (
            SELECT a.article_id AS src, b.article_id AS dst,
                   sum(1.0 / (1 + date_diff('day', b.t_dat, {start}) / 14.0)) AS co, count(*) AS raw
            FROM _cv_tx a JOIN _cv_tx b
              ON a.customer_idx = b.customer_idx AND a.article_id <> b.article_id
             AND b.t_dat >= a.t_dat AND b.t_dat <= a.t_dat + INTERVAL {gap} DAY
            GROUP BY 1, 2 HAVING count(*) >= {int(r.covis_min_count)})
        SELECT src, dst, co / (ns.n * pow(nd.n, {beta})) AS sim
        FROM co JOIN n ns ON ns.article_id = src JOIN n nd ON nd.article_id = dst
        -- `co` is a multi-threaded float sum whose last bits vary between runs; rounding before
        -- ordering keeps the neighbour cutoff deterministic (same fix as the merge, D-027).
        QUALIFY row_number() OVER (PARTITION BY src ORDER BY round(sim, 9) DESC, dst) <= {int(r.covis_neighbors)}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ch_covis AS
        SELECT s.customer_idx, p.dst AS article_id, sum(p.sim * s.w) AS score
        FROM (SELECT customer_idx, article_id, sum({_recency_weight(start)}) AS w FROM _hist
              WHERE t_dat >= {_d(week.start - timedelta(weeks=int(r.covis_seed_weeks)))} GROUP BY 1, 2) s
        JOIN _covis_pairs p ON p.src = s.article_id
        GROUP BY 1, 2
    """)
    con.execute("DROP TABLE _cv_tx")


@channel("als")
def ch_als(con, week: Week, r, users_table: str) -> None:
    """Collaborative latent factors (implicit ALS), see ``ensemble.candidates.als``."""
    from ensemble.candidates.als import build_als_channel
    build_als_channel(con, week, r, users_table)


@channel("new_arrival_pers")
def ch_new_arrival_pers(con, week: Week, r, users_table: str) -> None:
    """Personalised new arrivals: launch-proxy articles (first observed sale within
    ``new_arrival_days``) in the departments the customer buys, by affinity × sales."""
    start = _d(week.start)
    lo = _d(week.start - timedelta(weeks=int(r.new_arrival_pers_weeks)))
    days = int(r.get("new_arrival_days", r.pop_days))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _ch_new_arrival_pers AS
        WITH aff AS (
            SELECT h.customer_idx, a.department_no AS k, sum({_recency_weight(start, 'h.t_dat')}) AS w
            FROM _hist h JOIN articles a USING (article_id) WHERE h.t_dat >= {lo} GROUP BY 1, 2),
        top_aff AS (
            SELECT customer_idx, k, w / sum(w) OVER (PARTITION BY customer_idx) AS share FROM aff
            QUALIFY row_number() OVER (PARTITION BY customer_idx ORDER BY w DESC, k) <= {int(r.new_arrival_pers_types)}),
        fresh AS (
            SELECT a.department_no AS k, s.article_id, s.n_14d::DOUBLE AS n FROM _sales s JOIN articles a USING (article_id)
            WHERE s.first_sale >= {_days_before(week, days)}
            QUALIFY row_number() OVER (PARTITION BY a.department_no ORDER BY s.n_14d DESC, s.article_id) <= {int(r.new_arrival_pers_items)})
        SELECT t.customer_idx, f.article_id, max(t.share * f.n) AS score
        FROM top_aff t JOIN fresh f USING (k) GROUP BY 1, 2
    """)


def _neural(con, week: Week, r, users_table: str, model: str) -> None:
    """Reproduced BPR-MF / LightGCN / SASRec top-k for this week's cutoff (``ensemble.research``)."""
    from ensemble.research.channels import build_neural_channel
    build_neural_channel(con, week, r, users_table, model)


for _model in ("bpr", "lightgcn", "sasrec"):
    channel(_model)(lambda con, week, r, users_table, _m=_model: _neural(con, week, r, users_table, _m))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def cap_of(r, ch: str) -> int:
    return int(r.get(f"{ch}_k", 0))


def merge(con, channels: list[str], caps: dict[str, int], out: str = "cand") -> None:
    """Rank each channel per customer (deterministic ties), cap, union and pivot.

    Scores are rounded to 9 decimals first: multi-threaded float sums differ in the last bits
    between runs, which would otherwise make near-ties (and so the candidate set) non-deterministic.
    Capping happens per channel table, before the union, so the pivot only sees kept rows."""
    union = "\nUNION ALL ".join(f"""
        SELECT customer_idx, article_id, {i}::TINYINT AS ch, score, rnk FROM (
            SELECT customer_idx, article_id, round(score::DOUBLE, 9) AS score,
                   row_number() OVER (PARTITION BY customer_idx ORDER BY round(score::DOUBLE, 9) DESC, article_id) AS rnk
            FROM _ch_{c} QUALIFY rnk <= {int(caps[c])})""" for i, c in enumerate(channels))
    pivots = ",\n".join(
        f"max(CASE WHEN ch = {i} THEN score END) AS {c}_score, "
        f"min(CASE WHEN ch = {i} THEN rnk END) AS {c}_rank"
        for i, c in enumerate(channels))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {out} AS
        SELECT customer_idx, article_id, {pivots}
        FROM ({union}) GROUP BY 1, 2
    """)


def build_channels(con, week: Week, r, channels: list[str], users_table: str = "_users") -> dict[str, float]:
    """Build shared inputs and each channel table; return build seconds per channel."""
    secs = {}
    t = time.time()
    build_shared(con, week, r, users_table)
    secs["_shared"] = time.time() - t
    for c in channels:
        t = time.time()
        _REGISTRY[c](con, week, r, users_table)
        secs[c] = time.time() - t
    return secs


def drop_channels(con, channels: list[str]) -> None:
    for t in ("_hist", "_sales", *(f"_ch_{c}" for c in channels)):
        con.execute(f"DROP TABLE IF EXISTS {t}")


def build_candidates(con, week: Week, r, users_table: str = "_users") -> dict[str, float]:
    """Create TEMP TABLE ``cand`` for customers in ``users_table``. Returns build seconds per stage."""
    channels = enabled_channels(r)
    secs = build_channels(con, week, r, channels, users_table)
    t = time.time()
    merge(con, channels, {c: cap_of(r, c) for c in channels})
    secs["_merge"] = time.time() - t
    drop_channels(con, channels)
    return secs


def channel_columns(con, table: str = "cand") -> list[str]:
    """Channels present in a candidate table, in column order."""
    cols = [c[0] for c in con.execute(f"DESCRIBE {table}").fetchall()]
    return [c[:-5] for c in cols if c.endswith("_rank")]
