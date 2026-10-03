"""Track B ranker features (Round 3, Phases 2–4).

Every feature is built from data strictly before ``week.start``: ``week`` is a
required argument of :func:`build_context` and the only table that reads the
target week is the label table, which carries the truth.

Three groups:

*compatibility* — association evidence (raw co-count, time-decayed co-counts,
support, lift, PMI, NPMI, at article level and at style level with the backoff
level exposed), two-tower and FashionCLIP similarity, category/colour/price
agreement between anchor and candidate;

*candidate* — pre-cutoff popularity, how recently and how often the article sold,
its age in the catalogue, an availability proxy, price;

*personalization* (Phase 3) — the customer's own pre-cutoff affinities for the
candidate's article, style, product type, department, section, garment group,
colour and slot, plus their usual price point and distance from it, and explicit
interactions between compatibility evidence and customer preference. Customers
with no history get ``c_has_history = 0`` and null affinities, so the
non-personalized compatibility path still answers.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ensemble.completion.candidates import decay_names
from ensemble.data.splits import Week

SLOT_IDS = {s: i for i, s in enumerate(
    ["upper", "lower", "full", "shoes", "accessories", "socks", "swimwear"])}

# Low-cardinality integer attributes handed to LightGBM as categorical features.
CATEGORICAL = ["target_slot_id", "anchor_slot_id", "cand_product_type_no", "cand_index_group_no",
               "cand_garment_group_no", "cand_section_no", "cand_colour_group_code",
               "cand_perceived_colour_master_id", "anchor_product_type_no", "anchor_index_group_no",
               "anchor_garment_group_no"]
ID_COLS = ["qid", "article_id", "label", "customer_idx"]

PERSONAL_PREFIXES = ("c_", "cu_", "x_")


def _d(x) -> str:
    return f"DATE '{x}'"


ATTRS = """SELECT article_id, slot, product_code, product_type_no, graphical_appearance_no,
                  TRY_CAST(colour_group_code AS INTEGER) AS colour_group_code,
                  perceived_colour_value_id, perceived_colour_master_id, department_no,
                  index_group_no, section_no, garment_group_no, is_jewellery
           FROM articles"""


def register_attrs(con) -> None:
    """``_tb_attrs``: the article attributes every feature join needs (no dates involved)."""
    con.execute(f"CREATE OR REPLACE TEMP TABLE _tb_attrs AS {ATTRS}")


def build_context(con, week: Week, cfg, uni: pd.DataFrame, q: pd.DataFrame,
                  key_sims: pd.DataFrame | None = None) -> None:
    """Create the temp tables the per-chunk feature query joins against.

    ``key_sims`` holds two-tower / CLIP similarity for every (anchor, target_slot,
    article_id) candidate key, computed once per week.
    """
    register_attrs(con)
    con.register("_tb_uni_df", uni)
    con.execute("""CREATE OR REPLACE TEMP TABLE _tb_item AS
                   SELECT u.*, row_number() OVER (PARTITION BY u.slot ORDER BY u.pop_recent DESC,
                              u.pop_window DESC, u.article_id) AS slot_pop_rank,
                          count(*) OVER (PARTITION BY u.slot) AS slot_size
                   FROM _tb_uni_df u""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _tb_price AS
                    WITH p AS (SELECT article_id, avg(price) AS mean_price FROM transactions
                               WHERE t_dat BETWEEN {_d(week.start - timedelta(weeks=52))}
                                             AND {_d(week.cutoff)} GROUP BY 1)
                    SELECT p.article_id, p.mean_price,
                           ntile(5) OVER (PARTITION BY a.slot ORDER BY p.mean_price) AS price_tier
                    FROM p JOIN _tb_attrs a USING (article_id) WHERE a.slot IS NOT NULL""")
    register_key_sims(con, key_sims)
    truth = pd.DataFrame({"qid": np.repeat(q.qid.values, [len(t) for t in q.truth.values]),
                          "article_id": np.concatenate([np.asarray(t, dtype=np.int64)
                                                        for t in q.truth.values]) if len(q) else
                          np.empty(0, np.int64)})
    con.register("_tb_truth_df", truth)
    con.execute("CREATE OR REPLACE TEMP TABLE _tb_truth AS SELECT * FROM _tb_truth_df")
    _customer_tables(con, week, cfg)


def register_key_sims(con, key_sims: pd.DataFrame | None) -> None:
    """``_tb_keysim``: two-tower / FashionCLIP similarity per (anchor, target_slot, article_id).

    Separate from :func:`build_context` because serving covers every live anchor and
    swaps this table one batch of anchors at a time.
    """
    con.register("_tb_keysim_df", key_sims if key_sims is not None else pd.DataFrame(
        {"anchor": pd.Series(dtype="int64"), "target_slot": pd.Series(dtype="object"),
         "article_id": pd.Series(dtype="int64"), "tt_pair_sim": pd.Series(dtype="float32"),
         "ttc_pair_sim": pd.Series(dtype="float32"), "clip_sim": pd.Series(dtype="float32")}))
    con.execute("CREATE OR REPLACE TEMP TABLE _tb_keysim AS SELECT * FROM _tb_keysim_df")


def _customer_tables(con, week: Week, cfg) -> None:
    """Point-in-time customer affinities for the customers that have queries this week."""
    lo = week.start - timedelta(weeks=int(cfg.track_b.get("customer_weeks", 104)))
    con.execute(f"""CREATE OR REPLACE TEMP VIEW _tb_past AS
                    SELECT t.*, date_diff('day', t.t_dat, {_d(week.start)}) AS days_ago
                    FROM transactions t
                    WHERE t.t_dat >= {_d(lo)} AND t.t_dat < {_d(week.start)}
                      AND t.customer_idx IN (SELECT DISTINCT customer_idx FROM _tbq)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE _tb_hist AS
                   SELECT p.*, a.slot, a.product_code, a.product_type_no, a.department_no, a.section_no,
                          a.garment_group_no, TRY_CAST(a.colour_group_code AS INTEGER) AS colour_group_code,
                          a.perceived_colour_master_id,
                          1.0 / (1 + p.days_ago / 7.0) AS w
                   FROM _tb_past p JOIN _tb_attrs a USING (article_id)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE _tb_cust AS
                   SELECT customer_idx, count(*) AS c_n_tx, count(DISTINCT t_dat) AS c_n_days,
                          min(days_ago) AS c_days_since_last, avg(price) AS c_mean_price,
                          sum(w) AS c_w_total
                   FROM _tb_hist GROUP BY 1""")
    for name, col in (("type", "product_type_no"), ("dept", "department_no"), ("section", "section_no"),
                      ("ggroup", "garment_group_no"), ("colour", "colour_group_code"),
                      ("cmaster", "perceived_colour_master_id"), ("slot", "slot")):
        con.execute(f"""CREATE OR REPLACE TEMP TABLE _tb_cu_{name} AS
                        SELECT customer_idx, {col} AS key, count(*) AS n, sum(w) AS w
                        FROM _tb_hist WHERE {col} IS NOT NULL GROUP BY 1, 2""")
    con.execute("""CREATE OR REPLACE TEMP TABLE _tb_cu_article AS
                   SELECT customer_idx, article_id, count(*) AS n, sum(w) AS w, min(days_ago) AS days_ago
                   FROM _tb_hist GROUP BY 1, 2""")
    con.execute("""CREATE OR REPLACE TEMP TABLE _tb_cu_style AS
                   SELECT customer_idx, product_code, count(*) AS n, sum(w) AS w
                   FROM _tb_hist GROUP BY 1, 2""")


def _assoc_select(cfg) -> str:
    dcols = decay_names(cfg)
    art = ["a.co AS a_co", "a.last_co_days AS a_last_co_days", "a.n_src AS a_n_src", "a.n_dst AS a_n_dst",
           "a.lift AS a_lift", "a.pmi AS a_pmi", "a.npmi AS a_npmi",
           "a.rank_co AS a_rank_co", "a.rank_npmi AS a_rank_npmi",
           "a.co::DOUBLE / a.n_src AS a_co_share"]
    art += [f"a.{c} AS a_{c}" for c in dcols]
    art += [f"a.{c} / a.co AS a_{c}_ratio" for c in dcols]
    sty = ["y.s_co", "y.s_last_co_days", "y.s_n_src", "y.s_n_dst", "y.s_lift", "y.s_pmi", "y.s_npmi",
           "y.s_rank_co", "y.s_rank_npmi", "y.s_co::DOUBLE / y.s_n_src AS s_co_share"]
    sty += [f"y.s_{c}" for c in dcols]
    return ", ".join(art + sty)


PERSONAL_SQL = """
           (cust.customer_idx IS NOT NULL)::INT AS c_has_history,
           cust.c_n_tx, cust.c_n_days, cust.c_days_since_last, cust.c_mean_price,
           ca.n AS cu_article_n, ca.days_ago AS cu_article_days_ago,
           cs.n AS cu_style_n,
           ct.w / cust.c_w_total AS cu_type_share, cd.w / cust.c_w_total AS cu_dept_share,
           csec.w / cust.c_w_total AS cu_section_share, cg.w / cust.c_w_total AS cu_ggroup_share,
           cc.w / cust.c_w_total AS cu_colour_share, cm.w / cust.c_w_total AS cu_cmaster_share,
           csl.w / cust.c_w_total AS cu_slot_share,
           ct.n AS cu_type_n, cd.n AS cu_dept_n,
           abs(cp.mean_price - cust.c_mean_price) / nullif(cust.c_mean_price, 0) AS cu_price_dist,
           coalesce(a.npmi, 0) * coalesce(ct.w / cust.c_w_total, 0) AS x_npmi_type,
           coalesce(ks.tt_pair_sim, 0) * coalesce(ct.w / cust.c_w_total, 0) AS x_tt_type,
           coalesce(a.co, 0) * coalesce(cd.w / cust.c_w_total, 0) AS x_co_dept,
"""

PERSONAL_JOINS = """
        LEFT JOIN _tb_cust cust ON cust.customer_idx = q.customer_idx
        LEFT JOIN _tb_cu_article ca ON ca.customer_idx = q.customer_idx AND ca.article_id = u.article_id
        LEFT JOIN _tb_cu_style cs ON cs.customer_idx = q.customer_idx AND cs.product_code = ca_attr.product_code
        LEFT JOIN _tb_cu_type ct ON ct.customer_idx = q.customer_idx AND ct.key = ca_attr.product_type_no
        LEFT JOIN _tb_cu_dept cd ON cd.customer_idx = q.customer_idx AND cd.key = ca_attr.department_no
        LEFT JOIN _tb_cu_section csec ON csec.customer_idx = q.customer_idx AND csec.key = ca_attr.section_no
        LEFT JOIN _tb_cu_ggroup cg ON cg.customer_idx = q.customer_idx AND cg.key = ca_attr.garment_group_no
        LEFT JOIN _tb_cu_colour cc ON cc.customer_idx = q.customer_idx AND cc.key = ca_attr.colour_group_code
        LEFT JOIN _tb_cu_cmaster cm ON cm.customer_idx = q.customer_idx AND cm.key = ca_attr.perceived_colour_master_id
        LEFT JOIN _tb_cu_slot csl ON csl.customer_idx = q.customer_idx AND csl.key = q.target_slot
"""


def chunk_features(con, cfg, qid_lo: int, qid_hi: int, with_labels: bool, personalize: bool,
                   neg_rate: float = 1.0, seed: int = 0, sel_table: str | None = None) -> pd.DataFrame:
    """Feature matrix for queries ``qid_lo <= qid < qid_hi``, one row per candidate.

    ``neg_rate`` < 1 keeps every positive and a deterministic hashed share of the
    negatives (training weeks only). ``sel_table`` restricts the queries to a
    sampled subset (a temp table with a ``qid`` column).
    """
    label = ("(t.qid IS NOT NULL)::TINYINT AS label" if with_labels else "0::TINYINT AS label")
    neg_filter = ""
    if with_labels and neg_rate < 1.0:
        neg_filter = (f"AND (t.qid IS NOT NULL OR "
                      f"hash(u.qid * 1000003 + u.article_id + {int(seed)}) % 10000 < {int(neg_rate * 10000)})")
    personal = PERSONAL_SQL if personalize else ""
    joins = PERSONAL_JOINS if personalize else ""
    sel = f" AND qid IN (SELECT qid FROM {sel_table})" if sel_table else ""
    sql = f"""
        WITH qs AS (SELECT * FROM _tbq WHERE qid >= {qid_lo} AND qid < {qid_hi}{sel}),
        u AS (
            SELECT q.qid, a.dst AS article_id FROM qs q
                 JOIN _tb_assoc a ON a.src = q.anchor AND a.dst_slot = q.target_slot
            UNION
            SELECT q.qid, y.dst FROM qs q JOIN _tb_attrs ar ON ar.article_id = q.anchor
                 JOIN _tb_style y ON y.src_code = ar.product_code AND y.dst_slot = q.target_slot
            UNION
            SELECT q.qid, t.dst FROM qs q JOIN _tb_tt t ON t.anchor = q.anchor AND t.target_slot = q.target_slot
            UNION
            SELECT q.qid, c.dst FROM qs q JOIN _tb_ttc c ON c.anchor = q.anchor AND c.target_slot = q.target_slot
            UNION
            SELECT q.qid, p.dst FROM qs q JOIN _tb_pop p ON p.target_slot = q.target_slot)
        SELECT u.qid, u.article_id, q.customer_idx, {label},
               {_assoc_select(cfg)},
               (a.src IS NOT NULL)::INT AS has_article_evidence,
               (y.src_code IS NOT NULL)::INT AS has_style_evidence,
               CASE WHEN a.src IS NOT NULL THEN 0 WHEN y.src_code IS NOT NULL THEN 1 ELSE 2 END AS backoff_level,
               (tt.dst IS NOT NULL)::INT AS src_two_tower, tt.tt_rank,
               (ttc.dst IS NOT NULL)::INT AS src_two_tower_content, ttc.ttc_rank,
               (p.dst IS NOT NULL)::INT AS src_slot_pop,
               ks.tt_pair_sim, ks.ttc_pair_sim, ks.clip_sim,
               i.pop_recent AS cand_pop_recent, i.pop_window AS cand_pop_window,
               i.days_sold_window AS cand_days_sold, i.days_since_last_sale AS cand_days_since_last_sale,
               i.age_days AS cand_age_days, i.mean_price AS cand_mean_price,
               i.is_jewellery::INT AS cand_is_jewellery,
               i.slot_pop_rank AS cand_slot_pop_rank,
               i.slot_pop_rank::DOUBLE / i.slot_size AS cand_slot_pop_pct,
               ln(1 + i.pop_recent) AS cand_log_pop_recent, ln(1 + i.pop_window) AS cand_log_pop_window,
               {personal}
               (ca_attr.colour_group_code = an.colour_group_code)::INT AS same_colour_group,
               (ca_attr.perceived_colour_master_id = an.perceived_colour_master_id)::INT AS same_colour_master,
               (ca_attr.perceived_colour_value_id = an.perceived_colour_value_id)::INT AS same_colour_value,
               (ca_attr.index_group_no = an.index_group_no)::INT AS same_index_group,
               (ca_attr.section_no = an.section_no)::INT AS same_section,
               (ca_attr.department_no = an.department_no)::INT AS same_department,
               (ca_attr.garment_group_no = an.garment_group_no)::INT AS same_garment_group,
               (ca_attr.graphical_appearance_no = an.graphical_appearance_no)::INT AS same_graphical,
               (ca_attr.product_type_no = an.product_type_no)::INT AS same_product_type,
               ln((cp.mean_price + 1e-4) / (ap.mean_price + 1e-4)) AS price_log_ratio,
               abs(ln((cp.mean_price + 1e-4) / (ap.mean_price + 1e-4))) AS price_log_ratio_abs,
               cp.mean_price - ap.mean_price AS price_diff,
               cp.price_tier AS cand_price_tier, ap.price_tier AS anchor_price_tier,
               cp.price_tier - ap.price_tier AS price_tier_diff,
               {", ".join(f"ca_attr.{c[len('cand_'):]} AS {c}" for c in CATEGORICAL
                          if c.startswith("cand_"))},
               {", ".join(f"an.{c[len('anchor_'):]} AS {c}" for c in CATEGORICAL
                          if c.startswith("anchor_") and not c.endswith("slot_id"))},
               list_position({list(SLOT_IDS)}, q.target_slot) - 1 AS target_slot_id,
               list_position({list(SLOT_IDS)}, q.anchor_slot) - 1 AS anchor_slot_id
        FROM u
        JOIN qs q USING (qid)
        JOIN _tb_attrs an ON an.article_id = q.anchor
        JOIN _tb_attrs ca_attr ON ca_attr.article_id = u.article_id
        JOIN _tb_item i ON i.article_id = u.article_id
        LEFT JOIN _tb_price cp ON cp.article_id = u.article_id
        LEFT JOIN _tb_price ap ON ap.article_id = q.anchor
        LEFT JOIN _tb_assoc a ON a.src = q.anchor AND a.dst_slot = q.target_slot AND a.dst = u.article_id
        LEFT JOIN _tb_style y ON y.src_code = an.product_code AND y.dst_slot = q.target_slot AND y.dst = u.article_id
        LEFT JOIN _tb_tt tt ON tt.anchor = q.anchor AND tt.target_slot = q.target_slot AND tt.dst = u.article_id
        LEFT JOIN _tb_ttc ttc ON ttc.anchor = q.anchor AND ttc.target_slot = q.target_slot AND ttc.dst = u.article_id
        LEFT JOIN _tb_pop p ON p.target_slot = q.target_slot AND p.dst = u.article_id
        LEFT JOIN _tb_keysim ks ON ks.anchor = q.anchor AND ks.target_slot = q.target_slot
                                AND ks.article_id = u.article_id
        {joins}
        LEFT JOIN _tb_truth t ON t.qid = u.qid AND t.article_id = u.article_id
        WHERE TRUE {neg_filter}
        ORDER BY u.qid, u.article_id"""
    df = con.execute(sql).df()
    return downcast(df)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float32 / int32 everywhere except the id columns: the matrices are the memory budget."""
    for c in df.columns:
        if c in ("qid", "article_id", "customer_idx"):
            df[c] = df[c].astype(np.int64)
        elif c == "label":
            df[c] = df[c].astype(np.int8)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS]


def drop_personalization(features: list[str]) -> list[str]:
    return [f for f in features if not f.startswith(PERSONAL_PREFIXES)]


# Feature groups the ablation table switches off. Each value is the set of prefixes
# (or exact names) removed from the full schema; the ranker is refitted without them.
# A group has to take every feature that carries its signal, interactions included —
# leaving ``x_tt_type`` behind would let tower similarity back into "minus towers".
GROUPS: dict[str, tuple[str, ...]] = {
    "personalization": PERSONAL_PREFIXES,
    "repeat": ("cu_article_",),
    "clip": ("clip_sim",),
    "towers": ("tt_", "ttc_", "src_two_tower", "x_tt"),
    "style_backoff": ("s_", "has_style_evidence", "backoff_level"),
    "decay": ("a_co_d", "s_co_d"),
    "association": ("a_co", "a_lift", "a_pmi", "a_npmi", "a_n_", "a_rank", "a_last_co_days",
                    "has_article_evidence", "x_npmi", "x_co"),
    "popularity": ("cand_pop_", "cand_log_pop_", "cand_slot_pop", "src_slot_pop",
                   "cand_days_sold", "cand_days_since_last_sale", "cand_age_days"),
    "price": ("price_", "cand_price_tier", "anchor_price_tier", "cu_price_dist"),
    "colour": ("same_colour",),
}

# How many features each group must remove from the *full* production schema. A typo
# in a prefix, or a renamed feature, would otherwise silently produce an ablation
# identical to the full model, which would then be reported as "no effect".
# :func:`check_groups` enforces this; the backtest calls it on the real schema.
GROUP_MIN_FEATURES: dict[str, int] = {
    "personalization": 21, "repeat": 2, "clip": 1, "towers": 7, "style_backoff": 14,
    "decay": 6, "association": 17, "popularity": 10, "price": 7, "colour": 3,
}


def subset(features: list[str], drop: tuple[str, ...]) -> list[str]:
    """``features`` without the named groups (``drop`` holds :data:`GROUPS` keys).

    A group that removes nothing would make the ablation a copy of the full model,
    so that is an error rather than a silent "no effect" row in the table.
    """
    out = list(features)
    for g in drop:
        kept = [f for f in out if not f.startswith(GROUPS[g])]
        if len(kept) == len(out):
            raise ValueError(f"feature group {g!r} removed nothing from {len(out)} features")
        out = kept
    return out


def check_groups(features: list[str]) -> dict[str, int]:
    """Assert every group still bites the full schema; return how much each removes.

    Called on the real feature schema before any ablation is fitted, so a renamed
    feature fails the run instead of quietly weakening one row of the table.
    """
    counts = {g: len(features) - len([f for f in features if not f.startswith(GROUPS[g])])
              for g in GROUPS}
    short = {g: (n, GROUP_MIN_FEATURES[g]) for g, n in counts.items() if n < GROUP_MIN_FEATURES[g]}
    if short:
        raise ValueError(f"feature groups no longer match the schema (got, expected): {short}")
    return counts


def ablation_subsets(cfg, features: list[str]) -> dict[str, list[str]]:
    """Named feature subsets to fit on each fold, from ``track_b.feature_ablations``.

    ``lgbm_personalized`` (everything) and ``lgbm_compatibility`` (no customer
    feature) are always present: they are the two candidate shipping models.
    """
    out = {"lgbm_personalized": features, "lgbm_compatibility": subset(features, ("personalization",))}
    for name in cfg.track_b.get("feature_ablations", []) or []:
        groups = tuple(x.strip() for x in str(name).split("+") if x.strip())
        unknown = [g for g in groups if g not in GROUPS]
        if unknown:
            raise KeyError(f"unknown feature ablation group(s): {unknown}")
        out[f"lgbm_minus_{'_'.join(groups)}"] = subset(features, groups)
    return out
