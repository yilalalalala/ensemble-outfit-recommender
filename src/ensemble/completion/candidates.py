"""Track B candidate generation (Round 3, Phase 2).

The union of complementary retrieval sources, each with its own rank and score,
so the learned ranker can weigh them instead of a fixed RRF constant.

Sources are built at *key* level, not query level: article association and the
two towers depend only on (anchor, target_slot), and slot popularity only on
target_slot. One basket contributes several queries that share those keys, so
building per key and joining to queries is roughly 10× cheaper than per query.

``build_sources`` creates the key-level temp tables; ``candidates`` joins them to
a slice of the query table. Nothing here reads a row dated on or after
``week.start``: the association tables come from :mod:`ensemble.completion.association`
and the eligible catalogue from :func:`ensemble.completion.protocol.eligible_universe`,
both of which take the week as a required argument.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCES = ("assoc_co", "assoc_npmi", "style_co", "style_npmi", "two_tower", "two_tower_content", "slot_pop")


def decay_names(cfg) -> list[str]:
    return [f"co_d{int(h)}" for h in cfg.track_b.half_lives_days]


def register(con, q: pd.DataFrame, uni: pd.DataFrame) -> None:
    con.register("_tbq_df", q.drop(columns=["truth"], errors="ignore"))
    con.register("_tbuni_df", uni)
    con.execute("CREATE OR REPLACE TEMP TABLE _tbq AS SELECT * FROM _tbq_df")
    con.execute("CREATE OR REPLACE TEMP TABLE _tbuni AS SELECT * FROM _tbuni_df")
    con.execute("CREATE OR REPLACE TEMP TABLE _tb_keys AS "
                "SELECT DISTINCT anchor, target_slot FROM _tbq")
    con.execute("""CREATE OR REPLACE TEMP TABLE _tb_code_keys AS
                   SELECT DISTINCT a.product_code AS src_code, k.target_slot
                   FROM _tb_keys k JOIN articles a ON a.article_id = k.anchor""")


def build_sources(con, cfg, tt: pd.DataFrame | None = None, ttc: pd.DataFrame | None = None) -> dict:
    """Create ``_tb_assoc``, ``_tb_style``, ``_tb_tt`` and ``_tb_pop``; return row counts.

    ``tt`` / ``ttc`` are optional two-tower retrieval results with columns
    (anchor, target_slot, article_id, sim) ordered best-first per key.
    """
    b = cfg.track_b
    cand = b.candidates
    dcols = decay_names(cfg)
    dsel = "".join(f", p.{c}" for c in dcols)
    n_co, n_npmi = int(cand["assoc_co"]), int(cand["assoc_npmi"])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _tb_assoc AS
        WITH r AS (
            SELECT p.src, p.dst_slot, p.dst, p.co{dsel}, p.last_co_days, p.n_src, p.n_dst,
                   p.lift, p.pmi, p.npmi,
                   row_number() OVER (PARTITION BY p.src, p.dst_slot ORDER BY p.co DESC, p.npmi DESC, p.dst) AS rank_co,
                   row_number() OVER (PARTITION BY p.src, p.dst_slot ORDER BY p.npmi DESC, p.co DESC, p.dst) AS rank_npmi
            FROM tb_pairs p
            JOIN _tb_keys k ON p.src = k.anchor AND p.dst_slot = k.target_slot
            JOIN _tbuni u ON u.article_id = p.dst)
        SELECT * FROM r WHERE rank_co <= {n_co} OR rank_npmi <= {n_npmi}""")
    s_co, s_npmi = int(cand["style_co"]), int(cand["style_npmi"])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _tb_style AS
        WITH r AS (
            SELECT p.src_code, p.dst_slot, p.dst, p.co AS s_co,
                   {", ".join(f"p.{c} AS s_{c}" for c in dcols)},
                   p.last_co_days AS s_last_co_days, p.n_src AS s_n_src, p.n_dst AS s_n_dst,
                   p.lift AS s_lift, p.pmi AS s_pmi, p.npmi AS s_npmi,
                   row_number() OVER (PARTITION BY p.src_code, p.dst_slot ORDER BY p.co DESC, p.npmi DESC, p.dst) AS s_rank_co,
                   row_number() OVER (PARTITION BY p.src_code, p.dst_slot ORDER BY p.npmi DESC, p.co DESC, p.dst) AS s_rank_npmi
            FROM tb_style_pairs p
            JOIN _tb_code_keys k ON p.src_code = k.src_code AND p.dst_slot = k.target_slot
            JOIN _tbuni u ON u.article_id = p.dst)
        SELECT * FROM r WHERE s_rank_co <= {s_co} OR s_rank_npmi <= {s_npmi}""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _tb_pop AS
        SELECT slot AS target_slot, article_id AS dst,
               row_number() OVER (PARTITION BY slot ORDER BY pop_recent DESC, pop_window DESC, article_id) AS pop_rank
        FROM _tbuni QUALIFY pop_rank <= {int(cand["slot_pop"])}""")
    for name, df, col in (("_tb_tt", tt, "tt"), ("_tb_ttc", ttc, "ttc")):
        if df is None or not len(df):
            con.execute(f"CREATE OR REPLACE TEMP TABLE {name} (anchor INTEGER, target_slot VARCHAR, "
                        f"dst INTEGER, {col}_sim DOUBLE, {col}_rank INTEGER)")
        else:
            con.register(f"{name}_df", df)
            con.execute(f"""CREATE OR REPLACE TEMP TABLE {name} AS
                            SELECT anchor, target_slot, article_id AS dst, sim AS {col}_sim,
                                   row_number() OVER (PARTITION BY anchor, target_slot
                                                      ORDER BY sim DESC, article_id) AS {col}_rank
                            FROM {name}_df""")
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("_tb_assoc", "_tb_style", "_tb_pop", "_tb_tt", "_tb_ttc")}


def candidates(con, cfg, qid_lo: int, qid_hi: int) -> pd.DataFrame:
    """Candidate union for queries with ``qid_lo <= qid < qid_hi``, one row per (qid, article)."""
    dcols = decay_names(cfg)
    assoc_cols = ["co", *dcols, "last_co_days", "n_src", "n_dst", "lift", "pmi", "npmi", "rank_co", "rank_npmi"]
    style_cols = ["s_co", *[f"s_{c}" for c in dcols], "s_last_co_days", "s_n_src", "s_n_dst",
                  "s_lift", "s_pmi", "s_npmi", "s_rank_co", "s_rank_npmi"]
    sel = (", ".join(f"a.{c}" for c in assoc_cols) + ", " + ", ".join(f"y.{c}" for c in style_cols) +
           ", t.tt_sim, t.tt_rank, c.ttc_sim, c.ttc_rank, p.pop_rank")
    return con.execute(f"""
        WITH qs AS (SELECT * FROM _tbq WHERE qid >= {qid_lo} AND qid < {qid_hi}),
        u AS (
            SELECT q.qid, a.dst AS article_id FROM qs q JOIN _tb_assoc a ON a.src = q.anchor AND a.dst_slot = q.target_slot
            UNION
            SELECT q.qid, y.dst FROM qs q JOIN articles ar ON ar.article_id = q.anchor
                 JOIN _tb_style y ON y.src_code = ar.product_code AND y.dst_slot = q.target_slot
            UNION
            SELECT q.qid, t.dst FROM qs q JOIN _tb_tt t ON t.anchor = q.anchor AND t.target_slot = q.target_slot
            UNION
            SELECT q.qid, c.dst FROM qs q JOIN _tb_ttc c ON c.anchor = q.anchor AND c.target_slot = q.target_slot
            UNION
            SELECT q.qid, p.dst FROM qs q JOIN _tb_pop p ON p.target_slot = q.target_slot)
        SELECT u.qid, u.article_id, {sel}
        FROM u
        JOIN qs q USING (qid)
        JOIN articles ar ON ar.article_id = q.anchor
        LEFT JOIN _tb_assoc a ON a.src = q.anchor AND a.dst_slot = q.target_slot AND a.dst = u.article_id
        LEFT JOIN _tb_style y ON y.src_code = ar.product_code AND y.dst_slot = q.target_slot AND y.dst = u.article_id
        LEFT JOIN _tb_tt t ON t.anchor = q.anchor AND t.target_slot = q.target_slot AND t.dst = u.article_id
        LEFT JOIN _tb_ttc c ON c.anchor = q.anchor AND c.target_slot = q.target_slot AND c.dst = u.article_id
        LEFT JOIN _tb_pop p ON p.target_slot = q.target_slot AND p.dst = u.article_id
        ORDER BY u.qid, u.article_id""").df()


def union_recall(cand: pd.DataFrame, q: pd.DataFrame) -> dict:
    """Ceiling of the candidate union: share of truth pairs present, and set size."""
    truth = pd.DataFrame({"qid": np.repeat(q.qid.values, [len(t) for t in q.truth.values]),
                          "article_id": np.concatenate([np.asarray(t, dtype=np.int64) for t in q.truth.values])})
    hit = truth.merge(cand[["qid", "article_id"]].drop_duplicates(), on=["qid", "article_id"], how="left",
                      indicator=True)["_merge"].eq("both")
    return {"union_recall": float(hit.mean()), "candidates_per_query": len(cand) / max(1, cand.qid.nunique()),
            "n_queries": int(cand.qid.nunique()), "n_rows": int(len(cand))}
