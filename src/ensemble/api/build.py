"""M6: build the SQLite serving store from pipeline outputs (D-007).

This process loads LightGBM and must not load PyTorch: the two ship separate
OpenMP runtimes, and loading both in one process deadlocks on macOS. Complete
the Look is therefore precomputed by ``ensemble.completion.serve_ctl``.

  python -m ensemble.api.build

Everything the web app shows is precomputed here, so serving is indexed point
lookups only.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import time
from collections import defaultdict

import lightgbm as lgb
import pandas as pd

from ensemble.baselines import AGE_BINS, popular_by_age
from ensemble.config import load_config
from ensemble.data.splits import load_splits
from ensemble.db import connect
from ensemble.ranking.explain import explain
from ensemble.ranking.train import week_frame


def pick_demo_customers(con, cfg, week) -> pd.DataFrame:
    """A deterministic sample (ordered by hash) so demo links stay stable across rebuilds."""
    s = cfg.serving
    return con.execute(f"""
        (SELECT customer_idx, 'returning' AS segment FROM (
            SELECT customer_idx, count(*) n, max(t_dat) last_dat FROM transactions
            WHERE t_dat >= DATE '{week.start}' - INTERVAL 12 WEEK GROUP BY 1
            HAVING n BETWEEN 15 AND 80 AND last_dat >= DATE '{week.start}' - INTERVAL 2 WEEK)
         ORDER BY hash(customer_idx) LIMIT {int(s.demo_returning)})
        UNION ALL
        (SELECT customer_idx, 'new' FROM customers ANTI JOIN transactions USING (customer_idx)
         WHERE age IS NOT NULL ORDER BY hash(customer_idx) LIMIT {int(s.demo_new)})
    """).df()


def track_a_with_reasons(con, cfg, week, demo: pd.DataFrame) -> pd.DataFrame:
    booster = lgb.Booster(model_file=str(cfg.path("processed") / "models" / "ranker_submission.txt"))
    features = booster.feature_name()
    con.execute("CREATE OR REPLACE TEMP TABLE _demo AS SELECT customer_idx FROM demo")
    df = week_frame(con, cfg, week, "SELECT customer_idx FROM _demo", False, False)
    df["score"] = booster.predict(df[features], num_threads=8)
    serve = dict(cfg.get("rerank", {}).get("serve") or {})
    if serve.get("max_days_since_last_sale") is not None:   # availability proxy (rerank.py)
        df = df[df["a_days_since_last_sale"] <= serve["max_days_since_last_sale"]]
    df = (df.sort_values(["customer_idx", "score", "article_id"], ascending=[True, False, True])
            .groupby("customer_idx").head(12).reset_index(drop=True))
    ex = explain(booster, features, df)
    df["rank"] = df.groupby("customer_idx").cumcount() + 1
    df["reasons"] = [json.dumps(e["reasons"]) for e in ex]
    df["shap"] = [json.dumps(e["shap"]) for e in ex]
    return df[["customer_idx", "rank", "article_id", "score", "reasons", "shap"]]


def new_customer_reasons(con, week, recs: pd.DataFrame, k: int = 100) -> list[str]:
    """Reasons for customers with no history, from evidence only: an age-band best seller gets
    "Popular with customers your age", a global best seller "Trending this week", anything else none."""
    from ensemble.baselines import customer_age_bins, popular
    from ensemble.ranking.explain import REASON_TEXT, REASON_TYPE
    by_age = {b: set(v) for b, v in popular_by_age(con, week, k=k).items()}
    top = set(popular(con, week, k=k))
    ages = customer_age_bins(con, recs.customer_idx.unique().tolist())
    out = []
    for c, a in zip(recs.customer_idx, recs.article_id):
        key = "age_group" if a in by_age.get(ages.get(c, -1), set()) else "trending" if a in top else None
        out.append(json.dumps([{"key": key, "text": REASON_TEXT[key], "evidence": REASON_TYPE[key]}] if key else []))
    return out


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    con = connect(cfg, read_only=True)
    week = load_splits(con, cfg).submission
    out = cfg.path("serving_db")
    out.unlink(missing_ok=True)
    db = sqlite3.connect(out)

    demo = pick_demo_customers(con, cfg, week)
    print(f"demo customers: {len(demo)}", flush=True)
    ta = track_a_with_reasons(con, cfg, week, demo[demo.segment == "returning"])
    all_recs = pd.read_parquet(cfg.path("processed") / "track_a_recs.parquet")
    new_recs = all_recs[all_recs.customer_idx.isin(demo[demo.segment == "new"].customer_idx)].explode("articles")
    new_recs = new_recs.assign(rank=new_recs.groupby("customer_idx").cumcount() + 1, score=None,
                               shap="[]").rename(columns={"articles": "article_id"})
    new_recs["article_id"] = new_recs["article_id"].astype("int64")  # explode leaves objects → SQLite blobs
    new_recs["reasons"] = new_customer_reasons(con, week, new_recs)
    pd.concat([ta, new_recs[ta.columns]]).to_sql("for_you", db, index=False)
    print(f"for_you built ({time.time() - t0:.0f}s)", flush=True)

    con.execute("CREATE OR REPLACE TEMP TABLE _demo AS SELECT * FROM demo")
    con.execute(f"""SELECT d.customer_idx, d.segment, c.age, count(t.article_id) AS n_purchases, CAST(max(t.t_dat) AS VARCHAR) AS last_purchase
                    FROM _demo d JOIN customers c USING (customer_idx) LEFT JOIN transactions t USING (customer_idx)
                    GROUP BY ALL ORDER BY d.segment DESC, n_purchases DESC""").df().to_sql("demo_customers", db, index=False)
    con.execute("""SELECT customer_idx, article_id, CAST(max(t_dat) AS VARCHAR) AS last_bought, count(*) AS times FROM transactions
                   JOIN _demo USING (customer_idx) GROUP BY 1, 2""").df().to_sql("history", db, index=False)
    pa = popular_by_age(con, week, k=int(cfg.serving.trending_k))
    pd.DataFrame([(b, r + 1, a) for b, lst in pa.items() for r, a in enumerate(lst)],
                 columns=["age_bin", "rank", "article_id"]).to_sql("trending", db, index=False)
    con.execute(f"SELECT customer_idx, {AGE_BINS} AS age_bin FROM customers JOIN _demo USING (customer_idx)").df() \
        .to_sql("customer_age_bin", db, index=False)

    ctl = pd.read_parquet(cfg.path("processed") / "models" / "complete_the_look.parquet")
    ctl.to_sql("complete_the_look", db, index=False)
    print(f"complete_the_look: {len(ctl):,} rows ({time.time() - t0:.0f}s)", flush=True)

    used = set(ctl.anchor) | set(ctl.article_id) | set(ta.article_id) | set(new_recs.article_id)
    con.execute("CREATE OR REPLACE TEMP TABLE _hist_ids AS SELECT DISTINCT article_id FROM transactions JOIN _demo USING (customer_idx)")
    con.execute("""SELECT a.article_id, a.product_code, a.prod_name, a.product_type_name, a.product_group_name,
                          a.colour_group_name, a.department_name, a.index_group_name, a.garment_group_name,
                          a.slot, a.is_jewellery, a.detail_desc
                   FROM articles a""").df().to_sql("articles", db, index=False)
    uni = pd.read_parquet(cfg.path("processed") / "models" / "completion_universe.parquet")
    uni[["article_id", "pop_recent", "pop_window"]].to_sql("popularity", db, index=False)

    db.execute("""CREATE TABLE events (ts TEXT DEFAULT CURRENT_TIMESTAMP, customer_idx INTEGER,
                  event TEXT, surface TEXT, article_id INTEGER, anchor INTEGER)""")
    reports = {}
    for name in ["m1_baselines_val", "m1_baselines_test", "m2_retrieval_val", "m3_ranker_val", "m3_ranker_test", "m4_track_b_val", "m4_track_b_test"]:
        p = cfg.path("reports") / f"{name}.json"
        if p.exists():
            reports[name] = json.loads(p.read_text())
    db.execute("CREATE TABLE reports (name TEXT PRIMARY KEY, body TEXT)")
    db.executemany("INSERT INTO reports VALUES (?, ?)", [(k, json.dumps(v)) for k, v in reports.items()])
    for stmt in ["CREATE INDEX ix_for_you ON for_you(customer_idx, rank)",
                 "CREATE INDEX ix_hist ON history(customer_idx, last_bought)",
                 "CREATE INDEX ix_ctl ON complete_the_look(anchor, slot, rank)",
                 "CREATE UNIQUE INDEX ix_art ON articles(article_id)",
                 "CREATE INDEX ix_art_prod ON articles(product_code)",
                 "CREATE INDEX ix_art_type ON articles(product_type_name, index_group_name)",
                 "CREATE UNIQUE INDEX ix_pop ON popularity(article_id)"]:
        db.execute(stmt)
    db.commit()
    db.close()
    print(f"wrote {out} in {time.time() - t0:.0f}s; {len(used):,} articles referenced")


if __name__ == "__main__":
    main()
