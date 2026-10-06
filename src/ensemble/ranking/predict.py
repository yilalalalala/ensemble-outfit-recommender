"""Track A inference for the Kaggle week and the serving store.

  python -m ensemble.ranking.predict

Trains the M3 recipe on the ``train_weeks`` label weeks ending with the final
week of data, then ranks every customer in ``sample_submission.csv`` in chunks
and applies the shipped re-ranking policy (``rerank.serve``, e.g. availability).
Writes:
  reports/submission.csv                         Kaggle format, one row per customer
  data/processed/track_a_recs.parquet            top-12 per customer with scores
"""
from __future__ import annotations

import gc
import time

import pandas as pd

from ensemble.baselines import customer_age_bins, fill, popular, popular_by_age
from ensemble.config import load_config
from ensemble.data.splits import load_splits
from ensemble.db import connect
from ensemble.ranking.rerank import apply_rules
from ensemble.ranking.train import RERANK_COLS, score_users, train

CHUNK = 100_000          # customers per outer batch; score_users builds features 25k at a time


def main() -> None:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    week = splits.submission
    weeks = splits.train_weeks(int(cfg.ranker.train_weeks), before=week)
    print(f"training on label weeks {[str(w.start) for w in weeks]} for {week.start}..{week.end}", flush=True)
    booster, features = train(con, cfg, weeks)
    models = cfg.path("processed") / "models"
    models.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(models / "ranker_submission.txt"))

    all_customers = con.execute("SELECT customer_idx FROM customers ORDER BY customer_idx").fetchnumpy()["customer_idx"]
    pop = popular(con, week)
    pop_age = popular_by_age(con, week)
    articles = con.execute("SELECT article_id, product_code, product_type_no FROM articles").df().set_index("article_id")
    parts = []
    for i in range(0, len(all_customers), CHUNK):
        t = time.time()
        chunk = all_customers[i:i + CHUNK]
        con.execute("CREATE OR REPLACE TEMP TABLE _pred_chunk AS SELECT unnest(?::INTEGER[]) AS customer_idx", [chunk.tolist()])
        scored = score_users(con, cfg, booster, features, week, "SELECT customer_idx FROM _pred_chunk",
                             extra_cols=RERANK_COLS, chunk=25_000)
        ranked = apply_rules(scored, articles, **dict(cfg.rerank.get("serve") or {}))
        ages = customer_age_bins(con, chunk.tolist())
        rows = []
        for c in chunk.tolist():
            rec = fill(ranked.get(c, []), pop_age.get(ages.get(c, -1), pop))
            rows.append((c, rec))
        parts.append(pd.DataFrame(rows, columns=["customer_idx", "articles"]))
        del scored
        gc.collect()
        print(f"  customers {i:,}–{i + len(chunk):,} ranked ({time.time() - t:.0f}s)", flush=True)
    recs = pd.concat(parts, ignore_index=True)
    recs.to_parquet(cfg.path("processed") / "track_a_recs.parquet")

    ids = con.execute("SELECT customer_idx, customer_id FROM customers").df().set_index("customer_idx")["customer_id"]
    sub = pd.DataFrame({
        "customer_id": ids.loc[recs.customer_idx].values,
        "prediction": [" ".join(f"{a:010d}" for a in r) for r in recs.articles],
    })
    sample = pd.read_csv(cfg.path("raw") / "sample_submission.csv", usecols=["customer_id"])
    assert set(sample.customer_id) == set(sub.customer_id), "submission customers differ from sample_submission"
    assert sub.prediction.str.split().str.len().eq(12).all()
    out = cfg.path("reports") / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out} ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
