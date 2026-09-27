"""M2: Recall of each retrieval channel and of the merged candidate set."""
from __future__ import annotations

import json
import time

from ensemble import tracking
from ensemble.candidates.retrieval import CHANNELS, build_candidates
from ensemble.config import load_config
from ensemble.data.splits import load_splits
from ensemble.db import connect


def recall_report(con, week) -> dict:
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _truth AS SELECT DISTINCT customer_idx, article_id
                    FROM transactions WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}'""")
    total = con.execute("SELECT count(*) FROM _truth").fetchone()[0]
    n_users = con.execute("SELECT count(DISTINCT customer_idx) FROM _truth").fetchone()[0]
    out = {}
    for ch in (*CHANNELS, "merged"):
        where = "TRUE" if ch == "merged" else f"{ch}_rank IS NOT NULL"
        hit, size = con.execute(f"""
            SELECT (SELECT count(*) FROM _truth JOIN cand USING (customer_idx, article_id) WHERE {where}),
                   (SELECT count(*) FROM cand WHERE {where})""").fetchone()
        out[ch] = {"recall": hit / total, "candidates_per_customer": size / n_users,
                   "precision": hit / size if size else 0.0}
    out["unique_to_channel"] = {
        ch: con.execute(f"""SELECT count(*) FROM _truth JOIN cand USING (customer_idx, article_id)
                            WHERE {ch}_rank IS NOT NULL AND
                            {' AND '.join(f'{o}_rank IS NULL' for o in CHANNELS if o != ch)}""").fetchone()[0] / total
        for ch in CHANNELS
    }
    return out


def run(week_name: str = "val") -> dict:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    week = getattr(load_splits(con, cfg), week_name)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _users AS SELECT DISTINCT customer_idx FROM transactions
                    WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}'""")
    t0 = time.time()
    build_candidates(con, week, cfg.retrieval)
    secs = time.time() - t0
    rep = recall_report(con, week)
    rep["build_seconds"] = round(secs, 1)
    with tracking.run("track_a", f"m2/retrieval/{week_name}", dict(cfg.retrieval)):
        for ch, m in rep.items():
            if isinstance(m, dict) and "recall" in m:
                tracking.log_metrics(m, prefix=f"{ch}.")
    for ch in (*CHANNELS, "merged"):
        m = rep[ch]
        print(f"{ch:8s} recall={m['recall']:.4f}  cand/cust={m['candidates_per_customer']:6.1f}  "
              f"precision={m['precision']:.4f}  unique={rep['unique_to_channel'].get(ch, float('nan')):.4f}")
    print("build seconds", rep["build_seconds"])
    (cfg.path("reports") / f"m2_retrieval_{week_name}.json").write_text(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "val")
