"""M2: Recall of each retrieval channel and of the merged candidate set.

  python -m ensemble.candidates.evaluate [val|test|backtest] [n_weeks]

Per channel: recall, unique recall (hits no other enabled channel finds, i.e.
the drop in merged recall if the channel were removed), precision and
candidates per customer. Merged: recall, size, segment recall (new vs returning
customers; cold items, i.e. first sale inside the label week; recent launches,
i.e. first sale in the 28 days before it) and an overlap histogram (how many
channels find each hit). ``backtest`` evaluates the ``n_weeks`` label weeks
ending at validation; the test week is not touched.
"""
from __future__ import annotations

import json
import statistics
import time

from ensemble import tracking
from ensemble.candidates.retrieval import build_candidates, channel_columns
from ensemble.config import load_config
from ensemble.data.splits import load_splits
from ensemble.db import connect


def load_truth_table(con, week) -> None:
    """TEMP TABLE ``_truth``: distinct purchased (customer, article) pairs in the label week, with segments."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _truth AS
        WITH t AS (SELECT DISTINCT customer_idx, article_id FROM transactions
                   WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}'),
        ret AS (SELECT DISTINCT customer_idx FROM transactions
                WHERE t_dat < DATE '{week.start}' AND customer_idx IN (SELECT customer_idx FROM t)),
        first AS (SELECT article_id, min(t_dat) AS first_sale FROM transactions
                  WHERE article_id IN (SELECT article_id FROM t) GROUP BY 1)
        SELECT t.customer_idx, t.article_id,
               ret.customer_idx IS NOT NULL AS is_returning,
               first.first_sale >= DATE '{week.start}' AS is_cold_item,
               first.first_sale >= DATE '{week.start}' - INTERVAL 28 DAY AND first.first_sale < DATE '{week.start}' AS is_recent_item
        FROM t LEFT JOIN ret USING (customer_idx) JOIN first USING (article_id)
    """)


def recall_report(con, week, cand: str = "cand", truth_ready: bool = False) -> dict:
    if not truth_ready:
        load_truth_table(con, week)
    channels = channel_columns(con, cand)
    total, n_users = con.execute("SELECT count(*), count(DISTINCT customer_idx) FROM _truth").fetchone()
    flags = ", ".join(f"({c}_rank IS NOT NULL)::INT AS h_{c}" for c in channels)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _hits AS
                    SELECT t.*, {flags}, {' + '.join(f'({c}_rank IS NOT NULL)::INT' for c in channels)} AS n_ch
                    FROM _truth t JOIN {cand} c USING (customer_idx, article_id)""")
    out: dict = {}
    sizes = con.execute(f"SELECT {', '.join(f'count({c}_rank)' for c in channels)}, count(*) FROM {cand}").fetchone()
    hits = con.execute(f"SELECT {', '.join(f'sum(h_{c})' for c in channels)}, count(*) FROM _hits").fetchone()
    uniq = con.execute(f"SELECT {', '.join(f'sum((h_{c} = 1 AND n_ch = 1)::INT)' for c in channels)} FROM _hits").fetchone()
    for i, ch in enumerate([*channels, "merged"]):
        h, size = int(hits[i] or 0), int(sizes[i])
        out[ch] = {"recall": h / total, "candidates_per_customer": size / n_users, "precision": h / size if size else 0.0}
        if ch != "merged":
            out[ch]["unique_recall"] = int(uniq[i] or 0) / total
    out["unique_to_channel"] = {c: out[c]["unique_recall"] for c in channels}
    out["overlap_histogram"] = {str(k): v / total for k, v in con.execute(
        "SELECT n_ch, count(*) FROM _hits GROUP BY 1 ORDER BY 1").fetchall()}
    seg = con.execute("""
        SELECT count(*) FILTER (WHERE is_returning), count(*) FILTER (WHERE NOT is_returning),
               count(*) FILTER (WHERE is_cold_item), count(*) FILTER (WHERE is_recent_item) FROM _truth""").fetchone()
    seg_hits = con.execute("""
        SELECT count(*) FILTER (WHERE is_returning), count(*) FILTER (WHERE NOT is_returning),
               count(*) FILTER (WHERE is_cold_item), count(*) FILTER (WHERE is_recent_item) FROM _hits""").fetchone()
    out["segments"] = {name: {"recall": h / n if n else float("nan"), "share_of_truth": n / total}
                       for name, n, h in zip(("returning", "new_customers", "cold_items", "recent_items"), seg, seg_hits)}
    out["n_truth_pairs"], out["n_customers"] = int(total), int(n_users)
    con.execute("DROP TABLE _hits")
    return out


def users_table(con, week) -> None:
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _users AS SELECT DISTINCT customer_idx FROM transactions
                    WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}'""")


def evaluate_week(con, week, r) -> dict:
    users_table(con, week)
    t0 = time.time()
    secs = build_candidates(con, week, r)
    rep = recall_report(con, week)
    rep["build_seconds"] = round(time.time() - t0, 1)
    rep["build_seconds_by_stage"] = {k: round(v, 2) for k, v in secs.items()}
    return rep


def summarise(per_week: dict) -> dict:
    weeks = list(per_week.values())
    chs = [k for k, v in weeks[0].items() if isinstance(v, dict) and "recall" in v]
    out = {}
    for ch in chs:
        for m in ("recall", "candidates_per_customer", "precision", "unique_recall"):
            vals = [w[ch][m] for w in weeks if m in w[ch]]
            if vals:
                out[f"{ch}.{m}"] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    for seg in weeks[0]["segments"]:
        vals = [w["segments"][seg]["recall"] for w in weeks]
        out[f"segment.{seg}.recall"] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    out["build_seconds"] = {"mean": statistics.mean(w["build_seconds"] for w in weeks)}
    return out


def run(mode: str = "val", n_weeks: int = 4, tag: str | None = None) -> dict:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    if mode == "backtest":
        weeks = [splits.val.shift(-k) for k in range(n_weeks - 1, -1, -1)]
    else:
        weeks = [getattr(splits, mode)]
    per_week = {}
    for w in weeks:
        per_week[str(w.start)] = rep = evaluate_week(con, w, cfg.retrieval)
        print(f"{w.start}: merged recall={rep['merged']['recall']:.4f} cand/cust={rep['merged']['candidates_per_customer']:.1f} "
              f"({rep['build_seconds']}s)", flush=True)
    rep = per_week[str(weeks[-1].start)]
    for ch in [k for k, v in rep.items() if isinstance(v, dict) and "recall" in v]:
        m = rep[ch]
        print(f"{ch:17s} recall={m['recall']:.4f}  cand/cust={m['candidates_per_customer']:6.1f}  "
              f"precision={m['precision']:.4f}  unique={m.get('unique_recall', float('nan')):.4f}")
    print("segments", json.dumps(rep["segments"]))
    tag = tag or ("" if mode != "backtest" else f"_backtest{n_weeks}")
    name = f"m2_retrieval_{mode if mode != 'backtest' else 'val'}{tag}"
    with tracking.run("track_a", f"m2/retrieval/{mode}", dict(cfg.retrieval)):
        for ch, m in rep.items():
            if isinstance(m, dict) and "recall" in m:
                tracking.log_metrics({k: v for k, v in m.items()}, prefix=f"{ch}.")
    body = rep if mode != "backtest" else {"per_week": per_week, "summary": summarise(per_week)}
    (cfg.path("reports") / f"{name}.json").write_text(json.dumps(body, indent=2))
    return body


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "val", int(sys.argv[2]) if len(sys.argv) > 2 else 4)
