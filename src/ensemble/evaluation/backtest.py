"""Rolling backtest for Track A (D-017).

Evaluates the M3 recipe on several consecutive label weeks before the test week
(which stays untouched), retraining the ranker for each week. Reports mean ± std,
so decisions on small segments (new customers: ~8% of buyers) rest on more than
one week. Also evaluates a segment-level fallback policy: customers with no
purchase history get age-band popularity instead of the ranker.

Per week it records the candidate recall of the retrieval stage, the ranker's
MAP@12 by segment, and per-customer AP (``data/interim/ap/``) for paired
bootstrap comparisons between configurations (``ensemble.evaluation.compare``).

  python -m ensemble.evaluation.backtest [n_weeks] [tag]
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import time

from ensemble import tracking
from ensemble.baselines import customer_age_bins, fill, popular, popular_by_age, repeat_purchase
from ensemble.candidates.evaluate import evaluate_week as retrieval_week
from ensemble.config import load_config
from ensemble.data.splits import load_splits
from ensemble.db import connect
from ensemble.evaluation.track_a import evaluate, load_segments, load_truth, per_customer_ap
from ensemble.ranking.train import (RERANK_COLS, build_training_data, buyers_sql, fit, save_ap, save_scored,
                                    score_users, top_k)

METRICS = ("map@12", "map@12_returning", "map@12_new_customers", "recall@12", "recall@12_cold_items")


def evaluate_week(con, cfg, target, tag: str) -> dict:
    t0 = time.time()
    weeks = load_splits(con, cfg).train_weeks(int(cfg.ranker.train_weeks), before=target)
    booster, features, info = fit(cfg, build_training_data(con, cfg, weeks))
    gc.collect()
    scored = score_users(con, cfg, booster, features, target, buyers_sql(target), extra_cols=RERANK_COLS)
    save_scored(cfg, f"backtest{tag}_{target.start}", scored)
    ranked = top_k(scored)
    cand_per_cust = scored.attrs["candidates_per_customer"]
    del scored
    gc.collect()
    retrieval = retrieval_week(con, target, cfg.retrieval)
    truth, seg = load_truth(con, target), load_segments(con, target)
    users = list(truth)
    pop, pop_age, ages = popular(con, target), popular_by_age(con, target), customer_age_bins(con, users)
    rep = repeat_purchase(con, target, users)
    age_list = lambda u: pop_age.get(ages.get(u, -1), pop)  # noqa: E731
    systems = {
        "baseline_repeat+pop_age": {u: fill(rep.get(u, []), age_list(u)) for u in users},
        "ranker": {u: fill(ranked.get(u, []), age_list(u)) for u in users},
    }
    systems["ranker+new_customer_fallback"] = {
        u: (systems["ranker"][u] if u in seg["returning"] else age_list(u)) for u in users}
    save_ap(cfg, f"backtest{tag}_{target.start}", per_customer_ap(systems["ranker"], truth))
    out = {name: evaluate(p, truth, seg) for name, p in systems.items()}
    out["retrieval"] = {"recall": retrieval["merged"]["recall"], "candidates_per_customer": cand_per_cust,
                        "segments": retrieval["segments"],
                        "unique_recall": retrieval["unique_to_channel"]}
    out["fit"] = info
    out["seconds"] = round(time.time() - t0)
    return out


def _ms(vals):
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def summarise(per_week: dict) -> dict:
    weeks = list(per_week.values())
    systems = [k for k in weeks[0] if k not in ("seconds", "retrieval", "fit")]
    summary = {}
    for s in systems:
        for m in METRICS:
            summary[f"{s}.{m}"] = _ms([w[s][m] for w in weeks])
    summary["retrieval.recall"] = _ms([w["retrieval"]["recall"] for w in weeks])
    summary["retrieval.candidates_per_customer"] = _ms([w["retrieval"]["candidates_per_customer"] for w in weeks])
    lifts = [(w["ranker"]["map@12"] - w["baseline_repeat+pop_age"]["map@12"]) / w["baseline_repeat+pop_age"]["map@12"]
             for w in weeks]
    summary["ranker_relative_lift"] = {**_ms(lifts), "per_week": lifts}
    diffs = [w["ranker+new_customer_fallback"]["map@12_new_customers"] - w["ranker"]["map@12_new_customers"]
             for w in weeks]
    summary["fallback_minus_ranker_new_customers"] = {"mean": statistics.mean(diffs), "per_week": diffs,
                                                       "weeks_fallback_better": sum(d > 0 for d in diffs)}
    summary["seconds_per_week"] = _ms([w["seconds"] for w in weeks])
    return summary


def run(n_weeks: int = 4, tag: str | None = None) -> dict:
    cfg = load_config()
    if tag is None:
        name = os.environ.get("ENSEMBLE_CONFIG", "default")
        tag = "" if name == "default" else f"_{os.path.basename(name)}"
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    weeks = [splits.val.shift(-k) for k in range(n_weeks - 1, -1, -1)]  # ends at validation; test untouched
    per_week = {}
    for w in weeks:
        per_week[str(w.start)] = r = evaluate_week(con, cfg, w, tag)
        print(f"{w.start}: baseline {r['baseline_repeat+pop_age']['map@12']:.5f}  ranker {r['ranker']['map@12']:.5f}  "
              f"retrieval recall {r['retrieval']['recall']:.4f} @ {r['retrieval']['candidates_per_customer']:.1f}  "
              f"new: ranker {r['ranker']['map@12_new_customers']:.5f} vs fallback "
              f"{r['ranker+new_customer_fallback']['map@12_new_customers']:.5f}  ({r['seconds']}s)", flush=True)
    summary = summarise(per_week)
    with tracking.run("track_a", f"backtest/{n_weeks}w{tag}", {"weeks": [str(w.start) for w in weeks],
                                                               "channels": cfg.retrieval.get("channels")}):
        tracking.log_metrics({k: v["mean"] for k, v in summary.items() if isinstance(v.get("mean"), float)})
    out_dir = cfg.path("reports") if not tag else cfg.path("reports") / "phase2"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"backtest_track_a{tag}.json").write_text(json.dumps({"per_week": per_week, "summary": summary}, indent=2))
    for k, v in summary.items():
        print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return summary


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 4, sys.argv[2] if len(sys.argv) > 2 else None)
