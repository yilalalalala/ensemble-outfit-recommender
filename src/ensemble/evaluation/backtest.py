"""Rolling backtest for Track A (improvement plan step 1).

Evaluates the M3 recipe on several consecutive label weeks before the test week
(which stays untouched), retraining the ranker for each week. Reports mean ± std,
so decisions on small segments (new customers: ~8% of buyers) rest on more than
one week. Also evaluates a segment-level fallback policy: customers with no
purchase history get age-band popularity instead of the ranker.

  python -m ensemble.evaluation.backtest [n_weeks]
"""
from __future__ import annotations

import gc
import json
import statistics
import time

from ensemble import tracking
from ensemble.baselines import customer_age_bins, fill, popular, popular_by_age, repeat_purchase
from ensemble.config import load_config
from ensemble.data.splits import load_splits
from ensemble.db import connect
from ensemble.evaluation.track_a import evaluate, load_segments, load_truth
from ensemble.ranking.train import buyers_sql, rank, train, week_frame


def evaluate_week(con, cfg, target) -> dict:
    t0 = time.time()
    booster, features = train(con, cfg, load_splits(con, cfg).train_weeks(int(cfg.ranker.train_weeks), before=target))
    df = week_frame(con, cfg, target, buyers_sql(target), False, False)
    ranked = rank(booster, features, df)
    del df
    gc.collect()
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
    out = {name: evaluate(p, truth, seg) for name, p in systems.items()}
    out["seconds"] = round(time.time() - t0)
    return out


def summarise(per_week: dict) -> dict:
    systems = [k for k in next(iter(per_week.values())) if k != "seconds"]
    summary = {}
    for s in systems:
        for m in ("map@12", "map@12_returning", "map@12_new_customers"):
            vals = [w[s][m] for w in per_week.values()]
            summary[f"{s}.{m}"] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    lifts = [(w["ranker"]["map@12"] - w["baseline_repeat+pop_age"]["map@12"]) / w["baseline_repeat+pop_age"]["map@12"]
             for w in per_week.values()]
    summary["ranker_relative_lift"] = {"mean": statistics.mean(lifts), "std": statistics.stdev(lifts), "per_week": lifts}
    diffs = [w["ranker+new_customer_fallback"]["map@12_new_customers"] - w["ranker"]["map@12_new_customers"]
             for w in per_week.values()]
    summary["fallback_minus_ranker_new_customers"] = {"mean": statistics.mean(diffs), "per_week": diffs,
                                                       "weeks_fallback_better": sum(d > 0 for d in diffs)}
    return summary


def run(n_weeks: int = 4) -> dict:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    weeks = [splits.val.shift(-k) for k in range(n_weeks - 1, -1, -1)]  # ends at validation; test untouched
    per_week = {}
    for w in weeks:
        per_week[str(w.start)] = evaluate_week(con, cfg, w)
        r = per_week[str(w.start)]
        print(f"{w.start}: baseline {r['baseline_repeat+pop_age']['map@12']:.5f}  ranker {r['ranker']['map@12']:.5f}  "
              f"new: ranker {r['ranker']['map@12_new_customers']:.5f} vs fallback "
              f"{r['ranker+new_customer_fallback']['map@12_new_customers']:.5f}  ({r['seconds']}s)", flush=True)
    summary = summarise(per_week)
    with tracking.run("track_a", f"backtest/{n_weeks}w", {"weeks": [str(w.start) for w in weeks]}):
        tracking.log_metrics({k: v["mean"] for k, v in summary.items() if isinstance(v.get("mean"), float)})
    (cfg.path("reports") / "backtest_track_a.json").write_text(json.dumps({"per_week": per_week, "summary": summary}, indent=2))
    for k, v in summary.items():
        print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return summary


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
