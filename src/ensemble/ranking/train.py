"""M3: LightGBM LambdaRank ranker over retrieval candidates (DESIGN §4).

Modes (each week's features use only data before that week, D-003):
  val         train on the ``train_weeks`` label weeks before validation, score validation
  test        same recipe shifted one week: train through validation, score test (touched once)
  submission  shifted again: train through test, rank all customers for the Kaggle week
"""
from __future__ import annotations

import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ensemble import tracking
from ensemble.baselines import customer_age_bins, fill, popular, popular_by_age
from ensemble.candidates.retrieval import build_candidates
from ensemble.config import load_config
from ensemble.data.splits import Week, load_splits
from ensemble.db import connect
from ensemble.evaluation.track_a import evaluate, load_segments, load_truth
from ensemble.features.track_a import CATEGORICAL, build_features

ID_COLS = ["customer_idx", "article_id", "label", "_week"]


def week_frame(con, cfg, week: Week, users_sql: str, with_labels: bool, positives_only: bool) -> pd.DataFrame:
    con.execute(f"CREATE OR REPLACE TEMP TABLE _users AS {users_sql}")
    build_candidates(con, week, cfg.retrieval)
    df = build_features(con, week, with_labels)
    if cfg.get("features", {}).get("use_clip"):
        from ensemble.candidates.visual import add_clip_features
        df = add_clip_features(con, week, cfg, df)
    if positives_only:
        keep = df.groupby("customer_idx")["label"].transform("max") > 0
        df = df[keep].reset_index(drop=True)
    return df


def buyers_sql(week: Week) -> str:
    return (f"SELECT DISTINCT customer_idx FROM transactions "
            f"WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}'")


def train(con, cfg, weeks: list[Week]) -> tuple[lgb.Booster, list[str]]:
    frames = []
    for i, w in enumerate(weeks):
        t = time.time()
        frames.append(week_frame(con, cfg, w, buyers_sql(w), True, True).assign(_week=np.int8(i)))
        print(f"  train week {w.start}: {len(frames[-1]):,} rows, pos rate {frames[-1].label.mean():.4f} ({time.time()-t:.0f}s)", flush=True)
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    return _fit(cfg, df)


def _fit(cfg, df: pd.DataFrame):
    features = [c for c in df.columns if c not in ID_COLS]
    # One query group per (week, customer); rows arrive sorted by week then customer.
    c, w = df.customer_idx.values, df["_week"].values
    change = np.r_[True, (c[1:] != c[:-1]) | (w[1:] != w[:-1])]
    group_sizes = np.diff(np.r_[np.flatnonzero(change), len(df)])
    ds = lgb.Dataset(df[features], label=df["label"], group=group_sizes,
                     categorical_feature=[c for c in CATEGORICAL if c in features], free_raw_data=True)
    params = dict(cfg.ranker.params)
    booster = lgb.train(params, ds, num_boost_round=int(cfg.ranker.num_boost_round))
    return booster, features


def rank(booster, features, df: pd.DataFrame, k: int = 12, num_iteration: int | None = None) -> dict[int, list[int]]:
    df = df[["customer_idx", "article_id"]].assign(
        score=booster.predict(df[features], num_threads=8, num_iteration=num_iteration))
    df = df.sort_values(["customer_idx", "score"], ascending=[True, False])
    top = df.groupby("customer_idx", sort=False).head(k)
    return top.groupby("customer_idx", sort=False)["article_id"].agg(list).to_dict()


def importance(booster, features) -> list[tuple[str, float]]:
    gain = booster.feature_importance("gain")
    total = gain.sum()
    return sorted(((f, float(g / total)) for f, g in zip(features, gain)), key=lambda x: -x[1])


def run_eval(mode: str) -> dict:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    target = {"val": splits.val, "test": splits.test}[mode]
    weeks = splits.train_weeks(int(cfg.ranker.train_weeks), before=target)
    t0 = time.time()
    booster, features = train(con, cfg, weeks)
    train_secs = time.time() - t0

    df = week_frame(con, cfg, target, buyers_sql(target), False, False)
    truth = load_truth(con, target)
    segments = load_segments(con, target)
    pop_age = popular_by_age(con, target)
    pop = popular(con, target)
    ages = customer_age_bins(con, list(truth))
    by_rounds = {}
    # Report MAP at several ensemble sizes from the one booster (diagnostic for num_boost_round).
    for n in sorted({100, 200, 300, booster.current_iteration()}):
        preds = rank(booster, features, df, num_iteration=n)
        preds = {u: fill(preds.get(u, []), pop_age.get(ages.get(u, -1), pop)) for u in truth}
        by_rounds[n] = evaluate(preds, truth, segments)
        print(f"  rounds={n}: MAP@12={by_rounds[n]['map@12']:.5f}", flush=True)
    del df
    gc.collect()
    m = dict(by_rounds[booster.current_iteration()])
    m["map@12_by_rounds"] = {str(k): v["map@12"] for k, v in by_rounds.items()}
    m["train_seconds"] = round(train_secs, 1)
    imp = importance(booster, features)
    with tracking.run("track_a", f"m3/lgbm_lambdarank/{mode}", {**dict(cfg.ranker.params), "train_weeks": cfg.ranker.train_weeks,
                                                               "num_boost_round": cfg.ranker.num_boost_round}):
        tracking.log_metrics({k: v for k, v in m.items() if not isinstance(v, dict)})
    print(f"{mode}: MAP@12={m['map@12']:.5f} returning={m['map@12_returning']:.5f} new={m['map@12_new_customers']:.5f} "
          f"recall@12={m['recall@12']:.4f} ({m['train_seconds']}s train)")
    print("top features:", [(f, round(g, 3)) for f, g in imp[:15]])
    models = cfg.path("processed") / "models"
    models.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(models / f"ranker_{mode}.txt"))
    (cfg.path("reports") / f"m3_ranker_{mode}.json").write_text(json.dumps({"metrics": m, "importance": imp}, indent=2))
    return m


if __name__ == "__main__":
    import sys
    run_eval(sys.argv[1] if len(sys.argv) > 1 else "val")
