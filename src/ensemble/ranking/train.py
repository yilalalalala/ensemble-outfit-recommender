"""M3: LightGBM LambdaRank ranker over retrieval candidates (DESIGN §4).

Modes (each week's features use only data before that week, D-003):
  val         train on the ``train_weeks`` label weeks before validation, score validation
  test        same recipe shifted one week: train through validation, score test (touched once)
  submission  shifted again: train through test, rank all customers for the Kaggle week

**Number of trees.** Chosen by temporal early stopping: fit on the older
training weeks, stop on the most recent training week (still strictly before
the evaluation week), then refit on all training weeks with the best iteration
count (``ranker.early_stopping``). The evaluation week is never used to choose
the number of trees.

**Objective.** ``lambdarank`` optimises an NDCG-based surrogate; MAP@12 is the
reported metric and the early-stopping metric, not the training loss.

**Reproducibility.** LightGBM runs with fixed seeds and ``deterministic``;
rows are ordered by (week, customer, article) and ties in the predicted score
break on article_id. Each run writes a manifest (git commit, config hash, data
fingerprint, feature schema) next to its report.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ensemble import tracking
from ensemble.baselines import customer_age_bins, fill, popular, popular_by_age
from ensemble.candidates.retrieval import build_candidates
from ensemble.config import ROOT, load_config
from ensemble.data.splits import Week, load_splits
from ensemble.db import connect
from ensemble.evaluation.track_a import evaluate, load_segments, load_truth, per_customer_ap
from ensemble.features.track_a import CATEGORICAL, DEFAULT_GROUPS, build_features

ID_COLS = ["customer_idx", "article_id", "label", "_week"]


def feature_groups(cfg) -> tuple:
    return tuple(cfg.get("features", {}).get("groups", DEFAULT_GROUPS))


def week_frame(con, cfg, week: Week, users_sql: str, with_labels: bool, positives_only: bool,
               neg_rate: float = 1.0) -> pd.DataFrame:
    con.execute(f"CREATE OR REPLACE TEMP TABLE _users AS {users_sql}")
    t = time.time()
    build_candidates(con, week, cfg.retrieval)
    retrieval_secs = time.time() - t
    if positives_only:
        con.execute("""DELETE FROM cand WHERE customer_idx NOT IN (
                         SELECT DISTINCT customer_idx FROM cand JOIN (
                           SELECT DISTINCT customer_idx, article_id FROM transactions
                           WHERE t_dat BETWEEN ? AND ?) USING (customer_idx, article_id))""", [week.start, week.end])
    if neg_rate < 1.0:
        # Deterministic negative downsampling (training only): keep every positive and a hashed share of negatives.
        con.execute(f"""DELETE FROM cand WHERE (customer_idx, article_id) NOT IN (
                          SELECT customer_idx, article_id FROM transactions WHERE t_dat BETWEEN ? AND ?)
                        AND hash(customer_idx * 1000003 + article_id) % 10000 >= {int(neg_rate * 10000)}""",
                    [week.start, week.end])
    t = time.time()
    df = build_features(con, week, with_labels, groups=feature_groups(cfg), cfg=cfg)
    df.attrs["retrieval_seconds"], df.attrs["feature_seconds"] = retrieval_secs, time.time() - t
    if cfg.get("features", {}).get("use_clip"):
        from ensemble.candidates.visual import add_clip_features
        df = add_clip_features(con, week, cfg, df)
    return df


def buyers_sql(week: Week) -> str:
    return (f"SELECT DISTINCT customer_idx FROM transactions "
            f"WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}'")


def build_training_frame(con, cfg, weeks: list[Week]) -> pd.DataFrame:
    frames = []
    neg_rate = float(cfg.ranker.get("neg_sample_rate", 1.0))
    for i, w in enumerate(weeks):
        t = time.time()
        frames.append(week_frame(con, cfg, w, buyers_sql(w), True, True, neg_rate).assign(_week=np.int8(i)))
        f = frames[-1]
        print(f"  train week {w.start}: {len(f):,} rows, pos rate {f.label.mean():.4f} "
              f"(retrieval {f.attrs['retrieval_seconds']:.0f}s, features {f.attrs['feature_seconds']:.0f}s)", flush=True)
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    return df


def train(con, cfg, weeks: list[Week]) -> tuple[lgb.Booster, list[str]]:
    df = build_training_frame(con, cfg, weeks)
    booster, features, _ = fit(cfg, df)
    return booster, features


def _groups(df: pd.DataFrame) -> np.ndarray:
    """One query group per (week, customer); rows arrive sorted by week then customer."""
    c, w = df.customer_idx.to_numpy(), df["_week"].to_numpy()
    change = np.r_[True, (c[1:] != c[:-1]) | (w[1:] != w[:-1])]
    return np.diff(np.r_[np.flatnonzero(change), len(df)])


def params_of(cfg) -> dict:
    p = dict(cfg.ranker.params)
    seed = int(cfg.ranker.get("seed", 42))
    p.setdefault("seed", seed)
    p.setdefault("deterministic", True)
    p.setdefault("force_col_wise", True)
    return p


def _dataset(df, features, reference=None):
    return lgb.Dataset(df[features], label=df["label"], group=_groups(df),
                       categorical_feature=[c for c in CATEGORICAL if c in features],
                       reference=reference, free_raw_data=True)


def fit(cfg, df: pd.DataFrame) -> tuple[lgb.Booster, list[str], dict]:
    """Return (booster, features, info). ``info`` records how the number of trees was chosen."""
    features = [c for c in df.columns if c not in ID_COLS]
    params = params_of(cfg)
    es = cfg.ranker.get("early_stopping") or {}
    info: dict = {}
    if es and df["_week"].nunique() > 1:
        last = df["_week"].max()
        tr, va = df[df["_week"] < last], df[df["_week"] == last]
        dtr = _dataset(tr, features)
        dva = _dataset(va, features, reference=dtr)
        evals: dict = {}
        b = lgb.train(params, dtr, num_boost_round=int(es["max_rounds"]), valid_sets=[dva], valid_names=["last_week"],
                      callbacks=[lgb.early_stopping(int(es["patience"]), verbose=False), lgb.record_evaluation(evals)])
        best = int(b.best_iteration)
        metric = next(iter(evals["last_week"]))
        info = {"best_iteration": best, "early_stopping_metric": metric,
                "early_stopping_curve": [round(v, 6) for v in evals["last_week"][metric][::25]],
                "best_score": float(evals["last_week"][metric][best - 1])}
        del dtr, dva, tr, va
        gc.collect()
        if not es.get("refit", True):
            return b, features, info
        rounds = max(1, int(round(best * float(es.get("refit_multiplier", 1.0)))))
    else:
        rounds = int(cfg.ranker.num_boost_round)
    info["num_boost_round"] = rounds
    booster = lgb.train(params, _dataset(df, features), num_boost_round=rounds)
    return booster, features, info


def score_frame(booster, features, df: pd.DataFrame, num_iteration: int | None = None) -> pd.DataFrame:
    return df[["customer_idx", "article_id"]].assign(
        score=booster.predict(df[features], num_threads=8, num_iteration=num_iteration))


def top_k(scored: pd.DataFrame, k: int = 12) -> dict[int, list[int]]:
    """Top k per customer; ties in score break on the smaller article_id (deterministic)."""
    s = scored.sort_values(["customer_idx", "score", "article_id"], ascending=[True, False, True], kind="stable")
    top = s.groupby("customer_idx", sort=False).head(k)
    return top.groupby("customer_idx", sort=False)["article_id"].agg(list).to_dict()


def rank(booster, features, df: pd.DataFrame, k: int = 12, num_iteration: int | None = None) -> dict[int, list[int]]:
    return top_k(score_frame(booster, features, df, num_iteration), k)


def importance(booster, features) -> list[tuple[str, float]]:
    gain = booster.feature_importance("gain")
    total = gain.sum()
    return sorted(((f, float(g / total)) for f, g in zip(features, gain)), key=lambda x: -x[1])


def save_ap(cfg, name: str, ap: pd.DataFrame) -> None:
    d = cfg.path("interim") / "ap"
    d.mkdir(parents=True, exist_ok=True)
    ap.to_parquet(d / f"{name}.parquet")


def manifest(cfg, con, features: list[str], extra: dict | None = None) -> dict:
    """Identifiers that make a run traceable: code, config, data and feature schema."""
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - git may be unavailable; the manifest records that
            return None
    data = con.execute("SELECT count(*), min(t_dat)::VARCHAR, max(t_dat)::VARCHAR, sum(article_id::HUGEINT) FROM transactions").fetchone()
    body = json.dumps({k: v for k, v in cfg.items() if k in ("retrieval", "features", "ranker")}, sort_keys=True, default=str)
    from ensemble.features.vocab import load_vocab
    return {
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "config_name": os.environ.get("ENSEMBLE_CONFIG", "default"),
        "config_sha1": hashlib.sha1(body.encode()).hexdigest()[:12],
        "data_fingerprint": {"rows": data[0], "min_date": data[1], "max_date": data[2], "article_id_sum": str(data[3])},
        "vocab_version": load_vocab(con, cfg)["version"],
        "feature_schema": features, "feature_schema_sha1": hashlib.sha1(",".join(features).encode()).hexdigest()[:12],
        "lightgbm": lgb.__version__, **(extra or {}),
    }


def run_eval(mode: str, tag: str | None = None) -> dict:
    cfg = load_config()
    tag = tag if tag is not None else ("" if os.environ.get("ENSEMBLE_CONFIG", "default") == "default"
                                       else "_" + os.environ["ENSEMBLE_CONFIG"])
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    target = {"val": splits.val, "test": splits.test}[mode]
    weeks = splits.train_weeks(int(cfg.ranker.train_weeks), before=target)
    t0 = time.time()
    df_train = build_training_frame(con, cfg, weeks)
    build_secs = time.time() - t0
    n_train_rows = len(df_train)
    t1 = time.time()
    booster, features, fit_info = fit(cfg, df_train)
    fit_secs = time.time() - t1
    del df_train
    gc.collect()
    df = week_frame(con, cfg, target, buyers_sql(target), False, False)
    truth = load_truth(con, target)
    segments = load_segments(con, target)
    pop_age = popular_by_age(con, target)
    pop = popular(con, target)
    ages = customer_age_bins(con, list(truth))
    scored = score_frame(booster, features, df)
    preds = top_k(scored)
    preds = {u: fill(preds.get(u, []), pop_age.get(ages.get(u, -1), pop)) for u in truth}
    m = dict(evaluate(preds, truth, segments))
    save_ap(cfg, f"{mode}{tag}", per_customer_ap(preds, truth))
    # Diagnostic: MAP at smaller ensembles of the same booster (only iterations that exist).
    n_trees = booster.current_iteration()
    by_rounds = {}
    for n in sorted({x for x in (100, 200, 400, 800) if x < n_trees} | {n_trees}):
        p = top_k(score_frame(booster, features, df, num_iteration=n)) if n != n_trees else None
        by_rounds[str(n)] = (evaluate({u: fill(p.get(u, []), pop_age.get(ages.get(u, -1), pop)) for u in truth},
                                      truth, segments)["map@12"] if p is not None else m["map@12"])
    m["map@12_by_rounds"] = by_rounds
    m.update({"n_trees": n_trees, "train_rows": n_train_rows, "train_build_seconds": round(build_secs, 1),
              "fit_seconds": round(fit_secs, 1), "eval_retrieval_seconds": round(df.attrs["retrieval_seconds"], 1),
              "eval_feature_seconds": round(df.attrs["feature_seconds"], 1),
              "eval_candidates_per_customer": len(df) / df.customer_idx.nunique(),
              "train_seconds": round(build_secs + fit_secs, 1), **{f"fit.{k}": v for k, v in fit_info.items()}})
    del df, scored
    gc.collect()
    imp = importance(booster, features)
    with tracking.run("track_a", f"m3/lgbm_lambdarank/{mode}{tag or ''}",
                      {**params_of(cfg), "train_weeks": cfg.ranker.train_weeks, "channels": cfg.retrieval.get("channels")}):
        tracking.log_metrics({k: v for k, v in m.items() if not isinstance(v, (dict, list))})
    print(f"{mode}: MAP@12={m['map@12']:.5f} returning={m['map@12_returning']:.5f} new={m['map@12_new_customers']:.5f} "
          f"recall@12={m['recall@12']:.4f} trees={n_trees} ({m['train_seconds']}s train)", flush=True)
    print("top features:", [(f, round(g, 3)) for f, g in imp[:15]])
    models = cfg.path("processed") / "models"
    models.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(models / f"ranker_{mode}{tag}.txt"))
    man = manifest(cfg, con, features, {"mode": mode, "train_weeks": [str(w.start) for w in weeks], "target_week": str(target.start)})
    out_dir = cfg.path("reports") if not tag else cfg.path("reports") / "phase2"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"m3_ranker_{mode}{tag}.json").write_text(
        json.dumps({"metrics": m, "importance": imp, "manifest": man}, indent=2, default=str))
    return m


if __name__ == "__main__":
    import sys
    run_eval(sys.argv[1] if len(sys.argv) > 1 else "val", sys.argv[2] if len(sys.argv) > 2 else None)
