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
from dataclasses import dataclass, field

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


@dataclass
class TrainingData:
    """Training weeks as one float32 matrix per week (oldest first), passed to LightGBM as a list:
    no concatenated copy, and the early-stopping split is a split of the list."""
    features: list[str]
    X: list[np.ndarray] = field(default_factory=list)
    y: list[np.ndarray] = field(default_factory=list)
    groups: list[np.ndarray] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return sum(len(x) for x in self.X)


def _customer_groups(customer_idx: np.ndarray) -> np.ndarray:
    """Query group sizes; rows arrive sorted by customer."""
    change = np.r_[True, customer_idx[1:] != customer_idx[:-1]]
    return np.diff(np.r_[np.flatnonzero(change), len(customer_idx)])


def build_training_data(con, cfg, weeks: list[Week]) -> TrainingData:
    neg_rate = float(cfg.ranker.get("neg_sample_rate", 1.0))
    data = None
    for w in weeks:
        df = week_frame(con, cfg, w, buyers_sql(w), True, True, neg_rate)
        features = [c for c in df.columns if c not in ID_COLS]
        if data is None:
            data = TrainingData(features)
        assert features == data.features, "feature schema differs between training weeks"
        data.X.append(df[features].to_numpy(dtype=np.float32))
        data.y.append(df["label"].to_numpy(dtype=np.int8))
        data.groups.append(_customer_groups(df.customer_idx.to_numpy()))
        print(f"  train week {w.start}: {len(df):,} rows, pos rate {data.y[-1].mean():.4f} "
              f"(retrieval {df.attrs['retrieval_seconds']:.0f}s, features {df.attrs['feature_seconds']:.0f}s)", flush=True)
        del df
        gc.collect()
    return data


def train(con, cfg, weeks: list[Week]) -> tuple[lgb.Booster, list[str]]:
    booster, features, _ = fit(cfg, build_training_data(con, cfg, weeks))
    return booster, features


def params_of(cfg) -> dict:
    p = dict(cfg.ranker.params)
    seed = int(cfg.ranker.get("seed", 42))
    p.setdefault("seed", seed)
    p.setdefault("deterministic", True)
    p.setdefault("force_col_wise", True)
    return p


def _dataset(data: TrainingData, idx: list[int], reference=None) -> lgb.Dataset:
    X = [data.X[i] for i in idx]
    return lgb.Dataset(X if len(X) > 1 else X[0], label=np.concatenate([data.y[i] for i in idx]),
                       group=np.concatenate([data.groups[i] for i in idx]), feature_name=data.features,
                       categorical_feature=[c for c in CATEGORICAL if c in data.features],
                       reference=reference, free_raw_data=True)


def fit(cfg, data: TrainingData) -> tuple[lgb.Booster, list[str], dict]:
    """Return (booster, features, info). ``info`` records how the number of trees was chosen."""
    params = params_of(cfg)
    es = cfg.ranker.get("early_stopping") or {}
    n = len(data.X)
    info: dict = {}
    if es and n > 1:
        dtr = _dataset(data, list(range(n - 1)))
        dva = _dataset(data, [n - 1], reference=dtr)
        evals: dict = {}
        b = lgb.train(params, dtr, num_boost_round=int(es["max_rounds"]), valid_sets=[dva], valid_names=["last_week"],
                      callbacks=[lgb.early_stopping(int(es["patience"]), verbose=False), lgb.record_evaluation(evals)])
        best = int(b.best_iteration)
        metric = next(iter(evals["last_week"]))
        info = {"best_iteration": best, "early_stopping_metric": metric,
                "early_stopping_curve": [round(v, 6) for v in evals["last_week"][metric][::25]],
                "best_score": float(evals["last_week"][metric][best - 1])}
        del dtr, dva
        gc.collect()
        if not es.get("refit", True):
            return b, data.features, info
        rounds = max(1, int(round(best * float(es.get("refit_multiplier", 1.0)))))
    else:
        rounds = int(cfg.ranker.num_boost_round)
    info["num_boost_round"] = rounds
    booster = lgb.train(params, _dataset(data, list(range(n))), num_boost_round=rounds)
    return booster, data.features, info


def score_frame(booster, features, df: pd.DataFrame, num_iteration: int | None = None,
                chunk_rows: int = 2_000_000) -> pd.DataFrame:
    """Scores for ``df`` (predicted in row chunks to bound LightGBM's input copy)."""
    scores = np.concatenate([
        booster.predict(df[features].iloc[i:i + chunk_rows], num_threads=8, num_iteration=num_iteration)
        for i in range(0, len(df), chunk_rows)]) if len(df) else np.array([])
    return df[["customer_idx", "article_id"]].assign(score=scores)


def score_users(con, cfg, booster, features, week: Week, users_sql: str, iterations=(), extra_cols=(),
                chunk: int = 25_000) -> pd.DataFrame:
    """Build candidates and features for customers in chunks and score them.

    Returns customer_idx, article_id, score (all trees), ``score@<n>`` for each n in
    ``iterations``, and any ``extra_cols`` (raw features kept for re-ranking or explanations).
    ``attrs`` carries summed retrieval / feature seconds and the candidate count.
    """
    users = con.execute(f"SELECT DISTINCT customer_idx FROM ({users_sql}) ORDER BY 1").fetchnumpy()["customer_idx"]
    parts, secs = [], {"retrieval_seconds": 0.0, "feature_seconds": 0.0}
    for i in range(0, len(users), chunk):
        con.execute("CREATE OR REPLACE TEMP TABLE _chunk AS SELECT unnest(?::INTEGER[]) AS customer_idx",
                    [users[i:i + chunk].tolist()])
        df = week_frame(con, cfg, week, "SELECT customer_idx FROM _chunk", False, False)
        for k in secs:
            secs[k] += df.attrs[k]
        out = score_frame(booster, features, df)
        for n in iterations:
            out[f"score@{n}"] = score_frame(booster, features, df, num_iteration=n)["score"].to_numpy()
        for c in extra_cols:
            if c in df.columns:
                out[c] = df[c].to_numpy()
        parts.append(out)
        del df
        gc.collect()
    res = pd.concat(parts, ignore_index=True)
    res.attrs.update(secs)
    res.attrs["candidates_per_customer"] = len(res) / max(1, res.customer_idx.nunique())
    return res


def top_k(scored: pd.DataFrame, k: int = 12, score_col: str = "score") -> dict[int, list[int]]:
    """Top k per customer; ties in score break on the smaller article_id (deterministic)."""
    s = scored.sort_values(["customer_idx", score_col, "article_id"], ascending=[True, False, True], kind="stable")
    top = s.groupby("customer_idx", sort=False).head(k)
    return top.groupby("customer_idx", sort=False)["article_id"].agg(list).to_dict()


def rank(booster, features, df: pd.DataFrame, k: int = 12, num_iteration: int | None = None) -> dict[int, list[int]]:
    return top_k(score_frame(booster, features, df, num_iteration), k)


def importance(booster, features) -> list[tuple[str, float]]:
    gain = booster.feature_importance("gain")
    total = gain.sum()
    return sorted(((f, float(g / total)) for f, g in zip(features, gain)), key=lambda x: -x[1])


# Raw columns kept with the scores for re-ranking / availability experiments (ensemble.ranking.rerank).
RERANK_COLS = ("product_type_no", "colour_group_code", "department_no", "index_group_no", "garment_group_no",
               "a_days_since_last_sale", "a_days_since_first_sale", "a_n_1w", "a_n_4w", "ca_n_article_life")


def save_scored(cfg, name: str, scored: pd.DataFrame) -> None:
    d = cfg.path("interim") / "scored"
    d.mkdir(parents=True, exist_ok=True)
    scored.drop(columns=[c for c in scored.columns if c.startswith("score@")]).to_parquet(d / f"{name}.parquet")


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
                                       else "_" + os.path.basename(os.environ["ENSEMBLE_CONFIG"]))
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    target = {"val": splits.val, "test": splits.test}[mode]
    weeks = splits.train_weeks(int(cfg.ranker.train_weeks), before=target)
    t0 = time.time()
    data = build_training_data(con, cfg, weeks)
    build_secs = time.time() - t0
    n_train_rows = data.rows
    t1 = time.time()
    booster, features, fit_info = fit(cfg, data)
    fit_secs = time.time() - t1
    del data
    gc.collect()
    n_trees = booster.current_iteration()
    diag = sorted(x for x in (100, 200, 400, 800) if x < n_trees)
    t2 = time.time()
    scored = score_users(con, cfg, booster, features, target, buyers_sql(target), iterations=diag, extra_cols=RERANK_COLS)
    score_secs = time.time() - t2
    save_scored(cfg, f"{mode}{tag}", scored)
    truth = load_truth(con, target)
    segments = load_segments(con, target)
    pop_age = popular_by_age(con, target)
    pop = popular(con, target)
    ages = customer_age_bins(con, list(truth))

    def finish(p):
        return {u: fill(p.get(u, []), pop_age.get(ages.get(u, -1), pop)) for u in truth}

    preds = finish(top_k(scored))
    m = dict(evaluate(preds, truth, segments))
    save_ap(cfg, f"{mode}{tag}", per_customer_ap(preds, truth))
    # Diagnostic: MAP at smaller ensembles of the same booster (only iterations that exist).
    by_rounds = {str(n): evaluate(finish(top_k(scored, score_col=f"score@{n}")), truth, segments)["map@12"] for n in diag}
    by_rounds[str(n_trees)] = m["map@12"]
    m["map@12_by_rounds"] = by_rounds
    m.update({"n_trees": n_trees, "train_rows": n_train_rows, "train_build_seconds": round(build_secs, 1),
              "fit_seconds": round(fit_secs, 1), "eval_retrieval_seconds": round(scored.attrs["retrieval_seconds"], 1),
              "eval_feature_seconds": round(scored.attrs["feature_seconds"], 1), "eval_score_seconds": round(score_secs, 1),
              "eval_candidates_per_customer": scored.attrs["candidates_per_customer"],
              "train_seconds": round(build_secs + fit_secs, 1), **{f"fit.{k}": v for k, v in fit_info.items()}})
    del scored
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
