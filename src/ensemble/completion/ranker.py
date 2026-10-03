"""Learned Track B fusion: LightGBM LambdaRank over the candidate union (Phase 2).

One ranking group per query — a (basket, anchor, target_slot) triple — with the
basket's held-out articles in that slot as positives. This replaces the fixed
reciprocal-rank-fusion weight of the Round-1 hybrid, which could not use support,
lift, recency, backoff level or anything about the customer.

Number of trees is chosen by temporal early stopping: fit on the older training
weeks, stop on the most recent training week (still strictly before the fold's
label week), then refit on all training weeks with that count. The fold's own
week never influences the tree count.
"""
from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

from ensemble.completion.features import CATEGORICAL


def groups_of(qid: np.ndarray) -> np.ndarray:
    """Query group sizes; rows arrive sorted by qid."""
    if not len(qid):
        return np.array([], dtype=np.int64)
    change = np.r_[True, qid[1:] != qid[:-1]]
    return np.diff(np.r_[np.flatnonzero(change), len(qid)])


def params_of(cfg) -> dict:
    p = dict(cfg.track_b.ranker.params)
    p.setdefault("seed", int(cfg.track_b.ranker.get("seed", 42)))
    p.setdefault("deterministic", True)
    p.setdefault("force_col_wise", True)
    return p


def _dataset(mats: list[tuple[np.ndarray, np.ndarray, np.ndarray]], features: list[str], reference=None):
    X = [m[0] for m in mats]
    return lgb.Dataset(X if len(X) > 1 else X[0],
                       label=np.concatenate([m[1] for m in mats]),
                       group=np.concatenate([m[2] for m in mats]),
                       feature_name=features,
                       categorical_feature=[c for c in CATEGORICAL if c in features],
                       reference=reference, free_raw_data=True)


def to_matrix(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (df[features].to_numpy(dtype=np.float32), df["label"].to_numpy(dtype=np.int8),
            groups_of(df["qid"].to_numpy()))


def fit(cfg, weeks: list[pd.DataFrame], features: list[str], log=print) -> tuple[lgb.Booster, dict]:
    """Fit on ``weeks`` (oldest first). Returns the booster and how the tree count was chosen."""
    params = params_of(cfg)
    es = cfg.track_b.ranker.get("early_stopping") or {}
    mats = [to_matrix(df, features) for df in weeks]
    info: dict = {"train_rows": int(sum(len(m[0]) for m in mats)),
                  "train_groups": int(sum(len(m[2]) for m in mats)),
                  "positive_rate": float(np.concatenate([m[1] for m in mats]).mean())}
    if es and len(mats) > 1:
        dtr = _dataset(mats[:-1], features)
        dva = _dataset(mats[-1:], features, reference=dtr)
        evals: dict = {}
        b = lgb.train(params, dtr, num_boost_round=int(es["max_rounds"]), valid_sets=[dva],
                      valid_names=["last_train_week"],
                      callbacks=[lgb.early_stopping(int(es["patience"]), verbose=False),
                                 lgb.record_evaluation(evals)])
        metric = next(iter(evals["last_train_week"]))
        best = int(b.best_iteration)
        info.update({"best_iteration": best, "early_stopping_metric": metric,
                     "best_score": float(evals["last_train_week"][metric][best - 1]),
                     "curve": [round(v, 6) for v in evals["last_train_week"][metric][::25]]})
        del dtr, dva
        gc.collect()
        if not es.get("refit", True):
            return b, info
        rounds = max(1, best)
    else:
        rounds = int(cfg.track_b.ranker.num_boost_round)
    info["num_boost_round"] = rounds
    booster = lgb.train(params, _dataset(mats, features), num_boost_round=rounds)
    del mats
    gc.collect()
    return booster, info


def top_k(df: pd.DataFrame, score: np.ndarray, k: int) -> dict[int, np.ndarray]:
    """Top ``k`` article ids per qid; ties break on the smaller article_id."""
    order = np.lexsort((df.article_id.to_numpy(), -score, df.qid.to_numpy()))
    qid = df.qid.to_numpy()[order]
    art = df.article_id.to_numpy()[order]
    out: dict[int, np.ndarray] = {}
    start = 0
    bounds = np.r_[np.flatnonzero(np.r_[True, qid[1:] != qid[:-1]]), len(qid)]
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        out[int(qid[start])] = art[start:min(end, start + k)]
    return out


def importance(booster, features) -> list[tuple[str, float]]:
    gain = booster.feature_importance("gain")
    total = gain.sum() or 1.0
    return sorted(((f, float(g / total)) for f, g in zip(features, gain)), key=lambda x: -x[1])


def group_gain(imp: list[tuple[str, float]]) -> dict[str, float]:
    """Split total split gain over the feature groups in :data:`features.GROUPS`.

    A feature can belong to more than one group (``a_co_d14`` is association *and*
    decay), so the shares do not sum to 1; the point is to see at a glance whether
    the model is mostly association, mostly popularity or mostly the customer.
    """
    from ensemble.completion.features import GROUPS

    out = {}
    for name, prefixes in GROUPS.items():
        out[name] = round(sum(g for f, g in imp if f.startswith(prefixes)), 5)
    out["other"] = round(sum(g for f, g in imp
                             if not any(f.startswith(p) for p in GROUPS.values())), 5)
    return out
