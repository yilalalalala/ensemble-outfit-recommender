"""The frozen Track A research protocol (configs/track_a_research.yaml, ``research.protocol``).

Every function that reads interactions takes the label ``week`` as a required argument and
reads only ``t_dat < week.start`` unless it is explicitly building the label (``truth``).

Definitions shared by every system in the comparison:

- **Evaluation customers**: everyone with >= 1 purchase in the label week (D-014). No sampling.
- **Eligible catalogue** ``E(w)``: articles with >= 1 sale in the ``eligible_days`` before
  ``w.start`` (the D-029 availability proxy). Every list is restricted to it before the top-k
  cut, so no system can win by recommending stale stock.
- **Back-fill / new-user fallback**: lists shorter than k are completed with the M1 age-band
  best sellers of the 7 days before the cutoff, which are always in ``E(w)``.
- **Ties** break on ``article_id`` (smaller first) after the score.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import date, timedelta
from importlib import metadata

import numpy as np
import pandas as pd

from ensemble.config import ROOT, Config
from ensemble.data.splits import Week


def proto(cfg: Config) -> Config:
    return cfg.research.protocol


def week_of(start: str | date) -> Week:
    s = date.fromisoformat(start) if isinstance(start, str) else start
    return Week(s, s + timedelta(days=6))


def reporting_folds(cfg) -> list[Week]:
    return [week_of(s) for s in proto(cfg).reporting_folds]


def tuning_folds(cfg) -> list[Week]:
    return [week_of(s) for s in proto(cfg).tuning_folds]


def ranker_train_weeks(cfg, target: Week) -> list[Week]:
    """The ``ranker.train_weeks`` label weeks immediately before ``target`` (oldest first)."""
    return [target.shift(-k) for k in range(int(cfg.ranker.train_weeks), 0, -1)]


def all_model_weeks(cfg) -> list[Week]:
    """Every label week a per-week model is needed for: each reporting fold, each ranker training
    week of each fold, and the tuning folds. Sorted, unique."""
    weeks = {w for f in reporting_folds(cfg) for w in (f, *ranker_train_weeks(cfg, f))}
    weeks |= set(tuning_folds(cfg))
    return sorted(weeks, key=lambda w: w.start)


def check_protocol(cfg) -> None:
    """Structural guarantees of the frozen protocol; raises on violation."""
    p = proto(cfg)
    folds, tuning = reporting_folds(cfg), tuning_folds(cfg)
    conf = week_of(p.confirmation_week)
    starts = [f.start for f in folds]
    assert starts == sorted(starts) and all(b - a == timedelta(days=7) for a, b in zip(starts, starts[1:])), \
        "reporting folds must be consecutive weeks"
    assert folds[-1].end < conf.start, "a reporting fold reaches the confirmation week"
    assert all(t.end < folds[0].start for t in tuning), "tuning folds must precede every reporting fold"
    assert all(week_of(a).end < folds[0].start for a in p.allocator_weeks), "allocator weeks must precede reporting folds"
    for f in folds:
        assert all(w.end < f.start for w in ranker_train_weeks(cfg, f)), "a ranker training week overlaps its fold"


# ---------------------------------------------------------------------------
# Point-in-time catalogue and segments
# ---------------------------------------------------------------------------

def eligible(con, week: Week, days: int) -> np.ndarray:
    """Sorted article ids sold at least once in the ``days`` before ``week.start``."""
    return np.sort(con.execute(
        "SELECT DISTINCT article_id FROM transactions WHERE t_dat >= ? AND t_dat < ?",
        [week.start - timedelta(days=int(days)), week.start]).fetchnumpy()["article_id"].astype(np.int64))


def truth(con, week: Week) -> dict[int, set[int]]:
    """The label: distinct (customer, article) purchases inside the label week."""
    df = con.execute("SELECT DISTINCT customer_idx, article_id FROM transactions WHERE t_dat BETWEEN ? AND ?",
                     [week.start, week.end]).df()
    return {int(c): set(g.article_id.astype(int)) for c, g in df.groupby("customer_idx")}


def customer_segments(con, week: Week, customers, p) -> pd.DataFrame:
    """Per evaluation customer: returning flag and activity band from pre-cutoff history only."""
    con.execute("CREATE OR REPLACE TEMP TABLE _seg_u AS SELECT unnest(?::INTEGER[]) AS customer_idx", [list(customers)])
    df = con.execute(f"""
        SELECT u.customer_idx,
               count(t.customer_idx) > 0 AS returning,
               count(t.customer_idx) FILTER (WHERE t.t_dat >= DATE '{week.start}' - INTERVAL {7 * int(p.activity_weeks)} DAY) AS n_recent
        FROM _seg_u u LEFT JOIN transactions t ON t.customer_idx = u.customer_idx AND t.t_dat < DATE '{week.start}'
        GROUP BY 1 ORDER BY 1""").df()
    band = np.full(len(df), "new", dtype=object)
    for name, (lo, hi) in dict(p.activity_bands).items():
        band[(df.returning.to_numpy()) & (df.n_recent.to_numpy() >= lo) & (df.n_recent.to_numpy() <= hi)] = name
    df["activity"] = band
    return df


def item_segments(con, week: Week, elig: np.ndarray, p) -> dict[str, set[int]]:
    """Article sets for item-side slicing, all from pre-cutoff data: head (top ``head_share`` of
    the eligible catalogue by ``head_days`` sales), recently launched (first observed sale within
    ``recent_item_days`` before the cutoff) and cold (first observed sale inside the label week)."""
    sales = con.execute("""SELECT article_id, count(*) AS n FROM transactions WHERE t_dat >= ? AND t_dat < ?
                           GROUP BY 1""", [week.start - timedelta(days=int(p.head_days)), week.start]).df()
    sales = sales[sales.article_id.isin(elig)].sort_values(["n", "article_id"], ascending=[False, True])
    head = set(sales.article_id.head(int(round(float(p.head_share) * len(elig)))).astype(int))
    first = con.execute("SELECT article_id, min(t_dat) AS first FROM transactions GROUP BY 1").df()
    first["first"] = pd.to_datetime(first["first"]).dt.date
    recent = set(first.article_id[(first["first"] >= week.start - timedelta(days=int(p.recent_item_days)))
                                  & (first["first"] < week.start)].astype(int))
    cold = set(first.article_id[first["first"] >= week.start].astype(int))
    return {"head": head, "recent": recent, "cold": cold}


def prior_purchases(con, week: Week, customers) -> dict[int, set[int]]:
    """Articles each customer bought before the cutoff (for repeat vs non-repeat truth)."""
    con.execute("CREATE OR REPLACE TEMP TABLE _pp_u AS SELECT unnest(?::INTEGER[]) AS customer_idx", [list(customers)])
    df = con.execute("""SELECT DISTINCT t.customer_idx, t.article_id FROM transactions t JOIN _pp_u USING (customer_idx)
                        WHERE t.t_dat < ?""", [week.start]).df()
    return {int(c): set(g.article_id.astype(int)) for c, g in df.groupby("customer_idx")}


# ---------------------------------------------------------------------------
# Restriction, back-fill, deterministic top-k
# ---------------------------------------------------------------------------

def restrict_and_fill(lists: dict[int, list[int]], users, elig: set[int], fallback_of, k: int) -> dict[int, list[int]]:
    """Restrict every list to the eligible catalogue, keep order, de-duplicate, back-fill to k."""
    out = {}
    for u in users:
        seen, lst = set(), []
        for a in lists.get(u, ()):
            if a in elig and a not in seen:
                seen.add(a)
                lst.append(a)
                if len(lst) == k:
                    break
        if len(lst) < k:
            for a in fallback_of(u):
                if a not in seen:
                    seen.add(a)
                    lst.append(a)
                    if len(lst) == k:
                        break
        out[u] = lst
    return out


def top_k_from_scores(user_ids: np.ndarray, item_ids: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k per row of a dense score block; ties break on the smaller article id.

    ``item_ids`` must be sorted ascending, so a stable sort on -score keeps the smaller id first.
    Returns (items[n, k], scores[n, k])."""
    n, m = scores.shape
    k = min(k, m)
    if k < m:
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    else:
        part = np.tile(np.arange(m), (n, 1))
    ps = np.take_along_axis(scores, part, axis=1)
    if k < m:
        # argpartition may cut inside a tie at the boundary: for those (rare) rows keep every
        # column strictly above the k-th score plus the smallest-index columns equal to it.
        kth = ps.min(axis=1, keepdims=True)
        straddle = np.flatnonzero((scores >= kth).sum(axis=1) > k)
        for r in straddle:
            full = np.lexsort((np.arange(m), -scores[r]))[:k]
            part[r], ps[r] = full, scores[r, full]
    order = np.lexsort((part, -ps), axis=1)
    idx = np.take_along_axis(part, order, axis=1)
    return item_ids[idx], np.take_along_axis(scores, idx, axis=1)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def data_fingerprint(con) -> dict:
    r = con.execute("""SELECT count(*), min(t_dat)::VARCHAR, max(t_dat)::VARCHAR, sum(article_id::HUGEINT),
                              sum(customer_idx::HUGEINT), sum(hash(t_dat, customer_idx, article_id, price) % 1000000007)
                       FROM transactions""").fetchone()
    return {"transactions": r[0], "min_date": r[1], "max_date": r[2], "article_id_sum": str(r[3]),
            "customer_idx_sum": str(r[4]), "row_hash_sum": str(r[5]),
            "articles": con.execute("SELECT count(*) FROM articles").fetchone()[0],
            "customers": con.execute("SELECT count(*) FROM customers").fetchone()[0]}


def config_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def manifest(cfg, con, extra: dict | None = None) -> dict:
    pkgs = {}
    for name in ("torch", "lightgbm", "duckdb", "numpy", "pandas", "pyarrow", "scipy"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    p = dict(proto(cfg))
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "config_name": os.environ.get("ENSEMBLE_CONFIG", "default"),
        "protocol": p, "protocol_sha1": config_hash(p),
        "models_sha1": config_hash(dict(cfg.research.models)),
        "data_fingerprint": data_fingerprint(con),
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "packages": pkgs, **(extra or {}),
    }
