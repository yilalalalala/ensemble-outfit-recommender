"""Track B evaluation protocol (Round 3, Phase 1).

Three things make the Round-1 protocol indefensible, and this module fixes them.

**Catalogue leakage.** ``ensemble.completion.data.universe`` builds the eligible
catalogue from sales in ``[start - universe_weeks, week.end]`` — it reads the
target week, so eligibility is not knowable at prediction time and "item cold
start" becomes an artefact of the definition. Measured, that leak does *not*
flatter the numbers: it adds ~5.9% more truth articles with no pre-cutoff
footprint, which nothing can retrieve, and so *lowers* Recall@12 by ~5.1% for
every fixed-fusion system (``protocol_check``, D-031). Here the eligible
catalogue is built strictly before the cutoff (:func:`eligible_universe`); the
old definition stays reachable as ``oracle=True`` for that comparison only.

**One week of evidence.** Model selection used the single validation week.
:func:`folds` returns consecutive label weeks ending at validation, so every
decision is made on several weeks and the test week stays untouched.

**Correlated queries.** One basket produces several (anchor, target slot)
queries, so treating queries as independent understates the uncertainty.
:func:`customer_scores` collapses per-query scores to one score per customer for
the cluster bootstrap in ``ensemble.evaluation.metrics.paired_bootstrap``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import timedelta

import numpy as np
import pandas as pd

from ensemble.completion.association import basket_sql
from ensemble.config import ROOT
from ensemble.data.splits import Splits, Week

SLOTS = ["upper", "lower", "full", "shoes", "accessories", "socks", "swimwear"]


def _d(x) -> str:
    return f"DATE '{x}'"


def folds(splits: Splits, n: int, train_weeks: int = 0) -> list[Week]:
    """``n`` consecutive label weeks ending at validation (test week untouched).

    With ``train_weeks`` > 0 the list is extended backwards by that many weeks,
    which is the set of weeks whose point-in-time artifacts have to be built so
    each fold can be trained on earlier weeks only.
    """
    last = splits.val
    return [last.shift(-k) for k in range(n - 1 + train_weeks, -1, -1)]


def eligible_universe(con, week: Week, cfg, oracle: bool = False) -> pd.DataFrame:
    """Articles a merchandiser could have offered in ``week``, with pre-cutoff statistics.

    Eligibility: sold at least once in the ``universe_weeks`` before the cutoff.
    That is a *proxy* for availability — the dataset has no inventory or launch
    feed — but it uses no target-week information. ``oracle=True`` reproduces the
    leaky Round-1 universe (adds articles whose only sales are inside the target
    week) and exists so the leak can be quantified instead of hidden.
    """
    c = cfg.track_b
    cutoff = week.end if oracle else week.cutoff
    lo_live = week.start - timedelta(weeks=int(c.universe_weeks))
    lo_win = week.start - timedelta(weeks=int(c.window_weeks))
    return con.execute(f"""
        WITH live AS (
            SELECT DISTINCT article_id FROM transactions
            WHERE t_dat BETWEEN {_d(lo_live)} AND {_d(cutoff)}),
        stats AS (
            SELECT article_id,
                   count(*) FILTER (WHERE t_dat >= {_d(week.start - timedelta(days=int(c.pop_days)))}) AS pop_recent,
                   count(*) AS pop_window,
                   count(DISTINCT t_dat) AS days_sold_window,
                   date_diff('day', max(t_dat), {_d(week.start)}) AS days_since_last_sale,
                   avg(price) AS mean_price
            FROM transactions WHERE t_dat BETWEEN {_d(lo_win)} AND {_d(week.cutoff)} GROUP BY 1),
        first_seen AS (
            SELECT article_id, date_diff('day', min(t_dat), {_d(week.start)}) AS age_days
            FROM transactions WHERE t_dat < {_d(week.start)} GROUP BY 1)
        SELECT a.article_id, a.slot, a.is_jewellery,
               coalesce(s.pop_recent, 0) AS pop_recent,
               coalesce(s.pop_window, 0) AS pop_window,
               coalesce(s.days_sold_window, 0) AS days_sold_window,
               coalesce(s.days_since_last_sale, 999) AS days_since_last_sale,
               coalesce(f.age_days, -1) AS age_days,
               s.mean_price
        FROM live JOIN articles a USING (article_id)
        LEFT JOIN stats s USING (article_id) LEFT JOIN first_seen f USING (article_id)
        WHERE a.slot IS NOT NULL
        ORDER BY a.slot, pop_recent DESC, pop_window DESC, a.article_id""").df()


def queries(con, week: Week, cfg) -> pd.DataFrame:
    """One row per (basket, anchor, target slot); ``truth`` = that basket's articles in the slot.

    Reads the target week only — this is the label, not a feature.
    """
    c = cfg.track_b
    q = con.execute(f"""
        WITH b AS ({basket_sql(week.start, week.end, int(c.min_basket_size), int(c.max_basket_size), week.cutoff)})
        SELECT x.customer_idx, x.t_dat, x.article_id AS anchor, x.slot AS anchor_slot,
               y.slot AS target_slot, list(DISTINCT y.article_id) AS truth
        FROM b x JOIN b y ON x.customer_idx = y.customer_idx AND x.t_dat = y.t_dat AND x.slot <> y.slot
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY 1, 2, 3, 5""").df()
    return q.reset_index(drop=True).rename_axis("qid").reset_index()


def restrict_truth(q: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """Drop truth items outside the eligible catalogue and queries left with none.

    Under the leakage-free catalogue ~5.9% of truth articles are not eligible
    (never sold before the cutoff). Scoring them as guaranteed misses would bury
    an unreachable ceiling inside every number, so they are excluded from the
    primary protocol and counted in ``n_truth_dropped`` instead (D-031).
    """
    live = np.sort(uni.article_id.to_numpy(dtype=np.int64))
    lens = np.fromiter((len(t) for t in q.truth.values), dtype=np.int64, count=len(q))
    flat = np.concatenate([np.asarray(t, dtype=np.int64) for t in q.truth.values]) if len(q) else np.empty(0, np.int64)
    ok = np.isin(flat, live)
    kept, dropped, pos = [], int((~ok).sum()), 0
    for n in lens:
        kept.append(flat[pos:pos + n][ok[pos:pos + n]])
        pos += n
    out = q.assign(truth=kept)
    out = out[np.fromiter((len(t) > 0 for t in kept), dtype=bool, count=len(kept))].reset_index(drop=True)
    out.attrs["n_truth_dropped"] = dropped
    return out


def customer_scores(q: pd.DataFrame, per_query: np.ndarray) -> pd.DataFrame:
    """Mean per-query score per customer: the cluster bootstrap's resampling unit."""
    return (pd.DataFrame({"customer_idx": q.customer_idx.values, "score": per_query})
            .groupby("customer_idx", as_index=False)["score"].mean())


def sample_queries(q: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic customer-level subsample used for ranker training weeks.

    Customers (not queries) are sampled, so a basket is never split across the
    sample boundary.
    """
    if n <= 0 or len(q) <= n:
        return q
    cust = np.sort(q.customer_idx.unique())
    h = pd.util.hash_array(cust.astype(np.int64), hash_key=f"trackb{seed}".ljust(16)[:16])
    share = n / len(q)
    keep = set(cust[h % 10_000 < int(share * 10_000)].tolist())
    return q[q.customer_idx.isin(keep)].reset_index(drop=True)


def data_fingerprint(con) -> dict:
    row = con.execute("SELECT count(*), min(t_dat)::VARCHAR, max(t_dat)::VARCHAR, "
                      "sum(article_id::HUGEINT) FROM transactions").fetchone()
    return {"rows": row[0], "min_date": row[1], "max_date": row[2], "article_id_sum": str(row[3])}


def manifest(cfg, con, extra: dict | None = None) -> dict:
    """Code / config / data identifiers so every reported number is traceable."""
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - git may be unavailable; recorded as None
            return None
    body = json.dumps(cfg.get("track_b", {}), sort_keys=True, default=str)
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "config_name": os.environ.get("ENSEMBLE_CONFIG", "default"),
            "track_b_config_sha1": hashlib.sha1(body.encode()).hexdigest()[:12],
            "data_fingerprint": data_fingerprint(con), **(extra or {})}
