"""Per-customer evaluation under the frozen protocol, and the pooled customer-cluster bootstrap.

Every system is scored the same way (``score_lists``):

1. its raw ranked list is restricted to the eligible catalogue ``E(w)``;
2. back-filled to k with the age-band best sellers (``protocol.restrict_and_fill``);
3. compared with the label-week purchases of every buyer.

One row per (customer, fold) is stored with AP@12, NDCG@12, hits and truth counts overall
and per item segment, so any table or interval can be recomputed without re-scoring.

**Bootstrap.** Customers are the resampling unit *across* folds: a customer who buys in
three folds is one cluster carrying all three rows. The relative difference is computed
inside each resample (ratio of resampled means), so its interval is exact rather than a
difference interval divided by a fixed denominator.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from ensemble.baselines import customer_age_bins, popular, popular_by_age
from ensemble.data.splits import Week
from ensemble.research import protocol as P

ITEM_SEGMENTS = ("head", "tail", "recent", "cold", "ineligible", "repeat", "nonrepeat")


class FoldContext:
    """Everything about one label week that does not depend on the system being scored."""

    def __init__(self, con, cfg, week: Week):
        p = P.proto(cfg)
        self.week, self.k = week, int(p.k)
        self.truth = P.truth(con, week)
        self.users = sorted(self.truth)
        elig = P.eligible(con, week, int(p.eligible_days))
        self.eligible_arr = elig
        self.eligible = set(elig.tolist())
        pop, pop_age = popular(con, week), popular_by_age(con, week)
        ages = customer_age_bins(con, self.users)
        self._fallback = {u: pop_age.get(ages.get(u, -1), pop) for u in self.users}
        assert all(a in self.eligible for lst in pop_age.values() for a in lst), "fallback outside E(w)"
        seg = P.customer_segments(con, week, self.users, p)
        self.returning = dict(zip(seg.customer_idx.astype(int), seg.returning.astype(bool)))
        self.activity = dict(zip(seg.customer_idx.astype(int), seg.activity))
        self.items = P.item_segments(con, week, elig, p)
        self.prior = P.prior_purchases(con, week, self.users)
        sales = con.execute("""SELECT article_id, count(*) AS n FROM transactions WHERE t_dat >= ? AND t_dat < ?
                               GROUP BY 1""", [week.start - pd.Timedelta(days=int(p.eligible_days)), week.start]).df()
        n = dict(zip(sales.article_id.astype(int), sales.n.astype(float)))
        tot = sum(n.values()) + len(self.eligible)
        self.self_info = {a: -math.log2((n.get(a, 0.0) + 1.0) / tot) for a in self.eligible}

    def fallback(self, u: int) -> list[int]:
        return self._fallback[u]

    def finalize(self, lists: dict[int, list[int]]) -> dict[int, list[int]]:
        return P.restrict_and_fill(lists, self.users, self.eligible, self.fallback, self.k)

    def item_segment_of(self, u: int, a: int) -> list[str]:
        segs = []
        if a in self.items["cold"]:
            segs.append("cold")
        if a not in self.eligible:
            segs.append("ineligible")
        else:
            segs.append("head" if a in self.items["head"] else "tail")
        if a in self.items["recent"]:
            segs.append("recent")
        segs.append("repeat" if a in self.prior.get(u, ()) else "nonrepeat")
        return segs


def ap_at_k(pred: list[int], truth: set[int], k: int) -> float:
    hits, s = 0, 0.0
    for n, a in enumerate(pred[:k]):
        if a in truth:
            hits += 1
            s += hits / (n + 1)
    return s / min(len(truth), k)


def ndcg_at_k(pred: list[int], truth: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(n + 2) for n, a in enumerate(pred[:k]) if a in truth)
    idcg = sum(1.0 / math.log2(n + 2) for n in range(min(len(truth), k)))
    return dcg / idcg


def score_lists(ctx: FoldContext, raw: dict[int, list[int]], cand_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-customer metrics of a system's raw lists on one fold (restricted + back-filled first).

    ``cand_stats`` (optional, two-stage systems): customer_idx, cand_hits (distinct purchased
    articles in the candidate set), n_cand, to record the candidate-recall ceiling."""
    final = ctx.finalize(raw)
    k = ctx.k
    rows = []
    for u in ctx.users:
        t, pred = ctx.truth[u], final[u]
        top = set(pred[:k])
        r = {"customer_idx": u, "ap": ap_at_k(pred, t, k), "ndcg": ndcg_at_k(pred, t, k),
             "n_truth": len(t), "hits": len(t & top), "own_list": int(len(raw.get(u, ())) > 0),
             "returning": ctx.returning[u], "activity": ctx.activity[u]}
        for s in ITEM_SEGMENTS:
            r[f"t_{s}"] = 0
            r[f"h_{s}"] = 0
        for a in t:
            hit = a in top
            for s in ctx.item_segment_of(u, a):
                r[f"t_{s}"] += 1
                r[f"h_{s}"] += int(hit)
        rows.append(r)
    df = pd.DataFrame(rows)
    if cand_stats is not None:
        df = df.merge(cand_stats[["customer_idx", "cand_hits", "n_cand"]], on="customer_idx", how="left")
        df[["cand_hits", "n_cand"]] = df[["cand_hits", "n_cand"]].fillna(0).astype(int)
    df.attrs["coverage"] = len({a for u in ctx.users for a in final[u][:k]}) / len(ctx.eligible)
    df.attrs["novelty"] = float(np.mean([ctx.self_info[a] for u in ctx.users for a in final[u][:k]]))
    df.attrs["fallback_only_share"] = float(1 - df.own_list.mean())
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Headline numbers from per-customer rows (one fold, or several folds stacked)."""
    out = {"map@12": float(df.ap.mean()), "ndcg@12": float(df.ndcg.mean()),
           "recall@12": float(df.hits.sum() / df.n_truth.sum()), "n_rows": int(len(df)),
           "n_customers": int(df.customer_idx.nunique())}
    for flag, name in ((True, "returning"), (False, "new")):
        m = df.returning == flag
        out[f"map@12_{name}"] = float(df.ap[m].mean()) if m.any() else float("nan")
        out[f"share_{name}"] = float(m.mean())
    for band in ("low", "medium", "high"):
        m = df.activity == band
        out[f"map@12_activity_{band}"] = float(df.ap[m].mean()) if m.any() else float("nan")
    for s in ITEM_SEGMENTS:
        t = df[f"t_{s}"].sum()
        out[f"recall@12_{s}"] = float(df[f"h_{s}"].sum() / t) if t else float("nan")
        out[f"truth_share_{s}"] = float(t / df.n_truth.sum())
    if "cand_hits" in df:
        out["candidate_recall"] = float(df.cand_hits.sum() / df.n_truth.sum())
        out["candidates_per_customer"] = float(df.n_cand.mean())
        out["conversion"] = out["recall@12"] / out["candidate_recall"] if out["candidate_recall"] else float("nan")
    return out


def cluster_bootstrap(a: pd.DataFrame, b: pd.DataFrame, col: str = "ap", n_boot: int = 1000, seed: int = 0,
                      chunk: int = 50) -> dict:
    """Paired customer-cluster bootstrap of mean_b - mean_a and mean_b / mean_a - 1 over the
    (customer, fold) rows both systems scored. Clusters = customers across all folds."""
    m = a[["customer_idx", "fold", col]].merge(b[["customer_idx", "fold", col]], on=["customer_idx", "fold"],
                                                suffixes=("_a", "_b"), validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("systems were scored on different (customer, fold) populations")
    g = m.groupby("customer_idx").agg(sa=(f"{col}_a", "sum"), sb=(f"{col}_b", "sum"), n=(f"{col}_a", "size"))
    sa, sb, n = (g[c].to_numpy(dtype=np.float64) for c in ("sa", "sb", "n"))
    N = len(g)
    rng = np.random.default_rng(seed)
    d, rel = [], []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, N, size=(min(chunk, n_boot - start), N))
        nn_ = n[idx].sum(1)
        ma, mb = sa[idx].sum(1) / nn_, sb[idx].sum(1) / nn_
        d.append(mb - ma)
        rel.append(mb / ma - 1)
    d, rel = np.concatenate(d), np.concatenate(rel)
    mean_a, mean_b = sa.sum() / n.sum(), sb.sum() / n.sum()
    return {"metric": col, "mean_a": float(mean_a), "mean_b": float(mean_b), "diff": float(mean_b - mean_a),
            "diff_ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])],
            "relative": float(mean_b / mean_a - 1), "relative_ci95": [float(x) for x in np.percentile(rel, [2.5, 97.5])],
            "p_diff_gt_0": float((d > 0).mean()), "n_customers": int(N), "n_rows": int(n.sum()), "n_boot": int(n_boot)}


def cluster_bootstrap_ratio(a: pd.DataFrame, b: pd.DataFrame, num: str, den: str, n_boot: int = 1000,
                            seed: int = 0, chunk: int = 50) -> dict:
    """As ``cluster_bootstrap`` for a micro-averaged ratio metric sum(num) / sum(den), e.g. Recall@12
    = hits / truth. Customers (across folds) are the resampling unit; the ratio is recomputed in
    every resample, for both systems on the same resampled customers."""
    keys = ["customer_idx", "fold"]
    m = a[keys + [num, den]].merge(b[keys + [num]], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("systems were scored on different (customer, fold) populations")
    g = m.groupby("customer_idx").agg(na=(f"{num}_a", "sum"), nb=(f"{num}_b", "sum"), d=(den, "sum"))
    na, nb, dd = (g[c].to_numpy(dtype=np.float64) for c in ("na", "nb", "d"))
    rng = np.random.default_rng(seed)
    diff, rel = [], []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, len(g), size=(min(chunk, n_boot - start), len(g)))
        D = dd[idx].sum(1)
        ra, rb = na[idx].sum(1) / D, nb[idx].sum(1) / D
        diff.append(rb - ra)
        rel.append(rb / ra - 1)
    diff, rel = np.concatenate(diff), np.concatenate(rel)
    ra, rb = na.sum() / dd.sum(), nb.sum() / dd.sum()
    return {"metric": f"{num}/{den}", "mean_a": float(ra), "mean_b": float(rb), "diff": float(rb - ra),
            "diff_ci95": [float(x) for x in np.percentile(diff, [2.5, 97.5])], "relative": float(rb / ra - 1),
            "relative_ci95": [float(x) for x in np.percentile(rel, [2.5, 97.5])],
            "p_diff_gt_0": float((diff > 0).mean()), "n_customers": int(len(g)), "n_boot": int(n_boot)}


def fold_stats(per_fold: dict[str, dict], metric: str = "map@12") -> dict:
    vals = [v[metric] for v in per_fold.values()]
    return {"per_fold": {k: v[metric] for k, v in per_fold.items()}, "mean": float(np.mean(vals)),
            "std_between_folds": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}


def lists_from_topk(df: pd.DataFrame) -> dict[int, list[int]]:
    df = df.sort_values(["customer_idx", "rank"], kind="stable")
    return df.groupby("customer_idx", sort=False)["article_id"].agg(list).to_dict()


def seed_spread(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": [float(v) for v in values]}


SystemFn = Callable[[FoldContext], dict[int, list[int]]]
