"""Track B metrics for the Round-3 protocol (Phase 1.4).

Adds to the Round-1 set: Recall@5 alongside Recall@12, per-slot recall, a
recently-launched-item segment that replaces the leak-dependent "cold item"
segment, recommendation-list diversity, and per-query score vectors so model
comparisons can be bootstrapped over customers rather than over correlated
(anchor, slot) queries.

Why "recently launched" and not "cold". Under the leakage-free catalogue an
article is eligible only if it sold before the cutoff, so a never-sold article
can never be recommended and "item cold-start recall" is identically zero by
construction, not by model failure. The dataset has no launch or inventory feed,
so true pre-launch availability is unknowable; the honest substitute is recall on
articles whose first observed sale is within ``new_item_days`` of the cutoff.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def segments(uni: pd.DataFrame, cfg) -> dict:
    b = cfg.track_b
    head = set()
    for _, g in uni.groupby("slot"):
        g = g.sort_values("pop_window", ascending=False)
        head.update(g.article_id.to_numpy()[: max(1, int(len(g) * float(b.head_share)))].tolist())
    new_days = int(b.new_item_days)
    new_items = set(uni.article_id[(uni.age_days >= 0) & (uni.age_days <= new_days)].tolist())
    return {"head": head,
            "new_item": new_items,
            "jewellery": set(uni.article_id[uni.is_jewellery.astype(bool)].tolist()),
            "never_sold_before": set(uni.article_id[uni.pop_window == 0].tolist())}


def evaluate(recs, q: pd.DataFrame, uni: pd.DataFrame, cfg, attrs: pd.DataFrame | None = None,
             seg: dict | None = None) -> dict:
    """``recs[i]`` is the ranked article list for row ``i`` of ``q`` (same order)."""
    k = int(cfg.track_b.k)
    seg = seg or segments(uni, cfg)
    head, new_items, jew = seg["head"], seg["new_item"], seg["jewellery"]
    total = float(uni.pop_window.sum())
    pop_share = dict(zip(uni.article_id.to_numpy(), (uni.pop_window.to_numpy() + 1) / (total + len(uni))))
    ptype = {} if attrs is None else dict(zip(attrs.article_id.to_numpy(), attrs.product_type_no.to_numpy()))
    pcode = {} if attrs is None else dict(zip(attrs.article_id.to_numpy(), attrs.product_code.to_numpy()))

    n = len(q)
    r12 = np.zeros(n, dtype=np.float64)
    r5 = np.zeros(n, dtype=np.float64)
    nd = np.zeros(n, dtype=np.float64)
    seg_hits = {s: [0, 0] for s in ("tail", "new_item", "jewellery")}
    slots = q.target_slot.to_numpy()
    shown, novelty, div_type, div_code, list_len = set(), [], [], [], []
    for i, (r, truth) in enumerate(zip(recs, q.truth.values)):
        r = list(r[:k])
        t = set(np.asarray(truth).tolist())
        pos = [j for j, a in enumerate(r) if a in t]
        r12[i] = len(pos) / len(t)
        r5[i] = sum(1 for j in pos if j < 5) / len(t)
        dcg = sum(1 / math.log2(j + 2) for j in pos)
        idcg = sum(1 / math.log2(j + 2) for j in range(min(len(t), k)))
        nd[i] = dcg / idcg if idcg else 0.0
        top = set(r)
        for name, s, invert in (("tail", head, True), ("new_item", new_items, False),
                                ("jewellery", jew, False)):
            items = [a for a in t if (a not in s if invert else a in s)]
            seg_hits[name][0] += sum(1 for a in items if a in top)
            seg_hits[name][1] += len(items)
        shown.update(r)
        if r:
            novelty.append(float(np.mean([-math.log2(pop_share.get(a, 1e-9)) for a in r])))
            div_type.append(len({ptype.get(a) for a in r}) / len(r))
            div_code.append(len({pcode.get(a) for a in r}) / len(r))
        list_len.append(len(r))
    out = {f"recall@{k}": float(r12.mean()), "recall@5": float(r5.mean()), f"ndcg@{k}": float(nd.mean())}
    for name, (hit, tot) in seg_hits.items():
        out[f"recall@{k}_{name}"] = hit / tot if tot else float("nan")
        out[f"n_truth_{name}"] = tot
    for slot in sorted(set(slots.tolist())):
        m = slots == slot
        out[f"recall@{k}_slot_{slot}"] = float(r12[m].mean())
        out[f"n_queries_slot_{slot}"] = int(m.sum())
    out[f"catalog_coverage@{k}"] = len(shown) / len(uni)
    out["novelty"] = float(np.mean(novelty)) if novelty else float("nan")
    out[f"diversity_product_type@{k}"] = float(np.mean(div_type)) if div_type else float("nan")
    out[f"diversity_product_code@{k}"] = float(np.mean(div_code)) if div_code else float("nan")
    out["mean_list_length"] = float(np.mean(list_len)) if list_len else 0.0
    out["n_queries"] = n
    out["_per_query"] = {f"recall@{k}": r12, "recall@5": r5, f"ndcg@{k}": nd}
    return out


def public(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def cluster_bootstrap(customer_idx: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 1000,
                      seed: int = 0) -> dict:
    """Paired customer-cluster bootstrap of the query-averaged difference ``mean(b) - mean(a)``.

    One basket produces several (anchor, target slot) queries and one customer can
    produce several baskets, so queries are not independent. Customers are the
    resampling unit: a customer is drawn with replacement together with all of
    their queries, and the metric is recomputed as the query-weighted mean over
    the resampled customers. The point estimate therefore equals the reported
    query-averaged metric.
    """
    codes, _ = pd.factorize(customer_idx)
    n_c = codes.max() + 1 if len(codes) else 0
    if not n_c:
        return {"diff": float("nan"), "ci95": [float("nan"), float("nan")], "n_customers": 0}
    cnt = np.bincount(codes, minlength=n_c).astype(np.float64)
    sa = np.bincount(codes, weights=a, minlength=n_c)
    sb = np.bincount(codes, weights=b, minlength=n_c)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_c, size=n_c)
        w = np.bincount(idx, minlength=n_c).astype(np.float64)
        n = (w * cnt).sum()
        diffs[i] = ((w * sb).sum() - (w * sa).sum()) / n
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    base = a.mean()
    return {"diff": float(b.mean() - a.mean()), "ci95": [float(lo), float(hi)],
            "relative": float((b.mean() - a.mean()) / base) if base else float("nan"),
            "relative_ci95": [float(lo / base), float(hi / base)] if base else None,
            "p_gain_positive": float((diffs > 0).mean()), "n_customers": int(n_c), "n_queries": int(len(a))}
