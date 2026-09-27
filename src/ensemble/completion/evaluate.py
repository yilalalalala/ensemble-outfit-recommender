"""Track B metrics (DESIGN §5.5): Recall@K, NDCG@K, relative lift vs popularity,
tail-item recall, item cold-start recall, jewellery segment, coverage, novelty."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def evaluate(recs: list[np.ndarray], q: pd.DataFrame, uni: pd.DataFrame, cfg) -> dict:
    """``recs[i]`` is the ranked article list for query row ``i`` of ``q``."""
    k = int(cfg.completion.k)
    head_share = float(cfg.completion.head_share)
    head = set()
    for _, g in uni.groupby("slot"):
        g = g.sort_values("pop_window", ascending=False)
        head.update(g.article_id.values[: max(1, int(len(g) * head_share))])
    cold = set(uni.article_id[uni.pop_window == 0])
    jew = set(uni.article_id[uni.is_jewellery])
    total_sales = uni.pop_window.sum()
    pop_share = dict(zip(uni.article_id, (uni.pop_window + 1) / (total_sales + len(uni))))

    agg = {f"recall@{k}": [], "recall@5": [], f"ndcg@{k}": []}
    seg = {"tail": [0, 0], "cold": [0, 0], "jewellery": [0, 0]}
    shown, novelty = set(), []
    for r, truth in zip(recs, q.truth.values):
        r = list(r[:k])
        t = set(truth)
        pos = [i for i, a in enumerate(r) if a in t]
        agg[f"recall@{k}"].append(len(pos) / len(t))
        agg["recall@5"].append(sum(1 for i in pos if i < 5) / len(t))
        dcg = sum(1 / math.log2(i + 2) for i in pos)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(t), k)))
        agg[f"ndcg@{k}"].append(dcg / idcg)
        top = set(r)
        for name, s in (("tail", None), ("cold", cold), ("jewellery", jew)):
            items = [a for a in t if (a not in head if name == "tail" else a in s)]
            seg[name][0] += sum(1 for a in items if a in top)
            seg[name][1] += len(items)
        shown.update(r)
        novelty.append(np.mean([-math.log2(pop_share.get(a, 1e-9)) for a in r]) if r else 0.0)
    out = {m: float(np.mean(v)) for m, v in agg.items()}
    for name, (hit, n) in seg.items():
        out[f"recall@{k}_{name}"] = hit / n if n else float("nan")
        out[f"n_truth_{name}"] = n
    out[f"catalog_coverage@{k}"] = len(shown) / len(uni)
    out["novelty"] = float(np.mean(novelty))
    out["n_queries"] = len(q)
    return out
