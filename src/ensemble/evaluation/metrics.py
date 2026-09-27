"""Offline ranking metrics.

MAP@K follows the competition definition: average precision is normalised by
``min(|truth|, K)`` and averaged over customers who purchased at least once in
the label week (customers with no purchase have no defined AP).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def average_precision_at_k(pred: Sequence, truth: set, k: int = 12) -> float:
    if not truth:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for i, p in enumerate(pred[:k]):
        if p in truth and p not in seen:
            hits += 1
            score += hits / (i + 1)
        seen.add(p)
    return score / min(len(truth), k)


def map_at_k(preds: Mapping, truth: Mapping, k: int = 12, users: Iterable | None = None) -> float:
    users = list(truth if users is None else users)
    if not users:
        return float("nan")
    return sum(average_precision_at_k(preds.get(u, []), truth[u], k) for u in users) / len(users)


def recall_at_k(preds: Mapping, truth: Mapping, k: int | None = None, users: Iterable | None = None) -> float:
    """Micro-averaged recall: share of all true (user, item) pairs retrieved."""
    users = list(truth if users is None else users)
    found = total = 0
    for u in users:
        p = preds.get(u, [])
        p = set(p if k is None else p[:k])
        found += len(truth[u] & p)
        total += len(truth[u])
    return found / total if total else float("nan")


def ndcg_at_k(pred: Sequence, truth: set, k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(pred[:k]) if p in truth)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / idcg if idcg else 0.0


def relative_lift(model: float, baseline: float) -> float:
    """(model − baseline) / baseline, as a fraction (0.18 = +18%)."""
    return (model - baseline) / baseline if baseline else float("nan")
