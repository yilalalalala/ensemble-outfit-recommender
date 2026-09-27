"""Track A offline evaluation: MAP@12 overall and by segment (DESIGN §3)."""
from __future__ import annotations

from collections import defaultdict

from ensemble.data.splits import Week
from ensemble.evaluation.metrics import map_at_k, recall_at_k


def load_truth(con, week: Week) -> dict[int, set[int]]:
    rows = con.execute(
        "SELECT customer_idx, article_id FROM transactions WHERE t_dat BETWEEN ? AND ?",
        [week.start, week.end],
    ).fetchall()
    truth: dict[int, set[int]] = defaultdict(set)
    for c, a in rows:
        truth[c].add(a)
    return dict(truth)


def load_segments(con, week: Week) -> dict:
    """User cold start = no purchase before the week; item cold start = first sale inside it."""
    returning = {
        r[0]
        for r in con.execute(
            """SELECT DISTINCT customer_idx FROM transactions
               WHERE customer_idx IN (SELECT customer_idx FROM transactions WHERE t_dat BETWEEN ? AND ?)
                 AND t_dat < ?""",
            [week.start, week.end, week.start],
        ).fetchall()
    }
    cold_items = {
        r[0]
        for r in con.execute(
            """SELECT article_id FROM transactions GROUP BY 1 HAVING min(t_dat) >= ?""",
            [week.start],
        ).fetchall()
    }
    return {"returning": returning, "cold_items": cold_items}


def evaluate(preds: dict, truth: dict, segments: dict, k: int = 12) -> dict:
    users = list(truth)
    ret = [u for u in users if u in segments["returning"]]
    new = [u for u in users if u not in segments["returning"]]
    cold = segments["cold_items"]
    cold_truth = {u: t & cold for u, t in truth.items() if t & cold}
    return {
        f"map@{k}": map_at_k(preds, truth, k),
        f"map@{k}_returning": map_at_k(preds, truth, k, ret),
        f"map@{k}_new_customers": map_at_k(preds, truth, k, new),
        f"recall@{k}": recall_at_k(preds, truth, k),
        f"recall@{k}_cold_items": recall_at_k(preds, cold_truth, k) if cold_truth else float("nan"),
        "n_customers": len(users),
        "share_new_customers": len(new) / len(users),
        "share_purchases_cold_items": sum(len(t) for t in cold_truth.values()) / sum(len(t) for t in truth.values()),
    }
