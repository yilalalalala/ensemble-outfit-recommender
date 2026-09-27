"""M1 baselines (D-004): popularity and repeat purchase.

Every function takes the label week and reads only data before ``week.start``
(point-in-time correctness, D-003).
"""
from __future__ import annotations

import json
from datetime import timedelta

from ensemble.config import load_config
from ensemble.data.splits import Week, load_splits
from ensemble.db import connect
from ensemble.evaluation.track_a import evaluate, load_segments, load_truth
from ensemble import tracking

AGE_BINS = "CASE WHEN age IS NULL THEN -1 WHEN age < 25 THEN 0 WHEN age < 35 THEN 1 WHEN age < 50 THEN 2 ELSE 3 END"


def popular(con, week: Week, days: int = 7, k: int = 12) -> list[int]:
    return [r[0] for r in con.execute(
        """SELECT article_id FROM transactions WHERE t_dat >= ? AND t_dat < ?
           GROUP BY 1 ORDER BY count(*) DESC, article_id LIMIT ?""",
        [week.start - timedelta(days=days), week.start, k]).fetchall()]


def popular_by_age(con, week: Week, days: int = 7, k: int = 12) -> dict[int, list[int]]:
    rows = con.execute(f"""
        WITH s AS (
            SELECT {AGE_BINS} AS age_bin, t.article_id, count(*) n
            FROM transactions t JOIN customers c USING (customer_idx)
            WHERE t_dat >= ? AND t_dat < ? GROUP BY 1, 2)
        SELECT age_bin, article_id FROM s
        QUALIFY row_number() OVER (PARTITION BY age_bin ORDER BY n DESC, article_id) <= ?
        ORDER BY age_bin, n DESC, article_id""",
        [week.start - timedelta(days=days), week.start, k]).fetchall()
    out: dict[int, list[int]] = {}
    for b, a in rows:
        out.setdefault(b, []).append(a)
    return out


def customer_age_bins(con, customers) -> dict[int, int]:
    con.execute("CREATE OR REPLACE TEMP TABLE _u AS SELECT unnest(?) AS customer_idx", [list(customers)])
    return dict(con.execute(f"SELECT customer_idx, {AGE_BINS} FROM customers JOIN _u USING (customer_idx)").fetchall())


def repeat_purchase(con, week: Week, customers, weeks_back: int = 4, k: int = 12) -> dict[int, list[int]]:
    """The customer's own recent articles, most recent first, then most frequent."""
    con.execute("CREATE OR REPLACE TEMP TABLE _u AS SELECT unnest(?) AS customer_idx", [list(customers)])
    rows = con.execute(
        """SELECT customer_idx, article_id FROM (
               SELECT customer_idx, article_id, max(t_dat) AS last_dat, count(*) n
               FROM transactions JOIN _u USING (customer_idx)
               WHERE t_dat >= ? AND t_dat < ? GROUP BY 1, 2)
           QUALIFY row_number() OVER (PARTITION BY customer_idx ORDER BY last_dat DESC, n DESC, article_id) <= ?
           ORDER BY customer_idx, last_dat DESC, n DESC, article_id""",
        [week.start - timedelta(days=7 * weeks_back), week.start, k]).fetchall()
    out: dict[int, list[int]] = {}
    for c, a in rows:
        out.setdefault(c, []).append(a)
    return out


def fill(primary: list[int], backup: list[int], k: int = 12) -> list[int]:
    out = list(dict.fromkeys(primary))
    for a in backup:
        if len(out) >= k:
            break
        if a not in out:
            out.append(a)
    return out[:k]


def run(week_name: str = "val") -> dict:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    week = getattr(splits, week_name)
    truth = load_truth(con, week)
    seg = load_segments(con, week)
    users = list(truth)
    pop = popular(con, week)
    pop_age = popular_by_age(con, week)
    ages = customer_age_bins(con, users)
    rep = repeat_purchase(con, week, users)

    systems = {
        "popularity_global": {u: pop for u in users},
        "popularity_by_age": {u: pop_age.get(ages.get(u, -1), pop) for u in users},
        "repeat_purchase+popularity": {u: fill(rep.get(u, []), pop) for u in users},
        "repeat_purchase+popularity_by_age": {u: fill(rep.get(u, []), pop_age.get(ages.get(u, -1), pop)) for u in users},
    }
    results = {}
    for name, preds in systems.items():
        with tracking.run("track_a", f"m1/{name}/{week_name}", {"week": week.start, "k": 12}):
            m = evaluate(preds, truth, seg)
            tracking.log_metrics(m)
        results[name] = m
        print(f"{name:40s} MAP@12={m['map@12']:.5f}  returning={m['map@12_returning']:.5f}  new={m['map@12_new_customers']:.5f}")
    out = cfg.path("reports") / f"m1_baselines_{week_name}.json"
    out.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "val")
