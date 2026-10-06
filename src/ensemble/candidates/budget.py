"""Candidate-budget allocation across retrieval channels (recall / size Pareto frontier).

  python -m ensemble.candidates.budget [weeks_back=3,2,1] [sample=0.25]

For each selection week (label weeks before validation; the test week is never
read) every channel in ``budget.pool`` is built with its maximum cap. A greedy
allocator then grows one channel's cap at a time, always choosing the extension
that adds the most *new* true (customer, article) hits per *new* candidate,
i.e. the steepest step along the recall vs candidates-per-customer frontier.
Overlap between channels is accounted for exactly: a candidate another channel
already proposed costs nothing and earns nothing.

The output (``reports/phase2/budget_frontier.json``) lists the frontier and the
per-channel caps at each target budget. Customers are sampled deterministically
(hash of customer_idx) to bound memory; recall and size are ratios, so the
sample is unbiased for both.
"""
from __future__ import annotations

import json
import time

import numpy as np

from ensemble.candidates.evaluate import load_truth_table
from ensemble.candidates.retrieval import build_channels, drop_channels
from ensemble.config import Config, load_config
from ensemble.data.splits import load_splits
from ensemble.db import connect


def load_pool(con, week, r, pool: dict[str, int], sample: float, offset: int) -> tuple[dict, int, int]:
    """Return per-channel (rank, uid, hit) arrays for one week; uids are offset to stay unique across weeks."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _users AS SELECT DISTINCT customer_idx FROM transactions
                    WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}'
                      AND hash(customer_idx) % 1000 < {int(sample * 1000)}""")
    load_truth_table(con, week)
    con.execute("DELETE FROM _truth WHERE customer_idx NOT IN (SELECT customer_idx FROM _users)")
    n_users = con.execute("SELECT count(*) FROM _users").fetchone()[0]
    n_truth = con.execute("SELECT count(*) FROM _truth").fetchone()[0]
    chs = list(pool)
    build_channels(con, week, r, chs)
    union = " UNION ALL ".join(
        f"""SELECT {i}::TINYINT AS ch, customer_idx, article_id,
                   row_number() OVER (PARTITION BY customer_idx ORDER BY round(score::DOUBLE, 9) DESC, article_id) AS rnk
            FROM _ch_{c} QUALIFY rnk <= {int(pool[c])}""" for i, c in enumerate(chs))
    z = con.execute(f"""
        WITH p AS ({union}),
        u AS (SELECT *, dense_rank() OVER (ORDER BY customer_idx, article_id) - 1 AS uid FROM p)
        SELECT u.ch, u.rnk::SMALLINT AS rnk, (u.uid + {offset})::INTEGER AS uid, t.customer_idx IS NOT NULL AS hit
        FROM u LEFT JOIN _truth t USING (customer_idx, article_id)
    """).fetchnumpy()
    drop_channels(con, chs)
    out = {}
    for i, c in enumerate(chs):
        m = z["ch"] == i
        order = np.argsort(z["rnk"][m], kind="stable")
        out[c] = (z["rnk"][m][order], z["uid"][m][order], z["hit"][m][order])
    return out, n_users, n_truth


def greedy_frontier(pools: list[dict], n_users: int, n_truth: int, steps=(5, 10, 20)) -> list[dict]:
    chs = list(pools[0])
    rank = {c: np.concatenate([p[c][0] for p in pools]) for c in chs}
    uid = {c: np.concatenate([p[c][1] for p in pools]) for c in chs}
    hit_rows = {c: np.concatenate([p[c][2] for p in pools]) for c in chs}
    order = {c: np.argsort(rank[c], kind="stable") for c in chs}
    for c in chs:
        rank[c], uid[c], hit_rows[c] = rank[c][order[c]], uid[c][order[c]], hit_rows[c][order[c]]
    n_uid = int(max(u.max() for u in uid.values())) + 1
    hit = np.zeros(n_uid, bool)
    for c in chs:
        hit[uid[c][hit_rows[c]]] = True
    inset = np.zeros(n_uid, bool)
    caps = {c: 0 for c in chs}
    max_cap = {c: int(rank[c].max()) if len(rank[c]) else 0 for c in chs}
    n_in = n_hit = 0
    frontier = [{"cand": 0.0, "recall": 0.0, "caps": dict(caps)}]
    while True:
        best = None
        for c in chs:
            for s in steps:
                if caps[c] >= max_cap[c]:
                    continue
                lo, hi = np.searchsorted(rank[c], [caps[c] + 1, caps[c] + s + 1])
                u = uid[c][lo:hi]
                new = u[~inset[u]]
                cost = len(new)
                gain = int(hit[new].sum())
                ratio = gain / cost if cost else (np.inf if gain else 0.0)
                if cost == 0:   # free extension: take it immediately
                    ratio = np.inf
                if best is None or ratio > best[0]:
                    best = (ratio, c, s, new, gain)
        if best is None:
            break
        ratio, c, s, new, gain = best
        inset[new] = True
        n_in += len(new)
        n_hit += gain
        caps[c] = min(caps[c] + s, max_cap[c])
        frontier.append({"cand": n_in / n_users, "recall": n_hit / n_truth, "marginal_recall_per_cand": (gain / n_truth) / (len(new) / n_users) if len(new) else None,
                         "step": c, "caps": dict(caps)})
    return frontier


def caps_at(frontier: list[dict], budget: float) -> dict:
    pt = next((p for p in frontier if p["cand"] >= budget), frontier[-1])
    return {"cand": pt["cand"], "recall": pt["recall"], "caps": pt["caps"]}


def run(weeks_back=(3, 2, 1), sample: float = 0.25) -> dict:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    b = cfg.budget
    r = Config({**dict(cfg.retrieval), **dict(b.get("params") or {}), **{f"{c}_k": k for c, k in b.pool.items()}})
    pools, n_users, n_truth, offset = [], 0, 0, 0
    for k in weeks_back:
        w = splits.val.shift(-k)
        t = time.time()
        p, nu, nt = load_pool(con, w, r, dict(b.pool), sample, offset)
        offset = int(max(x[1].max() for x in p.values())) + 1
        pools.append(p)
        n_users, n_truth = n_users + nu, n_truth + nt
        print(f"pool {w.start}: {nu:,} customers, {nt:,} true pairs, {sum(len(x[0]) for x in p.values()):,} rows ({time.time() - t:.0f}s)", flush=True)
    frontier = greedy_frontier(pools, n_users, n_truth)
    targets = {str(t): caps_at(frontier, t) for t in b.targets}
    for t, v in targets.items():
        print(f"budget {t:>4}: recall={v['recall']:.4f} cand={v['cand']:.1f} caps={v['caps']}")
    out = {"weeks": [str(splits.val.shift(-k).start) for k in weeks_back], "sample": sample,
           "pool": dict(b.pool), "params": dict(b.get("params") or {}), "targets": targets, "frontier": frontier[::5] + [frontier[-1]]}
    d = cfg.path("reports") / "phase2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "budget_frontier.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    import sys
    wb = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (3, 2, 1)
    run(wb, float(sys.argv[2]) if len(sys.argv) > 2 else 0.25)
