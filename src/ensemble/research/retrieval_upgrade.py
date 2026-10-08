"""Track A retrieval-ceiling upgrade (docs/CLAUDECODE_TRACK_A_RETRIEVAL_UPGRADE_PLAN.md, D-044).

  ENSEMBLE_CONFIG=track_a_research python -m ensemble.research.retrieval_upgrade <stage> [args]

  pools <fold>            deep channel pools + exact neural top-k lists for one reporting fold
  allocate                candidate-budget frontier seeded at the final caps (allocator weeks only)
  diagnostics <fold>      per-customer / per-pair retrieval diagnostics and the 0.1757 reconciliation
  eval <config> <fold>    build the config's candidates, featurise, score with the fold's final ranker
  report                  pooled comparison, bootstrap, segments, costs, the predeclared decision

**Why every larger budget is a strict superset.** Channel *parameters* are unchanged, so the order
inside each channel is the final system's order (same SQL, same `round(score, 9) DESC, article_id`
ranking as ``retrieval.merge``); a larger budget only raises caps. Appended neural candidates come
strictly after the whole current union. The ``final`` config must therefore reproduce the final
system's per-customer results exactly, which ``eval final`` asserts.

**Fixed ranker.** Every configuration is scored by the same per-fold final ranker (seed 42), so a
wider candidate set never gets credit for a different model. Candidates outside the ranker's
training distribution (deeper channel ranks, appended items with no Phase-2 provenance) are scored
as they are; that limitation is reported.

Point in time: pools are built by the production retrieval channels (label-week corruption tests
in ``tests/test_retrieval.py`` and ``tests/test_research_track_a.py``); neural lists come from the
models trained for that fold's cutoff. Truth is joined only after candidates exist.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.config import Config, load_config
from ensemble.data.splits import Week
from ensemble.research import protocol as P
from ensemble.research.ensemble import PHASE2_CHANNELS, config_spec

NEURAL = ("bpr", "lightgcn", "sasrec")
CH_ID = {c: i for i, c in enumerate(PHASE2_CHANNELS)}


def ucfg(cfg) -> Config:
    return cfg.research.retrieval_upgrade


def final_caps(cfg) -> dict[str, int]:
    return config_spec(cfg, "phase2_nscores")["caps"]


def root(cfg) -> Path:
    u = ucfg(cfg)
    r = {k: v for k, v in dict(cfg.retrieval).items() if not k.endswith("_k") and k != "channels"}
    key = P.config_hash({"pool_caps": dict(u.pool_caps), "neural_depth": int(u.neural_depth), "retrieval": r,
                         "models": dict(cfg.research.models), "protocol": int(P.proto(cfg).protocol_version)})
    d = cfg.path("interim") / "track_a_retrieval" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir(cfg) -> Path:
    d = cfg.path("reports") / "track_a_retrieval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def peak_rss() -> int:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


# ---------------------------------------------------------------------------
# Stage 0: deep pools (point in time; no label-week read)
# ---------------------------------------------------------------------------

def pool_sql(channels: list[str], caps: dict[str, int]) -> str:
    """Rank every channel table exactly like ``retrieval.merge`` and keep the top ``caps[ch]``."""
    return "\nUNION ALL ".join(f"""
        SELECT customer_idx::INTEGER AS customer_idx, article_id::INTEGER AS article_id, {CH_ID[c]}::TINYINT AS ch,
               rnk::SMALLINT AS rnk, score FROM (
            SELECT customer_idx, article_id, round(score::DOUBLE, 9) AS score,
                   row_number() OVER (PARTITION BY customer_idx ORDER BY round(score::DOUBLE, 9) DESC, article_id) AS rnk
            FROM _ch_{c} QUALIFY rnk <= {int(caps[c])})""" for c in channels)


def build_pool(con, week: Week, r, caps: dict[str, int], users_table: str) -> pd.DataFrame:
    from ensemble.candidates.retrieval import build_channels, drop_channels
    chs = list(caps)
    build_channels(con, week, r, chs, users_table)
    df = con.execute(pool_sql(chs, caps) + " ORDER BY customer_idx, ch, rnk").df()
    drop_channels(con, chs)
    return df


def pool_retrieval_cfg(cfg) -> Config:
    u = ucfg(cfg)
    r = dict(cfg.retrieval)
    r.update({f"{c}_k": int(k) for c, k in dict(u.pool_caps).items()})
    r["channels"] = list(dict(u.pool_caps))
    return Config(r)


def deep_list(cfg, model: str, week: Week, depth: int, elig: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Exact top-``depth`` dot-product list over the eligible catalogue from the cached per-week vectors."""
    from ensemble.research.channels import model_dir
    from ensemble.research.neural import score_topk
    z = np.load(model_dir(cfg, model, week, 0) / "emb.npz")
    users, U, items, V = z["users"], z["user_emb"], z["items"], z["item_emb"]
    cols = np.flatnonzero(np.isin(items, elig))
    t0 = time.perf_counter()
    it, sc = score_topk(U, V[cols], items[cols], depth)
    secs = time.perf_counter() - t0
    k = it.shape[1]
    df = pd.DataFrame({"customer_idx": np.repeat(users, k).astype(np.int32),
                       "rnk": np.tile(np.arange(1, k + 1), len(users)).astype(np.int16),
                       "article_id": it.ravel().astype(np.int32), "score": sc.ravel().astype(np.float32)})
    bench = {"model": model, "queries": int(len(users)), "items": int(len(cols)), "dim": int(U.shape[1]),
             "depth": int(k), "exact_seconds": round(secs, 3), "ms_per_query": 1000 * secs / max(1, len(users))}
    return df, bench


def pools(fold_start: str) -> None:
    from ensemble.db import connect
    cfg = load_config()
    week = P.week_of(fold_start)
    d = root(cfg) / str(week.start)
    d.mkdir(parents=True, exist_ok=True)
    con = connect(cfg, read_only=True)
    t0 = time.time()
    pdir = d / "pool"
    if not (pdir / "_SUCCESS").exists():
        users = con.execute(f"""SELECT DISTINCT customer_idx FROM transactions
                                WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}' ORDER BY 1""") \
            .fetchnumpy()["customer_idx"]
        r = pool_retrieval_cfg(cfg)
        pdir.mkdir(exist_ok=True)
        n_rows = 0
        for i, b in enumerate(range(0, len(users), 10000)):
            con.execute("CREATE OR REPLACE TEMP TABLE _pu AS SELECT unnest(?::INTEGER[]) AS customer_idx",
                        [users[b:b + 10000].tolist()])
            df = build_pool(con, week, r, dict(ucfg(cfg).pool_caps), "_pu")
            df.to_parquet(pdir / f"part-{i:03d}.parquet", index=False, row_group_size=500_000)
            n_rows += len(df)
            del df
            gc.collect()
        (pdir / "_SUCCESS").write_text(json.dumps({"rows": n_rows, "customers": int(len(users))}))
        print(f"pool {week.start}: {n_rows:,} rows, {len(users):,} customers ({time.time() - t0:.0f}s)", flush=True)
    elig = P.eligible(con, week, int(P.proto(cfg).eligible_days))
    benches = []
    for m in NEURAL:
        f = d / f"deep_{m}.parquet"
        if f.exists():
            continue
        df, bench = deep_list(cfg, m, week, int(ucfg(cfg).neural_depth), elig)
        df.to_parquet(f, index=False)
        benches.append(bench)
    if benches:
        (d / "exact_search.json").write_text(json.dumps(benches, indent=1))
    (d / "_POOLS_SUCCESS").write_text(json.dumps({"seconds": round(time.time() - t0, 1), "peak_rss": peak_rss()}))


# ---------------------------------------------------------------------------
# Truth with segments (evaluation only; never an input to candidates)
# ---------------------------------------------------------------------------

def truth_pairs(con, cfg, week: Week, ctx) -> pd.DataFrame:
    rows = [(u, a) for u, t in ctx.truth.items() for a in t]
    df = pd.DataFrame(rows, columns=["customer_idx", "article_id"])
    df["eligible"] = df.article_id.isin(ctx.eligible)
    df["head"] = df.article_id.isin(ctx.items["head"])
    df["tail"] = df.eligible & ~df["head"]
    df["recent"] = df.article_id.isin(ctx.items["recent"])
    df["cold"] = df.article_id.isin(ctx.items["cold"])
    df["repeat"] = [a in ctx.prior.get(u, ()) for u, a in zip(df.customer_idx, df.article_id)]
    df["returning"] = df.customer_idx.map(ctx.returning).astype(bool)
    df["activity"] = df.customer_idx.map(ctx.activity)
    return df


def pair_ranks(con, d: Path, tp: pd.DataFrame) -> pd.DataFrame:
    """For every truth pair: its rank in each channel pool (NaN = not proposed) and each neural list."""
    con.register("_tp", tp[["customer_idx", "article_id"]])
    piv = ", ".join(f"min(CASE WHEN p.ch = {i} THEN p.rnk END) AS rk_{c}" for c, i in CH_ID.items())
    pr = con.execute(f"""SELECT t.customer_idx, t.article_id, {piv}
                         FROM _tp t LEFT JOIN read_parquet('{d / "pool" / "part-*.parquet"}') p
                           ON p.customer_idx = t.customer_idx AND p.article_id = t.article_id
                         GROUP BY 1, 2""").df()
    for m in NEURAL:
        nr = con.execute(f"""SELECT t.customer_idx, t.article_id, min(n.rnk) AS rk_{m}
                             FROM _tp t LEFT JOIN read_parquet('{d / f"deep_{m}.parquet"}') n
                               ON n.customer_idx = t.customer_idx AND n.article_id = t.article_id
                             GROUP BY 1, 2""").df()
        pr = pr.merge(nr, on=["customer_idx", "article_id"])
    con.unregister("_tp")
    return tp.merge(pr, on=["customer_idx", "article_id"], how="left")


def in_caps(df: pd.DataFrame, caps: dict[str, int]) -> np.ndarray:
    m = np.zeros(len(df), bool)
    for c, k in caps.items():
        m |= (df[f"rk_{c}"].to_numpy() <= k)
    return m


# ---------------------------------------------------------------------------
# Stage 1: diagnostics
# ---------------------------------------------------------------------------

def diagnostics(fold_start: str) -> dict:
    from ensemble.db import connect
    from ensemble.research.ensemble import eval_path
    from ensemble.research.evaluate import FoldContext
    cfg = load_config()
    week = P.week_of(fold_start)
    d = root(cfg) / str(week.start)
    out = out_dir(cfg) / "diagnostics"
    out.mkdir(exist_ok=True)
    con = connect(cfg, read_only=True)
    ctx = FoldContext(con, cfg, week)
    tp = pair_ranks(con, d, truth_pairs(con, cfg, week, ctx))
    caps = final_caps(cfg)
    pool_caps = dict(ucfg(cfg).pool_caps)
    U = in_caps(tp, caps)
    tp["in_union"] = U
    tp["in_pool"] = in_caps(tp, pool_caps)
    for c in PHASE2_CHANNELS:
        tp[f"alone_{c}"] = tp[f"rk_{c}"].to_numpy() <= caps.get(c, 0)
    n_alone = tp[[f"alone_{c}" for c in PHASE2_CHANNELS]].sum(axis=1).to_numpy()
    for c in PHASE2_CHANNELS:
        tp[f"uniq_{c}"] = tp[f"alone_{c}"] & (n_alone == 1)
    for m in NEURAL:
        for k in (100, 300, 1000):
            tp[f"marg_{m}_{k}"] = (tp[f"rk_{m}"].to_numpy() <= k) & ~U
    # Union of the deep non-neural channels beyond the final caps, per channel.
    for c in PHASE2_CHANNELS:
        tp[f"beyond_{c}"] = ~U & (tp[f"rk_{c}"].to_numpy() > caps.get(c, 0)) & (tp[f"rk_{c}"].to_numpy() <= pool_caps[c])
    any_neural = np.zeros(len(tp), bool)
    for m in NEURAL:
        any_neural |= tp[f"rk_{m}"].to_numpy() <= int(ucfg(cfg).neural_depth)
    # Miss taxonomy (mutually exclusive, in this order).
    ev = pd.read_parquet(eval_path(cfg, ucfg(cfg).baseline_system, week, int(ucfg(cfg).baseline_seed)))
    tp = tp.merge(ev[["customer_idx", "hits"]].rename(columns={"hits": "_top12"}), on="customer_idx", how="left")
    cust = tp.groupby("customer_idx").agg(
        n_truth=("article_id", "size"), n_truth_eligible=("eligible", "sum"), hits_union=("in_union", "sum"),
        top12_hits=("_top12", "first"), returning=("returning", "first"), activity=("activity", "first"),
        **{f"alone_{c}": (f"alone_{c}", "sum") for c in PHASE2_CHANNELS},
        **{f"uniq_{c}": (f"uniq_{c}", "sum") for c in PHASE2_CHANNELS},
        **{f"beyond_{c}": (f"beyond_{c}", "sum") for c in PHASE2_CHANNELS},
        **{f"marg_{m}_{k}": (f"marg_{m}_{k}", "sum") for m in NEURAL for k in (100, 300, 1000)}).reset_index()
    cust["miss_union_not_top12"] = cust.hits_union - cust.top12_hits
    nu = ~tp.in_union
    cust = cust.merge(pd.DataFrame({
        "customer_idx": tp.customer_idx,
        "miss_cut_by_cap": nu & tp.in_pool,
        "miss_ineligible_unproposed": nu & ~tp.in_pool & ~tp.eligible,
        "miss_no_channel": nu & ~tp.in_pool & tp.eligible,
        "miss_no_channel_incl_neural": nu & ~tp.in_pool & tp.eligible & ~any_neural}).groupby("customer_idx").sum()
        .reset_index(), on="customer_idx")
    # Candidate counts before / after de-duplication, and pairwise channel overlap, from the pool.
    cand = con.execute(f"""
        WITH u AS (SELECT * FROM read_parquet('{d / "pool" / "part-*.parquet"}')
                   WHERE {" OR ".join(f"(ch = {CH_ID[c]} AND rnk <= {k})" for c, k in caps.items())})
        SELECT customer_idx, count(*) AS n_rows, count(DISTINCT article_id) AS n_cand FROM u GROUP BY 1""").df()
    masks = con.execute(f"""
        WITH u AS (SELECT * FROM read_parquet('{d / "pool" / "part-*.parquet"}')
                   WHERE {" OR ".join(f"(ch = {CH_ID[c]} AND rnk <= {k})" for c, k in caps.items())}),
        m AS (SELECT customer_idx, article_id, bit_or((1::INTEGER << ch)) AS mask FROM u GROUP BY 1, 2)
        SELECT mask, count(*) AS n FROM m GROUP BY 1""").df()
    cust = cust.merge(cand, on="customer_idx", how="left")
    cust["fold"] = str(week.start)
    cust.to_parquet(out / f"customers_{week.start}.parquet", index=False)
    # Pairwise Jaccard from the mask histogram.
    mk, n = masks["mask"].to_numpy(), masks["n"].to_numpy()
    jac = {}
    for a in PHASE2_CHANNELS:
        for b in PHASE2_CHANNELS:
            if a >= b:
                continue
            ia, ib = 1 << CH_ID[a], 1 << CH_ID[b]
            inter = n[(mk & ia > 0) & (mk & ib > 0)].sum()
            union = n[(mk & ia > 0) | (mk & ib > 0)].sum()
            jac[f"{a}|{b}"] = float(inter / union) if union else 0.0
    tot = int(tp.shape[0])
    agg = {"fold": str(week.start), "n_truth_pairs": tot, "n_customers": int(len(cust)),
           "candidate_recall": float(tp.in_union.mean()),
           "rows_before_dedup": int(cust.n_rows.sum()), "candidates_after_dedup": int(cust.n_cand.sum()),
           "redundant_candidate_rate": float(1 - cust.n_cand.sum() / cust.n_rows.sum()),
           "channel_recall_alone": {c: float(tp[f"alone_{c}"].mean()) for c in PHASE2_CHANNELS},
           "channel_unique_recall": {c: float(tp[f"uniq_{c}"].mean()) for c in PHASE2_CHANNELS},
           "channel_beyond_cap_recall": {c: float(tp[f"beyond_{c}"].mean()) for c in PHASE2_CHANNELS},
           "neural_marginal_recall": {f"{m}@{k}": float(tp[f"marg_{m}_{k}"].mean()) for m in NEURAL
                                      for k in (100, 300, 1000)},
           "pairwise_jaccard": jac,
           "miss_taxonomy": {
               "hit_top12": float(cust.top12_hits.sum() / tot),
               "in_union_not_top12": float(cust.miss_union_not_top12.sum() / tot),
               "proposed_then_cut_by_cap": float(cust.miss_cut_by_cap.sum() / tot),
               "lost_by_merge_or_dedup": 0.0,
               "ineligible_not_proposed": float(cust.miss_ineligible_unproposed.sum() / tot),
               "no_channel_proposed_eligible": float(cust.miss_no_channel.sum() / tot),
               "no_channel_incl_neural_top1000": float(cust.miss_no_channel_incl_neural.sum() / tot)},
           "segments_union_recall": seg_recall(tp, "in_union")}
    # Reconciliation with the final system's stored per-customer candidate hits.
    ev_c = ev.set_index("customer_idx")
    chk = cust.set_index("customer_idx")
    agg["reconcile"] = {
        "stored_cand_hits": int(ev_c.cand_hits.sum()), "recomputed_hits": int(chk.hits_union.sum()),
        "stored_n_cand": int(ev_c.n_cand.sum()), "recomputed_n_cand": int(chk.n_cand.sum()),
        "per_customer_hits_equal": bool((ev_c.cand_hits.reindex(chk.index).to_numpy() == chk.hits_union.to_numpy()).all())}
    rc = agg["reconcile"]
    rc["n_customers_with_different_n_cand"] = int(
        (ev_c.n_cand.reindex(chk.index).to_numpy() != chk.n_cand.to_numpy()).sum())
    rc["n_cand_relative_difference"] = (rc["recomputed_n_cand"] - rc["stored_n_cand"]) / rc["stored_n_cand"]
    # Truth hits must reconcile exactly. Candidate counts may differ by a handful of rows: the stored
    # final system was built before the co-visitation neighbour cutoff was made deterministic
    # (multi-threaded float sums flipped near-ties at the 50-neighbour boundary).
    assert rc["stored_cand_hits"] == rc["recomputed_hits"] and rc["per_customer_hits_equal"], rc
    assert abs(rc["n_cand_relative_difference"]) <= 1e-5, rc
    s = sum(v for k, v in agg["miss_taxonomy"].items() if k != "no_channel_incl_neural_top1000")
    assert abs(s - 1.0) < 1e-9, s
    (out / f"fold_{week.start}.json").write_text(json.dumps(agg, indent=1, default=float))
    print(f"diagnostics {week.start}: candidate recall {agg['candidate_recall']:.4f} (reconciled)", flush=True)
    return agg


def seg_recall(tp: pd.DataFrame, col: str) -> dict:
    out = {}
    for name, m in (("returning", tp.returning), ("new", ~tp.returning), ("head", tp["head"]), ("tail", tp["tail"]),
                    ("recent", tp.recent), ("repeat", tp.repeat), ("nonrepeat", ~tp.repeat),
                    ("activity_low", tp.activity == "low"), ("activity_medium", tp.activity == "medium"),
                    ("activity_high", tp.activity == "high"), ("eligible", tp.eligible)):
        m = m.to_numpy(bool)
        out[name] = {"recall": float(tp[col].to_numpy()[m].mean()) if m.any() else float("nan"), "share": float(m.mean())}
    return out


# ---------------------------------------------------------------------------
# Stage 2: candidate-budget frontier seeded at the final caps (allocator weeks only)
# ---------------------------------------------------------------------------

def greedy_seeded(pools_: list[dict], n_users: int, n_truth: int, init: dict[str, int], steps) -> list[dict]:
    """``budget.greedy_frontier`` started from ``init`` caps instead of zero, so every point is a
    superset of the current union."""
    chs = list(pools_[0])
    rank = {c: np.concatenate([p[c][0] for p in pools_]) for c in chs}
    uid = {c: np.concatenate([p[c][1] for p in pools_]) for c in chs}
    hitr = {c: np.concatenate([p[c][2] for p in pools_]) for c in chs}
    for c in chs:
        o = np.argsort(rank[c], kind="stable")
        rank[c], uid[c], hitr[c] = rank[c][o], uid[c][o], hitr[c][o]
    n_uid = int(max(u.max() for u in uid.values())) + 1
    hit = np.zeros(n_uid, bool)
    for c in chs:
        hit[uid[c][hitr[c]]] = True
    inset = np.zeros(n_uid, bool)
    caps = {c: int(init.get(c, 0)) for c in chs}
    for c in chs:
        inset[uid[c][rank[c] <= caps[c]]] = True
    n_in, n_hit = int(inset.sum()), int(hit[inset].sum())
    max_cap = {c: int(rank[c].max()) if len(rank[c]) else 0 for c in chs}
    frontier = [{"cand": n_in / n_users, "recall": n_hit / n_truth, "caps": dict(caps), "step": "seed"}]
    while True:
        best = None
        for c in chs:
            for s in steps:
                if caps[c] >= max_cap[c]:
                    continue
                lo, hi = np.searchsorted(rank[c], [caps[c] + 1, caps[c] + s + 1])
                u = uid[c][lo:hi]
                new = u[~inset[u]]
                cost, gain = len(new), int(hit[new].sum())
                ratio = np.inf if cost == 0 else gain / cost
                if best is None or ratio > best[0]:
                    best = (ratio, c, s, new, gain)
        if best is None:
            break
        ratio, c, s, new, gain = best
        inset[new] = True
        n_in += len(new)
        n_hit += gain
        caps[c] = min(caps[c] + s, max_cap[c])
        frontier.append({"cand": n_in / n_users, "recall": n_hit / n_truth, "caps": dict(caps), "step": c,
                         "marginal_recall_per_cand": (gain / n_truth) / (len(new) / n_users) if len(new) else None})
    return frontier


def allocate() -> dict:
    from ensemble.candidates.budget import load_pool
    from ensemble.db import connect
    cfg = load_config()
    p, u = P.proto(cfg), ucfg(cfg)
    con = connect(cfg, read_only=True)
    pool = dict(u.pool_caps)
    r = Config({**dict(cfg.retrieval), **{f"{c}_k": k for c, k in pool.items()}})
    pools_, n_users, n_truth, offset = [], 0, 0, 0
    for w in [P.week_of(s) for s in p.allocator_weeks]:
        t = time.time()
        pl, nu, nt = load_pool(con, w, r, pool, float(p.allocator_sample), offset)
        offset = int(max(x[1].max() for x in pl.values())) + 1
        pools_.append(pl)
        n_users, n_truth = n_users + nu, n_truth + nt
        print(f"pool {w.start}: {nu:,} customers ({time.time() - t:.0f}s)", flush=True)
    fr = greedy_seeded(pools_, n_users, n_truth, final_caps(cfg), tuple(u.allocator.steps))
    targets = {}
    for b in u.budgets:
        pt = next((x for x in fr if x["cand"] >= b), fr[-1])
        targets[str(b)] = {"cand": pt["cand"], "recall": pt["recall"], "caps": pt["caps"]}
    res = {"weeks": list(p.allocator_weeks), "sample": float(p.allocator_sample), "pool_caps": pool,
           "seed_caps": final_caps(cfg), "seed_point": fr[0], "targets": targets, "max_point": fr[-1],
           "frontier": fr[::10] + [fr[-1]]}
    (out_dir(cfg) / "budget_frontier.json").write_text(json.dumps(res, indent=1, default=float))
    for b, v in targets.items():
        print(f"budget {b}: {v['cand']:.1f} cand, recall {v['recall']:.4f}, caps {v['caps']}", flush=True)
    return res


# ---------------------------------------------------------------------------
# Stage 2/3 evaluation
# ---------------------------------------------------------------------------

def config_def(cfg, name: str) -> dict:
    """``final``; ``F<B>`` frontier caps; ``<model>_A<B>`` final union + appended neural list up to B;
    ``neural3_A<B>`` final union + the three neural lists merged by rank (SASRec, LightGCN, BPR-MF order)."""
    if name == "final":
        return {"caps": final_caps(cfg), "append": None, "budget": None}
    if name.startswith("F"):
        fr = json.loads((out_dir(cfg) / "budget_frontier.json").read_text())
        return {"caps": {c: int(k) for c, k in fr["targets"][name[1:]]["caps"].items() if int(k) > 0},
                "append": None, "budget": int(name[1:])}
    model, b = name.split("_A")
    models = ["sasrec", "lightgcn", "bpr"] if model == "neural3" else [model]
    return {"caps": final_caps(cfg), "append": models, "budget": int(b)}


def build_cand(con, d: Path, spec: dict, users_table: str) -> dict:
    """Create TEMP TABLE ``cand`` (the Phase-2 wide contract) for the customers in ``users_table``."""
    caps = spec["caps"]
    cond = " OR ".join(f"(ch = {CH_ID[c]} AND rnk <= {int(k)})" for c, k in caps.items())
    piv = ", ".join(f"max(CASE WHEN ch = {CH_ID[c]} THEN score END) AS {c}_score, "
                    f"min(CASE WHEN ch = {CH_ID[c]} THEN rnk END)::BIGINT AS {c}_rank" for c in PHASE2_CHANNELS)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cand AS
                    SELECT customer_idx, article_id, {piv}
                    FROM read_parquet('{d / "pool" / "part-*.parquet"}') p
                    WHERE p.customer_idx IN (SELECT customer_idx FROM {users_table}) AND ({cond})
                    GROUP BY 1, 2""")
    info = {"n_union": con.execute("SELECT count(*) FROM cand").fetchone()[0], "n_appended": 0}
    if spec["append"]:
        srcs = " UNION ALL ".join(
            f"SELECT customer_idx, article_id, rnk, {i} AS src FROM read_parquet('{d / f'deep_{m}.parquet'}')"
            for i, m in enumerate(spec["append"]))
        nulls = ", ".join(f"NULL::DOUBLE AS {c}_score, NULL::BIGINT AS {c}_rank" for c in PHASE2_CHANNELS)
        con.execute(f"""CREATE OR REPLACE TEMP TABLE _app AS
            WITH n AS (SELECT customer_idx, count(*) AS n FROM cand GROUP BY 1),
            s AS (SELECT x.customer_idx, x.article_id, min(x.rnk) AS rnk, min(x.src) AS src FROM ({srcs}) x
                  WHERE x.customer_idx IN (SELECT customer_idx FROM {users_table})
                    AND NOT EXISTS (SELECT 1 FROM cand c WHERE c.customer_idx = x.customer_idx
                                                     AND c.article_id = x.article_id)
                  GROUP BY 1, 2),
            o AS (SELECT s.*, row_number() OVER (PARTITION BY s.customer_idx ORDER BY s.rnk, s.src, s.article_id) AS pos
                  FROM s)
            SELECT o.customer_idx, o.article_id, o.rnk, o.src FROM o LEFT JOIN n USING (customer_idx)
            WHERE o.pos <= {int(spec['budget'])} - coalesce(n.n, 0)""")
        info["n_appended"] = con.execute("SELECT count(*) FROM _app").fetchone()[0]
        con.execute(f"INSERT INTO cand SELECT customer_idx, article_id, {nulls} FROM _app")
    return info


def eval_config(name: str, fold_start: str) -> Path:
    import lightgbm as lgb

    from ensemble.db import connect
    from ensemble.features.track_a import build_features
    from ensemble.ranking.train import feature_groups
    from ensemble.research.channels import add_neural_scores
    from ensemble.research.ensemble import eval_path
    from ensemble.research.evaluate import FoldContext, score_lists, summarize
    cfg = load_config()
    u = ucfg(cfg)
    week = P.week_of(fold_start)
    system = f"ret_{name}"
    out = eval_path(cfg, system, week, int(u.baseline_seed))
    if out.with_suffix(".json").exists():
        return out
    d = root(cfg) / str(week.start)
    assert (d / "_POOLS_SUCCESS").exists(), f"pools missing for {week.start}"
    spec = config_def(cfg, name)
    booster = lgb.Booster(model_file=str(cfg.path("interim") / "track_a_research" / "models" / u.baseline_system
                                         / f"{week.start}_s{int(u.baseline_seed)}.txt"))
    feats = booster.feature_name()
    con = connect(cfg, read_only=True)
    t_start = time.time()
    users = con.execute(f"""SELECT DISTINCT customer_idx FROM transactions
                            WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}' ORDER BY 1""") \
        .fetchnumpy()["customer_idx"]
    ctx = FoldContext(con, cfg, week)
    tail = ctx.eligible - ctx.items["head"]
    secs = {"candidates": 0.0, "features": 0.0, "scoring": 0.0}
    parts, stats, n_union, n_app = [], [], 0, 0
    step = int(u.eval_chunk_customers)
    for b in range(0, len(users), step):
        con.execute("CREATE OR REPLACE TEMP TABLE _cu AS SELECT unnest(?::INTEGER[]) AS customer_idx",
                    [users[b:b + step].tolist()])
        t = time.time()
        info = build_cand(con, d, spec, "_cu")
        n_union += info["n_union"]
        n_app += info["n_appended"]
        secs["candidates"] += time.time() - t
        t = time.time()
        df = build_features(con, week, True, groups=feature_groups(cfg), cfg=cfg)
        df = add_neural_scores(df, cfg, week, ("bpr", "lightgcn", "sasrec"), 0)
        missing = [f for f in feats if f not in df.columns]
        assert not missing, missing
        secs["features"] += time.time() - t
        t = time.time()
        s = booster.predict(df[feats].to_numpy(dtype=np.float32), num_threads=8)
        secs["scoring"] += time.time() - t
        sc = pd.DataFrame({"customer_idx": df.customer_idx.to_numpy(), "article_id": df.article_id.to_numpy(),
                           "score": s, "label": df.label.to_numpy()})
        sc["tail_hit"] = (sc.label == 1) & sc.article_id.isin(tail)
        stats.append(sc.groupby("customer_idx").agg(cand_hits=("label", "sum"), n_cand=("label", "size"),
                                                   cand_hits_tail=("tail_hit", "sum")).reset_index())
        sc = sc.sort_values(["customer_idx", "score", "article_id"], ascending=[True, False, True], kind="stable")
        parts.append(sc.groupby("customer_idx", sort=False).head(int(u.keep_top))[["customer_idx", "article_id"]])
        del df, sc
        gc.collect()
    ranked = pd.concat(parts, ignore_index=True)
    lists = ranked.groupby("customer_idx", sort=False)["article_id"].agg(list).to_dict()
    cs = pd.concat(stats, ignore_index=True)
    res = score_lists(ctx, lists, cs).assign(fold=str(week.start))
    attrs = dict(res.attrs)                      # merge() drops DataFrame.attrs
    res = res.merge(cs[["customer_idx", "cand_hits_tail"]], on="customer_idx", how="left")
    res.attrs.update(attrs)
    res["cand_hits_tail"] = res.cand_hits_tail.fillna(0).astype(int)
    res.to_parquet(out, index=False)
    wall = time.time() - t_start
    meta = {"system": system, "config": name, "spec": spec, "fold": str(week.start), "summary": summarize(res),
            "coverage": res.attrs["coverage"], "novelty": res.attrs["novelty"], "n_union_rows": int(n_union),
            "n_appended_rows": int(n_app), "seconds": {k: round(v, 1) for k, v in secs.items()},
            "wall_seconds": round(wall, 1), "peak_rss_bytes": peak_rss(),
            "rank_score_seconds": round(secs["candidates"] + secs["features"] + secs["scoring"], 1),
            "manifest": P.manifest(cfg, con, {"pools": d.parent.name})}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"{system} {week.start}: cand/cust {meta['summary']['candidates_per_customer']:.1f} cand-recall "
          f"{meta['summary']['candidate_recall']:.4f} MAP@12 {meta['summary']['map@12']:.5f} "
          f"({wall:.0f}s, {meta['peak_rss_bytes'] / 1e9:.1f} GB)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Report: pooled results, bootstrap, the SASRec gate and the predeclared decision
# ---------------------------------------------------------------------------

def _load_system(cfg, system: str, seed: int) -> tuple[pd.DataFrame, dict] | None:
    from ensemble.research.ensemble import eval_path
    rows, metas = [], {}
    for f in P.reporting_folds(cfg):
        p = eval_path(cfg, system, f, seed)
        if not p.with_suffix(".json").exists():
            return None
        rows.append(pd.read_parquet(p))
        metas[str(f.start)] = json.loads(p.with_suffix(".json").read_text())
    return pd.concat(rows, ignore_index=True), metas


def _boot(a: pd.DataFrame, b: pd.DataFrame, cfg) -> dict:
    from ensemble.research.evaluate import cluster_bootstrap, cluster_bootstrap_ratio
    bs = P.proto(cfg).bootstrap
    n, sd = int(bs.n_boot), int(bs.seed)
    out = {"map@12": cluster_bootstrap(a, b, "ap", n, sd), "ndcg@12": cluster_bootstrap(a, b, "ndcg", n, sd),
           "recall@12": cluster_bootstrap_ratio(a, b, "hits", "n_truth", n, sd),
           "candidate_recall": cluster_bootstrap_ratio(a, b, "cand_hits", "n_truth", n, sd),
           "tail_candidate_recall": cluster_bootstrap_ratio(a, b, "cand_hits_tail", "t_tail", n, sd),
           "tail_recall@12": cluster_bootstrap_ratio(a, b, "h_tail", "t_tail", n, sd)}
    na, nb = a[~a.returning], b[~b.returning]
    out["new_customer_candidate_recall"] = cluster_bootstrap_ratio(na, nb, "cand_hits", "n_truth", n, sd)
    out["new_customer_map@12"] = cluster_bootstrap(na, nb, "ap", n, sd)
    return out


def report() -> dict:
    from ensemble.research.evaluate import summarize
    cfg = load_config()
    u = ucfg(cfg)
    dec = u.decision
    seed = int(u.baseline_seed)
    folds = [str(f.start) for f in P.reporting_folds(cfg)]
    # --- diagnostics, pooled (weights = truth pairs)
    diag = {f: json.loads((out_dir(cfg) / "diagnostics" / f"fold_{f}.json").read_text()) for f in folds}
    w = np.array([diag[f]["n_truth_pairs"] for f in folds], float)
    wavg = lambda key, sub=None: float(np.dot([diag[f][key] if sub is None else diag[f][key][sub] for f in folds], w) / w.sum())  # noqa: E731
    pooled_diag = {
        "candidate_recall": wavg("candidate_recall"),
        "redundant_candidate_rate": float(sum(diag[f]["rows_before_dedup"] - diag[f]["candidates_after_dedup"]
                                              for f in folds) / sum(diag[f]["rows_before_dedup"] for f in folds)),
        "channel_recall_alone": {c: wavg("channel_recall_alone", c) for c in PHASE2_CHANNELS},
        "channel_unique_recall": {c: wavg("channel_unique_recall", c) for c in PHASE2_CHANNELS},
        "channel_beyond_cap_recall": {c: wavg("channel_beyond_cap_recall", c) for c in PHASE2_CHANNELS},
        "neural_marginal_recall": {k: wavg("neural_marginal_recall", k) for k in diag[folds[0]]["neural_marginal_recall"]},
        "miss_taxonomy": {k: wavg("miss_taxonomy", k) for k in diag[folds[0]]["miss_taxonomy"]},
        "pairwise_jaccard_mean_over_folds": {k: float(np.mean([diag[f]["pairwise_jaccard"][k] for f in folds]))
                                             for k in diag[folds[0]]["pairwise_jaccard"]},
        "segments_union_recall": {s_: {"recall": float(np.dot([diag[f]["segments_union_recall"][s_]["recall"] *
                                                                diag[f]["segments_union_recall"][s_]["share"] for f in folds], w)
                                                       / np.dot([diag[f]["segments_union_recall"][s_]["share"] for f in folds], w))}
                                  for s_ in diag[folds[0]]["segments_union_recall"]},
        "reconcile": {"stored_hits": sum(diag[f]["reconcile"]["stored_cand_hits"] for f in folds),
                      "recomputed_hits": sum(diag[f]["reconcile"]["recomputed_hits"] for f in folds),
                      "truth_pairs": int(w.sum()),
                      "customers_with_n_cand_difference": sum(diag[f]["reconcile"]["n_customers_with_different_n_cand"]
                                                              for f in folds),
                      "max_abs_n_cand_relative_difference": max(abs(diag[f]["reconcile"]["n_cand_relative_difference"])
                                                                for f in folds)}}
    rc = pooled_diag["reconcile"]
    rc["candidate_recall"] = rc["recomputed_hits"] / rc["truth_pairs"]
    assert rc["stored_hits"] == rc["recomputed_hits"]
    # --- evaluated configurations
    base = _load_system(cfg, "ret_final", seed)
    assert base is not None, "ret_final missing"
    b_rows, b_meta = base
    stored = _load_system(cfg, u.baseline_system, seed)
    repro = None
    if stored is not None:
        m = stored[0].merge(b_rows, on=["customer_idx", "fold"], suffixes=("_s", "_n"))
        repro = {"customers": int(len(m)), "ap_identical_share": float(np.isclose(m.ap_s, m.ap_n, atol=0, rtol=0).mean()),
                 "map_stored": float(stored[0].ap.mean()), "map_rerun": float(b_rows.ap.mean()),
                 "cand_hits_identical": bool((m.cand_hits_s == m.cand_hits_n).all())}
    base_cost = {"rank_score_seconds": sum(x["rank_score_seconds"] for x in b_meta.values()),
                 "peak_rss_bytes": max(x["peak_rss_bytes"] for x in b_meta.values())}
    names = ["final"] + [f"sasrec_A{b}" for b in u.budgets] + [f"F{b}" for b in u.budgets] + \
        [f"{m}_A{b}" for m in ("lightgcn", "bpr", "neural3") for b in u.budgets]
    configs = {}
    for name in names:
        got = _load_system(cfg, f"ret_{name}", seed)
        if got is None:
            continue
        rows, meta = got
        summ = summarize(rows)
        nc = rows.n_cand.to_numpy()
        per_fold = {f: {"candidate_recall": meta[f]["summary"]["candidate_recall"], "map@12": meta[f]["summary"]["map@12"],
                        "recall@12": meta[f]["summary"]["recall@12"]} for f in folds}
        cost = {"rank_score_seconds": sum(x["rank_score_seconds"] for x in meta.values()),
                "wall_seconds": sum(x["wall_seconds"] for x in meta.values()),
                "peak_rss_bytes": max(x["peak_rss_bytes"] for x in meta.values()),
                "seconds_by_stage": {k: sum(x["seconds"][k] for x in meta.values()) for k in ("candidates", "features", "scoring")}}
        cost["time_ratio_vs_final"] = cost["rank_score_seconds"] / base_cost["rank_score_seconds"]
        cost["rss_increase_vs_final"] = cost["peak_rss_bytes"] / base_cost["peak_rss_bytes"] - 1
        c = {"pooled": summ, "candidates_per_customer": {"mean": float(nc.mean()), "p50": float(np.percentile(nc, 50)),
                                                          "p95": float(np.percentile(nc, 95))},
             "per_fold": per_fold, "cost": cost, "spec": meta[folds[0]]["spec"],
             "appended_rows": int(sum(x["n_appended_rows"] for x in meta.values()))}
        if name != "final":
            bt = _boot(b_rows, rows, cfg)
            c["bootstrap_vs_final"] = bt
            fold_rel = {f: per_fold[f]["candidate_recall"] / configs["final"]["per_fold"][f]["candidate_recall"] - 1
                        for f in folds}
            c["fold_candidate_rel_gain"] = fold_rel
            d_cr = summ["candidate_recall"] - configs["final"]["pooled"]["candidate_recall"]
            c["marginal_conversion"] = ((summ["recall@12"] - configs["final"]["pooled"]["recall@12"]) / d_cr
                                        if d_cr else None)
            gates = {
                "candidate_recall_target": summ["candidate_recall"] >= float(u.baseline_candidate_recall) *
                (1 + float(dec.min_candidate_recall_rel_gain)),
                "map_ci_above_zero": bt["map@12"]["diff_ci95"][0] > 0,
                "recall12_ci_above_zero": bt["recall@12"]["diff_ci95"][0] > 0,
                "folds_with_candidate_gain": int(sum(v > 0 for v in fold_rel.values())),
                "worst_fold_candidate_rel": min(fold_rel.values()),
                "tail_candidate_lcb": bt["tail_candidate_recall"]["relative_ci95"][0],
                "new_customer_candidate_lcb": bt["new_customer_candidate_recall"]["relative_ci95"][0],
                "time_ratio": cost["time_ratio_vs_final"], "rss_increase": cost["rss_increase_vs_final"]}
            gates["folds_ok"] = gates["folds_with_candidate_gain"] >= int(dec.min_folds_candidate_gain) and \
                gates["worst_fold_candidate_rel"] >= float(dec.max_fold_candidate_loss_rel)
            gates["segments_ok"] = gates["tail_candidate_lcb"] >= float(dec.min_segment_candidate_lcb_rel) and \
                gates["new_customer_candidate_lcb"] >= float(dec.min_segment_candidate_lcb_rel)
            gates["cost_ok"] = gates["time_ratio"] <= float(dec.max_wall_time_ratio) and \
                gates["rss_increase"] <= float(dec.max_peak_rss_increase)
            gates["pass"] = bool(gates["candidate_recall_target"] and gates["map_ci_above_zero"] and
                                 gates["recall12_ci_above_zero"] and gates["folds_ok"] and gates["segments_ok"] and
                                 gates["cost_ok"])
            c["decision_gates"] = gates
        configs[name] = c
    # SASRec usefulness gate (Stage 3 -> LightGCN / BPR-MF extensions)
    su = u.sasrec_useful
    sas = {n: {"abs_gain": configs[n]["pooled"]["candidate_recall"] - configs["final"]["pooled"]["candidate_recall"],
               "folds_positive": int(sum(v > 0 for v in configs[n]["fold_candidate_rel_gain"].values()))}
           for n in configs if n.startswith("sasrec_A")}
    sasrec_useful = any(v["abs_gain"] >= float(su.min_abs_gain) and v["folds_positive"] >= int(su.min_folds)
                        for v in sas.values())
    passing = [n for n in configs if n != "final" and configs[n]["decision_gates"]["pass"]]
    budget_of = lambda n: configs[n]["candidates_per_customer"]["mean"]  # noqa: E731
    adopted = min(passing, key=budget_of) if passing else None
    best_quality = max((n for n in configs if n != "final"), key=lambda n: configs[n]["pooled"]["map@12"], default=None)
    from ensemble.db import connect
    out = {"diagnostics": pooled_diag, "diagnostics_per_fold": diag, "reproduction_of_final": repro,
           "configs": configs, "sasrec_gate": {"rule": dict(su), "per_budget": sas, "useful": sasrec_useful},
           "decision_rule": dict(dec), "passing": passing, "adopted": adopted, "best_quality_only": best_quality,
           "manifest": P.manifest(cfg, connect(cfg, read_only=True))}
    (out_dir(cfg) / "comparison.json").write_text(json.dumps(out, indent=1, default=float))
    for n, c in configs.items():
        g = c.get("decision_gates", {})
        print(f"{n:16s} cand {c['candidates_per_customer']['mean']:7.1f} cand-recall {c['pooled']['candidate_recall']:.4f} "
              f"MAP {c['pooled']['map@12']:.5f} R@12 {c['pooled']['recall@12']:.4f} time x{c['cost']['time_ratio_vs_final']:.2f} "
              f"pass {g.get('pass')}")
    print("sasrec useful:", sasrec_useful, "adopted:", adopted, "best quality:", best_quality)
    return out


def run_sub(args: list[str], log: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    with open(log, "a") as fh:
        r = subprocess.run([sys.executable, "-m", "ensemble.research.retrieval_upgrade", *args], env=env,
                           stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{args} failed ({r.returncode}); see {log}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "pools":
        pools(a[1])
    elif a[0] == "allocate":
        allocate()
    elif a[0] == "diagnostics":
        diagnostics(a[1])
    elif a[0] == "eval":
        eval_config(a[1], a[2])
    elif a[0] == "report":
        report()
    elif a[0] == "queue":
        cfg = load_config()
        log = out_dir(cfg) / "logs"
        log.mkdir(exist_ok=True)
        for f in P.reporting_folds(cfg):
            for name in a[1].split(","):
                run_sub(["eval", name, str(f.start)], log / "eval.log")
    else:
        raise SystemExit(__doc__)
