"""Rolling temporal backtest for Track B (Round 3).

  python -m ensemble.completion.backtest [n_folds] [tag]

Each fold is one label week. For fold *w* everything — basket mining, the
eligible catalogue, the two towers, every feature, and the ranker's training
labels — comes from weeks strictly before *w*; the ranker trains on the
``train_weeks`` label weeks before *w*. Folds end at the validation week, so the
test week is never touched here (``mode="test"`` runs the single final
evaluation).

Systems compared on every fold:

``slot_popularity``      per-slot recent popularity (the mandated reference, D-004)
``assoc_npmi``           Round-1 association rules (support >= 3, lift > 1) + popularity backfill
``shipped_rrf_hybrid``   the Round-1 shipped model: RRF(w=0.5) of association and the two towers
``rrf_all_sources``      fixed equal-weight RRF over the full candidate union (fair-retrieval control)
``lgbm_compatibility``   the learned ranker without any customer feature
``lgbm_personalized``    the learned ranker with customer affinity features (Phase 3)

Per-week feature matrices are cached under ``data/interim/track_b/<hash>/`` so
ranker-side ablations do not rebuild candidates or retrain the towers; the hash
covers every setting that changes the matrices.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from ensemble import tracking
from ensemble.completion import candidates as C
from ensemble.completion import features as FE
from ensemble.completion import fusion as FU
from ensemble.completion import metrics as MET
from ensemble.completion import pipeline as PL
from ensemble.completion import protocol as P
from ensemble.completion import ranker as R
from ensemble.completion.pipeline import cache_dir, cache_key
from ensemble.config import ROOT, load_config
from ensemble.data.splits import Week, load_splits
from ensemble.db import connect
from ensemble.completion.fusion import BASELINES, RRF_C
from ensemble.evaluation.metrics import relative_lift

# Equal-weight RRF control over the union: (column holding the source rank, weight).
RRF_SOURCES = (("a_rank_co", 1.0), ("a_rank_npmi", 1.0), ("s_rank_co", 1.0), ("s_rank_npmi", 1.0),
               ("tt_rank", 1.0), ("ttc_rank", 1.0), ("cand_slot_pop_rank", 1.0))


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ------------------------------------------------------------------ per-week artifacts


def tower_artifacts(con, cfg, week: Week, log=print) -> dict:
    """Load this week's cached tower artifacts, building them in a subprocess if absent."""
    paths = PL.tower_paths(cfg, week)
    if not all(p.exists() for p in paths.values()):
        log(f"  training towers for {week.start} in a subprocess (PyTorch / LightGBM OpenMP clash)")
        subprocess.run([sys.executable, "-m", "ensemble.completion.towers", str(week.start)],
                       check=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    tt = pd.read_parquet(paths["tt"])
    ttc = pd.read_parquet(paths["ttc"])
    return {"tt": tt if len(tt) else None, "ttc": ttc if len(ttc) else None,
            "key_sims": pd.read_parquet(paths["key_sims"]),
            "info": json.loads(paths["info"].read_text())}


def build_week(con, cfg, week: Week, log=print) -> dict:
    """Mine, retrieve and register everything needed for ``week`` (nothing after its cutoff)."""
    t0 = time.time()
    prep = PL.prepare_sql(con, cfg, week)
    tow = tower_artifacts(con, cfg, week, log=log)
    counts = C.build_sources(con, cfg, tow["tt"], tow["ttc"])
    FE.build_context(con, week, cfg, prep["uni"], prep["q"], tow["key_sims"])
    base = FU.baseline_lists(con, cfg, prep["uni"], tow["tt"], tow["ttc"])
    out = {"week": week, "uni": prep["uni"], "q": prep["q"], "funnel": prep["funnel"],
           "basket_audit": prep["basket_audit"], "n_truth_dropped": prep["n_truth_dropped"],
           "source_rows": counts, "tower_info": tow["info"], "baselines": base,
           "n_candidate_keys": int(len(tow["key_sims"])), "prepare_seconds": round(time.time() - t0, 1)}
    del tow
    gc.collect()
    return out


# ------------------------------------------------------------------ baselines


def rrf_from_ranks(df: pd.DataFrame) -> np.ndarray:
    """Equal-weight RRF score computed from the union's per-source rank columns."""
    s = np.zeros(len(df), dtype=np.float64)
    for col, w in RRF_SOURCES:
        if col in df.columns:
            r = df[col].to_numpy(dtype=np.float64)
            ok = np.isfinite(r)
            s[ok] += w / (RRF_C + r[ok])
    return s


def pooled_rerank(chunk: pd.DataFrame, pool_score: np.ndarray, rank_score: np.ndarray, pool: int) -> dict:
    """Per qid: keep the ``pool`` best candidates by ``pool_score`` (ties -> smaller article id),
    then order them by ``rank_score``. Returns the whole re-ordered pool per qid."""
    qid, art = chunk.qid.to_numpy(), chunk.article_id.to_numpy()
    order = np.lexsort((art, -pool_score, qid))
    q_s = qid[order]
    starts = np.r_[0, np.flatnonzero(q_s[1:] != q_s[:-1]) + 1]
    pos = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = order[pos < pool]
    sub = chunk.iloc[keep][["qid", "article_id"]]
    return R.top_k(sub, rank_score[keep], pool)


# ------------------------------------------------------------------ matrices


def training_matrix(con, cfg, art: dict, log=print) -> pd.DataFrame:
    """Sampled, negative-downsampled labelled matrix for one week (cached on disk)."""
    week = art["week"]
    path = cache_dir(cfg) / f"train_{week.start}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    b = cfg.track_b
    sel = P.sample_queries(art["q"], int(b.train_queries))
    con.register("_tb_sel_df", sel[["qid"]])
    con.execute("CREATE OR REPLACE TEMP TABLE _tb_sel AS SELECT * FROM _tb_sel_df")
    t = time.time()
    step = int(b.eval_chunk_queries)
    hi = int(art["q"].qid.max()) + 1
    parts = [FE.chunk_features(con, cfg, lo, lo + step, with_labels=True, personalize=True,
                               neg_rate=float(b.neg_sample_rate), sel_table="_tb_sel")
             for lo in range(0, hi, step)]
    df = pd.concat([x for x in parts if len(x)], ignore_index=True)
    del parts
    gc.collect()
    df.to_parquet(path, index=False)
    log(f"  train matrix {week.start}: {len(df):,} rows, {df.qid.nunique():,} groups, "
        f"pos rate {df.label.mean():.4f} ({time.time() - t:.0f}s)")
    return df


def eval_chunks(con, cfg, art: dict):
    """Yield labelled feature chunks covering every query of the fold week."""
    q = art["q"]
    step = int(cfg.track_b.eval_chunk_queries)
    hi = int(q.qid.max()) + 1
    for lo in range(0, hi, step):
        df = FE.chunk_features(con, cfg, lo, lo + step, with_labels=True, personalize=True)
        if len(df):
            yield df


# ------------------------------------------------------------------ folds


def run_fold(con, cfg, art: dict, train_frames: list[pd.DataFrame], log=print) -> dict:
    q, uni = art["q"], art["uni"]
    k = int(cfg.track_b.k)
    attrs = con.execute("SELECT article_id, product_type_no, product_code FROM articles").df()
    seg = MET.segments(uni, cfg)
    # Phase 3 segment split: customers with any pre-cutoff purchase vs the no-history path.
    with_history = con.execute("SELECT customer_idx FROM _tb_cust").fetchnumpy()["customer_idx"]
    returning = np.isin(q.customer_idx.to_numpy(), with_history)

    recs = FU.baseline_recs(q, art["baselines"], cfg)
    fit_info: dict = {}
    boosters: dict = {}
    if train_frames:
        all_features = FE.feature_names(train_frames[0])
        group_sizes = FE.check_groups(all_features)   # fail loudly on a renamed feature
        fit_info["_feature_groups"] = {"n_features": len(all_features), "removed_by_group": group_sizes}
        subsets = FE.ablation_subsets(cfg, all_features)
        for name, feats in subsets.items():
            t = time.time()
            booster, info = R.fit(cfg, train_frames, feats, log=log)
            info["fit_seconds"] = round(time.time() - t, 1)
            info["n_features"] = len(feats)
            imp = R.importance(booster, feats)
            info["top_features"] = [{"feature": f, "gain_share": round(g, 5)} for f, g in imp[:25]]
            info["group_gain_share"] = R.group_gain(imp)
            boosters[name] = (booster, feats)
            fit_info[name] = info
            log(f"  {name}: {info['train_rows']:,} rows, {len(feats)} features, "
                f"{booster.current_iteration()} trees ({info['fit_seconds']:.0f}s)")
        for name in boosters:
            recs[name] = [None] * len(q)

    # Phase 7: the serving model's deeper pool, kept so the diversity rules can be
    # re-ranked out of it without a second scoring pass.
    rules = cfg.track_b.get("serving_rules") or {}
    serve_model = rules.get("model") if rules.get("model") in boosters else None
    pool_k = max(k, int(rules.get("pool", 4 * k))) if serve_model else k
    pools: list = [None] * len(q) if serve_model else []

    # Request-time serving (D-041): the personalized ranker re-orders only the compatibility
    # ranker's top-P pool per (anchor, target slot), because serving stores a bounded pool per key.
    sp = cfg.track_b.get("serving_pools") or {}
    sp_on = bool(sp) and sp.get("model") in boosters and sp.get("pool_model") in boosters
    sp_pools = [int(x) for x in sp.get("pools", [])] if sp_on else []
    sp_lists: dict = {P: [None] * len(q) for P in sp_pools}

    t_eval = time.time()
    n_rows, n_hits = 0, 0
    rrf_union: list = [None] * len(q)
    pos_of = {int(v): i for i, v in enumerate(q.qid.to_numpy())}
    for chunk in eval_chunks(con, cfg, art):
        n_rows += len(chunk)
        n_hits += int(chunk.label.sum())
        scores = {}
        for name, (booster, feats) in boosters.items():
            score = booster.predict(chunk[feats], num_threads=8)
            scores[name] = score
            take = pool_k if name == serve_model else k
            for qid, arr in R.top_k(chunk, score, take).items():
                i = pos_of[qid]
                recs[name][i] = arr[:k]
                if name == serve_model:
                    pools[i] = arr
        for P_ in sp_pools:
            for qid, arr in pooled_rerank(chunk, scores[sp["pool_model"]], scores[sp["model"]], P_).items():
                sp_lists[P_][pos_of[qid]] = arr
        for qid, arr in R.top_k(chunk, rrf_from_ranks(chunk), k).items():
            rrf_union[pos_of[qid]] = arr
        del chunk
        gc.collect()
    recs["rrf_all_sources_union"] = [a if a is not None else np.empty(0, dtype=np.int64) for a in rrf_union]
    for name in boosters:
        recs[name] = [a if a is not None else np.empty(0, dtype=np.int64) for a in recs[name]]
    if sp_on:
        ptype_ = dict(zip(attrs.article_id.to_numpy(), attrs.product_type_no.to_numpy()))
        pcode_ = dict(zip(attrs.article_id.to_numpy(), attrs.product_code.to_numpy()))
        empty = np.empty(0, dtype=np.int64)
        rules_ = dict(sp.get("rules") or {})
        for P_ in sp_pools:
            lists = [a if a is not None else empty for a in sp_lists[P_]]
            recs[f"{sp['model']}@pool{P_}"] = [a[:k] for a in lists]
            if rules_:
                recs[f"{sp['model']}@pool{P_}+shipped"] = [FU.apply_diversity(a, k, ptype_, pcode_, **rules_)
                                                          for a in lists]
        del sp_lists
        gc.collect()
    if serve_model:
        ptype = dict(zip(attrs.article_id.to_numpy(), attrs.product_type_no.to_numpy()))
        pcode = dict(zip(attrs.article_id.to_numpy(), attrs.product_code.to_numpy()))
        empty = np.empty(0, dtype=np.int64)
        for variant, opts in (rules.get("variants") or {}).items():
            recs[f"{serve_model}+{variant}"] = [
                FU.apply_diversity(pl, k, ptype, pcode, **dict(opts)) if pl is not None else empty
                for pl in pools]
        del pools
        gc.collect()

    n_truth = int(sum(len(t) for t in q.truth.values))
    out: dict = {"metrics": {}, "fit": fit_info, "n_eval_rows": int(n_rows),
                 "candidates_per_query": n_rows / max(1, len(q)),
                 "union_recall": n_hits / n_truth if n_truth else float("nan"),
                 "n_truth_pairs": n_truth, "eval_seconds": round(time.time() - t_eval, 1)}
    per_query: dict = {}
    for name, r in recs.items():
        m = MET.evaluate(r, q, uni, cfg, attrs=attrs, seg=seg)
        pq = m.pop("_per_query")
        for label, mask in (("returning", returning), ("new_customer", ~returning)):
            m[f"recall@{k}_{label}"] = float(pq[f"recall@{k}"][mask].mean()) if mask.any() else float("nan")
            m[f"ndcg@{k}_{label}"] = float(pq[f"ndcg@{k}"][mask].mean()) if mask.any() else float("nan")
            m[f"n_queries_{label}"] = int(mask.sum())
        per_query[name] = pq
        out["metrics"][name] = m
    p0 = out["metrics"]["slot_popularity"][f"recall@{k}"]
    h0 = out["metrics"]["shipped_rrf_hybrid"][f"recall@{k}"]
    for name, m in out["metrics"].items():
        m[f"relative_lift_recall@{k}_vs_popularity"] = relative_lift(m[f"recall@{k}"], p0)
        m[f"relative_lift_recall@{k}_vs_shipped"] = relative_lift(m[f"recall@{k}"], h0)
        m[f"relative_lift_ndcg@{k}_vs_shipped"] = relative_lift(
            m[f"ndcg@{k}"], out["metrics"]["shipped_rrf_hybrid"][f"ndcg@{k}"])
    bs = cfg.track_b.bootstrap
    cust = q.customer_idx.to_numpy()
    out["bootstrap_vs_shipped"] = {
        name: {metric: MET.cluster_bootstrap(cust, per_query["shipped_rrf_hybrid"][metric],
                                             per_query[name][metric], int(bs["n_boot"]), int(bs["seed"]))
               for metric in (f"recall@{k}", f"ndcg@{k}")}
        for name in recs if name != "shipped_rrf_hybrid"}
    out["per_query"] = {name: {m: v.astype(np.float32) for m, v in d.items()} for name, d in per_query.items()}
    return out


def _ms(vals) -> dict:
    vals = [v for v in vals if v == v]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "per_fold": []}
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "per_fold": [round(v, 6) for v in vals]}


def summarise(folds: dict, cfg) -> dict:
    k = int(cfg.track_b.k)
    names = list(next(iter(folds.values()))["metrics"])
    first = next(iter(folds.values()))["metrics"][names[0]]
    slot_keys = sorted(kk for kk in first if kk.startswith(f"recall@{k}_slot_"))
    keys = [f"recall@{k}", "recall@5", f"ndcg@{k}", f"recall@{k}_tail", f"recall@{k}_new_item",
            f"recall@{k}_jewellery", f"recall@{k}_returning", f"recall@{k}_new_customer",
            f"catalog_coverage@{k}", "novelty", f"diversity_product_type@{k}",
            f"diversity_product_code@{k}", f"relative_lift_recall@{k}_vs_popularity",
            f"relative_lift_recall@{k}_vs_shipped", f"relative_lift_ndcg@{k}_vs_shipped",
            "mean_list_length", *slot_keys]
    out: dict = {"systems": {}}
    for name in names:
        out["systems"][name] = {kk: _ms([f["metrics"][name].get(kk) for f in folds.values()]) for kk in keys}
    out["wins_vs_shipped"] = {
        name: sum(1 for f in folds.values()
                  if f["metrics"][name][f"recall@{k}"] > f["metrics"]["shipped_rrf_hybrid"][f"recall@{k}"])
        for name in names if name != "shipped_rrf_hybrid"}
    out["n_folds"] = len(folds)
    out["candidates_per_query"] = _ms([f["candidates_per_query"] for f in folds.values()])
    out["union_recall"] = _ms([f["union_recall"] for f in folds.values()])
    return out


def pooled_bootstrap(folds: dict, cfg, baseline: str = "shipped_rrf_hybrid") -> dict:
    """Customer-cluster bootstrap pooled over folds (customers are nested in folds)."""
    k = int(cfg.track_b.k)
    bs = cfg.track_b.bootstrap
    names = [n for n in next(iter(folds.values()))["per_query"] if n != baseline]
    out = {}
    for name in names:
        res = {}
        for metric in (f"recall@{k}", f"ndcg@{k}"):
            cust, a, b = [], [], []
            for i, f in enumerate(folds.values()):
                cust.append(f["_customer_idx"] + i * 10_000_000)
                a.append(f["per_query"][baseline][metric])
                b.append(f["per_query"][name][metric])
            res[metric] = MET.cluster_bootstrap(np.concatenate(cust), np.concatenate(a), np.concatenate(b),
                                                int(bs["n_boot"]), int(bs["seed"]))
        out[name] = res
    return out


def run(n_folds: int | None = None, tag: str = "", mode: str = "val") -> dict:
    cfg = load_config()
    b = cfg.track_b
    n_folds = int(b.folds) if n_folds is None else int(n_folds)
    train_weeks = int(b.train_weeks)
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    if mode == "test":
        weeks = [splits.test.shift(-k) for k in range(train_weeks, -1, -1)]
        fold_start = train_weeks
    else:
        weeks = P.folds(splits, n_folds, train_weeks)
        fold_start = train_weeks
    print(f"Track B rolling backtest ({mode}): weeks {weeks[0].start}..{weeks[-1].start}, "
          f"{len(weeks) - fold_start} folds, cache {cache_key(cfg)}", flush=True)
    t0 = time.time()
    cached: dict[str, pd.DataFrame] = {}
    folds: dict[str, dict] = {}
    week_meta: dict[str, dict] = {}
    for i, week in enumerate(weeks):
        print(f"[{week.start}] preparing", flush=True)
        art = build_week(con, cfg, week, log=lambda s: print(s, flush=True))
        print(f"  {len(art['q']):,} queries, {len(art['uni']):,} eligible items, "
              f"towers best epoch {art['tower_info']['best_epoch']} "
              f"(holdout R@12 {art['tower_info']['best_holdout_recall@12']:.4f}), "
              f"{art['prepare_seconds']:.0f}s", flush=True)
        week_meta[str(week.start)] = {
            "funnel": art["funnel"], "basket_audit": art["basket_audit"], "tower": art["tower_info"],
            "n_queries": int(len(art["q"])), "n_eligible_items": int(len(art["uni"])),
            "n_truth_dropped": art["n_truth_dropped"], "source_rows": art["source_rows"],
            "n_candidate_keys": art["n_candidate_keys"], "prepare_seconds": art["prepare_seconds"]}
        if i < len(weeks) - 1:
            cached[str(week.start)] = training_matrix(con, cfg, art, log=lambda s: print(s, flush=True))
        if i >= fold_start:
            frames = [cached[str(weeks[j].start)] for j in range(i - train_weeks, i)]
            res = run_fold(con, cfg, art, frames, log=lambda s: print(s, flush=True))
            res["_customer_idx"] = art["q"].customer_idx.to_numpy()
            folds[str(week.start)] = res
            k = int(b.k)
            print(f"    candidate union: recall {res['union_recall']:.4f} at "
                  f"{res['candidates_per_query']:.0f} candidates/query ({res['n_eval_rows']:,} rows)", flush=True)
            for name, m in res["metrics"].items():
                print(f"    {name:26s} R@{k}={m[f'recall@{k}']:.4f} R@5={m['recall@5']:.4f} "
                      f"NDCG={m[f'ndcg@{k}']:.4f} vs_pop={m[f'relative_lift_recall@{k}_vs_popularity']:+.1%} "
                      f"vs_shipped={m[f'relative_lift_recall@{k}_vs_shipped']:+.1%} "
                      f"tail={m[f'recall@{k}_tail']:.4f} cov={m[f'catalog_coverage@{k}']:.3f}", flush=True)
        for key in ("towers", "uni", "q", "baselines"):
            art.pop(key, None)
        del art
        gc.collect()
        # Keep only the training matrices still needed by a later fold.
        for wk in list(cached):
            if wk < str(weeks[max(0, i + 1 - train_weeks)].start):
                del cached[wk]
        gc.collect()
    summary = summarise(folds, cfg)
    summary["pooled_bootstrap_vs_shipped"] = pooled_bootstrap(folds, cfg)
    summary["total_seconds"] = round(time.time() - t0, 1)
    summary["peak_memory_mb"] = round(peak_mb(), 1)
    out = {"mode": mode, "weeks": [str(w.start) for w in weeks], "folds": [str(w.start) for w in weeks[fold_start:]],
           "cache_key": cache_key(cfg), "week_meta": week_meta,
           "per_fold": {w: {kk: v for kk, v in f.items() if kk not in ("per_query", "_customer_idx")}
                        for w, f in folds.items()},
           "summary": summary, "manifest": P.manifest(cfg, con, {"mode": mode, "tag": tag})}
    d = cfg.path("reports") / "track_b_round3"
    d.mkdir(parents=True, exist_ok=True)
    name = f"backtest_{mode}{('_' + tag) if tag else ''}"
    (d / f"{name}.json").write_text(json.dumps(out, indent=2, default=str))
    _save_per_query(cfg, name, folds)
    with tracking.run("track_b", f"round3/{name}", {"folds": n_folds, "cache_key": cache_key(cfg),
                                                    "candidates": dict(b.candidates)}):
        for system, mm in summary["systems"].items():
            tracking.log_metrics({kk: v["mean"] for kk, v in mm.items()}, prefix=f"{system}.")
    print(f"\n== summary ({summary['total_seconds']:.0f}s, peak {summary['peak_memory_mb']:.0f} MB) ==")
    for system, mm in summary["systems"].items():
        kk = f"recall@{int(b.k)}"
        print(f"{system:26s} R@12 {mm[kk]['mean']:.4f}±{mm[kk]['std']:.4f} "
              f"NDCG {mm[f'ndcg@{int(b.k)}']['mean']:.4f} "
              f"vs_shipped {mm[f'relative_lift_recall@{int(b.k)}_vs_shipped']['mean']:+.2%} "
              f"wins {summary['wins_vs_shipped'].get(system, '-')}/{summary['n_folds']}")
    return out


def _save_per_query(cfg, name: str, folds: dict) -> None:
    d = cfg.path("interim") / "track_b" / "per_query"
    d.mkdir(parents=True, exist_ok=True)
    for week, f in folds.items():
        frame = pd.DataFrame({"customer_idx": f["_customer_idx"]})
        for system, metrics in f["per_query"].items():
            for metric, v in metrics.items():
                frame[f"{system}|{metric}"] = v
        frame.to_parquet(d / f"{name}_{week}.parquet", index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = "val"
    if args and args[0] in ("val", "test"):
        mode = args.pop(0)
    run(int(args[0]) if args else None, args[1] if len(args) > 1 else "", mode=mode)
