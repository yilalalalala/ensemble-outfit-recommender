"""Track A ranker retrain for the SASRec-append-to-300 candidates (docs/CLAUDECODE_TRACK_A_RANKER_RETRAIN_PLAN.md, D-045).

  ENSEMBLE_CONFIG=track_a_research python -m ensemble.research.ranker_retrain <stage> [args]

  pools <week>                 channel pools + exact neural lists for a training week (reuses retrieval_upgrade)
  train_matrix <week>          sasrec_A300 training matrix of one label week (positives-only groups, 50% negatives)
  eval_matrix <fold>           sasrec_A300 evaluation matrix of one reporting fold (all candidates, labels kept)
  fit <variant> <fold> <seed>  fit R1/R2/R3 on the four training weeks before the fold; save the model
  score <variant> <fold> <seed>  score the fold's evaluation matrix in its own process (clean time / RSS)
  queue <variants> <seeds>     fit + score every reporting fold, sequentially
  report                       pooled comparison against R0, bootstrap, segments, costs, SHAP, decision

**Distribution matching.** Training and evaluation candidates come from the same function,
``retrieval_upgrade.build_cand`` with the adopted ``sasrec_A300`` spec: the Phase-2 union from
pools built before the week, then the de-duplicated exact SASRec list of the model trained for that
week's cutoff, up to 300 per customer. Training weeks then keep customers with a retrieved purchase
and a deterministic hashed half of the negatives (``ranking.train.downsample_negatives``), exactly as
the current final ranker was trained.

**Provenance (R2).** ``sasrec_rank`` = the candidate's rank in the customer's exact SASRec list over the
eligible catalogue (≤ 1,000, else NULL); ``sasrec_rank_pct`` = that rank / the number of eligible
articles SASRec scored that week; ``src_sasrec_append`` = 1 when the candidate entered only through the
append (no Phase-2 channel proposed it within its cap). All are point-in-time and online-computable.
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
from ensemble.research import retrieval_upgrade as RU
from ensemble.research.ensemble import PHASE2_CHANNELS

PROV = ["sasrec_rank", "sasrec_rank_pct", "src_sasrec_append"]
ID = ("customer_idx", "article_id", "label")
VARIANTS = ("R0", "R1", "R2", "R3")
SPEC_NAME = "sasrec_A300"


def rcfg(cfg) -> Config:
    return cfg.research.ranker_retrain


def cache(cfg) -> Path:
    u = RU.ucfg(cfg)
    key = P.config_hash({"spec": SPEC_NAME, "pools": RU.root(cfg).name, "features": dict(cfg.features),
                         "neg": float(cfg.ranker.neg_sample_rate), "prov": PROV,
                         "protocol": int(P.proto(cfg).protocol_version), "baseline": u.baseline_system})
    d = cfg.path("interim") / "track_a_retrain" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def base_model_path(cfg, fold: Week) -> Path:
    u = RU.ucfg(cfg)
    return cfg.path("interim") / "track_a_research" / "models" / u.baseline_system / f"{fold.start}_s{int(u.baseline_seed)}.txt"


def base_features(cfg) -> list[str]:
    import lightgbm as lgb
    return lgb.Booster(model_file=str(base_model_path(cfg, P.reporting_folds(cfg)[0]))).feature_name()


def variant_features(cfg, variant: str) -> list[str]:
    f = base_features(cfg)
    return f if variant in ("R0", "R1") else f + PROV


def n_scored(cfg, week: Week) -> int:
    d = RU.root(cfg) / str(week.start)
    bench = json.loads((d / "exact_search.json").read_text()) if (d / "exact_search.json").exists() else []
    hit = [b for b in bench if b["model"] == "sasrec"]
    if hit:
        return int(hit[0]["items"])
    # exact_search.json is only written on a fresh deep-list build; fall back to recomputing the count
    from ensemble.db import connect
    from ensemble.research.channels import model_dir
    con = connect(cfg, read_only=True)
    elig = P.eligible(con, week, int(P.proto(cfg).eligible_days))
    items = np.load(model_dir(cfg, "sasrec", week, 0) / "emb.npz")["items"]
    return int(np.isin(items, elig).sum())


def add_provenance(con, df: pd.DataFrame, d: Path, n_items: int) -> pd.DataFrame:
    """Join the point-in-time SASRec rank of every (customer, article) row; derive the R2 features."""
    con.register("_prov_df", df[["customer_idx", "article_id"]])
    r = con.execute(f"""SELECT p.customer_idx, p.article_id, s.rnk AS sasrec_rank
                        FROM _prov_df p LEFT JOIN read_parquet('{d / "deep_sasrec.parquet"}') s
                          ON s.customer_idx = p.customer_idx AND s.article_id = p.article_id""").df()
    con.unregister("_prov_df")
    df = df.merge(r, on=["customer_idx", "article_id"], how="left", validate="one_to_one")
    df["sasrec_rank"] = df["sasrec_rank"].astype(np.float32)
    df["sasrec_rank_pct"] = (df["sasrec_rank"] / float(n_items)).astype(np.float32)
    ranks = df[[f"{c}_rank" for c in PHASE2_CHANNELS]].to_numpy()
    df["src_sasrec_append"] = np.isnan(ranks).all(axis=1).astype(np.float32)
    return df


def keep_training_rows(con, week: Week, rate: float) -> None:
    """Training-week rule on ``cand``: customers with >= 1 retrieved purchase, every positive kept,
    a deterministic hashed share of negatives (the D-014 / D-028 rule, shared with ranking.train)."""
    from ensemble.ranking.train import downsample_negatives
    con.execute("""DELETE FROM cand WHERE customer_idx NOT IN (
                     SELECT DISTINCT customer_idx FROM cand JOIN (
                       SELECT DISTINCT customer_idx, article_id FROM transactions
                       WHERE t_dat BETWEEN ? AND ?) USING (customer_idx, article_id))""", [week.start, week.end])
    downsample_negatives(con, week, rate)


def build_matrix(week: Week, kind: str) -> Path:
    from ensemble.db import connect
    from ensemble.features.track_a import build_features
    from ensemble.ranking.train import feature_groups
    from ensemble.research.channels import add_neural_scores
    cfg = load_config()
    out = cache(cfg) / f"{kind}_{week.start}"
    if (out / "_SUCCESS").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    d = RU.root(cfg) / str(week.start)
    assert (d / "_POOLS_SUCCESS").exists(), f"pools missing for {week.start}"
    spec = RU.config_def(cfg, SPEC_NAME)
    con = connect(cfg, read_only=True)
    t0 = time.time()
    users = con.execute(f"""SELECT DISTINCT customer_idx FROM transactions
                            WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}' ORDER BY 1""") \
        .fetchnumpy()["customer_idx"]
    n_items = n_scored(cfg, week)
    step = int(rcfg(cfg).chunk_customers)
    rows, n_cand = 0, 0
    for i, b in enumerate(range(0, len(users), step)):
        con.execute("CREATE OR REPLACE TEMP TABLE _cu AS SELECT unnest(?::INTEGER[]) AS customer_idx",
                    [users[b:b + step].tolist()])
        info = RU.build_cand(con, d, spec, "_cu")
        n_cand += info["n_union"] + info["n_appended"]
        if kind == "train":
            keep_training_rows(con, week, float(cfg.ranker.neg_sample_rate))
        df = build_features(con, week, True, groups=feature_groups(cfg), cfg=cfg)
        df = add_neural_scores(df, cfg, week, ("bpr", "lightgcn", "sasrec"), 0)
        df = add_provenance(con, df, d, n_items)
        df.to_parquet(out / f"part-{i:03d}.parquet", index=False)
        rows += len(df)
        del df
        gc.collect()
    meta = {"week": str(week.start), "kind": kind, "rows": rows, "candidates_before_filter": n_cand,
            "customers": int(len(users)), "n_scored_items": n_items, "seconds": round(time.time() - t0, 1),
            "peak_rss_bytes": RU.peak_rss()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "_SUCCESS").write_text("")
    print(f"{kind} matrix {week.start}: {rows:,} rows ({meta['seconds']}s, {meta['peak_rss_bytes'] / 1e9:.1f} GB)",
          flush=True)
    return out


# ---------------------------------------------------------------------------
# Fit and score
# ---------------------------------------------------------------------------

def model_path(cfg, variant: str, fold: Week, seed: int) -> Path:
    if variant == "R0":
        return base_model_path(cfg, fold)
    d = cache(cfg) / "models" / variant
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{fold.start}_s{seed}.txt"


def variant_cfg(cfg, variant: str, seed: int) -> Config:
    rc = dict(cfg.ranker)
    rc["seed"] = int(seed)
    if variant == "R3":
        grid = json.loads((out_dir(cfg) / "r3_tuning.json").read_text())
        rc["params"] = {**dict(cfg.ranker.params), **grid["selected"]}
    return Config({**cfg, "ranker": rc})


def fit(variant: str, fold_start: str, seed: int, train_weeks: list[Week] | None = None,
        target: Path | None = None) -> dict:
    from ensemble.ranking.train import TrainingData, _customer_groups
    from ensemble.ranking.train import fit as lgb_fit
    cfg = load_config()
    fold = P.week_of(fold_start)
    mp = target or model_path(cfg, variant, fold, seed)
    if mp.exists() and mp.with_suffix(".json").exists():
        return json.loads(mp.with_suffix(".json").read_text())
    feats = variant_features(cfg, "R2" if variant == "R3" and json.loads(
        (out_dir(cfg) / "r3_tuning.json").read_text())["base_variant"] == "R2" else variant)
    weeks = train_weeks or P.ranker_train_weeks(cfg, fold)
    t0 = time.time()
    data = TrainingData(feats)
    for w in weeks:
        parts = sorted((cache(cfg) / f"train_{w.start}").glob("part-*.parquet"))
        assert parts, f"training matrix missing for {w.start}"
        df = pd.concat([pd.read_parquet(p, columns=["customer_idx", "label", *feats]) for p in parts],
                       ignore_index=True)
        assert df.customer_idx.is_monotonic_increasing, "group boundaries must be contiguous"
        data.X.append(df[feats].to_numpy(dtype=np.float32))
        data.y.append(df["label"].to_numpy(dtype=np.int8))
        data.groups.append(_customer_groups(df.customer_idx.to_numpy()))
        del df
        gc.collect()
    load_s = time.time() - t0
    t1 = time.time()
    booster, features, info = lgb_fit(variant_cfg(cfg, variant, seed), data)
    fit_s = time.time() - t1
    rows = data.rows
    del data
    gc.collect()
    booster.save_model(str(mp))
    meta = {"variant": variant, "fold": str(fold.start), "seed": int(seed), "features": features,
            "n_features": len(features), "n_trees": booster.current_iteration(), "train_rows": int(rows),
            "train_weeks": [str(w.start) for w in weeks], "fit_info": info, "load_seconds": round(load_s, 1),
            "fit_seconds": round(fit_s, 1), "train_seconds": round(load_s + fit_s, 1),
            "peak_rss_bytes": RU.peak_rss(), "model_bytes": mp.stat().st_size}
    mp.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"fit {variant} {fold.start} s{seed}: {rows:,} rows, {meta['n_trees']} trees, "
          f"{meta['train_seconds']:.0f}s, {meta['peak_rss_bytes'] / 1e9:.1f} GB", flush=True)
    return meta


def system_name(variant: str) -> str:
    return f"rr_{variant}"


def score(variant: str, fold_start: str, seed: int) -> Path:
    import lightgbm as lgb

    from ensemble.db import connect
    from ensemble.research.ensemble import eval_path
    from ensemble.research.evaluate import FoldContext, score_lists, summarize
    cfg = load_config()
    fold = P.week_of(fold_start)
    out = eval_path(cfg, system_name(variant), fold, seed)
    if out.with_suffix(".json").exists():
        return out
    booster = lgb.Booster(model_file=str(model_path(cfg, variant, fold, seed)))
    feats = booster.feature_name()
    parts = sorted((cache(cfg) / f"eval_{fold.start}").glob("part-*.parquet"))
    assert parts, f"evaluation matrix missing for {fold.start}"
    con = connect(cfg, read_only=True)
    ctx = FoldContext(con, cfg, fold)
    tail = ctx.eligible - ctx.items["head"]
    t_read = t_pred = 0.0
    keep, stats = [], []
    for p in parts:
        t = time.time()
        df = pd.read_parquet(p, columns=["customer_idx", "article_id", "label", *feats])
        t_read += time.time() - t
        t = time.time()
        s = booster.predict(df[feats].to_numpy(dtype=np.float32), num_threads=8)
        t_pred += time.time() - t
        sc = pd.DataFrame({"customer_idx": df.customer_idx.to_numpy(), "article_id": df.article_id.to_numpy(),
                           "score": s, "label": df.label.to_numpy()})
        sc["tail_hit"] = (sc.label == 1) & sc.article_id.isin(tail)
        stats.append(sc.groupby("customer_idx").agg(cand_hits=("label", "sum"), n_cand=("label", "size"),
                                                   cand_hits_tail=("tail_hit", "sum")).reset_index())
        sc = sc.sort_values(["customer_idx", "score", "article_id"], ascending=[True, False, True], kind="stable")
        keep.append(sc.groupby("customer_idx", sort=False).head(50)[["customer_idx", "article_id"]])
        del df, sc
        gc.collect()
    lists = pd.concat(keep, ignore_index=True).groupby("customer_idx", sort=False)["article_id"].agg(list).to_dict()
    cs = pd.concat(stats, ignore_index=True)
    res = score_lists(ctx, lists, cs).assign(fold=str(fold.start))
    attrs = dict(res.attrs)
    res = res.merge(cs[["customer_idx", "cand_hits_tail"]], on="customer_idx", how="left")
    res.attrs.update(attrs)
    res["cand_hits_tail"] = res.cand_hits_tail.fillna(0).astype(int)
    res.to_parquet(out, index=False)
    meta = {"system": system_name(variant), "variant": variant, "fold": str(fold.start), "seed": int(seed),
            "summary": summarize(res), "coverage": attrs["coverage"], "novelty": attrs["novelty"],
            "read_seconds": round(t_read, 2), "scoring_seconds": round(t_pred, 2),
            "peak_rss_bytes": RU.peak_rss(), "n_trees": booster.current_iteration(), "n_features": len(feats)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"score {variant} {fold.start} s{seed}: MAP@12 {meta['summary']['map@12']:.5f} cand-recall "
          f"{meta['summary']['candidate_recall']:.4f} predict {t_pred:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# Report: the frozen decision rule, applied mechanically
# ---------------------------------------------------------------------------

def _rows(cfg, system: str, seed: int):
    from ensemble.research.ensemble import eval_path
    rows, metas = [], {}
    for f in P.reporting_folds(cfg):
        p = eval_path(cfg, system, f, seed)
        if not p.with_suffix(".json").exists():
            return None
        rows.append(pd.read_parquet(p))
        metas[str(f.start)] = json.loads(p.with_suffix(".json").read_text())
    return pd.concat(rows, ignore_index=True), metas


def _fit_metas(cfg, variant: str, seed: int) -> dict:
    return {str(f.start): json.loads(model_path(cfg, variant, f, seed).with_suffix(".json").read_text())
            for f in P.reporting_folds(cfg) if model_path(cfg, variant, f, seed).with_suffix(".json").exists()}


def compare_to_r0(cfg, r0, rows) -> dict:
    from ensemble.research.evaluate import cluster_bootstrap, cluster_bootstrap_ratio
    bs = P.proto(cfg).bootstrap
    n, sd = int(bs.n_boot), int(bs.seed)
    out = {"map@12": cluster_bootstrap(r0, rows, "ap", n, sd), "ndcg@12": cluster_bootstrap(r0, rows, "ndcg", n, sd),
           "recall@12": cluster_bootstrap_ratio(r0, rows, "hits", "n_truth", n, sd)}
    for seg in ("tail", "repeat", "nonrepeat", "head", "recent"):
        out[f"recall@12_{seg}"] = cluster_bootstrap_ratio(r0, rows, f"h_{seg}", f"t_{seg}", n, sd)
    for name, flag in (("new", False), ("returning", True)):
        out[f"map@12_{name}"] = cluster_bootstrap(r0[r0.returning == flag], rows[rows.returning == flag], "ap", n, sd)
    return out


def report() -> dict:
    from ensemble.research.evaluate import summarize
    cfg = load_config()
    dec = rcfg(cfg).decision
    folds = [str(f.start) for f in P.reporting_folds(cfg)]
    r0, r0m = _rows(cfg, system_name("R0"), 42)
    # R0 through this pipeline must reproduce the D-044 evaluation of sasrec_A300 exactly.
    ref = _rows(cfg, "ret_sasrec_A300", 42)
    m = ref[0].merge(r0, on=["customer_idx", "fold"], suffixes=("_ref", "_r0"))
    repro = {"rows": int(len(m)), "ap_identical": bool((m.ap_ref == m.ap_r0).all()),
             "cand_hits_identical": bool((m.cand_hits_ref == m.cand_hits_r0).all()),
             "map_ref": float(ref[0].ap.mean()), "map_r0": float(r0.ap.mean())}
    base_cost = {"scoring_seconds": sum(x["scoring_seconds"] for x in r0m.values()),
                 "scoring_peak_rss": max(x["peak_rss_bytes"] for x in r0m.values())}
    variants = {}
    for v in ("R1", "R2", "R3"):
        got = _rows(cfg, system_name(v), 42)
        if got is None:
            continue
        rows, meta = got
        fm = _fit_metas(cfg, v, 42)
        summ = summarize(rows)
        b = compare_to_r0(cfg, r0, rows)
        per_fold = {f: {"map@12": meta[f]["summary"]["map@12"], "r0_map@12": r0m[f]["summary"]["map@12"],
                        "candidate_recall": meta[f]["summary"]["candidate_recall"]} for f in folds}
        won = sum(per_fold[f]["map@12"] > per_fold[f]["r0_map@12"] for f in folds)
        cost = {"scoring_seconds": sum(x["scoring_seconds"] for x in meta.values()),
                "scoring_peak_rss": max(x["peak_rss_bytes"] for x in meta.values()),
                "training_seconds_mean": float(np.mean([x["train_seconds"] for x in fm.values()])),
                "training_peak_rss": max(x["peak_rss_bytes"] for x in fm.values()),
                "train_rows_mean": float(np.mean([x["train_rows"] for x in fm.values()])),
                "trees_mean": float(np.mean([x["n_trees"] for x in fm.values()]))}
        cost["scoring_time_increase"] = cost["scoring_seconds"] / base_cost["scoring_seconds"] - 1
        cost["scoring_rss_increase"] = cost["scoring_peak_rss"] / base_cost["scoring_peak_rss"] - 1
        cost["training_time_ratio"] = cost["training_seconds_mean"] / float(dec.baseline_training_seconds)
        segs = {k: b[k]["relative_ci95"][0] for k in ("map@12_new", "map@12_returning", "recall@12_tail",
                                                      "recall@12_repeat", "recall@12_nonrepeat")}
        g = {"map_gain": b["map@12"]["relative"], "map_gain_ok": b["map@12"]["relative"] >= float(dec.min_map_rel_gain),
             "map_ci_above_zero": b["map@12"]["diff_ci95"][0] > 0, "folds_won": int(won),
             "folds_ok": won >= int(dec.min_folds_won),
             "recall12_lcb_ok": b["recall@12"]["diff_ci95"][0] >= 0, "ndcg12_lcb_ok": b["ndcg@12"]["diff_ci95"][0] >= 0,
             "segment_lcbs": segs, "segments_ok": all(x >= float(dec.min_segment_lcb_rel) for x in segs.values()),
             "candidate_recall_identical": bool((rows.sort_values(["fold", "customer_idx"]).cand_hits.to_numpy() ==
                                                 r0.sort_values(["fold", "customer_idx"]).cand_hits.to_numpy()).all()),
             "online_compatible": True,
             "cost_ok": cost["scoring_time_increase"] <= float(dec.max_scoring_time_increase) and
             cost["scoring_rss_increase"] <= float(dec.max_scoring_rss_increase) and
             cost["training_time_ratio"] <= float(dec.max_training_time_ratio)}
        g["pass"] = bool(g["map_gain_ok"] and g["map_ci_above_zero"] and g["folds_ok"] and g["recall12_lcb_ok"] and
                         g["ndcg12_lcb_ok"] and g["segments_ok"] and g["candidate_recall_identical"] and g["cost_ok"])
        importance = {}
        for f in folds:
            import lightgbm as lgb
            bst = lgb.Booster(model_file=str(model_path(cfg, v, P.week_of(f), 42)))
            gain = bst.feature_importance("gain")
            for name, val in zip(bst.feature_name(), gain / gain.sum()):
                importance[name] = importance.get(name, 0.0) + val / len(folds)
        variants[v] = {"pooled": summ, "bootstrap_vs_r0": b, "per_fold": per_fold, "cost": cost, "gates": g,
                       "importance_top": dict(sorted(importance.items(), key=lambda x: -x[1])[:25]),
                       "prov_gain_share": {p_: importance.get(p_, 0.0) for p_ in PROV}}
    seeds = {}
    for v in variants:
        vals = {}
        for sd in rcfg(cfg).seeds:
            got = _rows(cfg, system_name(v), int(sd))
            if got is not None:
                vals[str(sd)] = float(got[0].ap.mean())
        if len(vals) > 1:
            seeds[v] = {"per_seed_map@12": vals, "mean": float(np.mean(list(vals.values()))),
                        "sd": float(np.std(list(vals.values()), ddof=1))}
    passing = [v for v in ("R1", "R2", "R3") if v in variants and variants[v]["gates"]["pass"]]
    adopted = passing[0] if passing else None
    best = max(variants, key=lambda v: variants[v]["pooled"]["map@12"]) if variants else None
    trig = rcfg(cfg).r3_trigger
    cand = [v for v in ("R1", "R2") if v in variants]
    better = max(cand, key=lambda v: variants[v]["pooled"]["map@12"]) if cand else None
    r3 = None
    if better:
        g = variants[better]["gates"]
        others = all(g[k] for k in ("map_ci_above_zero", "folds_ok", "recall12_lcb_ok", "ndcg12_lcb_ok",
                                    "segments_ok", "candidate_recall_identical", "cost_ok"))
        r3 = {"better_of_r1_r2": better, "map_gain": g["map_gain"], "other_gates_pass": others,
              "triggered": bool(others and float(trig.min_map_rel_gain) <= g["map_gain"] < float(trig.max_map_rel_gain))}
    out = {"reproduction_of_r0": repro, "r0": {"pooled": summarize(r0), "cost": base_cost}, "variants": variants,
           "seeds": seeds, "decision_rule": dict(dec), "passing": passing, "adopted": adopted,
           "best_quality_only": best, "r3_trigger": r3}
    (out_dir(cfg) / "comparison.json").write_text(json.dumps(out, indent=1, default=float))
    for v, x in variants.items():
        g = x["gates"]
        print(f"{v}: MAP {x['pooled']['map@12']:.5f} ({g['map_gain'] * 100:+.2f}%) folds {g['folds_won']}/6 "
              f"pass {g['pass']}", flush=True)
    print("adopted", adopted, "best", best, "R3", r3)
    return out


def out_dir(cfg) -> Path:
    d = cfg.path("reports") / "track_a_retrain"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_sub(args: list[str]) -> None:
    cfg = load_config()
    log = out_dir(cfg) / "logs"
    log.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    with open(log / "runs.log", "a") as fh:
        r = subprocess.run([sys.executable, "-m", "ensemble.research.ranker_retrain", *args], env=env,
                           stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{args} failed ({r.returncode}); see {log / 'runs.log'}")


def queue(variants: list[str], seeds: list[int]) -> None:
    cfg = load_config()
    for seed in seeds:
        for v in variants:
            for f in P.reporting_folds(cfg):
                if v != "R0":
                    run_sub(["fit", v, str(f.start), str(seed)])
                run_sub(["score", v, str(f.start), str(seed)])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "pools":
        RU.pools(a[1])
    elif a[0] in ("train_matrix", "eval_matrix"):
        build_matrix(P.week_of(a[1]), a[0].split("_")[0])
    elif a[0] == "fit":
        fit(a[1], a[2], int(a[3]))
    elif a[0] == "score":
        score(a[1], a[2], int(a[3]))
    elif a[0] == "report":
        report()
    elif a[0] == "queue":
        queue(a[1].split(","), [int(x) for x in a[2].split(",")])
    else:
        raise SystemExit(__doc__)
