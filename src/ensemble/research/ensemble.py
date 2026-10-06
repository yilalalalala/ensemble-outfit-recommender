"""The Track A ensemble on the frozen research protocol: superset matrices, exact derivation of
each configuration, refitted ablations, seeds.

  ENSEMBLE_CONFIG=track_a_research python -m ensemble.research.ensemble build <week> <train|eval>
  ENSEMBLE_CONFIG=track_a_research python -m ensemble.research.ensemble fit <config> <fold> <seed> [ablation]

**Superset matrix.** For every label week one candidate table is built with every channel any
configuration uses, at the largest cap any configuration gives it, and the full feature set
(Phase-2 features + ``nn_<model>_dot``). Training weeks keep customers with >= 1 retrieved
purchase and a deterministic 50% of negatives (D-014, D-028); evaluation weeks keep every
candidate of every buyer, with the label stored for scoring only.

**Derivation.** A configuration is a set of channel caps and neural feature models. A row is
kept if at least one of the configuration's channels ranked it within that channel's cap;
provenance outside a cap is nulled; ``n_channels`` is recomputed; training weeks re-apply the
positives-only rule. Because channel ranks, feature values and the downsampling hash are all
per (customer, article), this reproduces a direct build of the configuration exactly
(``test_derivation_matches_direct_build``), so one build serves every configuration,
ablation and seed.

**Ablations** remove a named feature group (and, for channel signals, the candidates only
those channels proposed) and **refit** the ranker; nothing is zeroed in a fitted model.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.config import Config, load_config
from ensemble.data.splits import Week
from ensemble.research import protocol as P

PHASE2_CHANNELS = ("repeat", "pop", "pop_age", "cf", "variant", "dept_pop", "section_pop", "covis", "new_arrival_pers")
NEURAL = ("bpr", "lightgcn", "sasrec")
RECENCY_AFFINITY = [f"ca_w_{n}" for n in ("prod", "type", "colour", "igroup", "ggroup")] + \
    [f"ca_wshare_{n}" for n in ("prod", "type", "colour", "igroup", "ggroup")] + \
    ["ca_days_since_type", "ca_days_since_dept", "ca_wshare_dept", "ca_w_article"]
SHORT_VELOCITY = ["a_n_1d", "a_n_3d", "a_mean_price_1w", "a_trend_3d_1w", "a_price_change_1w_4w"]
REPEAT = ["repeat_score", "repeat_rank", "ca_n_article_life", "ca_days_since_article", "ca_w_article"]

# Each ablation: (channels whose candidates and provenance are removed, feature columns removed).
ABLATIONS = {
    "collaborative": (("bpr", "lightgcn"), ["nn_bpr_dot", "nn_lightgcn_dot"]),
    "sequential": (("sasrec",), ["nn_sasrec_dot"]),
    "repeat": ((), REPEAT),
    "recency_velocity": ((), RECENCY_AFFINITY + SHORT_VELOCITY),
    "provenance": ((), "PROVENANCE"),
}


def ecfg(cfg) -> Config:
    return cfg.research.ensemble


def config_spec(cfg, name: str) -> dict:
    c = dict(ecfg(cfg).configs[name])
    return {"caps": {k: int(v) for k, v in dict(c["caps"]).items() if int(v) > 0},
            "neural_features": list(c.get("neural_features") or [])}


def superset(cfg) -> dict:
    caps: dict[str, int] = {}
    feats: set[str] = set()
    for name in ecfg(cfg).configs:
        s = config_spec(cfg, name)
        for ch, k in s["caps"].items():
            caps[ch] = max(caps.get(ch, 0), k)
        feats |= set(s["neural_features"])
    return {"caps": caps, "neural_features": sorted(feats)}


def superset_dir(cfg) -> Path:
    sup = superset(cfg)
    r = {k: v for k, v in dict(cfg.retrieval).items() if not k.endswith("_k") and k != "channels"}
    models = {m: dict(cfg.research.models[m]) for m in sup["neural_features"] or ()}
    models.update({m: dict(cfg.research.models[m]) for m in sup["caps"] if m in NEURAL})
    key = P.config_hash({"superset": sup, "retrieval": r, "features": dict(cfg.features), "models": models,
                         "neural_seed": int(ecfg(cfg).neural_seed), "ranker_neg": cfg.ranker.neg_sample_rate,
                         "protocol_version": int(P.proto(cfg).protocol_version)})
    d = cfg.path("interim") / "track_a_research" / "matrices" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _retrieval_cfg(cfg) -> Config:
    sup = superset(cfg)
    r = dict(cfg.retrieval)
    for k in list(r):
        if k.endswith("_k"):
            r[k] = 0
    r["channels"] = list(sup["caps"])
    r.update({f"{c}_k": k for c, k in sup["caps"].items()})
    r["neural_seed"] = int(ecfg(cfg).neural_seed)
    return Config(r)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _frame(con, cfg, week: Week, users_sql: str, train: bool) -> pd.DataFrame:
    from ensemble.candidates.retrieval import build_candidates
    from ensemble.features.track_a import build_features
    from ensemble.ranking.train import downsample_negatives, feature_groups
    from ensemble.research.channels import add_neural_scores
    con.execute(f"CREATE OR REPLACE TEMP TABLE _users AS {users_sql}")
    build_candidates(con, week, _retrieval_cfg(cfg))
    if train:
        con.execute("""DELETE FROM cand WHERE customer_idx NOT IN (
                         SELECT DISTINCT customer_idx FROM cand JOIN (
                           SELECT DISTINCT customer_idx, article_id FROM transactions
                           WHERE t_dat BETWEEN ? AND ?) USING (customer_idx, article_id))""", [week.start, week.end])
        downsample_negatives(con, week, float(cfg.ranker.neg_sample_rate))
    df = build_features(con, week, True, groups=feature_groups(cfg), cfg=cfg)
    return add_neural_scores(df, cfg, week, superset(cfg)["neural_features"], int(ecfg(cfg).neural_seed))


def build(week: Week, kind: str) -> Path:
    from ensemble.db import connect
    cfg = load_config()
    d = superset_dir(cfg)
    out = d / (f"train_{week.start}.parquet" if kind == "train" else f"eval_{week.start}")
    done = out if kind == "train" else out / "_SUCCESS"
    if done.exists():
        return out
    con = connect(cfg, read_only=True)
    t0 = time.time()
    if kind == "train":
        df = _frame(con, cfg, week, P_buyers(week), True)
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out)
        meta = {"rows": len(df), "customers": int(df.customer_idx.nunique()), "pos_rate": float(df.label.mean())}
    else:
        out.mkdir(parents=True, exist_ok=True)
        users = con.execute(f"SELECT DISTINCT customer_idx FROM ({P_buyers(week)}) ORDER BY 1").fetchnumpy()["customer_idx"]
        step = int(ecfg(cfg).eval_chunk_customers)
        rows = 0
        for n, b in enumerate(range(0, len(users), step)):
            con.execute("CREATE OR REPLACE TEMP TABLE _chunk AS SELECT unnest(?::INTEGER[]) AS customer_idx",
                        [users[b:b + step].tolist()])
            df = _frame(con, cfg, week, "SELECT customer_idx FROM _chunk", False)
            df.to_parquet(out / f"part-{n:03d}.parquet", index=False)
            rows += len(df)
            del df
            gc.collect()
        meta = {"rows": rows, "customers": int(len(users))}
        (out / "_SUCCESS").write_text("")
    meta.update({"week": str(week.start), "kind": kind, "seconds": round(time.time() - t0, 1),
                 "superset": superset(cfg)})
    (d / f"meta_{kind}_{week.start}.json").write_text(json.dumps(meta, indent=1))
    print(f"built {kind} {week.start}: {meta['rows']:,} rows ({meta['seconds']}s)", flush=True)
    return out


def P_buyers(week: Week) -> str:
    return (f"SELECT DISTINCT customer_idx FROM transactions "
            f"WHERE t_dat BETWEEN DATE '{week.start}' AND DATE '{week.end}'")


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

def provenance_cols(cols) -> list[str]:
    return [c for c in cols if c.endswith("_score") or c.endswith("_rank")] + (["n_channels"] if "n_channels" in cols else [])


def derive(df: pd.DataFrame, spec: dict, all_channels: list[str], train: bool,
           drop_channels=(), drop_features=()) -> pd.DataFrame:
    """Rows and columns of configuration ``spec`` (optionally minus an ablation) from a superset frame."""
    caps = {c: k for c, k in spec["caps"].items() if c not in drop_channels}
    keep = np.zeros(len(df), bool)
    within = {}
    for ch in all_channels:
        r = df[f"{ch}_rank"].to_numpy()
        ok = (r <= caps[ch]) if ch in caps else np.zeros(len(df), bool)
        within[ch] = ok
        keep |= ok
    out = df.loc[keep].copy()
    drop = []
    for ch in all_channels:
        if ch in caps:
            ok = within[ch][keep]
            out.loc[~ok, [f"{ch}_score", f"{ch}_rank"]] = np.nan
        else:
            drop += [f"{ch}_score", f"{ch}_rank"]
    out["n_channels"] = np.sum([within[ch][keep] for ch in caps], axis=0).astype(np.float32)
    drop += [f"nn_{m}_dot" for m in NEURAL if m not in spec["neural_features"] and f"nn_{m}_dot" in out]
    out = out.drop(columns=[c for c in drop if c in out])
    if drop_features == "PROVENANCE":
        out = out.drop(columns=provenance_cols(out.columns))
    elif drop_features:
        out = out.drop(columns=[c for c in drop_features if c in out])
    if train:
        pos = out.groupby("customer_idx")["label"].transform("max") > 0
        out = out.loc[pos.to_numpy()]
    return out


def channels_in(df_cols) -> list[str]:
    return [c[:-5] for c in df_cols if c.endswith("_rank")]


# ---------------------------------------------------------------------------
# Fit + score one (config, fold, seed[, ablation])
# ---------------------------------------------------------------------------

ID = ("customer_idx", "article_id", "label")


def system_name(config: str, ablation: str | None) -> str:
    return f"ens_{config}" + (f"__minus_{ablation}" if ablation else "")


def eval_path(cfg, system: str, fold: Week, seed: int) -> Path:
    d = cfg.path("interim") / "track_a_research" / "eval" / system
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{fold.start}_s{seed}.parquet"


def fit_one(config: str, fold_start: str, seed: int, ablation: str | None = None) -> Path:
    import lightgbm as lgb

    from ensemble.db import connect
    from ensemble.ranking.train import TrainingData, _customer_groups, fit
    from ensemble.research.evaluate import FoldContext, score_lists, summarize
    from ensemble.research.neural import peak_rss_bytes
    cfg = load_config()
    fold = P.week_of(fold_start)
    system = system_name(config, ablation)
    out = eval_path(cfg, system, fold, seed)
    if out.with_suffix(".json").exists():
        return out
    spec = config_spec(cfg, config)
    drop_ch, drop_f = ABLATIONS[ablation] if ablation else ((), ())
    if ablation in ("collaborative", "sequential") and not set(drop_ch) & (set(spec["caps"]) | set(spec["neural_features"])):
        raise SystemExit(f"ablation {ablation} removes nothing from {config}")
    d = superset_dir(cfg)
    t0 = time.time()
    data, all_ch = None, None
    for w in P.ranker_train_weeks(cfg, fold):
        df = pd.read_parquet(d / f"train_{w.start}.parquet")
        all_ch = channels_in(df.columns)
        df = derive(df, spec, all_ch, True, drop_ch, drop_f)
        feats = [c for c in df.columns if c not in ID]
        if data is None:
            data = TrainingData(feats)
        assert feats == data.features, "feature schema differs between training weeks"
        data.X.append(df[feats].to_numpy(dtype=np.float32))
        data.y.append(df["label"].to_numpy(dtype=np.int8))
        data.groups.append(_customer_groups(df.customer_idx.to_numpy()))
        del df
        gc.collect()
    load_s = time.time() - t0
    rc = dict(cfg.ranker)
    rc["seed"] = int(seed)
    fcfg = Config({**cfg, "ranker": rc})
    t1 = time.time()
    n_rows = data.rows
    booster, features, info = fit(fcfg, data)
    fit_s = time.time() - t1
    del data
    gc.collect()
    # Score the fold.
    t2 = time.time()
    keep_top = int(ecfg(cfg).keep_top)
    parts, stats = [], []
    for f in sorted((d / f"eval_{fold.start}").glob("part-*.parquet")):
        df = derive(pd.read_parquet(f), spec, all_ch, False, drop_ch, drop_f)
        assert [c for c in df.columns if c not in ID] == features, "evaluation schema differs from training"
        s = booster.predict(df[features].to_numpy(dtype=np.float32), num_threads=8)
        sc = pd.DataFrame({"customer_idx": df.customer_idx.to_numpy(), "article_id": df.article_id.to_numpy(), "score": s})
        stats.append(df.groupby("customer_idx").agg(cand_hits=("label", "sum"), n_cand=("label", "size")).reset_index())
        sc = sc.sort_values(["customer_idx", "score", "article_id"], ascending=[True, False, True], kind="stable")
        parts.append(sc.groupby("customer_idx", sort=False).head(keep_top))
        del df, sc
        gc.collect()
    score_s = time.time() - t2
    ranked = pd.concat(parts, ignore_index=True)
    lists = ranked.groupby("customer_idx", sort=False)["article_id"].agg(list).to_dict()
    con = connect(cfg, read_only=True)
    ctx = FoldContext(con, cfg, fold)
    res = score_lists(ctx, lists, pd.concat(stats, ignore_index=True)).assign(fold=str(fold.start))
    res.to_parquet(out, index=False)
    models = cfg.path("interim") / "track_a_research" / "models" / system
    models.mkdir(parents=True, exist_ok=True)
    mpath = models / f"{fold.start}_s{seed}.txt"
    booster.save_model(str(mpath))
    gain = booster.feature_importance("gain")
    meta = {"system": system, "config": config, "ablation": ablation, "fold": str(fold.start), "seed": int(seed),
            "summary": summarize(res), "coverage": res.attrs["coverage"], "novelty": res.attrs["novelty"],
            "n_features": len(features), "features": features, "n_trees": booster.current_iteration(),
            "fit_info": info, "train_rows": int(n_rows), "load_seconds": round(load_s, 1),
            "fit_seconds": round(fit_s, 1), "score_seconds": round(score_s, 1), "peak_rss_bytes": peak_rss_bytes(),
            "model_bytes": mpath.stat().st_size, "spec": spec, "lightgbm": lgb.__version__,
            "importance_gain_share": dict(sorted(zip(features, (gain / gain.sum()).round(6).tolist()),
                                                 key=lambda x: -x[1])[:40]),
            "manifest": P.manifest(cfg, con, {"superset_dir": d.name})}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"{system} {fold.start} s{seed}: MAP@12 {meta['summary']['map@12']:.5f} cand-recall "
          f"{meta['summary']['candidate_recall']:.4f} trees {meta['n_trees']} fit {fit_s:.0f}s", flush=True)
    return out


def fit_subprocess(config: str, fold: str, seed: int, ablation: str | None = None, log: Path | None = None) -> Path:
    import subprocess
    cfg = load_config()
    out = eval_path(cfg, system_name(config, ablation), P.week_of(fold), seed)
    if out.with_suffix(".json").exists():
        return out
    cmd = [sys.executable, "-m", "ensemble.research.ensemble", "fit", config, fold, str(seed)] + ([ablation] if ablation else [])
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    with open(log or os.devnull, "a") as fh:
        r = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"fit {config} {fold} s{seed} {ablation} failed ({r.returncode}); see {log}")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "build":
        build(P.week_of(a[1]), a[2])
    elif a[0] == "fit":
        fit_one(a[1], a[2], int(a[3]), a[4] if len(a) > 4 else None)
    else:
        raise SystemExit(__doc__)
