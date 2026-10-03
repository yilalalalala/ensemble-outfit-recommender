"""The authoritative Track A comparison artifact: reports/track_a_research/comparison.json.

  ENSEMBLE_CONFIG=track_a_research python -m ensemble.research.report

Reads every per-customer evaluation file under data/interim/track_a_research/eval/ (systems that
cover all six reporting folds) and writes, per system and seed: per-fold and pooled metrics,
segments, coverage, novelty and resource cost; seed mean / SD; paired customer-cluster bootstrap
comparisons (MAP@12, NDCG@12, Recall@12) for the predeclared pairs; the ablation table; the
adoption decisions under D-038's rule; and a failure analysis of the final system (every truth
pair classified, plus deterministic anonymized examples).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.config import load_config
from ensemble.research import protocol as P
from ensemble.research.evaluate import cluster_bootstrap, cluster_bootstrap_ratio, summarize

NEURAL = ("bpr", "lightgcn", "sasrec")
RULES = ("popularity_global", "popularity_age", "repeat_pop_age", "item_cf", "covis")
PRIMARY_SEED = {"neural": 0, "rule": 0, "ensemble": 42}


def kind(system: str) -> str:
    return "neural" if system in NEURAL else ("ensemble" if system.startswith("ens_") else "rule")


def load_eval(cfg) -> dict:
    root = cfg.path("interim") / "track_a_research" / "eval"
    folds = [str(f.start) for f in P.reporting_folds(cfg)]
    out: dict = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        by_seed: dict = {}
        for f in d.glob("*.json"):
            fold, seed = f.stem.split("_s")
            by_seed.setdefault(int(seed), {})[fold] = f
        for seed, files in by_seed.items():
            if sorted(files) != folds:
                continue
            metas = {fd: json.loads(files[fd].read_text()) for fd in folds}
            rows = pd.concat([pd.read_parquet(files[fd].with_suffix(".parquet")) for fd in folds], ignore_index=True)
            out.setdefault(d.name, {})[seed] = {"meta": metas, "rows": rows}
    return out


def resources(system: str, metas: dict) -> dict:
    m = list(metas.values())
    if kind(system) == "neural":
        runs = [x["model_run"] for x in m]
        keys = ("train_seconds", "score_seconds", "wall_seconds", "peak_rss_bytes", "mps_driver_bytes",
                "n_parameters", "artifact_bytes")
        return {k: float(np.mean([r[k] for r in runs])) for k in keys} | {
            "device": runs[0]["device"], "coverage_of_buyers": float(np.mean(
                [r["n_targets_represented"] / r["n_targets"] for r in runs]))}
    if kind(system) == "ensemble":
        keys = ("fit_seconds", "load_seconds", "score_seconds", "peak_rss_bytes", "model_bytes", "n_trees",
                "n_features", "train_rows")
        return {k: float(np.mean([x[k] for x in m])) for k in keys}
    return {}


def system_block(system: str, seeds: dict) -> dict:
    blk = {"kind": kind(system), "seeds": {}}
    for seed, d in sorted(seeds.items()):
        per_fold = {fd: {**x["summary"], "coverage": x["coverage"], "novelty": x["novelty"]}
                    for fd, x in d["meta"].items()}
        pooled = summarize(d["rows"])
        pooled["coverage_mean"] = float(np.mean([x["coverage"] for x in d["meta"].values()]))
        pooled["novelty_mean"] = float(np.mean([x["novelty"] for x in d["meta"].values()]))
        blk["seeds"][str(seed)] = {"per_fold": per_fold, "pooled": pooled, "resources": resources(system, d["meta"])}
    vals = {m: [blk["seeds"][s]["pooled"][m] for s in blk["seeds"]] for m in ("map@12", "ndcg@12", "recall@12")}
    blk["seed_stats"] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                             "n_seeds": len(v)} for m, v in vals.items()}
    folds = list(next(iter(blk["seeds"].values()))["per_fold"])
    blk["per_fold_seed_mean"] = {fd: {"mean": float(np.mean([blk["seeds"][s]["per_fold"][fd]["map@12"]
                                                               for s in blk["seeds"]])),
                                      "std": float(np.std([blk["seeds"][s]["per_fold"][fd]["map@12"]
                                                           for s in blk["seeds"]], ddof=1))
                                      if len(blk["seeds"]) > 1 else 0.0} for fd in folds}
    return blk


def primary(system: str, seeds: dict):
    s = PRIMARY_SEED[kind(system)]
    return seeds.get(s) or seeds[min(seeds)]


def compare(ev: dict, a: str, b: str, bs: dict) -> dict:
    A, B = primary(a, ev[a]), primary(b, ev[b])
    ra, rb = A["rows"], B["rows"]
    out = {"a": a, "b": b,
           "map@12": cluster_bootstrap(ra, rb, "ap", int(bs["n_boot"]), int(bs["seed"])),
           "ndcg@12": cluster_bootstrap(ra, rb, "ndcg", int(bs["n_boot"]), int(bs["seed"])),
           "recall@12": cluster_bootstrap_ratio(ra, rb, "hits", "n_truth", int(bs["n_boot"]), int(bs["seed"]))}
    pa = {fd: x["summary"]["map@12"] for fd, x in A["meta"].items()}
    pb = {fd: x["summary"]["map@12"] for fd, x in B["meta"].items()}
    out["per_fold_relative"] = {fd: pb[fd] / pa[fd] - 1 for fd in pa}
    out["folds_won_by_b"] = int(sum(pb[fd] > pa[fd] for fd in pa))
    return out


def failure_analysis(cfg, ev: dict, system: str, n_examples: int = 24) -> dict:
    """Classify every truth pair of the final system's primary run, and draw anonymized examples."""
    rows = primary(system, ev[system])["rows"]
    t = {s: int(rows[f"t_{s}"].sum()) for s in ("cold", "ineligible")}
    total = int(rows.n_truth.sum())
    out = {"system": system, "n_truth_pairs": total, "n_customers_rows": int(len(rows))}
    if "cand_hits" in rows:
        hits = int(rows.hits.sum())
        cand = int(rows.cand_hits.sum())
        out["pair_modes"] = {"hit_in_top12": hits / total, "retrieved_not_in_top12": (cand - hits) / total,
                             "not_retrieved": (total - cand) / total}
        out["pair_ceilings"] = {"ineligible_share (no sale in 28 days before the cutoff, incl. cold)":
                                t["ineligible"] / total, "cold_share (first sale inside the label week)": t["cold"] / total}
    out["customer_modes"] = {
        "no_hit_share": float((rows.hits == 0).mean()),
        "no_hit_share_new_customers": float((rows.hits[~rows.returning] == 0).mean()),
        "no_hit_share_returning": float((rows.hits[rows.returning] == 0).mean()),
        "repeat_truth_recall": float(rows.h_repeat.sum() / max(1, rows.t_repeat.sum())),
        "nonrepeat_truth_recall": float(rows.h_nonrepeat.sum() / max(1, rows.t_nonrepeat.sum()))}
    # Deterministic anonymized examples: customers with no hit, ordered by a salted hash.
    miss = rows[rows.hits == 0].copy()
    miss["h"] = [hashlib.sha256(f"tA-failures:{c}:{f}".encode()).hexdigest() for c, f in zip(miss.customer_idx, miss.fold)]
    miss = miss.sort_values("h").head(n_examples)
    ex = []
    for i, r in enumerate(miss.itertuples(index=False), 1):
        if r.t_cold == r.n_truth:
            mode = "only cold items bought (unreachable by any system)"
        elif r.t_ineligible == r.n_truth:
            mode = "only stale (ineligible) items bought"
        elif not r.returning:
            mode = "new customer: fallback list, no personal signal"
        elif getattr(r, "cand_hits", 0) > 0:
            mode = "retrieved but ranked below 12"
        elif r.t_repeat > 0:
            mode = "repeat purchase missed"
        else:
            mode = "not retrieved: new-to-customer items outside every channel"
        ex.append({"case": f"C{i:02d}", "fold": r.fold, "activity": r.activity, "returning": bool(r.returning),
                   "n_truth": int(r.n_truth), "truth_cold": int(r.t_cold), "truth_ineligible": int(r.t_ineligible),
                   "truth_repeat": int(r.t_repeat), "truth_head": int(r.t_head), "truth_tail": int(r.t_tail),
                   "retrieved_truth": int(getattr(r, "cand_hits", -1)), "candidates": int(getattr(r, "n_cand", -1)),
                   "failure_mode": mode})
    out["examples"] = ex
    out["example_modes"] = pd.Series([e["failure_mode"] for e in ex]).value_counts().to_dict()
    return out


def adoption(cmp: dict, res_a: dict, res_b: dict, rule: dict) -> dict:
    m = cmp["map@12"]
    t_inc = (res_b["fit_seconds"] + res_b["load_seconds"]) / (res_a["fit_seconds"] + res_a["load_seconds"]) - 1
    r_inc = res_b["peak_rss_bytes"] / res_a["peak_rss_bytes"] - 1
    checks = {"ci_above_zero": m["diff_ci95"][0] > 0, "folds_won": cmp["folds_won_by_b"],
              "folds_ok": cmp["folds_won_by_b"] >= int(rule["min_folds_won"]),
              "train_time_increase": t_inc, "time_ok": t_inc <= float(rule["max_train_time_increase"]),
              "peak_rss_increase": r_inc, "rss_ok": r_inc <= float(rule["max_peak_rss_increase"])}
    checks["adopt"] = bool(checks["ci_above_zero"] and checks["folds_ok"] and checks["time_ok"] and checks["rss_ok"])
    return checks


def run() -> dict:
    cfg = load_config()
    p = P.proto(cfg)
    ev = load_eval(cfg)
    bs = dict(p.bootstrap)
    systems = {s: system_block(s, seeds) for s, seeds in ev.items()}
    neural = [s for s in NEURAL if s in systems]
    strongest = max(neural, key=lambda s: systems[s]["seed_stats"]["map@12"]["mean"]) if neural else None
    baselines_all = [s for s in systems if kind(s) != "ensemble"]
    strongest_any = max(baselines_all, key=lambda s: systems[s]["seed_stats"]["map@12"]["mean"])
    ens = cfg.research.ensemble
    final = f"ens_{ens.get('final', 'phase2')}"
    pairs = []
    if strongest:
        for s in neural:
            for ref in ("popularity_global", "repeat_pop_age"):
                if ref in ev:
                    pairs.append((ref, s))
    for e in [s for s in systems if kind(s) == "ensemble" and "__minus_" not in s]:
        if e != "ens_phase2" and "ens_phase2" in ev:
            pairs.append(("ens_phase2", e))
        if strongest:
            pairs.append((strongest, e))
        pairs.append((strongest_any, e))
    comparisons = [compare(ev, a, b, bs) for a, b in dict.fromkeys(pairs) if a in ev and b in ev]
    decisions = {}
    for c in comparisons:
        if c["a"] == "ens_phase2" and c["b"].startswith("ens_") and "__minus_" not in c["b"]:
            ra = systems["ens_phase2"]["seeds"]["42"]["resources"]
            rb = systems[c["b"]]["seeds"]["42"]["resources"]
            decisions[c["b"]] = adoption(c, ra, rb, dict(p.adoption))
    ablations = {}
    for s in systems:
        if s.startswith(final + "__minus_"):
            c = compare(ev, final, s, bs)
            ablations[s.split("__minus_")[1]] = {"map@12": c["map@12"], "ndcg@12": c["ndcg@12"],
                                                 "recall@12": c["recall@12"], "folds_ablation_wins": c["folds_won_by_b"],
                                                 "pooled": systems[s]["seeds"]["42"]["pooled"],
                                                 "resources": systems[s]["seeds"]["42"]["resources"]}
    target = None
    if strongest and final in ev:
        c = next((c for c in comparisons if c["a"] == strongest and c["b"] == final), None)
        if c:
            target = {"vs": strongest, "relative": c["map@12"]["relative"], "relative_ci95": c["map@12"]["relative_ci95"],
                      "met_5pct_and_ci": bool(c["map@12"]["relative"] >= 0.05 and c["map@12"]["relative_ci95"][0] > 0)}
    from ensemble.db import connect
    con = connect(cfg, read_only=True)
    out = {"protocol": dict(p), "systems": systems, "strongest_reproduced_baseline": strongest,
           "strongest_baseline_any": strongest_any, "final_system": final, "comparisons": comparisons,
           "adoption_decisions": decisions, "ablations": ablations, "target": target,
           "failure_analysis": failure_analysis(cfg, ev, final) if final in ev else None,
           "manifest": P.manifest(cfg, con)}
    path = cfg.path("reports") / "track_a_research" / "comparison.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}")
    for s, b in sorted(systems.items(), key=lambda x: -x[1]["seed_stats"]["map@12"]["mean"]):
        ss = b["seed_stats"]["map@12"]
        print(f"{s:34s} MAP@12 {ss['mean']:.5f} ± {ss['std']:.5f} (n={ss['n_seeds']})")
    return out


if __name__ == "__main__":
    run()
