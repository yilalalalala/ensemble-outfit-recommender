"""How much the Round-1 catalogue leak was worth (Round 3, Phase 1.2).

  python -m ensemble.completion.protocol_check [n_weeks]

The Round-1 eligible catalogue was "sold between four weeks before the target
week and the *end* of the target week", so an article whose only sales were
inside the target week was still a legal recommendation. Every truth article was
therefore eligible by construction. This script scores the same fixed-fusion
baselines twice per week — once over the leakage-free catalogue (sales strictly
before the cutoff) and once over the Round-1 oracle catalogue — and reports the
difference, so the Round-1 numbers can be compared to Round 3 honestly instead of
being quietly restated.

Runs in its own process because it trains the towers (PyTorch, no LightGBM).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ensemble.completion import association as A
from ensemble.completion import candidates as C
from ensemble.completion import features as FE
from ensemble.completion import fusion as FU
from ensemble.completion import metrics as MET
from ensemble.completion import pipeline as PL
from ensemble.completion import protocol as P
from ensemble.completion import towers as TW
from ensemble.completion.data import item_table
from ensemble.completion.models import pop_buckets
from ensemble.config import load_config
from ensemble.data.splits import Week, load_splits
from ensemble.db import connect
from ensemble.evaluation.metrics import relative_lift


def evaluate_universe(con, cfg, week: Week, tw: TW.Towers, uni: pd.DataFrame, q_raw: pd.DataFrame,
                      attrs: pd.DataFrame, label: str, log=print) -> dict:
    """Score every fixed-fusion baseline over ``uni``."""
    k = int(cfg.track_b.k)
    q = P.restrict_truth(q_raw, uni)
    dropped = int(q.attrs.get("n_truth_dropped", 0))
    C.register(con, q, uni)
    FE.register_attrs(con)
    keys = q[["anchor", "target_slot"]].drop_duplicates().reset_index(drop=True)
    cand = cfg.track_b.candidates
    tt = tw.retrieve(keys, uni, int(cand["two_tower"]))
    ttc = tw.retrieve(keys, uni, int(cand["two_tower_content"]), use_ids=False)
    C.build_sources(con, cfg, tt, ttc)
    recs = FU.baseline_recs(q, FU.baseline_lists(con, cfg, uni, tt, ttc), cfg)
    seg = MET.segments(uni, cfg)
    out = {"universe": label, "n_eligible_items": int(len(uni)), "n_queries": int(len(q)),
           "n_truth_pairs": int(sum(len(t) for t in q.truth.values)),
           "n_truth_dropped_as_ineligible": dropped, "metrics": {}}
    for name, r in recs.items():
        m = MET.public(MET.evaluate(r, q, uni, cfg, attrs=attrs, seg=seg))
        out["metrics"][name] = m
    p0 = out["metrics"]["slot_popularity"][f"recall@{k}"]
    for m in out["metrics"].values():
        m[f"relative_lift_recall@{k}_vs_popularity"] = relative_lift(m[f"recall@{k}"], p0)
    log(f"  {label:7s}: {len(uni):,} items, {len(q):,} queries, "
        + "  ".join(f"{n}={out['metrics'][n][f'recall@{k}']:.4f}" for n in FU.BASELINES))
    return out


def run(n_weeks: int = 2) -> dict:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    weeks = [splits.val.shift(-k) for k in range(n_weeks - 1, -1, -1)]
    k = int(cfg.track_b.k)
    attrs = con.execute("SELECT article_id, product_type_no, product_code FROM articles").df()
    per_week: dict = {}
    for week in weeks:
        t0 = time.time()
        A.mine(con, week, cfg, cfg.track_b.get("basket_filter", ""))
        honest = P.eligible_universe(con, week, cfg)
        oracle = P.eligible_universe(con, week, cfg, oracle=True)
        q_raw = P.queries(con, week, cfg)
        tr, hold = PL.mining_pairs(con, int(cfg.track_b.two_tower.holdout_days))
        enc = TW.Encoder(item_table(con), PL.price_tiers(con, week),
                         np.unique(np.r_[tr.src.to_numpy(), tr.dst.to_numpy()]),
                         pop_buckets(honest), clip=PL.load_clip(cfg))
        print(f"[{week.start}] training towers once, scoring both catalogues", flush=True)
        tw, info = TW.train(con, tr, enc, honest, cfg, holdout=hold, log=lambda s: print(s, flush=True))
        res = {}
        for label, uni in (("honest", honest), ("oracle", oracle)):
            res[label] = evaluate_universe(con, cfg, week, tw, uni, q_raw, attrs, label,
                                           log=lambda s: print(s, flush=True))
        res["oracle_inflation"] = {
            name: {"recall@12_honest": res["honest"]["metrics"][name][f"recall@{k}"],
                   "recall@12_oracle": res["oracle"]["metrics"][name][f"recall@{k}"],
                   "relative": relative_lift(res["oracle"]["metrics"][name][f"recall@{k}"],
                                             res["honest"]["metrics"][name][f"recall@{k}"])}
            for name in FU.BASELINES}
        res["seconds"] = round(time.time() - t0, 1)
        res["tower_best_epoch"] = info["best_epoch"]
        per_week[str(week.start)] = res
        for name, v in res["oracle_inflation"].items():
            print(f"    {name:20s} honest {v['recall@12_honest']:.4f} -> oracle "
                  f"{v['recall@12_oracle']:.4f} ({v['relative']:+.1%})", flush=True)
    summary = {name: {"mean_oracle_inflation": float(np.mean(
        [per_week[w]["oracle_inflation"][name]["relative"] for w in per_week]))} for name in FU.BASELINES}
    summary["truth_coverage"] = {
        "honest_mean_truth_dropped_share": float(np.mean(
            [per_week[w]["honest"]["n_truth_dropped_as_ineligible"]
             / (per_week[w]["honest"]["n_truth_dropped_as_ineligible"] + per_week[w]["honest"]["n_truth_pairs"])
             for w in per_week])),
        "oracle_mean_truth_dropped_share": float(np.mean(
            [per_week[w]["oracle"]["n_truth_dropped_as_ineligible"]
             / (per_week[w]["oracle"]["n_truth_dropped_as_ineligible"] + per_week[w]["oracle"]["n_truth_pairs"])
             for w in per_week]))}
    out = {"per_week": per_week, "summary": summary,
           "manifest": P.manifest(cfg, con, {"n_weeks": n_weeks})}
    d = cfg.path("reports") / "track_b_round3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "protocol_leak.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n== oracle (Round-1) catalogue inflation, mean over weeks ==")
    for name, v in summary.items():
        if isinstance(v, dict) and "mean_oracle_inflation" in v:
            print(f"{name:22s} {v['mean_oracle_inflation']:+.2%}")
    print("truth dropped as ineligible:", summary["truth_coverage"])
    return out


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
