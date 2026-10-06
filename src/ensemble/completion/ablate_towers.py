"""Two-tower negative-sampling and determinism ablation (Round 3, Phase 5).

  python -m ensemble.completion.ablate_towers [n_weeks] [n_seeds]

Runs in its own process (PyTorch, no LightGBM). For each label week, negative
configuration and seed it trains the towers point-in-time and reports

* ``holdout_recall@12`` — Recall@12 on baskets from the tail of the mining window,
  which is the rule used to pick the number of epochs;
* ``fold_recall@12`` — Recall@12 of the two-tower list alone on the target week's
  held-out baskets, over the leakage-free eligible catalogue;
* ``fold_recall@12_content_only`` — the same with the article-ID embedding switched
  off, i.e. what a brand-new article could be retrieved by.

The Round-1 ablation compared popularity-sampled and (product type, price tier)
hard negatives and found no gain. Those are gone; what is tested here is
retrieval-informed hard negatives — the highest-scoring wrong items from a
popularity-sampled pool, with the positive, its colourways and articles that
co-occur with the anchor in a basket masked out as likely false negatives.

Seed spread is reported because the MPS backend is not bit-deterministic; the
``cpu`` row shows what the same configuration does on a deterministic backend.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ensemble.completion import pipeline as PL
from ensemble.completion import protocol as P
from ensemble.completion import towers as TW
from ensemble.completion.data import item_table
from ensemble.completion.models import pop_buckets
from ensemble.config import Config, load_config
from ensemble.data.splits import Week, load_splits
from ensemble.db import connect

CONFIGS = {
    "in_batch": {"negatives": ["in_batch"], "hard_negatives": 0},
    "in_batch+logq": {"negatives": ["in_batch", "logq"], "hard_negatives": 0},
    "in_batch+logq+hard4": {"negatives": ["in_batch", "logq", "hard"], "hard_negatives": 4},
}
# Extra single runs: is the six-epoch budget binding, and is the backend reproducible?
PROBES = {
    "in_batch+logq_12epochs": {"negatives": ["in_batch", "logq"], "hard_negatives": 0, "epochs": 12},
}
DETERMINISM = {"negatives": ["in_batch", "logq"], "hard_negatives": 0, "epochs": 1}


def tower_cfg(cfg, device: str | None = None, **over) -> Config:
    tt = {**dict(cfg.track_b.two_tower), **over}
    b = {**dict(cfg.track_b), "two_tower": tt}
    if device:
        b["device"] = device
    return Config({**cfg, "track_b": b})


def fold_recall(tw: TW.Towers, q: pd.DataFrame, uni: pd.DataFrame, k: int, use_ids: bool) -> float:
    keys = q[["anchor", "target_slot"]].drop_duplicates().reset_index(drop=True)
    recs = tw.retrieve(keys, uni, k, use_ids=use_ids)
    got = recs.groupby(["anchor", "target_slot"])["article_id"].agg(set).to_dict()
    hit = tot = 0
    for a, s, t in q[["anchor", "target_slot", "truth"]].itertuples(index=False):
        p = got.get((a, s), set())
        hit += sum(1 for x in np.asarray(t).tolist() if x in p)
        tot += len(t)
    return hit / tot if tot else float("nan")


def run_one(con, cfg, week: Week, prep: dict, tr: pd.DataFrame, hold: pd.DataFrame, enc: TW.Encoder,
            name: str, seed: int, device: str | None, over: dict | None = None, log=print) -> dict:
    c = tower_cfg(cfg, device=device, **(over if over is not None else CONFIGS[name]))
    t0 = time.time()
    tw, info = TW.train(con, tr, enc, prep["uni"], c, holdout=hold, seed=seed, log=lambda s: None)
    k = int(cfg.track_b.k)
    out = {"negatives": name, "seed": seed, "device": info["device"],
           "holdout_recall@12": info["best_holdout_recall@12"], "best_epoch": info["best_epoch"],
           "epochs_run": info["epochs_run"],
           "fold_recall@12": fold_recall(tw, prep["q"], prep["uni"], k, use_ids=True),
           "fold_recall@12_content_only": fold_recall(tw, prep["q"], prep["uni"], k, use_ids=False),
           "seconds": round(time.time() - t0, 1)}
    log(f"  {week.start} {name:22s} seed {seed} {out['device']:3s}: holdout {out['holdout_recall@12']:.4f} "
        f"fold {out['fold_recall@12']:.4f} content {out['fold_recall@12_content_only']:.4f} "
        f"(epoch {out['best_epoch']}/{out['epochs_run']}, {out['seconds']:.0f}s)")
    return out


def run(n_weeks: int = 2, n_seeds: int = 2) -> dict:
    cfg = load_config()
    con = connect(cfg, read_only=True)
    splits = load_splits(con, cfg)
    weeks = [splits.val.shift(-k) for k in range(n_weeks - 1, -1, -1)]
    rows: list[dict] = []
    for week in weeks:
        prep = PL.prepare_sql(con, cfg, week)
        tr, hold = PL.mining_pairs(con, int(cfg.track_b.two_tower.holdout_days))
        enc = TW.Encoder(item_table(con), PL.price_tiers(con, week),
                         np.unique(np.r_[tr.src.to_numpy(), tr.dst.to_numpy()]),
                         pop_buckets(prep["uni"]), clip=PL.load_clip(cfg))
        print(f"[{week.start}] {len(tr):,} training pairs, {len(hold):,} holdout pairs, "
              f"{len(prep['q']):,} queries", flush=True)
        for name in CONFIGS:
            for seed in range(n_seeds):
                rows.append({"week": str(week.start),
                             **run_one(con, cfg, week, prep, tr, hold, enc, name, seed, None,
                                       log=lambda s: print(s, flush=True))})
        if week == weeks[-1]:
            # Is the epoch budget binding? One long run on the most recent fold.
            for name, over in PROBES.items():
                rows.append({"week": str(week.start), "probe": name,
                             **run_one(con, cfg, week, prep, tr, hold, enc, name, 0, None, over=over,
                                       log=lambda s: print(s, flush=True))})
            # Determinism: one epoch, same seed, repeated on each backend.
            for dev in ("cpu", "cpu", "mps", "mps"):
                rows.append({"week": str(week.start), "probe": "determinism",
                             **run_one(con, cfg, week, prep, tr, hold, enc, "determinism_1epoch", 0, dev,
                                       over=DETERMINISM, log=lambda s: print(s, flush=True))})
    df = pd.DataFrame(rows)
    summary = {}
    neg = df[df.get("probe").isna()] if "probe" in df.columns else df
    for (name, dev), g in neg.groupby(["negatives", "device"]):
        summary[f"{name}|{dev}"] = {
            "fold_recall@12_mean": float(g["fold_recall@12"].mean()),
            "fold_recall@12_std": float(g["fold_recall@12"].std(ddof=0)),
            "fold_recall@12_content_only_mean": float(g["fold_recall@12_content_only"].mean()),
            "holdout_recall@12_mean": float(g["holdout_recall@12"].mean()),
            "mean_seconds": float(g["seconds"].mean()), "n_runs": int(len(g))}
    base = summary.get("in_batch+logq|mps") or next(iter(summary.values()))
    for key, v in summary.items():
        v["relative_vs_in_batch+logq"] = ((v["fold_recall@12_mean"] - base["fold_recall@12_mean"])
                                          / base["fold_recall@12_mean"])
    det = df[(df.get("probe") == "determinism") & (df.device == "cpu")]
    mps = df[(df.get("probe") == "determinism") & (df.device == "mps")]
    probes = (df[df.get("probe").notna() & (df.get("probe") != "determinism")].to_dict("records")
              if "probe" in df.columns else [])
    out = {"runs": rows, "summary": summary, "epoch_budget_probe": probes,
           "determinism": {
               "cpu_repeat_same_seed_fold_recall": det["fold_recall@12"].round(8).tolist(),
               "cpu_identical": bool(det["fold_recall@12"].nunique() == 1) if len(det) else None,
               "mps_repeat_same_seed_fold_recall": mps["fold_recall@12"].round(8).tolist(),
               "mps_identical": bool(mps["fold_recall@12"].nunique() == 1) if len(mps) else None,
               "note": "One epoch, identical seed, repeated twice per backend. MPS kernels are not "
                       "bit-reproducible, so decisive comparisons use the seed spread above."},
           "manifest": P.manifest(cfg, con, {"n_weeks": n_weeks, "n_seeds": n_seeds})}
    d = cfg.path("reports") / "track_b_round3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ablation_towers.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n== two-tower negatives ==")
    for key, v in summary.items():
        print(f"{key:30s} fold R@12 {v['fold_recall@12_mean']:.4f}±{v['fold_recall@12_std']:.4f} "
              f"content {v['fold_recall@12_content_only_mean']:.4f} "
              f"vs in_batch+logq {v['relative_vs_in_batch+logq']:+.2%} ({v['mean_seconds']:.0f}s)")
    return out


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 2, int(sys.argv[2]) if len(sys.argv) > 2 else 2)
