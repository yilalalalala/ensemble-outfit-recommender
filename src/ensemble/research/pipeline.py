"""Track A research pipeline, stage by stage (restartable; every stage skips cached work).

  ENSEMBLE_CONFIG=track_a_research python -m ensemble.research.pipeline <stage> [...]

  apply-tuning   write reports/track_a_research/tuning.json's selection into research.models
  models         per-week neural models: seed 0 for every week the ensemble needs, then
                 seeds 1 and 2 for the six reporting folds (CPU lane: BPR-MF; GPU lane:
                 LightGCN, SASRec)
  allocate       candidate-budget frontier with the neural channels, on allocator weeks only
  matrices       superset candidate/feature matrices (train weeks + reporting folds)
  baselines      single-stage systems on the reporting folds
  fit <config> [seeds] [ablation]   ensemble fits on the six folds, one subprocess per fit
"""
from __future__ import annotations

import json
import sys
import time

import yaml

from ensemble.config import ROOT, Config, load_config
from ensemble.research import protocol as P


def log_dir(cfg):
    d = cfg.path("reports") / "track_a_research" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def apply_tuning() -> dict:
    """Replace the ``research.models`` block with the tuning selection (other text untouched)."""
    cfg = load_config()
    sel = json.loads((cfg.path("reports") / "track_a_research" / "tuning.json").read_text())["selected"]
    path = ROOT / "configs" / "track_a_research.yaml"
    text = path.read_text()
    start = text.index("  models:\n")
    end = text.index("  top_k:")
    body = yaml.safe_dump({"models": sel}, sort_keys=False, default_flow_style=False)
    body = "\n".join("  " + line if line else line for line in body.splitlines())
    header = ("  # Selected on tuning folds only by ensemble.research.tune (reports/track_a_research/tuning.json).\n")
    path.write_text(text[:start] + header + body + "\n" + text[end:])
    return sel


def models(seeds_extra=(1, 2)) -> None:
    from ensemble.research.tune import run_lanes
    cfg = load_config()
    weeks = [str(w.start) for w in P.all_model_weeks(cfg)]
    folds = [str(w.start) for w in P.reporting_folds(cfg)]
    jobs = [(m, w, 0, None, False) for w in weeks for m in ("bpr", "lightgcn", "sasrec")]
    jobs += [(m, w, s, None, False) for s in seeds_extra for w in folds for m in ("bpr", "lightgcn", "sasrec")]
    t = time.time()
    # Three lanes: BPR-MF on the CPU; LightGCN and SASRec each on its own MPS lane (measured: the
    # two GPU jobs overlap usefully because both spend part of each step on host-side work).
    run_lanes(jobs, {"bpr": "cpu", "lightgcn": "gpu", "sasrec": "gpu2"})
    print(f"models done ({time.time() - t:.0f}s)")


def allocate() -> dict:
    """Greedy recall-per-candidate frontier over the Phase-2 channels plus the neural channels."""
    from ensemble.candidates.budget import caps_at, greedy_frontier, load_pool
    from ensemble.db import connect
    cfg = load_config()
    p = P.proto(cfg)
    con = connect(cfg, read_only=True)
    pool = {**dict(cfg.budget.pool), "bpr": 100, "lightgcn": 100, "sasrec": 100}
    pool.pop("als", None)
    pool.pop("new_arrival", None)
    r = Config({**dict(cfg.retrieval), **dict(cfg.budget.get("params") or {}), "neural_seed": 0,
                **{f"{c}_k": k for c, k in pool.items()}})
    pools, n_users, n_truth, offset = [], 0, 0, 0
    for w in [P.week_of(s) for s in p.allocator_weeks]:
        t = time.time()
        pl, nu, nt = load_pool(con, w, r, dict(pool), float(p.allocator_sample), offset)
        offset = int(max(x[1].max() for x in pl.values())) + 1
        pools.append(pl)
        n_users, n_truth = n_users + nu, n_truth + nt
        print(f"pool {w.start}: {nu:,} customers ({time.time() - t:.0f}s)", flush=True)
    frontier = greedy_frontier(pools, n_users, n_truth)
    targets = {str(t): caps_at(frontier, t) for t in (116, 130, 160, 200)}
    # The same allocator without the neural channels on the same weeks: separates "new channels"
    # from "re-allocated old channels".
    base_pools = [{c: v for c, v in pl.items() if c not in ("bpr", "lightgcn", "sasrec")} for pl in pools]
    base = greedy_frontier(base_pools, n_users, n_truth)
    out = {"weeks": list(p.allocator_weeks), "sample": float(p.allocator_sample), "pool": pool,
           "targets": targets, "targets_without_neural": {str(t): caps_at(base, t) for t in (116, 130, 160, 200)},
           "frontier": frontier[::5] + [frontier[-1]], "frontier_without_neural": base[::5] + [base[-1]]}
    path = cfg.path("reports") / "track_a_research" / "budget_frontier.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    for t, v in targets.items():
        print(f"budget {t}: recall {v['recall']:.4f} (without neural "
              f"{out['targets_without_neural'][t]['recall']:.4f}) caps {v['caps']}")
    return out


def matrices() -> None:
    from ensemble.research.ensemble import build
    cfg = load_config()
    weeks = sorted({w for f in P.reporting_folds(cfg) for w in P.ranker_train_weeks(cfg, f)}, key=lambda w: w.start)
    for w in weeks:
        build(w, "train")
    for f in P.reporting_folds(cfg):
        build(f, "eval")


def baselines() -> None:
    from ensemble.research.benchmark import run
    run()


def fit(config: str, seeds: list[int], ablation: str | None = None) -> None:
    from ensemble.research.ensemble import fit_subprocess
    cfg = load_config()
    log = log_dir(cfg) / "fits.log"
    for seed in seeds:
        for f in P.reporting_folds(cfg):
            t = time.time()
            fit_subprocess(config, str(f.start), seed, ablation, log)
            print(f"fit {config} {ablation or ''} {f.start} s{seed} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    stage = a[0]
    if stage == "apply-tuning":
        print(json.dumps(apply_tuning(), indent=1))
    elif stage == "models":
        models()
    elif stage == "allocate":
        allocate()
    elif stage == "matrices":
        matrices()
    elif stage == "baselines":
        baselines()
    elif stage == "fit":
        fit(a[1], [int(x) for x in a[2].split(",")] if len(a) > 2 else [42], a[3] if len(a) > 3 else None)
    else:
        raise SystemExit(__doc__)
