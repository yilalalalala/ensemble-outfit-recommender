"""Bounded hyperparameter selection for the neural baselines, on tuning folds only.

  ENSEMBLE_CONFIG=track_a_research python -m ensemble.research.tune [stage]

The grid is declared here, before any run, and is deliberately small (compute-aware
successive halving, not open-ended search):

- **Stage A** (tuning fold 2020-07-22, seed 0, learning curve of raw MAP@12 every few
  epochs, with and without masking the customer's training items). Each run trains for the
  maximum epoch budget; the curve picks the epoch.
- **Stage A2** two refinements around the best Stage-A configuration per model.
- **Stage B** (tuning fold 2020-07-29): the two best configurations per model are rerun with
  their chosen epoch count; the winner has the higher mean of the curve-selected raw MAP@12
  over both tuning folds (ties -> fewer epochs, then cheaper).
- **Stage C** (convergence guard, declared before Stage A2 results were read): a winner whose
  curve-selected epoch is the last epoch of its budget is retrained on tuning fold 2020-07-22
  with twice the budget; the longer count is adopted if its curve-selected raw MAP@12 is
  higher. A baseline is never reported at a budget its own learning curve says is binding.

Two lanes run side by side: BPR-MF on the CPU, LightGCN and SASRec on the GPU (MPS). Their
measured peaks are ~2.8 GB and ~6 GB, so the pair stays well inside 16 GB; nothing else
memory-heavy runs at the same time. Every run is cached (``ensemble.research.neural``), so the
driver is restartable. Output: ``reports/track_a_research/tuning.json``.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

from ensemble.config import load_config
from ensemble.research.neural import cache_dir, params_for, run_subprocess

LANE = {"bpr": "cpu", "lightgcn": "gpu", "sasrec": "gpu"}
MAX_EPOCHS = {"bpr": 30, "lightgcn": 30, "sasrec": 12}
CURVE_EVERY = {"bpr": 3, "lightgcn": 3, "sasrec": 1}

STAGE_A = {
    "bpr": [{"lr": 0.005, "reg": 1e-4}, {"lr": 0.02, "reg": 1e-4}, {"lr": 0.005, "reg": 1e-5}, {"lr": 0.02, "reg": 1e-5}],
    "lightgcn": [{"window_weeks": 12}, {"window_weeks": 26}],
    "sasrec": [{"loss": "gbce"}, {"loss": "bce"}],
}


def stage_a2(model: str, best: dict) -> list[dict]:
    """Two refinements around the best Stage-A configuration (declared before Stage A ran)."""
    if model == "bpr":
        return [{**best, "window_weeks": 12}, {**best, "dim": 128}]
    if model == "lightgcn":
        return [{**best, "lr": 0.02}, {**best, "layers": 2}]
    return [{**best, "lr": 0.002}, {**best, "dropout": 0.5}]


def log_path(cfg) -> Path:
    d = cfg.path("reports") / "track_a_research" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "tuning_runs.log"


def run_lanes(jobs: list[tuple]) -> None:
    """jobs: (model, week, seed, overrides, curve). Each lane runs its jobs sequentially."""
    cfg = load_config()
    lanes: dict[str, list] = {}
    for j in jobs:
        lanes.setdefault(LANE[j[0]], []).append(j)
    errors = []

    def worker(lane_jobs):
        for model, week, seed, ov, curve in lane_jobs:
            t = time.time()
            try:
                run_subprocess(model, week, seed, ov, curve, log_path(cfg))
                print(f"  done {model} {week} s{seed} {ov} ({time.time() - t:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001 - re-raised after the other lane finishes
                errors.append((model, week, ov, repr(e)))
                traceback.print_exc()

    threads = [threading.Thread(target=worker, args=(js,)) for js in lanes.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"failed runs: {errors}")


def result(model: str, week: str, seed: int, ov: dict) -> dict:
    cfg = load_config()
    return json.loads((cache_dir(cfg, model, week, seed, params_for(cfg, model, ov)) / "result.json").read_text())


def curve_best(res: dict) -> dict:
    """Best (epoch, mask) on the run's learning curve; ties -> fewer epochs, then no masking."""
    best = None
    for rec in res["epochs_log"]:
        for key, mask in (("raw_map@12", False), ("raw_map@12_mask_seen", True)):
            if key in rec:
                cand = (rec[key], -rec["epoch"], not mask)
                if best is None or cand > best[0]:
                    best = (cand, {"epoch": rec["epoch"], "mask_seen": mask, "raw_map@12": rec[key]})
    return best[1]


def a_overrides(model: str, base: dict) -> dict:
    return {**base, "epochs": MAX_EPOCHS[model], "curve_every": CURVE_EVERY[model]}


def run(stage: str = "all") -> dict:
    cfg = load_config()
    tf = list(cfg.research.protocol.tuning_folds)
    out_path = cfg.path("reports") / "track_a_research" / "tuning.json"
    state = json.loads(out_path.read_text()) if out_path.exists() else {}
    models = list(STAGE_A)

    print("stage A", flush=True)
    run_lanes([(m, tf[0], 0, a_overrides(m, ov), True) for m in models for ov in STAGE_A[m]])
    state["stage_a"] = {m: [{"overrides": ov, "best": curve_best(result(m, tf[0], 0, a_overrides(m, ov))),
                             "train_seconds": result(m, tf[0], 0, a_overrides(m, ov))["train_seconds"],
                             "curve": result(m, tf[0], 0, a_overrides(m, ov))["epochs_log"]}
                            for ov in STAGE_A[m]] for m in models}
    best_a = {m: max(state["stage_a"][m], key=lambda r: (r["best"]["raw_map@12"], -r["train_seconds"]))
              for m in models}

    print("stage A2", flush=True)
    a2 = {m: stage_a2(m, best_a[m]["overrides"]) for m in models}
    run_lanes([(m, tf[0], 0, a_overrides(m, ov), True) for m in models for ov in a2[m]])
    state["stage_a2"] = {m: [{"overrides": ov, "best": curve_best(result(m, tf[0], 0, a_overrides(m, ov))),
                              "train_seconds": result(m, tf[0], 0, a_overrides(m, ov))["train_seconds"],
                              "curve": result(m, tf[0], 0, a_overrides(m, ov))["epochs_log"]}
                             for ov in a2[m]] for m in models}

    print("stage B", flush=True)
    top2 = {}
    for m in models:
        pool = state["stage_a"][m] + state["stage_a2"][m]
        top2[m] = sorted(pool, key=lambda r: (-r["best"]["raw_map@12"], r["train_seconds"]))[:2]
    jobs = []
    for m in models:
        for r in top2[m]:
            ov = {**r["overrides"], "epochs": r["best"]["epoch"], "curve_every": r["best"]["epoch"]}
            jobs.append((m, tf[1], 0, ov, True))
    run_lanes(jobs)
    state["stage_b"] = {}
    selected = {}
    for m in models:
        rows = []
        for r in top2[m]:
            ov = {**r["overrides"], "epochs": r["best"]["epoch"], "curve_every": r["best"]["epoch"]}
            rb = result(m, tf[1], 0, ov)
            last = rb["epochs_log"][-1]
            key = "raw_map@12_mask_seen" if r["best"]["mask_seen"] else "raw_map@12"
            rows.append({"overrides": r["overrides"], "epochs": r["best"]["epoch"], "mask_seen": r["best"]["mask_seen"],
                         "fold_a_raw_map@12": r["best"]["raw_map@12"], "fold_b_raw_map@12": last[key],
                         "mean_raw_map@12": (r["best"]["raw_map@12"] + last[key]) / 2,
                         "fold_b_train_seconds": rb["train_seconds"]})
        state["stage_b"][m] = rows
        win = max(rows, key=lambda r: (r["mean_raw_map@12"], -r["epochs"], -r["fold_b_train_seconds"]))
        selected[m] = {**params_for(cfg, m, win["overrides"]), "epochs": win["epochs"], "mask_seen": win["mask_seen"]}
        selected[m].pop("curve_every", None)
    state["selected"] = selected
    state["declared_grid"] = {"stage_a": STAGE_A, "max_epochs": MAX_EPOCHS, "curve_every": CURVE_EVERY,
                              "stage_a2": "see ensemble.research.tune.stage_a2", "tuning_folds": tf}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(state, indent=1, default=float))
    print(json.dumps(selected, indent=1))
    return state


def stage_c() -> dict:
    cfg = load_config()
    tf = list(cfg.research.protocol.tuning_folds)
    out_path = cfg.path("reports") / "track_a_research" / "tuning.json"
    state = json.loads(out_path.read_text())
    jobs, plan = [], {}
    for m, sel in state["selected"].items():
        win = next(r for r in state["stage_b"][m] if r["epochs"] == sel["epochs"]
                   and all(sel.get(k) == v for k, v in r["overrides"].items()))
        budget = MAX_EPOCHS[m]
        if win["epochs"] < budget:
            plan[m] = {"binding": False}
            continue
        ov = {**win["overrides"], "epochs": 2 * budget, "curve_every": CURVE_EVERY[m]}
        plan[m] = {"binding": True, "overrides": ov}
        jobs.append((m, tf[0], 0, ov, True))
    run_lanes(jobs)
    for m, pl in plan.items():
        if not pl["binding"]:
            continue
        best = curve_best(result(m, tf[0], 0, pl["overrides"]))
        prev = next(r for r in state["stage_b"][m] if r["epochs"] == state["selected"][m]["epochs"])
        pl["best"] = best
        pl["adopted"] = best["raw_map@12"] > prev["fold_a_raw_map@12"]
        if pl["adopted"]:
            state["selected"][m]["epochs"] = best["epoch"]
            state["selected"][m]["mask_seen"] = best["mask_seen"]
    state["stage_c"] = plan
    out_path.write_text(json.dumps(state, indent=1, default=float))
    print(json.dumps({m: state["selected"][m] for m in state["selected"]}, indent=1))
    return state


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stage_c":
        stage_c()
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else "all")
