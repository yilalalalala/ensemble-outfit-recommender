"""Restartable queue: fit configurations on every reporting fold as soon as its inputs exist.

  ENSEMBLE_CONFIG=track_a_research python -m ensemble.research.queue <configs> <seeds> [ablations]

For each fold (oldest first) it waits until the seed-0 neural models of every week the fold
needs exist, builds any missing superset matrix, then fits every (config, seed, ablation) that
has no result yet. One fit at a time, each in its own subprocess (ensemble.fit_subprocess).
"""
from __future__ import annotations

import sys
import time

from ensemble.config import load_config
from ensemble.research import protocol as P
from ensemble.research.ensemble import build, eval_path, fit_subprocess, superset_dir, system_name
from ensemble.research.neural import cache_dir, params_for


def models_ready(cfg, weeks) -> bool:
    return all((cache_dir(cfg, m, str(w.start), 0, params_for(cfg, m)) / "result.json").exists()
               for w in weeks for m in ("bpr", "lightgcn", "sasrec"))


def run(configs: list[str], seeds: list[int], ablations: list[str | None]) -> None:
    cfg = load_config()
    log = cfg.path("reports") / "track_a_research" / "logs" / "fits.log"
    d = superset_dir(cfg)
    for fold in P.reporting_folds(cfg):
        weeks = [*P.ranker_train_weeks(cfg, fold), fold]
        while not models_ready(cfg, weeks):
            time.sleep(120)
        for w in P.ranker_train_weeks(cfg, fold):
            if not (d / f"train_{w.start}.parquet").exists():
                build(w, "train")
        if not (d / f"eval_{fold.start}" / "_SUCCESS").exists():
            build(fold, "eval")
        for c in configs:
            for a in ablations:
                for s in seeds:
                    if eval_path(cfg, system_name(c, a), fold, s).with_suffix(".json").exists():
                        continue
                    t = time.time()
                    fit_subprocess(c, str(fold.start), s, a, log)
                    print(f"fit {system_name(c, a)} {fold.start} s{s} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0].split(","), [int(x) for x in a[1].split(",")],
        [None if x == "-" else x for x in (a[2].split(",") if len(a) > 2 else ["-"])])
