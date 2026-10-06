#!/usr/bin/env bash
# Run Track A validation experiments sequentially: scripts/run_experiments.sh <mode> <config>...
# mode: val (one validation run per config) or backtest (4-week rolling backtest per config).
# "default" means configs/default.yaml with the given tag; others are configs/experiments/<name>.yaml.
set -u
mode=$1; shift
cd "$(dirname "$0")/.."
mkdir -p reports/phase2/logs
for c in "$@"; do
  if [ "$c" = "default" ]; then cfg=default; tag=_p2_default; else cfg=experiments/$c; tag=_$c; fi
  log=reports/phase2/logs/${mode}${tag}.log
  echo "$(date '+%H:%M:%S') start $mode $c" >> reports/phase2/logs/queue.log
  if [ "$mode" = "val" ]; then
    ENSEMBLE_CONFIG=$cfg PYTHONPATH=src /usr/bin/time -l .venv/bin/python -m ensemble.ranking.train val "$tag" > "$log" 2>&1
    rc=$?
  else
    ENSEMBLE_CONFIG=$cfg PYTHONPATH=src /usr/bin/time -l .venv/bin/python -m ensemble.evaluation.backtest 4 "$tag" > "$log" 2>&1
    rc=$?
  fi
  echo "$(date '+%H:%M:%S') done $mode $c exit=$rc" >> reports/phase2/logs/queue.log
done
