#!/bin/sh
# Sequential, restartable chain for the ranker retrain (every step skips cached work).
set -e
cd "$(dirname "$0")/../.."
export ENSEMBLE_CONFIG=track_a_research PYTHONPATH=src
PY=.venv/bin/python
while pgrep -f "ranker_retrain pools" >/dev/null; do sleep 15; done
for w in 2020-07-08 2020-07-15 2020-07-22 2020-07-29 2020-08-05 2020-08-12 2020-08-19 2020-08-26 2020-09-02; do
  $PY -m ensemble.research.ranker_retrain train_matrix $w
done
for f in 2020-08-05 2020-08-12 2020-08-19 2020-08-26 2020-09-02 2020-09-09; do
  $PY -m ensemble.research.ranker_retrain eval_matrix $f
done
$PY -m ensemble.research.ranker_retrain queue R0 42
$PY -m ensemble.research.ranker_retrain queue R1,R2 42
echo CHAIN_DONE
