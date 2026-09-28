.PHONY: help setup data ingest test baselines retrieval ranker track-b submission serving serve mvp mlflow
PYTHON := .venv/bin/python
PY := PYTHONPATH=src $(PYTHON)
PORT ?= 8010

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:      ## create the venv (Python 3.11) and install dependencies; macOS also needs `brew install libomp`
	uv venv -p 3.11 .venv && uv pip install -p $(PYTHON) -r requirements.txt -e .

data:       ## download the full dataset from Kaggle (see README for credentials)
	./scripts/download_data.sh --images

ingest:     ## M0: load the CSVs into DuckDB and run integrity checks
	$(PY) -m ensemble.data.ingest

baselines:  ## M1: popularity and repeat-purchase baselines on validation and test
	$(PY) -m ensemble.baselines val
	$(PY) -m ensemble.baselines test

retrieval:  ## M2: recall per retrieval channel on validation
	$(PY) -m ensemble.candidates.evaluate val

ranker:     ## M3: LightGBM LambdaRank on validation, then the test week once
	$(PY) -m ensemble.ranking.train val
	$(PY) -m ensemble.ranking.train test

track-b:    ## M4/M5: outfit completion, model selection and ablations on validation, then test once, then serving artifacts
	$(PY) -m ensemble.completion.run val
	$(PY) -m ensemble.completion.run test
	$(PY) -m ensemble.completion.run serve

submission: ## Track A for all customers + reports/submission.csv (Kaggle format)
	$(PY) -m ensemble.ranking.predict

serving:    ## M6: build the SQLite serving store
	$(PY) -m ensemble.api.build

serve:      ## run the web app at http://localhost:8010 (override with PORT=...)
	PYTHONPATH=src .venv/bin/uvicorn ensemble.api.app:app --port $(PORT)

mvp: ingest baselines retrieval ranker track-b submission serving test  ## the full MVP pipeline, end to end

test:       ## run the test suite
	$(PY) -m pytest -q

mlflow:     ## browse experiment runs
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
