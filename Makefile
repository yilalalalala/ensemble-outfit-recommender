.PHONY: help setup data ingest test baselines retrieval ranker track-b submission serving serve mvp mlflow backtest clip df2 visual-eval assistant-eval research-a tb-serving-regression serving-bundle serve-v2 serve-smoke bench-serving report-verify
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

backtest:   ## 4-week rolling backtest for Track A (D-017)
	$(PY) -m ensemble.evaluation.backtest 4

clip:       ## FashionCLIP embeddings for every article image (M7a)
	$(PY) -m ensemble.vision.clip

df2:        ## DeepFashion2 street-to-shop adapters: manifest, embeddings, learning curve, evaluation (D-020)
	$(PY) -m ensemble.vision.deepfashion2 manifest
	$(PY) -m ensemble.vision.deepfashion2 embed validation
	$(PY) -m ensemble.vision.deepfashion2 embed train
	$(PY) -m ensemble.vision.deepfashion2 curve nobg
	$(PY) -m ensemble.vision.deepfashion2 train crop
	$(PY) -m ensemble.vision.deepfashion2 evaluate

visual-eval: ## detect garments (local VLM), match every query mode, judge (paid, Batch API), report
	$(PY) -m ensemble.vision.eval_visual_search detect ollama
	$(PY) -m ensemble.vision.eval_visual_search match
	$(PY) -m ensemble.vision.eval_visual_search judge
	$(PY) -m ensemble.vision.eval_visual_search report

assistant-eval: ## assistant eval set on the local model (free); run with `claude` for the paid comparison
	$(PY) -m ensemble.assistant.eval ollama

mlflow:     ## browse experiment runs
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# ---------------------------------------------------------------- Track A research / Track B production
RA := ENSEMBLE_CONFIG=track_a_research $(PY)

research-a: ## Track A research benchmark: tuning, per-week baselines, allocator, matrices, fits, report (many hours)
	$(RA) -m ensemble.research.tune
	$(RA) -m ensemble.research.tune stage_c
	$(RA) -m ensemble.research.pipeline apply-tuning
	$(RA) -m ensemble.research.pipeline models
	$(RA) -m ensemble.research.pipeline baselines
	$(RA) -m ensemble.research.pipeline allocate
	$(RA) -m ensemble.research.pipeline matrices
	$(RA) -m ensemble.research.pipeline fit phase2 42,43,44
	$(RA) -m ensemble.research.report

tb-serving-regression: ## Track B: personalized re-rank of a bounded compatibility pool on the six rolling folds (D-041)
	ENSEMBLE_CONFIG=experiments/tb_serving_pools $(PY) -m ensemble.completion.backtest val 6 serving_pools

serving-bundle: ## build, verify and publish a versioned Track B serving bundle (needs the ingested data)
	$(PY) -m ensemble.serving.bundle build

serve-v2:   ## run the API on the CURRENT serving bundle (override with ENSEMBLE_BUNDLE=...)
	PYTHONPATH=src .venv/bin/uvicorn ensemble.api.app:app --port $(PORT)

serve-smoke: ## end-to-end smoke test in a real server on a synthetic bundle (no H&M data needed)
	$(PY) scripts/serve_smoke.py

bench-serving: ## local latency / throughput / memory benchmark of the serving API on the CURRENT bundle
	$(PY) scripts/bench_serving.py

report-verify: ## assert every headline figure of the final report against stored artifacts
	$(PYTHON) scripts/verify_final_report.py
