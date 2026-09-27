.PHONY: help setup data ingest test lint
help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## create the venv and install dependencies
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

data:   ## download the CSVs from Kaggle (see README for credentials)
	./scripts/download_data.sh

ingest: ## load the CSVs into DuckDB and run integrity checks
	.venv/bin/python -m ensemble.data.ingest

test:   ## run the test suite
	.venv/bin/python -m pytest tests/ -q
