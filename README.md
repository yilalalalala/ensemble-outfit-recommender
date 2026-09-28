# Ensemble

A fashion recommender on the H&M Group transaction dataset — 31.8M purchases,
1.37M customers, 105k articles with images.

Two tracks:

**Track A — next-purchase recommendation.** The task published by the H&M
Personalized Fashion Recommendations competition, adopted unchanged: predict the
12 articles each customer buys in the following seven days, scored by MAP@12
(Mean Average Precision at 12). Running the published task on the published data
means the result is checkable against 3,006 competing teams rather than against
a number of our own making.

**Track B — outfit completion ("Complete the Look").** Given a garment, which
bottoms, shoes, bag or accessory completes it? Jewellery is the showcase slice.
The competition never asks this: it models *what will
this person buy next*, not *what goes with what*. The signal comes from basket
structure — items one customer bought on one day — filtered by Pointwise Mutual
Information so that "these two are both popular" cannot masquerade as "these two
go together".

Full specification in [docs/DESIGN.md](docs/DESIGN.md); vocabulary in
[docs/GLOSSARY.md](docs/GLOSSARY.md).

## Status

MVP complete (M0–M6): [reports/MVP_REPORT.md](reports/MVP_REPORT.md). Kaggle private MAP@12 0.0319.
Post-MVP (rolling backtest, M7a FashionCLIP + DeepFashion2 visual search, M7b assistant):
[reports/IMPROVEMENTS_REPORT.md](reports/IMPROVEMENTS_REPORT.md). Next: human gold labels, owner pair
benchmark, optional M8.

## Running it

```bash
brew install libomp            # macOS: LightGBM's OpenMP runtime
make setup                     # Python 3.11 venv via uv
make data                      # full dataset from Kaggle (see below)
make mvp                       # ingest → baselines → retrieval → ranker → Track B → submission → serving → tests
make serve                     # http://localhost:8010
make mlflow                    # experiment runs at http://localhost:5000
```

`make mvp` takes about 1.5 hours on an M-series laptop with 16 GB RAM. Each
stage can also be run on its own (`make help`).

**Troubleshooting.** If `import ensemble` fails inside the venv on macOS, the
editable-install `.pth` file may have been given the "hidden" flag (common in
synced folders), and Python skips hidden `.pth` files. The Makefile sets
`PYTHONPATH=src`, so it is unaffected; to fix the venv itself, run
`chflags nohidden .venv/lib/python3.11/site-packages/*.pth`.

## Getting the data

The dataset requires a Kaggle account and acceptance of the competition rules.

1. Open the competition and accept the rules:
   <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations>
2. Install the Kaggle CLI (v2+) and log in:

```bash
uv tool install kaggle      # or: pip install kaggle
kaggle auth login           # browser-based login, credentials cached locally
```

3. Alternatively, generate a token at <https://www.kaggle.com/settings/api> and
   save it to `~/.kaggle/access_token` (or export `KAGGLE_API_TOKEN`). The legacy
   `~/.kaggle/kaggle.json` also still works.

4. Download:

```bash
./scripts/download_data.sh            # CSVs only, ~3.5 GB
./scripts/download_data.sh --images   # adds ~30 GB of article images
```

The MVP needs only the CSVs. Images are shown in the web UI (with a placeholder
if missing) and are used for visual search from M7a.

## Layout

```
docs/                design, data model, decisions
src/ensemble/
  data/              ingestion, schema, temporal splits
  features/          feature construction with an enforced cutoff date
  candidates/        Track A candidate generation
  ranking/           Track A ranker
  completion/        Track B pair mining and two-tower model
  evaluation/        MAP@12, Recall@K, NDCG@K, relative lift vs. popularity
  api/               serving store builder, FastAPI app, web UI
configs/             all tunables (default.yaml)
reports/             evaluation readouts, model cards, submission.csv
scripts/             data acquisition
tests/
data/                gitignored; raw / interim / processed
```

## Conventions

- **Temporal splits only.** Random splits leak the future into the past and
  every metric becomes fiction. Feature builders take a cutoff date as a
  required argument.
- **Popularity is the baseline everywhere.** A recommender that cannot beat
  "show the best sellers" has not been shown to work.
- **Sampling parameters are configuration, not constants.** A full-data run is a
  config change.
