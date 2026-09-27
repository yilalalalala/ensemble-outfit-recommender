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

**Track B — accessory completion.** Given a garment, which piece of jewellery or
accessory completes it? The competition never asks this: it models *what will
this person buy next*, not *what goes with what*. The signal comes from basket
structure — items one customer bought on one day — filtered by Pointwise Mutual
Information so that "these two are both popular" cannot masquerade as "these two
go together".

Full specification in [docs/DESIGN.md](docs/DESIGN.md).

## Status

Scaffolding. No results yet. See the milestones in the design document.

## Getting the data

The dataset requires a Kaggle account and acceptance of the competition rules.

1. Open the competition and accept the rules:
   <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations>
2. Get an API token: Kaggle → your profile → Settings → API → **Create New
   Token**. This downloads `kaggle.json`.
3. Install it:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
```

4. Download:

```bash
./scripts/download_data.sh            # CSVs only, ~3.5 GB
./scripts/download_data.sh --images   # adds ~30 GB of article images
```

Start without images. They are only needed from M4, and the CSVs are enough to
reach the first benchmarked score.

## Layout

```
docs/                design, data model, decisions
src/ensemble/
  data/              ingestion, schema, temporal splits
  features/          feature construction with an enforced cutoff date
  candidates/        Track A candidate generation
  ranking/           Track A ranker
  completion/        Track B pair mining and two-tower model
  evaluation/        MAP@12, Recall@K, NDCG@K, lift over popularity
  api/               serving
scripts/             data acquisition and pipeline entry points
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
