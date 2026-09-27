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

Scaffolding. No results yet. See the milestones in the design document.

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
  evaluation/        MAP@12, Recall@K, NDCG@K, relative lift vs. popularity
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
