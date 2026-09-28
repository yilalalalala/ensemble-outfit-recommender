# Model card: Track A next-purchase ranker

| | |
| --- | --- |
| **Task** | Predict the 12 articles each customer buys in the next 7 days (H&M Kaggle task, unchanged, D-001) |
| **Architecture** | Two-stage: 6 retrieval channels (~116 candidates per customer) → LightGBM LambdaRank (200 trees, 63 leaves) |
| **Training data** | 4 label weeks before the target week; features strictly from earlier data (D-003); groups with ≥1 retrieved positive (D-014) |
| **Artifacts** | `data/processed/models/ranker_{val,test,submission}.txt`; config `configs/default.yaml` (`retrieval`, `ranker`) |
| **Owner / date** | Ensemble project, 2026-09-27 |

## Intended use

Personalised "Recommended for you" module on the home page (DESIGN §7.2). Not
intended for new customers without a fallback: see limitations.

## Offline metrics (MAP@12)

| system | validation (2020-09-09..15) | test (2020-09-16..22) |
| --- | ---: | ---: |
| popularity (global) | 0.00660 | 0.00875 |
| popularity by age band | 0.00756 | 0.00960 |
| repeat purchase + age-band popularity (best baseline) | 0.02524 | 0.02689 |
| **LightGBM LambdaRank** | **0.03546** | **0.03699** |
| relative lift vs best baseline | +40.5% | +37.6% |

Segments (test): returning customers 0.03951; new customers 0.00831 (vs 0.00909
for age-band popularity); item cold-start Recall@12 is 0 for every system.

**External check — Kaggle late submission (2026-09-28):** private MAP@12
**0.03189**, public 0.03102. For reference, private leaderboard 1st place scored
0.0379 and ~45th place (silver) 0.0300. The local test score (0.0370) runs about
14% above the leaderboard.

## Top features (gain share)

Days since the customer last bought the article (13%), product type (7%),
last-week sales (7%), days since the customer's last purchase (6%), repeat
purchase score and rank (10%), item-CF rank (4%), share of the customer's
purchases in this product (4%).

## Limitations

- Retrieval caps performance: merged candidate recall is 13.1% at 116
  candidates.
- New customers: the ranker does not beat age-band popularity on the test week.
  A segment-level fallback policy should be A/B tested.
- Item cold start: articles first sold in the target week (3.9% of test purchases)
  cannot be retrieved without content-based retrieval.
- Offline only. Online impact needs an A/B test (CTR, conversion).
