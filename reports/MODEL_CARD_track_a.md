# Model card: Track A next-purchase ranker

| | |
| --- | --- |
| **Task** | Predict the 12 articles each customer buys in the next 7 days (H&M Kaggle task, unchanged, D-001) |
| **Architecture** | Three stages. **Retrieval:** 9 channels, ~160 candidates per customer, caps chosen on a recall-vs-size frontier (D-027). **Ranking:** LightGBM LambdaRank, 63 leaves, number of trees from temporal early stopping (≈250–450) (D-028). **Re-ranking:** availability proxy (D-029) |
| **Training data** | 4 label weeks before the target week; features strictly from earlier data (D-003); groups with ≥1 retrieved positive (D-014); 50% of negatives kept (deterministic hash) |
| **Features** | 88: customer, article, customer×article (lifetime `_life` and recency-weighted `ca_w_*`), short-term sales velocity, per-channel retrieval score and rank. Schema hash `ad12512a7cbc`, vocabulary version in `data/processed/vocab/vocab.json` |
| **Artifacts** | `data/processed/models/ranker_{val,test,submission}.txt`; config `configs/default.yaml` (`retrieval`, `features`, `ranker`, `rerank`); run manifests inside `reports/m3_ranker_{val,test}.json` |
| **Owner / date** | Ensemble project, Phase 2 upgrade 2026-10-02 (MVP version 2026-09-27) |

## Intended use

Personalised "Recommended for you" module on the home page (DESIGN §7.2), with
evidence-gated reason chips (D-030).

## Offline metrics (MAP@12)

| system | validation (2020-09-09..15) | test (2020-09-16..22) |
| --- | ---: | ---: |
| popularity (global) | 0.00660 | 0.00875 |
| repeat purchase + age-band popularity (best rule baseline) | 0.02524 | 0.02689 |
| MVP ranker (116 candidates, 6 channels) | 0.03546 | 0.03699 |
| **Phase 2 ranker** | **0.03772** | **0.03918** |
| relative lift vs best rule baseline | +49.4% | +45.7% |

**4-week rolling backtest** (2020-08-19 … 2020-09-09): 0.03542 ± 0.00284 vs
0.03298 ± 0.00325 for the MVP recipe; paired customer bootstrap +7.5% [+6.9, +8.2]
pooled; better in every week. The test week was evaluated once, after every choice was
frozen.

Segments (test): returning customers 0.04188; new customers 0.00839 (the 4-week
backtest gives 0.01053 ± 0.00217 vs 0.00837 for the MVP; the age-band fallback is
worse in 3 of 4 weeks, D-017). Candidate recall 20.1% on test (MVP 13.1% on
validation); item cold-start recall is 0 for every system.

**External check.** The MVP's Kaggle late submission scored private MAP@12
**0.03189** (public 0.03102); local test then ran ~14% above the leaderboard. The
Phase 2 submission file has been regenerated but **not submitted**; its leaderboard
score is unknown.

## Top features (gain share, validation)

Recency-weighted share and sum of the customer's purchases of this style
(`ca_wshare_prod` 9.5%, `ca_w_prod` 6.4%), product type (7.9%), days since the
customer last bought the article (6.4%), days since the customer's last purchase
(4.4%), colour (4.2%), days since the article's first sale (3.5%), yesterday's sales
(2.8%), number of channels that proposed the candidate (2.5%).

## Limitations

- **Recall above ~160 candidates does not convert:** 200 candidates raise recall to 21%
  but not MAP@12, and do not fit in 16 GB for training.
- **Item cold start:** articles first sold in the target week (3.9% of test purchases)
  cannot be retrieved; there is no launch calendar.
- **Availability is a proxy** (days since last observed sale), not stock.
- **Training cost:** ≈10–14 minutes per week at 160 candidates on a 16 GB laptop
  (early stopping + refit, deterministic mode).
- Offline only. Online impact needs an A/B test (CTR, conversion).
