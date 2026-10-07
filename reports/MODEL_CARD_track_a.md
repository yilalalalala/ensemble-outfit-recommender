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
Phase 2 late submission (2026-10-02) scored private **0.03330**, public 0.03263:
+4.4% / +5.2% over the MVP, in line with the offline gain. Local test (0.03918) again ran
~15% above the private leaderboard.

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

---

## Update 2026-10-05 — research benchmark and neural score features (D-038 … D-040)

Full readout: [`TRACK_A_RESEARCH_TRACK_B_PRODUCTION_REPORT.md`](TRACK_A_RESEARCH_TRACK_B_PRODUCTION_REPORT.md).
The numbers above are kept as history; they come from a different protocol (one validation week,
no eligibility restriction) and are not directly comparable with the table below.

| | |
| --- | --- |
| **Change** | Three features added to the Phase-2 ranker: the dot product of customer and candidate vectors from reproduced **BPR-MF**, **LightGCN** and **SASRec** (gSASRec loss), each trained per label week on data before its cutoff (`nn_<model>_dot`). Candidates, other features and the ranker recipe are unchanged (91 features). |
| **Not adopted** | The same models as retrieval channels (+0.72% MAP@12 over the final system, but +197% training time against a +50% limit) and the allocator's re-allocated caps (+0.74% over Phase 2, cost limit missed by 0.01 pt). |
| **Protocol** | Six rolling folds 2020-08-05 … 2020-09-09, eligible catalogue = sold in the 28 days before the cutoff, shared age-band fallback, customer-cluster bootstrap across folds (D-038). The test week 2020-09-16 is confirmation evidence and was not used. |

Six-fold pooled MAP@12 (`reports/track_a_research/comparison.json`):

| system | MAP@12 |
| --- | ---: |
| repeat purchase + age-band popularity | 0.02212 |
| BPR-MF / LightGCN / SASRec (3 seeds each) | 0.00924 / 0.00937 / 0.01432 |
| Phase-2 ensemble, reproduced (3 seeds) | 0.03409 ± 0.00006 |
| **Final: + neural score features (3 seeds)** | **0.03525 ± 0.00004** |

Paired bootstrap (seed 42): vs Phase 2 **+3.55% [+3.10, +3.97]**, 6/6 folds; vs SASRec, the strongest
reproduced baseline, +147% [+143, +151]. Every ablation hurts: recency/velocity −3.1%, SASRec score
−1.9%, repeat features −1.6%, retrieval provenance −0.9%, BPR-MF + LightGCN scores −0.4%.

**Serving status.** The served Track A recommendations (`make submission serving`) are still the
Phase-2 model: shipping the new features needs the three neural models retrained on all data and
scored for every customer, which this round did not do.

---

## Update 2026-10-06 — retrieval ceiling: SASRec appended to 300 candidates (D-044)

Readout: [`TRACK_A_RETRIEVAL_UPGRADE_REPORT.md`](TRACK_A_RETRIEVAL_UPGRADE_REPORT.md).

| | |
| --- | --- |
| **Change** | Retrieval: the current union (Phase-2 caps, ~160 candidates) followed by the customer's exact SASRec top list, de-duplicated, up to 300 candidates. Nothing is evicted; the ranker is unchanged. |
| **Six-fold result** (vs the 2026-10-05 final system, same ranker) | candidate recall 0.1757 → 0.2247 (+27.9%); MAP@12 0.03524 → 0.03535 (+0.33% [+0.18, +0.48]); Recall@12 +0.43%; 6/6 folds |
| **Cost** | rank + score time 1.63×, peak RSS +17% (idle machine, one fold at a time) |
| **Limitation** | The ranker was trained on ~160 candidates and converts only ~0.5% of the added candidate recall into Recall@12; 55.7% of purchased pairs are proposed by no channel. New customers are unchanged. Not yet in the served recommendations. |
