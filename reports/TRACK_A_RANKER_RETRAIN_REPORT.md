# Track A ranker retrain for the SASRec-append-to-300 candidates

**Offline evaluation readout.** 2026-10-08 · branch `track-a-ranker-retrain` (from `9e3fd92`) ·
local only (nothing pushed, merged, published or submitted). The protocol, the ablation ladder and
the adoption rule were committed before any result was computed (`f552c22`,
`docs/CLAUDECODE_TRACK_A_RANKER_RETRAIN_PLAN.md`). Every figure below is rendered from
`reports/track_a_retrain/comparison.json` and `shap_checks.json` by
`scripts/ranker_retrain_report_tables.py`. `scripts/verify_ranker_retrain_report.py` asserts the
decision and the headline figures. Measured facts are stated plainly; *reading* marks inference.

## 1. Result

**Nothing is adopted (D-045): the fixed ranker R0 stays.** Retraining the LambdaRank ranker on
the distribution it now ranks (R1) and adding SASRec provenance features (R2) both fail the
predeclared rule. R3 (a hyper-parameter grid) was not triggered.

- **R1** (retrained, same 91 features): MAP@12 0.03535 → **0.03544**, **+0.25% [-0.07, +0.60]**. Recall@12
  +0.68% [+0.37, +0.97] and NDCG@12 +0.39% [+0.14, +0.66] are significant. Wins 5/6 folds. It fails
  the +1.0% MAP threshold, the MAP interval and the segment gate: the new-customer MAP@12 lower bound is
  -2.04%, against a -2% limit.
- **R2** (R1 + `sasrec_rank`, `sasrec_rank_pct`, `src_sasrec_append`): MAP@12 0.03544, **+0.23% [-0.10, +0.58]**,
  4/6 folds. It fails only the MAP threshold and the MAP interval.
- **The R1 gain is the size of seed noise.** With ranker seeds 43 and 44, R1's gain over R0 is
  +0.05% and +0.13% (seed 42: +0.25%). The three seeds average 0.03541 MAP@12 (SD 0.00003) and a
  +0.14% gain.
- **Best quality-only variant:** R1 (it beats R2 at the fifth decimal). Neither variant costs more
  than the limits allow; both fail on quality.

**The honest headline.** The low conversion of the added candidates is not mainly a
training-distribution problem. Retraining on the 300-candidate distribution raises Recall@12 by
about 0.7%, but MAP@12 moves by less than seed noise. The ranker already ranks most of what it can
from these features. A retrained ranker trusts appended candidates *less* than the fixed one, not
more: they are 7.7% of R0's top 12 but 3.9% of R1's.

## 2. Protocol and controls

- **Frozen D-038 protocol.**
  - Six reporting folds 2020-08-05 … 2020-09-09.
  - Eligible catalogue = sold in the 28 days before the cutoff, plus age-band back-fill.
  - Ties broken on `article_id`.
  - Customer-cluster bootstrap across folds (1,000 resamples, seed 0).
  - The observed week 2020-09-16 was neither read nor scored.
- **Distribution matching.**
  - Each ranker training week (2020-07-08 … 2020-09-02) gets exactly the inference candidate set, built
    by the same function (`retrieval_upgrade.build_cand`, spec `sasrec_A300`). That set is the Phase-2
    union from pools built before the week, followed by the de-duplicated exact SASRec list of the model
    trained for that week's cutoff, up to 300 per customer.
  - The training recipe is otherwise unchanged:
    - label-week buyers with ≥ 1 retrieved positive;
    - every positive kept;
    - 50% deterministic hashed negative downsampling;
    - early stopping on the most recent week, then a refit on all four weeks.
- **Identical retrieval.** Every variant scores the same cached evaluation candidates. Candidate
  recall is 0.2247 for all of them, asserted per customer.
- **Reproduction.** R0 rescored through the new pipeline reproduces the D-044 `sasrec_A300` evaluation
  exactly: identical AP and candidate hits for all 446,056 (customer, fold) rows.
- **Online-compatible features.**
  - The three provenance features come from the request-time SASRec query vector and the candidate
    list.
  - No label-week data is used. Tests show that label-week rows outside the candidate set cannot
    change the training rows.
- **Decision on seed 42, as declared.** Seeds 43 and 44 were run for the best variant, to measure
  stability only.

## 3. Ablation ladder (six folds, seed 42)

| variant | MAP@12 | vs R0 | Recall@12 vs R0 | NDCG@12 vs R0 | folds won | adopted by rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| R0 (fixed ranker) | 0.03535 | — | — | — | — | baseline |
| R1 | 0.03544 | +0.25% [-0.07, +0.60] | +0.68% [+0.37, +0.97] | +0.39% [+0.14, +0.66] | 5/6 | no |
| R2 | 0.03544 | +0.23% [-0.10, +0.58] | +0.75% [+0.43, +1.05] | +0.40% [+0.15, +0.66] | 4/6 | no |

### MAP@12 per fold

| fold | R0 | R1 | R2 |
| --- | ---: | ---: | ---: |
| 2020-08-05 | 0.03069 | 0.03076 | 0.03083 |
| 2020-08-12 | 0.03440 | 0.03459 | 0.03431 |
| 2020-08-19 | 0.03429 | 0.03460 | 0.03441 |
| 2020-08-26 | 0.03444 | 0.03395 | 0.03442 |
| 2020-09-02 | 0.03938 | 0.03966 | 0.03966 |
| 2020-09-09 | 0.03899 | 0.03922 | 0.03904 |

### Predeclared gates

| gate | R1 | R2 |
| --- | --- | --- |
| MAP@12 gain ≥ +1.0% | +0.25% ✗ | +0.23% ✗ |
| MAP@12 difference CI above 0 | [-0.07, +0.60] ✗ | [-0.10, +0.58] ✗ |
| folds won ≥ 4/6 | 5/6 ✓ | 4/6 ✓ |
| Recall@12 lower bound ≥ 0 | [+0.37, +0.97] ✓ | [+0.43, +1.05] ✓ |
| NDCG@12 lower bound ≥ 0 | [+0.14, +0.66] ✓ | [+0.15, +0.66] ✓ |
| segment lower bounds ≥ −2% | min -2.04% (map@12_new) ✗ | min -0.51% (recall@12_tail) ✓ |
| candidate recall identical | 0.2247 ✓ | 0.2247 ✓ |
| scoring time ≤ +20% | +0.7% | +10.7% |
| scoring RSS ≤ +25% | +14.4% | +19.2% |
| training time ≤ 2× | 0.98× ✓ | 1.06× ✓ |
| **all gates** | **fail** | **fail** |

**R3 was not run.** The trigger requires the better of R1/R2 (R1) to pass every gate except the
MAP threshold, with a MAP gain in [+0.5%, +1.0%). R1 fails the MAP interval and segment gates, and
its gain is +0.25%. The trigger is evaluated mechanically in `comparison.json` (`r3_trigger`).

## 4. Segments (relative change vs R0, 95% interval)

| segment | share | R0 | R1 vs R0 | R2 vs R0 |
| --- | ---: | ---: | ---: | ---: |
| new-customer MAP@12 | 7.1% | 0.00940 | +1.94% [-2.04, +6.17] | +3.58% [-0.32, +7.70] |
| returning MAP@12 | 92.9% | 0.03734 | +0.21% [-0.11, +0.54] | +0.16% [-0.15, +0.50] |
| tail-item Recall@12 | 32.5% | 0.01945 | +0.40% [-0.38, +1.19] | +0.26% [-0.51, +1.08] |
| repeat-purchase Recall@12 | 3.8% | 0.63892 | +0.89% [+0.64, +1.14] | +0.54% [+0.29, +0.80] |
| non-repeat Recall@12 | 96.2% | 0.03842 | +0.54% [+0.07, +0.99] | +0.88% [+0.44, +1.36] |
| head-item Recall@12 | 63.4% | 0.08650 | +0.71% [+0.39, +1.03] | +0.80% [+0.46, +1.12] |
| recent-item Recall@12 | 26.0% | 0.06897 | +1.81% [+1.19, +2.41] | +1.76% [+1.18, +2.32] |

*Reading.*
- The retrained rankers gain mostly on recently-introduced and repeat-purchase items. Both are
  significant, and on R1 they are the largest Recall@12 gains (+1.81% and +0.89%).
- Tail items do not move significantly, although the appended SASRec candidates raised tail candidate
  recall by 71% (D-044).
- The new-customer interval is wide (7.1% of rows; SASRec has no vector for them, so they receive
  no appended candidates). R1's point estimate is +1.94%, but its lower bound sits just past the
  -2% limit.

## 5. Seed stability (R1, the best quality-only variant)

| R1 seed | MAP@12 | vs R0 | 2020-08-05 | 2020-08-12 | 2020-08-19 | 2020-08-26 | 2020-09-02 | 2020-09-09 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.03544 | +0.25% | 0.03076 | 0.03459 | 0.03460 | 0.03395 | 0.03966 | 0.03922 |
| 43 | 0.03537 | +0.05% | 0.03079 | 0.03423 | 0.03439 | 0.03454 | 0.03939 | 0.03894 |
| 44 | 0.03540 | +0.13% | 0.03084 | 0.03449 | 0.03450 | 0.03430 | 0.03933 | 0.03902 |

R1: mean 0.03541, SD 0.00003 over 3 seeds; mean gain vs R0 +0.14%.

*Reading.* Fold-level differences between R1 and R0 (for example 2020-08-26: -0.00049 at seed 42,
+0.00010 at seed 43) are of the same size as the variation between seeds. None of the three seeds
reaches the +1.0% threshold.

## 6. Feature importance and SHAP checks

Gain share, mean over the six seed-42 models:

| # | R1 | R2 |
| ---: | --- | --- |
| 1 | `ca_w_prod` 9.4% | `ca_w_prod` 9.7% |
| 2 | `product_type_no` 7.5% | `product_type_no` 7.5% |
| 3 | `ca_wshare_prod` 6.8% | `ca_wshare_prod` 5.8% |
| 4 | `ca_days_since_article` 5.8% | `ca_days_since_article` 5.4% |
| 5 | `nn_sasrec_dot` 5.3% | `c_days_since_last` 4.5% |
| 6 | `a_days_since_first_sale` 4.5% | `nn_sasrec_dot` 4.2% |
| 7 | `colour_group_code` 4.1% | `colour_group_code` 4.1% |
| 8 | `nn_lightgcn_dot` 3.9% | `nn_lightgcn_dot` 3.9% |
| 9 | `c_days_since_last` 3.7% | `garment_group_no` 3.8% |
| 10 | `ca_w_article` 3.2% | `ca_w_article` 3.2% |

R2 provenance gain share (mean over folds): `sasrec_rank` 0.91%, `sasrec_rank_pct` 0.85%, `src_sasrec_append` 0.43%; total 2.20%.

Sample: the top-12 list of the first 2,000 customers (by id) of fold 2020-09-09's first evaluation part, seed 42 models.

| | R0 | R1 | R2 |
| --- | ---: | ---: | ---: |
| rows explained | 24,000 | 24,000 | 24,000 |
| additivity, max abs error | 2.2e-14 | 1.4e-14 | 1.7e-14 |
| top-12 rows that are SASRec-appended | 7.7% | 3.9% | 4.2% |
| provenance share of mean absolute SHAP | 0.00% | 0.00% | 3.56% |
| rows with a provenance feature in the top-3 absolute SHAP | 0 | 0 | 233 |
| reason chips shown | 47,044 | 47,246 | 47,256 |
| rows without any chip | 58 | 37 | 25 |
| unsupported chips (evidence-gated, D-030) | 0 | 0 | 0 |
| unsupported chips if the gate were off | 8,168 | 10,356 | 10,442 |

- **Additivity.** TreeSHAP contributions sum to the raw score to within 2.2e-14 for all three
  models.
- **Provenance attribution.** In R2 the three provenance features carry 3.56% of the mean absolute SHAP
  and 2.20% of the gain. They rank in the top 3 for 233 of 24,000 explained rows.
- **Reason chips (D-030).**
  - The provenance features map to no reason chip, so they cannot create one.
  - With the evidence gate on, no shown chip is unsupported in any variant.
  - With the gate off, about one chip in five (8,168–10,442) would claim something the raw features
    do not support. The gate stays necessary.

## 7. Cost (one job at a time on the Apple M5 / 16 GB laptop)

| | R0 | R1 | R2 |
| --- | ---: | ---: | ---: |
| scoring time, six folds (s) | 544 | 548 | 602 |
| peak scoring RSS (GB) | 2.85 | 3.26 | 3.40 |
| training time per fold, mean (s) | 843 | 827 | 890 |
| peak training RSS (GB) | — | 4.72 | 4.70 |
| training rows per fold, mean | — | 19.98M | 19.98M |
| trees, mean | — | 341 | 382 |

Shared, built once for all variants: 9 training matrices (4.6–5.4M rows after the training-row rule, 245 s in total, peak 4.4 GB) and 6 evaluation matrices (20.2–22.6M rows, 434 s, peak 4.9 GB).

- Scoring is the model prediction over the six evaluation matrices. Training is matrix load plus
  early stopping plus refit, per fold. Both are within the limits for R1 and R2.
- R2's extra scoring time (+10.7%) comes from 41 more trees on average and three more features.
- Per-fold timings are noisy on a laptop: the seed-43 2020-08-05 fit took 1,456 s because the report
  stage ran next to it (see §8). Seed-42 timings, which the gate uses, ran with no other job of this
  project.

## 8. Incidents

| incident | handling |
| --- | --- |
| The `report` stage (bootstrap) was rerun while the seed-43 fit of 2020-08-05 was training | that fit's time (1,456 s vs 749–1,203 s for the other 11 seed-43/44 fits) is not used by any gate; the result (model, MAP@12) is unaffected |
| The plan anticipated possible memory pressure from 20M-row training sets | the peak was 4.7 GB; no change was needed |
| The full test suite segfaulted in a new test that loaded a LightGBM model after torch (two OpenMP runtimes in one process) | `base_features` reads feature names from the model text header; the new fit/SHAP test runs LightGBM in a fresh interpreter, as `tests/test_ranker.py` already does. No experiment job was affected |

## 9. Limitations and next experiment

- Offline only. The served Track A recommendations still come from the Phase-2 model.
- The decision uses one ranker seed, as declared. The seed replicates show that the R1 effect is
  within seed noise, which makes the "no" robust.
- The SHAP sample is one fold and the first 2,000 customers by id. It is a fixed, label-blind sample,
  not a random one.
- New customers are 7% of rows and get no SASRec candidates, so their interval is wide.
- R3 was not run (trigger not met), so a better-tuned ranker is not excluded. The trigger was set
  before results to avoid tuning on the reporting folds.

**Next recommended experiment.** The ranker is not the bottleneck at this candidate set. D-044's
diagnostics found that 55.7% of purchased pairs are proposed by no channel and 40.8% are absent from
all three neural top-1000 lists. Raising the ceiling therefore needs a new signal in retrieval, such
as a content/two-tower channel that uses article metadata and image embeddings. A ranker change will
not do it. A smaller, cheaper follow-up is a features experiment on what drives the significant
repeat and recent-item gains, before any ranker change is reconsidered.

## 10. Reproduction

```bash
export ENSEMBLE_CONFIG=track_a_research PYTHONPATH=src
for w in 2020-07-08 2020-07-15 2020-07-22 2020-07-29 2020-08-05 2020-08-12 2020-08-19 2020-08-26 2020-09-02 2020-09-09; do
  .venv/bin/python -m ensemble.research.ranker_retrain pools $w; done      # cached weeks are skipped
bash reports/track_a_retrain/run_chain.sh                                  # matrices + R0/R1/R2 seed 42 (~3.5 h)
.venv/bin/python -m ensemble.research.ranker_retrain queue R1 43,44        # seed replicates (~3 h)
.venv/bin/python -m ensemble.research.ranker_retrain report
.venv/bin/python -m ensemble.research.ranker_retrain shap 2020-09-09 42
.venv/bin/python scripts/ranker_retrain_report_tables.py --render
.venv/bin/python scripts/verify_ranker_retrain_report.py
```
