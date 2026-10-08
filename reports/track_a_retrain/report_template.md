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

{{LADDER}}

### MAP@12 per fold

{{PER_FOLD}}

### Predeclared gates

{{GATES}}

**R3 was not run.** The trigger requires the better of R1/R2 (R1) to pass every gate except the
MAP threshold, with a MAP gain in [+0.5%, +1.0%). R1 fails the MAP interval and segment gates, and
its gain is +0.25%. The trigger is evaluated mechanically in `comparison.json` (`r3_trigger`).

## 4. Segments (relative change vs R0, 95% interval)

{{SEGMENTS}}

*Reading.*
- The retrained rankers gain mostly on recently-introduced and repeat-purchase items. Both are
  significant, and on R1 they are the largest Recall@12 gains (+1.81% and +0.89%).
- Tail items do not move significantly, although the appended SASRec candidates raised tail candidate
  recall by 71% (D-044).
- The new-customer interval is wide (7.1% of rows; SASRec has no vector for them, so they receive
  no appended candidates). R1's point estimate is +1.94%, but its lower bound sits just past the
  -2% limit.

## 5. Seed stability (R1, the best quality-only variant)

{{SEEDS}}

*Reading.* Fold-level differences between R1 and R0 (for example 2020-08-26: -0.00049 at seed 42,
+0.00010 at seed 43) are of the same size as the variation between seeds. None of the three seeds
reaches the +1.0% threshold.

## 6. Feature importance and SHAP checks

Gain share, mean over the six seed-42 models:

{{IMPORTANCE}}

{{SHAP}}

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

{{COSTS}}

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
