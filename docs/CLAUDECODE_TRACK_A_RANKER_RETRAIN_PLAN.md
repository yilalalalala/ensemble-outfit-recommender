# Track A ranker retrain for the SASRec-append-to-300 candidate distribution

**Status:** frozen before any result of this milestone was computed or read
**Branch:** `track-a-ranker-retrain`, from `9e3fd92` (`track-a-retrieval-upgrade`)
**Baseline (R0):** `sasrec_A300` candidates scored by the old fixed per-fold ranker (D-044):
candidate recall **0.2247**, MAP@12 **0.03535** (six reporting folds, seed 42)

## Why

D-044 widened retrieval to 300 candidates per customer, but the ranker was trained on the ~160
candidate distribution and converted only ~0.5% of the added candidate recall into Recall@12. The
appended SASRec candidates carry no Phase-2 provenance (`n_channels = 0`, every channel rank NULL),
a region the old ranker never saw. This milestone retrains the ranker on the distribution it is
asked to rank, then tests whether SASRec provenance features add more.

## Frozen protocol (D-038, unchanged)

- Six reporting folds 2020-08-05 … 2020-09-09; eligible catalogue = sold in the 28 days before the
  cutoff; restrict + age-band back-fill; ties on `article_id`; customer-cluster bootstrap across
  folds, 1,000 resamples, seed 0. **2020-09-16 is never read or scored.**
- **Distribution matching.** For every ranker training label week *w* (the four weeks before each
  fold), build exactly the inference candidate set: the Phase-2 union at the Phase-2 caps from
  channel pools computed with data before *w*, followed by the de-duplicated exact SASRec list of
  the model trained for *w*'s cutoff, up to 300 per customer (`retrieval_upgrade.build_cand`). The
  same function builds training and evaluation candidates.
- **Training recipe unchanged** apart from the candidate set: label-week buyers with ≥ 1 retrieved
  positive (D-014), every positive kept, 50% deterministic hashed negative downsampling (D-028),
  LambdaRank with the `ranker` parameters in `configs/default.yaml`, early stopping on the most
  recent training week, refit on all four weeks.
- **Retrieval is identical across variants**: every variant scores the same cached evaluation
  candidates, so candidate recall must equal 0.2247 exactly (asserted).
- Choices (R3 only) use the D-038 tuning folds 2020-07-22 and 2020-07-29, whose training weeks
  2020-07-08 … 2020-07-22 already have point-in-time models. They train on the 2 and 3 available
  earlier weeks rather than 4; this is a declared deviation that affects only R3's grid choice.

## Ablation ladder

| | candidates | ranker training data | features |
| --- | --- | --- | --- |
| **R0** | sasrec_A300 | old ~160-candidate distribution (existing models) | the 91 final-system features |
| **R1** | sasrec_A300 | retrained on sasrec_A300 for every training week | the same 91 |
| **R2** | sasrec_A300 | as R1 | 91 + `sasrec_rank` (rank in the customer's exact SASRec list over the eligible catalogue, ≤ 1,000, else NULL), `sasrec_rank_pct` (that rank divided by the number of eligible scored articles), `src_sasrec_append` (1 if the candidate entered only through the SASRec append). The raw SASRec score is already `nn_sasrec_dot`. All three are point-in-time and computable online from the request-time SASRec query vector. |
| **R3** | sasrec_A300 | as the better of R1/R2 | same, with LambdaRank `num_leaves` ∈ {31, 63, 127} × `min_child_samples` ∈ {100, 400} chosen on the tuning folds, then frozen |

R3 runs **only if** the better of R1/R2 passes every gate except the MAP@12 threshold and its
MAP@12 gain over R0 is between +0.5% and +1.0% (tuning might close a near miss); otherwise it is
not run and the reason is recorded.

## Predeclared adoption rule (versus R0, six folds, seed 42 for the decision)

A variant is adopted only if **all** hold:

1. MAP@12 relative gain ≥ **+1.0%**, with the paired customer-cluster bootstrap 95% interval of the
   difference entirely above zero;
2. MAP@12 higher in ≥ **4 of 6** folds;
3. Recall@12 and NDCG@12 difference intervals with lower bound ≥ 0;
4. no material segment regression: lower bound of the relative change ≥ **−2%** for new-customer
   MAP@12, returning-customer MAP@12, tail-item Recall@12, repeat-purchase Recall@12 and
   non-repeat Recall@12;
5. candidate recall identical to R0 (0.2247);
6. online-compatible features only;
7. scoring time (model prediction over the evaluation candidates) ≤ **+20%** of R0's, peak scoring
   RSS ≤ **+25%** of R0's, and ranker training time (load + early stopping + refit) ≤ **2×** the
   current final ranker's (mean 843 s per fold), all measured on an otherwise idle machine.

If several pass, adopt the simplest (R1 before R2 before R3). If none passes, keep R0 and report the
best quality-only variant. The rule is not relaxed after results are seen.

## Stability, explanation, cost

- The adopted (or best) variant is refitted with ranker seeds 42, 43, 44; mean, SD and per-fold
  values are reported, and the decision uses seed 42 as declared.
- Feature importance (gain share) per variant; SHAP (`pred_contrib`) on a fixed sample: additivity
  check (contributions sum to the raw score), the share of attribution carried by the new
  provenance features, and the D-030 evidence-gated reason-chip audit (no unsupported chip).
- Costs per fold: training-matrix build, fit, evaluation-matrix build, scoring time, peak RSS.

## Engineering

- Restartable per-week training matrices and per-fold evaluation matrices under
  `data/interim/track_a_retrain/<hash>/`; one fit per subprocess; sequential queue; Mac kept awake
  with `caffeinate` tied to the job.
- Tests: training-candidate construction ignores the label week; training and evaluation use the
  same candidate function; provenance features ignore the label week; feature contract
  (R1 = the 91 final features, R2 = R1 + the 3 named); identical-seed fits reproduce.

## Deliverables

ADR D-045, Track A model-card update, `reports/TRACK_A_RANKER_RETRAIN_REPORT.md` with tables
rendered from stored artifacts, `scripts/verify_ranker_retrain_report.py`, full test suite, git
audit. Nothing is pushed, merged, published or submitted.
