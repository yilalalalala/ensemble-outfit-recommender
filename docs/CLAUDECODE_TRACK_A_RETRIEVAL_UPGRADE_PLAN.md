# Track A retrieval-ceiling upgrade plan

**Status:** execution plan fixed before new experiment results are read  
**Baseline:** commit `9442f1d`, final system = Phase-2 ensemble + BPR-MF/LightGCN/SASRec score features  
**Scope:** high-value first round only: diagnostics, candidate-budget frontier, and append-only neural retrieval  

## Objective

Raise Track A's candidate-recall ceiling without weakening the already verified ranking result.
The current final system averages 159.8 candidates/customer, candidate recall 0.1757 and
Recall@12 0.0609. The ranker converts candidate recall well enough that retrieval is now the
binding constraint.

Do not re-tune the ranker merely to manufacture a gain. First make more relevant items available;
then verify that the existing ranking pipeline converts the additional coverage into top-12 gains.

## Non-negotiable controls

- Reuse the frozen D-038 six-fold protocol (2020-08-05 through 2020-09-09).
- Use only the existing earlier tuning folds for choices. Do not read or score 2020-09-16.
- Every interaction, sequence, graph, vocabulary, index, statistic and eligible catalogue must be
  point-in-time correct at each cutoff.
- Preserve the current final result and artifacts; new work must be restartable and separately
  cached.
- Preserve all user-owned untracked `* 2.*` files unchanged.
- Do not push, merge, publish, submit, or alter external systems.
- Fix ordinary failures autonomously. Record failures and exclusions honestly.

## Stage 1 — Retrieval diagnostics

For every reporting fold and customer, produce a compact artifact containing:

1. eligible truth count and retrieved-truth count;
2. candidate count before and after deduplication;
3. recall for every channel alone;
4. each channel's unique relevant-item contribution to the union;
5. pairwise overlap/Jaccard and redundant-candidate rate;
6. marginal recall when each channel is appended to the current final union;
7. recall by returning/new customer, activity band, head/tail, recent launch, repeat/non-repeat;
8. miss taxonomy: ineligible truth, no channel proposed it, proposed then removed by cap, proposed
   then lost by merge/dedup, present in candidates but missed by top 12.

Programmatically reconcile aggregate counts with the already reported 0.1757 candidate recall.
Add tests proving diagnostic computation does not read the label week when generating candidates.

## Stage 2 — Candidate-budget frontier

Rebuild the existing non-neural union at target budgets approximately 160, 300, 500 and 1,000.
Keep the current merge order reproducible and preserve channel provenance. For each budget report:

- actual mean/p50/p95 candidate count;
- candidate Recall, Recall@12, MAP@12 and NDCG@12;
- conversion from candidate recall to Recall@12;
- scoring time and peak RSS;
- segment metrics and bootstrap intervals versus the current final system.

Use cached matrices/models where valid. Do not silently give a larger-budget system credit for a
different ranker or different eligible catalogue.

## Stage 3 — Append-only neural retrieval

Use the already trained point-in-time SASRec representations as the first new channel.

1. Establish exact dot-product top-k as the quality reference over each fold's eligible catalogue.
2. Benchmark exact retrieval before adding an ANN dependency. The catalogue is small enough that
   exact search may be preferable; adopt ANN only if it materially lowers cost while retaining at
   least 99.5% of exact channel recall.
3. For each customer, preserve every candidate from the current final union first, then append
   deduplicated SASRec candidates. Do not let a channel quota evict an existing candidate in the
   primary append-only experiment.
4. Evaluate appended total budgets 300, 500 and 1,000. Retain source rank, raw score and calibrated
   score as provenance features where the current ranker contract supports them.
5. Measure SASRec's unique truth contribution and the downstream ranker's conversion of those
   additions.

If SASRec adds useful unique recall, repeat the append-only comparison for LightGCN and BPR-MF,
then their union. Do not run these expensive extensions if SASRec's six-fold result has no useful
marginal coverage.

## Stage 4 — Fixed-budget allocation, only after append-only evidence

If a wider union improves quality but cost is excessive, learn or tune a fixed-budget merge using
only the two earlier tuning folds. Prefer calibrated per-source scores or marginal-recall-aware
allocation over hand-picked equal quotas. Always retain a control that preserves the original
candidate set before adding new candidates.

Do not train a new two-tower model in this round. Recommend that follow-up only if diagnostics show
a large `no channel proposed it` population that cannot be addressed by the existing sequential,
graph, co-visitation, repeat and popularity channels.

## Predeclared decision rules

Select the smallest candidate budget/configuration that satisfies all of the following:

- candidate recall improves by at least 15% relative to 0.1757 (target at least 0.2021);
- MAP@12 and Recall@12 paired customer-cluster bootstrap intervals are entirely above zero versus
  the current final system;
- the candidate-recall improvement occurs in at least 4 of 6 folds, with no fold losing more than
  1% relative candidate recall;
- tail-item and new-customer candidate recall do not materially regress (lower confidence bound
  no worse than -2% relative);
- rank/score wall time is no more than 2x the current final pipeline and peak RSS no more than
  +25%, measured on an otherwise idle machine.

If no configuration passes, keep the current system and report the frontier rather than relaxing
the rule after seeing results. Also report the best quality-only point separately.

## Definition of done

- Restartable diagnostics and experiment runner with tests.
- Stored per-fold and pooled JSON/Parquet artifacts, including source overlap and marginal lift.
- Candidate-budget and SASRec append-only comparisons across all six reporting folds.
- Paired customer-cluster bootstrap, segment analysis and resource measurements.
- ADR documenting the adoption/rejection decision and why.
- Updated Track A model card and a concise final report at
  `reports/TRACK_A_RETRIEVAL_UPGRADE_REPORT.md`.
- Programmatic report verifier, full test suite, and final `git status` audit.
- Final response must state: what changed, exact metrics, uncertainty, cost, limitations, incidents,
  commits, and the next recommended experiment.

