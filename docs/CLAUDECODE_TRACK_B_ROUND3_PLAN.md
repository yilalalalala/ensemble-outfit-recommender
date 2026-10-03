# Claude Code Task — Track B Round 3 Upgrade

## Mission

Upgrade Track B (Complete the Look) end to end. Work autonomously: inspect the current implementation and evidence, make a staged plan, implement each stage, run the relevant experiments and tests, diagnose failures, correct them, and continue until the strongest defensible result has been reached.

The objective is a **large, repeatable improvement over the current shipped Track B hybrid**, not a cosmetic refactor or a gain on one selected week. Prefer simple methods that win consistently over complex methods that do not. Do not claim a large improvement unless the measurements support it.

At completion, provide a detailed English upgrade report covering code changes, experiment design, ablations, results, failures, limitations, runtime/memory, reproduction commands, and recommended next steps.

## Repository state and safety rules

- Work from the current local branch and inspect the repository before editing.
- Preserve all unrelated user changes and all untracked `* 2.*` sync-duplicate files. Do not delete, rename, stage, or modify them.
- Make small, coherent local commits as milestones are completed.
- Do **not** push, merge, open a pull request, submit to Kaggle, publish artifacts, contact external services, or take any other external action.
- Do not replace evidence with optimistic prose. If a target is not achieved, report that honestly and retain only changes justified by the measurements.
- Stay within the existing local hardware constraints. Record peak memory and wall time for important runs.

## Current baseline to freeze first

Before model selection, reproduce or validate the current Track B baseline and save a frozen baseline report.

Current reference results are approximately:

- Validation popularity Recall@12: `0.07498`
- Validation shipped RRF + FashionCLIP Recall@12: `0.12144` (about `+62%` vs popularity)
- Test popularity Recall@12: `0.08374`
- Test shipped RRF + FashionCLIP Recall@12: `0.13032` (about `+55.6%` vs popularity)
- Test shipped hybrid NDCG@12: `0.07915`
- Item cold-start Recall@12: approximately `0`

Treat the existing validation and test results as already observed. Do not present the current test week as a fresh untouched holdout. Model selection must use historical rolling folds, and the report must state this limitation.

## Phase 1 — Make Track B evaluation defensible

Implement this before selecting new models.

1. Add a 4–8 week rolling temporal backtest for Track B. Every fold must train/mine only before its target week.
2. Remove target-week availability leakage from the primary protocol. Build the eligible catalogue strictly from information available before the target-week cutoff. If the old target-week-sales universe is retained, report it only as a separate oracle/proxy comparison.
3. Compute confidence intervals by resampling at the customer or basket level, not by treating correlated anchor/slot queries as independent.
4. Report at least:
   - Recall@5 and Recall@12;
   - NDCG@12;
   - relative lift over slot popularity;
   - tail-item, cold-item, jewellery, and per-slot recall;
   - catalogue coverage and novelty;
   - recommendation-list diversity;
   - runtime and peak memory.
5. Add leakage, determinism, metric, and fold-boundary tests.
6. Store per-fold and aggregate reports in stable JSON files with config/code/data manifests.

## Phase 2 — Replace fixed RRF with a learned Track B ranker

Build a candidate union from the existing complementary signals and train a point-in-time query-group ranker. Compare LightGBM LambdaRank and a simpler calibrated/logistic alternative if useful; keep the smallest model that wins consistently.

Candidate sources should include, where available:

- article-level association ranked by raw co-count and by NPMI;
- style/product-code association backoff;
- FashionCLIP two-tower retrieval;
- content-only two-tower retrieval;
- slot popularity fallback.

Use leakage-safe features such as:

- source presence, source rank, and normalized source score;
- two-tower similarity and FashionCLIP cosine similarity;
- co-count, recent/decayed co-count, support, lift, PMI/NPMI;
- exact-article versus style-backoff evidence;
- anchor/candidate category and slot attributes;
- colour compatibility signals;
- price-tier compatibility and price ratio/distance;
- candidate popularity, age/recency, and availability proxy known before cutoff;
- pair recency and seasonality where justified;
- customer affinity features from Phase 3.

Train with one ranking group per `(basket/customer, anchor, target_slot)` query. Positives are the held-out target-slot articles. Ensure no evaluation-week information enters retrieval, features, vocabulary fitting, or candidate eligibility.

Compare against current fixed RRF on every rolling fold. Do not adopt the learned ranker unless the gain is consistent and its complexity is justified.

## Phase 3 — Add user personalization

Track B currently models item compatibility but barely models the person. Add point-in-time customer features derived only from history before the target basket:

- preferred colours and perceived colour families;
- product type, department, section, garment-group, and slot affinities;
- usual price tier and distance from it;
- recency-weighted category/attribute counts;
- repeat/variant affinity where relevant;
- interaction between compatibility evidence and customer preference.

Include an explicit no-history path so anonymous/new customers still receive non-personalized compatibility results. Run an ablation of compatibility-only versus compatibility-plus-personalization, overall and by returning/new customer segments.

## Phase 4 — Improve association evidence

1. Keep raw co-count, lift, PMI/NPMI, and support as separate features rather than forcing one global choice.
2. Add recency-weighted or time-decayed co-occurrence and test multiple sensible half-lives on historical folds only.
3. Gate style-level backoff: use it when exact article evidence is weak or absent, and expose the backoff level as a ranker feature.
4. Preserve a popularity fallback, but make fallback provenance explicit.
5. Audit the current same-customer/same-day basket label. Quantify noise proxies such as basics, repeated quantities, and unusually heterogeneous baskets. Apply a filter or confidence weight only if it improves rolling results.

## Phase 5 — Fix two-tower negatives and training

The existing ablation shows logQ is essential, while the current popularity and hard negatives add no measurable overall gain. Redesign rather than merely retaining them.

1. Keep a correct, tested logQ correction.
2. Mine retrieval-informed hard negatives: high-scoring wrong items in the same slot, product type, colour family, and price range.
3. Filter likely false negatives using basket/association evidence and known positive relations.
4. Compare in-batch + logQ against redesigned hard negatives across multiple seeds and rolling folds.
5. Add early stopping or another defensible model-selection rule.
6. Check determinism. If the MPS backend is nondeterministic, run decisive comparisons on a deterministic backend or report seed variation explicitly.
7. Remove negative components that do not demonstrate value.

## Phase 6 — Improve content and cold-item handling

Use only catalogue eligibility information legitimately available before the cutoff.

1. Evaluate an explicit content-only path for articles without interaction history.
2. Use FashionCLIP/DeepFashion2-adapted signals where they are relevant, but do not assume they help; demonstrate the effect by ablation.
3. Consider global plus local/multi-region visual features or a small top-N pairwise reranker only if feasible on the current hardware.
4. Keep cold-item metrics separate. If true pre-launch availability cannot be known without an inventory/launch feed, state that limitation instead of constructing an invalid proxy.
5. Do not spend most of the round on visual complexity if personalization and learned fusion provide larger gains.

## Phase 7 — Serving and explanation integration

If a new model is adopted:

- rebuild Track B serving artifacts;
- preserve the existing diversity rules and measure their accuracy/diversity trade-off;
- expose evidence provenance: association, style backoff, visual compatibility, user preference, or fallback;
- ensure explanations only cite features/evidence actually present for that recommendation;
- keep API and assistant interfaces backward compatible;
- add or update tests for the new serving path.

## Required experiment discipline

- Change one major factor at a time and keep an ablation table.
- Select on rolling historical folds, not the existing test week.
- Report fold-by-fold results; do not show only the best mean.
- Prefer paired customer/basket bootstrap intervals for model comparisons.
- Track candidate-set size so ranking gains are not secretly caused by a much larger retrieval budget.
- Record random seeds, configuration hashes, git commit, feature schema, data fingerprint, runtime, and peak memory.
- Investigate suspiciously large gains for leakage before accepting them.
- Keep rejected experiments and the reason for rejection in the final report.

## Success criteria

Aim for all of the following:

1. A consistent and statistically supported improvement over the current shipped RRF hybrid across rolling folds.
2. Target at least **+5% relative Recall@12 and NDCG@12 over the current hybrid**, while seeking a larger gain through personalization and learned fusion. This is an ambitious target, not permission to overfit or misreport.
3. No material regression in tail, jewellery, coverage, diversity, or serving correctness without an explicitly justified trade-off.
4. A defensible leakage-free catalogue protocol.
5. All canonical tests passing, plus new tests for the upgrade.

If the +5% target cannot be reached honestly, stop adding unjustified complexity. Keep the strongest reproducible configuration, explain why the target was not reached, identify the limiting data, and provide the best next experiment.

## Final deliverables

1. Updated implementation, configuration, tests, and local commits.
2. Frozen baseline and all accepted/rejected experiment reports.
3. Updated Track B model card and relevant architecture/decision documentation.
4. Rebuilt local serving artifacts if the selected model changes.
5. A final `reports/TRACK_B_ROUND3_UPGRADE_REPORT.md` containing:
   - executive summary;
   - before/after metrics and confidence intervals;
   - per-fold and per-segment results;
   - architecture and code changes;
   - ablation table;
   - leakage/reproducibility checks;
   - runtime/memory impact;
   - problems encountered and how they were fixed;
   - rejected approaches;
   - remaining limitations;
   - exact reproduction commands;
   - local commit list and repository status;
   - a clear statement that nothing was pushed, merged, published, or externally submitted.

Begin by auditing the current Track B code, reports, tests, branch status, and available artifacts. Then execute the phases autonomously in the order that maximizes evidence and avoids wasted computation.
