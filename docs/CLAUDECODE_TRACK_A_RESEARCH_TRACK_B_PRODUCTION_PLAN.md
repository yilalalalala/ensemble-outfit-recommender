# Claude Code execution plan: Track A research benchmark + Track B production-oriented prototype

**Date:** 2026-10-03  
**Repository:** `ensemble`  
**Working branch:** `phase2-retrieval-ranking-upgrade`  
**Execution mode:** autonomous, evidence-driven, local only

## 0. Mission and non-negotiable outcome

Continue this repository in two deliberately different directions:

1. **Track A becomes the research-oriented next-purchase system.** Build a fair,
   reproducible comparison against strong published recommender families and improve
   the current ensemble if evidence supports doing so. The defensible claim is not
   "better than recommender-system papers in general"; it is "better than the
   reproduced strong baselines under the same H&M temporal protocol."
2. **Track B becomes a production-oriented multimodal Complete-the-Look prototype.**
   Preserve its existing offline research evidence, then turn the best justified
   pipeline into a robust end-to-end service with online per-customer ranking,
   fallbacks, provenance, observability, packaging and measured local performance.

Work autonomously through the plan. Diagnose and fix ordinary code, data, dependency,
memory and performance problems without asking the user. Stop only for a genuine
external blocker: missing required data/credentials, insufficient disk, a paid service,
or a decision that would overturn the core leakage-safe protocol.

The final deliverable is a detailed report at:

`reports/TRACK_A_RESEARCH_TRACK_B_PRODUCTION_REPORT.md`

Do not push, merge, publish, submit to Kaggle, contact anyone, or use a paid API. Local,
small Conventional Commits are allowed and preferred. Never stage or modify the user's
untracked `* 2.*` sync-duplicate files. Do not use `git add -A` or `git add .`; stage
explicit paths only.

---

## 1. Read first, preserve prior evidence

Before changing code, read in full:

- `CLAUDE.md`
- `docs/DESIGN.md`
- `docs/DATA.md`
- `docs/DECISIONS.md`
- `docs/GLOSSARY.md`
- `reports/MODEL_CARD_track_a.md`
- `reports/MODEL_CARD_track_b.md`
- `reports/PHASE2_UPGRADE_REPORT.md`
- `reports/TRACK_B_ROUND3_UPGRADE_REPORT.md`

Then:

1. Record the starting commit, branch, Python/dependency versions, data fingerprints,
   hardware and `git status` in the final report.
2. Run the existing tests before implementation and preserve the baseline result.
3. Inventory existing artifacts before recomputing expensive outputs.
4. Treat the current authoritative Track A and Track B reports as immutable historical
   evidence. Do not overwrite them. New artifacts go under dedicated new directories.
5. Append new ADRs after the current highest ADR number for every material protocol,
   model-selection or serving decision. Add missing terminology to `docs/GLOSSARY.md`.

All reported figures must be machine-verifiable from stored artifacts. Add a report
verification script that asserts every headline number against the source JSON/Parquet
artifacts. Never type an unverified metric into the final report.

---

# Part I — Track A: research-oriented next-purchase recommendation

## 2. Research question and claims

Primary question:

> Under one point-in-time H&M evaluation protocol, does the existing multi-channel
> retrieval + LambdaRank ensemble outperform strong collaborative-filtering and
> sequential-recommendation baselines in MAP@12, and where do the gains come from?

The primary metric is **MAP@12**. Secondary metrics are Recall@12, NDCG@12,
candidate/truth recall, catalogue coverage, novelty, training time, inference time and
peak memory. MAP@12 is the selection metric; do not select a model on a secondary
metric and then present MAP as if it were the selection target.

Target, not a guarantee:

- The final Track A system should beat the strongest correctly reproduced baseline on
  pooled rolling-fold MAP@12, preferably by at least **5% relative**, with a paired
  customer-cluster bootstrap confidence interval above zero.
- It should not trade away returning/new-customer or tail/recent-item performance
  without reporting that trade-off.
- If this target is not met, do not manipulate the protocol. Report the strongest
  result, the failed hypotheses and the engineering/accuracy trade-off honestly.

No claim of broad state of the art is allowed. A claim may name only the exact
baselines, dataset, split, candidate universe and metric actually compared.

## 3. Freeze the evaluation protocol before model work

Create a versioned Track A research configuration and protocol artifact. Use rolling
temporal folds ending before the existing final week. Prefer six feasible consecutive
weekly folds; if the earliest folds lack sufficient history, use the maximum feasible
count and document the objective reason. For every fold:

- all interactions, retrieval statistics, vocabularies, graphs, sequences and features
  must stop strictly before the label week;
- catalogue eligibility must be knowable at the cutoff;
- training/early-stopping weeks must precede the evaluation week;
- no customer downsampling for final metrics;
- customer-cluster bootstrap must keep all queries/purchases for one customer together;
- deterministic tie-breaking must end with `article_id`;
- label-week corruption tests must show that candidates/features/model inputs do not
  change.

The existing 2020-09-16 week and Kaggle scores have already been observed. They are
**confirmation evidence, not a fresh untouched holdout**. Do not use them for selection,
debugging, thresholds or stopping. Do not make another Kaggle submission. Run no final
confirmation-week evaluation until every Track A choice is frozen; if existing cached
outputs suffice, do not touch the week again. State this limitation prominently.

Persist a protocol manifest containing exact dates, data hashes, eligible-catalogue
logic, metric definitions, bootstrap seed, model seeds, dependency versions and git
commit.

## 4. Baseline ladder

Implement or validate the following ladder. All systems must use the same point-in-time
data and evaluation customers. A model may score the full eligible catalogue or use an
ANN/exact top-k stage, but its end-to-end retrieval limitation must count against it.
Do not give one model oracle candidates unavailable to another. If exact full-catalogue
scoring is computationally infeasible, document the approximation and validate its
top-k agreement/recall on a representative exact-scoring sample.

### 4.1 Existing controls

- global/segment popularity;
- repeat purchase + age-band popularity;
- existing item-to-item CF/covisitation controls;
- current Phase-2 ensemble, reproduced from the authoritative configuration.

### 4.2 BPR-MF

Implement **Bayesian Personalized Ranking with Matrix Factorization** as the classical
personalized-ranking baseline.

- positives: observed pre-cutoff purchases;
- negatives: clearly specified unobserved eligible items, sampled without label-week
  information;
- objective: pairwise BPR loss;
- score: user/item latent-vector dot product;
- report embedding dimension, negative sampler, regularization, epochs, early stopping,
  parameter count, runtime and peak memory;
- define and evaluate a deterministic new-user fallback.

### 4.3 LightGCN

Implement **Light Graph Convolutional Network** as the main strong collaborative-
filtering baseline on the pre-cutoff user-item bipartite graph.

- keep the implementation faithful and minimal; do not silently turn it into the final
  feature-rich ensemble;
- tune embedding dimension, layers, regularization and training length only on rolling
  development folds;
- use BPR or the paper-consistent ranking objective and state it;
- evaluate both personalization quality and retrieval coverage;
- define new-user fallback and unseen/recent-item behavior;
- avoid a heavy graph framework if a tested sparse PyTorch implementation is more
  reproducible on the 16 GB machine.

### 4.4 One sequence-aware baseline

Implement one feasible, task-aligned sequence model, preferably **SASRec**. BERT4Rec is
an acceptable substitution only if the reason is documented before reading results.

- construct sequences strictly before each cutoff;
- define repeat handling, maximum sequence length and time-gap treatment;
- train with a published objective and non-leaking negative sampling;
- report full-catalogue/ANN retrieval behavior and new-user fallback;
- do not build both SASRec and BERT4Rec merely to increase model count.

### 4.5 Reproduction validation

For each neural baseline:

- include unit tests on a toy graph/sequence where expected ranking behavior is known;
- run at least three fixed seeds for the selected configuration;
- report mean, standard deviation and per-fold values;
- verify that identical CPU seed/config runs reproduce within an explicitly stated
  tolerance;
- retain learning curves and selection logs.

If a baseline cannot be reproduced faithfully within local constraints, do not publish
a misleading weak number. Document the blocker, demonstrate attempted validation and
exclude it from claims about outperforming strong baselines.

## 5. Improve the final Track A system only from development evidence

After the baseline ladder works, test whether its signals improve the current ensemble.
Prioritize high-value, bounded experiments:

1. Use BPR-MF, LightGCN and/or SASRec top-k lists as retrieval channels under the same
   channel contract, with scores/ranks/source flags exposed to the ranker.
2. Measure marginal recall per added candidate and unique recall, not just standalone
   model quality.
3. Re-run the candidate-budget allocator only on rolling development folds.
4. Test whether embedding similarities or sequence scores improve LambdaRank after
   controlling for candidate-set size.
5. Preserve the existing point-in-time recency, velocity and retrieval-provenance
   features; add new features only through named feature groups and ablation.
6. Reject channels that add cost without statistically reliable downstream MAP.

Run a compact ablation table for the selected final system:

- remove collaborative baseline signals;
- remove sequential signals;
- remove repeat features;
- remove recency/velocity;
- remove retrieval provenance;
- remove content/FashionCLIP if present in the final configuration.

Refit every ablation; do not zero columns in a fitted model and call it an ablation.

## 6. Track A statistical and segment analysis

Produce one authoritative comparison artifact and table containing:

- per-fold and pooled MAP@12, Recall@12 and NDCG@12;
- relative/absolute lift vs current ensemble and strongest reproduced baseline;
- paired customer-cluster bootstrap 95% confidence intervals;
- three-seed mean and standard deviation for stochastic models;
- returning vs new customers;
- low/medium/high activity customers;
- head vs tail items using a predeclared pre-cutoff popularity definition;
- recently launched items using the existing honest launch proxy;
- repeat vs non-repeat truth;
- candidate recall ceiling and conversion from candidate recall to top-12 hits;
- training time, scoring time, parameter count, peak memory and artifact size.

Also include at least 20 deterministic, anonymized failure cases grouped into useful
failure modes, without exposing customer identifiers or copying dataset images into git.

Update `reports/MODEL_CARD_track_a.md` only after the new authoritative artifacts exist.
Preserve the previous numbers in a dated history section or reference the earlier report.

## 7. Track A stopping rule

Stop Track A experimentation when all of the following are true:

- the protocol and leakage tests pass;
- BPR-MF, LightGCN and one sequence-aware baseline are either faithfully reproduced or
  explicitly excluded with evidence;
- stochastic stability is measured;
- the final system and ablations have run on the rolling development folds;
- further tested additions fail the predefined improvement/cost threshold;
- one configuration is frozen and the report verifier passes.

Do not continue open-ended hyperparameter searching. Use bounded grids/successive
halving or another documented compute-aware strategy.

---

# Part II — Track B: production-oriented multimodal Complete-the-Look prototype

## 8. Product contract

Track B's user-facing job is:

> Given an anchor item or an uploaded outfit photo, optional customer context and a
> requested/missing slot, return available complementary products with truthful reasons,
> provenance and graceful fallbacks.

Preserve the existing Round-3 offline comparison as the quality basis. Do not start a
new academic-model arms race and do not add Polyvore/POG or another dataset in this
round. Run offline regression checks only where production changes can alter ranking.

The prototype must support two modes:

1. **Anonymous compatibility mode:** anchor + target slot -> compatibility ranker.
2. **Personalized mode:** customer + anchor + target slot -> personalized ranker at
   request time, with explicit fallback when customer history is unavailable.

The current static anchor-level table is not sufficient for the personalized mode.

## 9. Serving architecture

Design and implement a clear offline/online split.

### Offline build

- versioned eligible catalogue and availability snapshot;
- association evidence and two-tower candidate pools;
- FashionCLIP catalogue embeddings and DeepFashion2 adapter artifacts where applicable;
- anchor/slot candidate pools with compatibility features;
- customer profile features or bounded online-readable aggregates;
- ranker artifacts, feature schema, vocabularies and artifact manifest;
- atomic publication into a versioned serving bundle.

### Request-time path

1. validate request and resolve anchor/garment crops;
2. determine or accept target slot;
3. retrieve a bounded candidate pool through precomputed lists and/or ANN;
4. filter by catalogue eligibility/availability proxy before top-k;
5. add customer features when history exists;
6. score with compatibility or personalized ranker;
7. apply measured diversity reorder/backfill, never an unmeasured destructive filter;
8. attach row-level evidence and truthful reason chips;
9. return results plus model/catalog version and fallback/provenance fields.

No request may require retraining. Avoid loading large training tables in the API process.
Heavy models must load once and be reused. Confirm that PyTorch/LightGBM runtime conflicts
remain isolated.

## 10. API contract and safety

Use typed request/response schemas and keep backward compatibility where reasonable.
At minimum provide or validate:

- health/readiness endpoint;
- model/catalog metadata endpoint;
- Complete-the-Look endpoint with optional customer;
- visual-search endpoint for a garment crop/photo;
- outfit-photo endpoint that can identify garments and request one or more complementary
  slots without inventing a "missing" item when the outfit is already complete;
- stable error schema and request ID.

Every recommendation row should expose machine-readable fields such as:

- `article_id`, `target_slot`, `score` or calibrated rank position;
- `model_version`, `catalog_version`;
- `personalized` and `fallback_level`;
- `evidence_types`/source provenance;
- user-facing reasons that are gated by the evidence on that row.

For uploads, enforce documented file-type, image-decoding, pixel/dimension and size
limits. Reject malformed files cleanly. Do not retain user images by default. Do not log
image bytes, raw personal identifiers or full customer histories.

## 11. ANN, cache and fallback behavior

- Benchmark the current exact/vector-search path before selecting an ANN dependency.
- If ANN materially improves latency/memory, use a reproducible local index with a
  stored build configuration, version and recall-vs-exact test. Do not adopt ANN merely
  for the label.
- Build bounded caches with explicit keys that include relevant model, catalogue,
  customer-profile and availability versions. Prevent stale recommendations after a
  bundle swap.
- Implement deterministic fallback levels, for example: personalized ranker ->
  compatibility ranker -> article association -> style association -> two-tower ->
  slot popularity. Use the actual evidence ordering supported by the existing model,
  and report fallback usage.
- Preserve the Round-3 rule that reason chips are allowed only when the individual row
  has supporting evidence. Add tests for every fallback/reason combination.

## 12. Observability, reliability and model lifecycle

Implement a lightweight local equivalent of production controls:

- structured request logs with request ID, latency, endpoint, model/catalog version,
  result count, fallback level and error class;
- aggregated counters/timers or a local metrics endpoint;
- startup validation of artifact presence, schema hashes and compatible versions;
- readiness false until required artifacts are loaded;
- atomic serving-bundle swap or safe restart procedure;
- timeouts and bounded work for image/model paths;
- deterministic degraded mode when optional FashionCLIP/adapter/VLM artifacts are absent;
- no silent exception swallowing;
- concise runbook covering build, start, health check, rollback and common failures.

Add contract, integration and failure-injection tests for missing/corrupt artifacts,
unknown customer, unknown article, unavailable candidate, empty target slot, malformed
image, dependency/model-load failure and cache invalidation.

## 13. Performance and quality gates

Create reproducible local benchmarks on the current Apple Silicon/16 GB machine.
Measure rather than invent targets. Report cold and warm performance separately:

- p50/p95/p99 latency per endpoint;
- throughput at documented concurrency levels;
- startup/model-load time;
- steady-state and peak RSS;
- serving-bundle/index sizes;
- cache hit rate under a reproducible workload;
- ANN recall versus exact search if ANN is used;
- percentage of results at every fallback/provenance level;
- failure rate and timeout behavior.

Define explicit acceptance thresholds after measuring the current baseline but before
optimizing. Thresholds must be demanding yet achievable on this laptop and recorded in
an ADR. Do not optimize by changing recommendation semantics without running the
offline regression suite.

For any serving change that can change ordering, rerun the relevant Track B rolling-fold
evaluation or a proven ranking-equivalence test. Reject regressions outside a
predeclared tolerance unless there is a documented product trade-off.

## 14. Packaging and reproducibility

- Provide a Dockerfile and/or compose setup that starts the API with documented mounted
  artifacts; do not bake restricted H&M/DeepFashion2 data or images into the image.
- Pin the runtime dependency set without breaking the existing local environment.
- Add `make` targets or equivalent documented commands for building the serving bundle,
  starting the service, smoke testing it and benchmarking it.
- Ensure a new user gets a clear error explaining which non-versioned data/artifacts are
  required.
- Add a minimal end-to-end smoke fixture using synthetic/tiny data so CI tests do not
  require restricted datasets.
- Preserve licensing/data-use notes: H&M and DeepFashion2 raw data and derived images
  remain outside git.

Do not redesign the fashion UI in this round. Make only the minimal UI/client changes
needed to exercise and visibly verify the new API behavior and provenance. A separate
UI polish round will follow.

## 15. Track B completion criteria

Track B is complete for this round when:

- anonymous and personalized Complete-the-Look requests work end to end;
- visual and outfit-photo flows degrade safely when optional models are unavailable;
- availability filtering, fallback and evidence-gated reasons are tested;
- the service loads a versioned bundle and reports its versions;
- local performance/load benchmarks and memory measurements exist;
- packaging, smoke tests and runbook are verified;
- relevant offline quality has not regressed beyond the declared tolerance;
- limitations explicitly say no real inventory feed, no live traffic and no online A/B
  test, so the result is a **production-oriented prototype**, not production-proven.

---

# Part III — Execution, verification and final report

## 16. Work order

Follow this order unless measured evidence justifies a documented change:

1. Repository/data/artifact audit and clean baseline test run.
2. Freeze Track A protocol and add leakage/reproducibility tests.
3. Implement/evaluate BPR-MF.
4. Implement/evaluate LightGCN.
5. Implement/evaluate one sequence-aware baseline.
6. Test those signals in the current Track A retrieval/ranking pipeline.
7. Run seeds, rolling folds, statistics, segments and ablations; freeze Track A.
8. Design/freeze Track B API and serving-bundle contracts.
9. Implement request-time personalization, caching/ANN decision, availability,
   fallbacks, provenance and observability.
10. Add packaging, reliability tests, smoke flow and performance benchmarks.
11. Run the complete test suite and report-verification script.
12. Audit git status so only intentional paths are tracked; confirm the user's `* 2.*`
    files remain untouched and untracked.
13. Write and verify the final report.

Commit after coherent, tested units. Keep expensive experiments restartable and cache
artifacts with configuration/data hashes. Do not run multiple memory-heavy jobs in
parallel on the 16 GB machine.

## 17. Required final report

`reports/TRACK_A_RESEARCH_TRACK_B_PRODUCTION_REPORT.md` must be detailed but decision-
oriented and contain:

1. Executive summary: what shipped, what did not, and whether each target was met.
2. Starting state and exact code/data/environment versions.
3. Track A protocol and leakage controls.
4. Baseline definitions and fidelity checks.
5. Track A authoritative results: per fold, pooled, confidence intervals, seeds and
   resource cost.
6. Track A final-vs-baseline table and honest claim wording.
7. Track A segments, ablations, failure analysis and rejected experiments.
8. Track B final architecture diagram and offline/online boundary.
9. Track B API, personalization, ANN/cache choice, availability, fallbacks and reason
   provenance.
10. Track B reliability, packaging and security/privacy behavior.
11. Track B latency/throughput/memory/index-size benchmark tables.
12. Test results, exact reproduction commands and expected runtimes.
13. Every incident/error encountered and the real fix.
14. ADRs and material files changed.
15. Known limitations and the next three highest-value actions.
16. Git commits/status and explicit confirmation that nothing was pushed, merged,
    published or submitted.

The report must distinguish clearly among measured fact, inference and future proposal.
Include no invented business impact, online result or production claim. Verify all
headline figures programmatically before declaring completion.

## 18. Definition of done

Do not declare the work finished merely because code exists. It is done only when:

- all applicable existing and new tests pass;
- authoritative experiment and benchmark artifacts exist;
- report figures are programmatically verified;
- reproduction commands have been run successfully or any genuine external limitation
  is explicitly proven;
- Track A claims are fair and protocol-specific;
- Track B works as a measured production-oriented prototype;
- the final report is complete;
- git status has been audited and user-owned sync duplicates are untouched.

