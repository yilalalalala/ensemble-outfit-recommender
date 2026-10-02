# Phase 2 upgrade report: Track A retrieval and ranking

*Offline evaluation readout, 2026-10-02. Branch `phase2-retrieval-ranking-upgrade` (local only, not pushed).
All results are offline; a production decision needs an online A/B test.*

## 1. Executive summary

**The primary goal was met.** Candidate recall rose from **13.2% to 18.4%** (4-week backtest
mean) at ~160 candidates/customer, against 116 before, and the gain carried through to ranking
quality without overfitting the validation week:

- **4-week rolling backtest MAP@12: 0.03542 ± 0.00284 vs 0.03298 ± 0.00325**, i.e.
  **+7.5% [+6.9%, +8.2%]** (paired customer bootstrap, pooled). Every week improved, by +5.5% to +10.3%.
- **Test week, evaluated once after all choices were frozen: 0.03918 vs 0.03699 (+5.9%)**;
  validation 0.03772 vs 0.03555 (+6.1%).
- **Lift over the repeat + age-band popularity rule went from +42.4% to +53.2%**
  (paired bootstrap [+51.7%, +54.7%]).
- Recall stretch target: **20.1% on the test week**, 18.6% on validation and 17.1–19.2% per backtest
  week, reached without inflating the budget: at the *old* size (116) the new channel mix already
  gives 15.6%. Budgets of 200+ reach 21–26% recall but **do not improve MAP@12** and do not fit
  in 16 GB, so they were rejected.

What did the work:

1. **Retrieval** (D-027): a channel contract, six new or re-parameterised channels (department-
   and section-conditioned popularity, directional co-visitation, colourway variants seeded from
   104 weeks, personalised new arrivals, 10-band age popularity), and **caps chosen by a greedy
   recall-per-candidate allocator** on historical weeks instead of by hand.
2. **Ranker** (D-028): temporal early stopping, seeds and determinism, stable vocabularies,
   explicit lifetime vs recency features, and **recency-weighted affinity + short-velocity
   features (+3.4% [+2.9, +3.9] on the backtest)**, plus 50% negative downsampling.
3. **Re-ranking** (D-029): an availability proxy that is accuracy-neutral; diversity caps
   measured and left off (they cost 1.5–11% MAP@12).
4. **Explanations** (D-030): **16% of SHAP-only reason chips were not supported by the data**;
   every chip now needs evidence.

The main costs: training is ~3× slower (early stopping + refit, deterministic mode), and item
cold start (articles first sold in the target week, 4–5% of purchases) is still unreachable.

## 2. Baseline vs final

Frozen baseline = `main` at `2d78709`, re-run from a code snapshot before any change
(`reports/phase2/baseline_*`). "Final" = `configs/default.yaml` on this branch.

### Ranking quality (MAP@12)

| | frozen baseline | final | change |
| --- | ---: | ---: | ---: |
| validation | 0.03555 (recorded 0.03546) | **0.03772** | +6.1% |
| validation, returning customers | 0.03774 | 0.03986 | +5.6% |
| validation, new customers | 0.00851 | 0.01133 | +33% |
| validation Recall@12 | 0.0568 | 0.0625 | +10% |
| **test (single touch)** | 0.03699 | **0.03918** | **+5.9%** |
| test, returning / new customers | 0.03951 / 0.00831 | 0.04188 / 0.00839 | +6.0% / +1% |
| test Recall@12 | 0.0616 | 0.0656 | +6.5% |
| **4-week backtest mean ± std** | 0.03298 ± 0.00325 | **0.03542 ± 0.00284** | **+7.5% [+6.9, +8.2]** |
| backtest, returning customers | 0.03492 ± 0.00341 | 0.03738 ± 0.00297 | +7.0% |
| backtest, new customers | 0.00837 ± 0.00273 | 0.01053 ± 0.00217 | +26% |
| backtest Recall@12 | 0.0549 | 0.0609 ± 0.0024 | +11% |
| lift over repeat + age-band popularity (backtest) | +42.4% ± 4.9% | **+53.1% ± 5.5%** | |
| item cold-start Recall@12 | 0 | 0 | — |

Per backtest week (frozen → final): 08-19 0.03032 → 0.03289 · 08-26 0.03002 → 0.03304 ·
09-02 0.03592 → 0.03813 · 09-09 0.03566 → 0.03760. Paired CIs per week: +8.1% [+6.9, +9.5],
+10.3% [+9.0, +11.6], +6.6% [+5.3, +7.8], +5.5% [+4.2, +6.6] (`reports/phase2/paired_comparisons.json`).

### Retrieval

| | frozen baseline | final |
| --- | ---: | ---: |
| merged recall, validation | 13.06% | **18.58%** |
| merged recall, 4-week backtest | 13.23% ± 0.27% | **18.41% ± 0.94%** |
| merged recall, test | (13.1% on val; not re-measured on test) | **20.09%** |
| candidates/customer | 115.9 | 159.7 (test 156.6) |
| precision of the candidate set | 0.36% | 0.37% |
| new-customer recall (backtest) | 6.5% | 8.3% |
| recently launched items, first sale ≤ 28 days before (backtest; 32% of purchases) | 13.8% | 21.5% |
| cold items (first sale inside the label week) | 0% | 0% |
| historical report incl. experimental visual channel | 13.33% @ 130.5 | — |

**Per channel, validation week, final configuration** (unique = recall lost if the channel were removed):

| channel | cap | cand/cust | recall | unique recall | precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat | 10 | 7.8 | 2.77% | 1.14% | 1.13% |
| pop | 5 | 5.0 | 0.78% | 0.00% | 0.49% |
| pop_age (10 bands) | 65 | 65.0 | 7.54% | 3.05% | 0.37% |
| cf (cosine) | 5 | 3.8 | 1.28% | 0.02% | 1.05% |
| variant (104-week seeds) | 20 | 16.8 | 3.41% | 0.99% | 0.64% |
| dept_pop | 40 | 36.7 | 5.78% | 1.16% | 0.50% |
| section_pop | 65 | 58.9 | 8.08% | 1.67% | 0.43% |
| covis | 25 | 21.4 | 4.40% | 0.88% | 0.65% |
| new_arrival_pers | 10 | 9.1 | 1.09% | 0.66% | 0.38% |
| **merged** | | **159.7** | **18.58%** | | 0.37% |

Frozen baseline, same week: repeat 3.47% (unique 2.05%), pop 4.73% (1.16%), pop_age 4.28% (0.93%),
new_arrival 0.94% (0.35%), cf 4.81% (1.98%), variant 2.47% (0.47%).

`pop` and `cf` add ~0 unique recall at 5 slots each but stay: they are cheap, and their ranks are
ranker features (`covis_rank`, `n_channels` rank in the top 15) and the evidence for "Trending" /
"Often bought with" chips.

### Runtime and memory

| | frozen | final |
| --- | ---: | ---: |
| retrieval build, one week (standalone) | 2.4–2.9 s | 5.5 s (12–22 s when another job runs) |
| training rows (4 weeks) | 11.3M | 9.5M (50% negatives) |
| validation run, end to end | 271 s | 949 s (incl. diagnostic scoring at 100/200/400 trees) |
| test run, end to end | — | 793 s (build 76 s, fit 528 s, scoring 186 s) |
| 4-week backtest | 1,202 s | 3,454 s |
| peak RSS, backtest | 7.96 GB | 6.22 GB |

## 3. Code and configuration changes

| component | change |
| --- | --- |
| `candidates/retrieval.py` | Channel registry with one contract `(customer_idx, article_id, score)`; deterministic ties (score rounded to 9 dp, then `article_id`); per-channel capping before the union (merge at 250 candidates: 417 s → 11 s); new channels `dept_pop`, `section_pop`, `type_pop` (shared `_affinity_pop`), `covis`, `als`, `new_arrival_pers`; `pop_metric`, fine age bands, variant seed window |
| `candidates/als.py` | Implicit ALS fitted in a subprocess (OpenMP isolation), cached per cutoff and parameter hash (rejected as a channel; kept for reproducibility) |
| `candidates/budget.py` | Greedy recall-per-candidate allocator; frontier and caps per target budget |
| `candidates/evaluate.py` | Per-channel recall, unique recall, precision, size; overlap histogram; segment recall (new / returning customers, cold and recent items); build seconds per stage; `backtest` mode |
| `features/vocab.py` | Versioned vocabularies replace `hash() % N` |
| `features/track_a.py` | `_life` suffix for lifetime features; `recency_affinity` and `short_velocity` groups; float32 cast in SQL; deterministic row order |
| `ranking/train.py` | Seeds + `deterministic`; temporal early stopping + refit; rounds diagnostic only on existing trees; per-week float32 matrices passed to LightGBM as a list; chunked scoring; negative downsampling; run manifest; per-customer AP and scored-candidate files |
| `ranking/rerank.py` | Availability proxy and diversity caps with diversity / coverage / novelty metrics |
| `ranking/explain.py`, `explain_audit.py` | Evidence-gated reasons with evidence types; faithfulness audit |
| `evaluation/backtest.py`, `compare.py`, `metrics.py` | Backtest records retrieval recall, segments, fit info, AP files; paired customer bootstrap |
| `ranking/predict.py`, `api/build.py` | Chunked scoring, shipped re-ranking policy, evidence-based new-customer chips |
| `configs/default.yaml` | New channel parameters and caps; `budget`; `features.groups`; `ranker.seed / neg_sample_rate / early_stopping`; `rerank` |
| `configs/experiments/` | One file per experiment in this report |
| `scripts/run_experiments.sh` | Sequential val / backtest runner with `/usr/bin/time -l` |
| `db.py`, `pyproject.toml` | DuckDB progress bar off; pytest ignores `* 2.py` sync duplicates |
| docs | ADRs D-027–D-030; DESIGN §4 (contract, frontier, re-ranking stage); DATA.md (lifetime vs windowed history, launch and availability proxies); GLOSSARY (14 terms) |

## 4. Retrieval experiments

Exploration ran on the week before validation (2020-09-02); the allocator on 2020-08-19, 08-26
and 09-02 (25% customer sample); confirmation on validation; selection on validation MAP@12 and
the 4-week backtest. Marginal efficiency = added recall per added candidate over the old merged set.

**Accepted**

| experiment | evidence |
| --- | --- |
| Colourway variants seeded from 104 weeks (was 16) | +1.16 pt recall for 10 added candidates at cap 20 (~0.12 pt/cand, the most efficient source) |
| Department popularity × affinity (104 weeks, top 5 departments, age-banded) | +1.36 pt for 12 cands at cap 20; +5.5 pt at cap 100 |
| Section popularity × affinity (age-banded) | +5.2 pt at cap 100 (age-banding +0.35 pt over plain) |
| Directional co-visitation, gap 7 days, 8 weeks, seeds 52 weeks | +1.40 pt for 17.5 cands at cap 50; better than cosine CF (+1.14 pt for 30 cands) |
| Age popularity with 10 bands (was 4) | +0.3 pt at the same cap |
| Personalised new arrivals (launch proxy × department) | +0.44 pt for 7 cands at cap 10 |
| Greedy budget allocation | Same size (116): 13.06% → 15.64% recall and +4.3% [+3.1, +5.5] MAP@12 |
| Operating point 160 | Backtest +0.55% [+0.14, +0.98] MAP@12 over 116, recall +2.9 pt |
| Join-order fix for age-banded affinity channels; per-channel capping | Same output; 160-candidate build 43 s → 5.5 s |

**Rejected**

| experiment | evidence |
| --- | --- |
| Implicit ALS (12/26 weeks, 64/128 factors) | ~0.06 pt/cand, like extended popularity; 80–130 s per week; ≤ 5 slots from the allocator; +0.06 pt at 160 |
| FashionCLIP visual channel (D-023) | 0 slots at every budget |
| Global new arrivals (7/14/28 days) | Covered by personalised new arrivals and pop channels; 0 slots at ≤ 200 |
| Product-type / garment-group / type×index affinity keys | Below department and section at every cap |
| Repeat over all history, deeper repeat caps | 0.018 pt/cand beyond rank 10 |
| Popularity time decay, 3-day / 14-day windows | ≈ 7-day count (±0.3 pt at 200); kept count |
| Co-visitation same-day / 14 / 28-day gaps, popularity damping β 0.25–0.5 | All ≤ the 7-day, β 0 variant |
| Budgets 200 / 250 / 300 | Recall 21.0% / 23.8% / 26.3% on validation, but MAP@12 at 200 was −0.51% [−1.40, +0.37] vs 160 and training swapped (25.6M rows) |

**Why the old visual channel added unique recall but no MAP.** Its 0.27 pt of unique recall came at
the lowest precision of any channel (0.18%), and next purchases at H&M are rarely look-alikes of past
purchases (D-023). The allocator confirms it: at every budget, a slot spent on another channel finds
more true purchases. Improving it would need a different pool (unsold launches, which needs a launch calendar).

## 5. Ranker and feature experiments

All on validation at the stated budget, same retrieval within a row pair; CIs are paired customer bootstraps.

| experiment | MAP@12 | vs reference | decision |
| --- | ---: | --- | --- |
| Frozen code, rerun | 0.03555 | recorded 0.03546 | reference |
| Phase A infrastructure only (seeds, early stopping, vocab) | 0.03562 | ≈ frozen | accepted (correctness) |
| + recency_affinity, short_velocity (old retrieval) | 0.03582 | +0.56% [−0.29, +1.41] | needed backtest ↓ |
| same groups at 160, **4-week backtest** | 0.03542 vs 0.03426 | **+3.39% [+2.93, +3.89]** | **accepted** |
| 160 candidates, all negatives | 0.03743 | ref | |
| **160, 50% negatives** | **0.03772** | +0.8%, half the rows | **accepted** |
| 160, 50% negatives, 6 training weeks | 0.03762 | ≈ 4 weeks | rejected (cost) |
| 160, `rank_xendcg` objective | 0.03645 | −2.61% [−3.63, −1.59] | rejected |
| New-customer fallback to age-band popularity | backtest 0.00847 vs ranker 0.01053 | ranker better in 3 of 4 weeks | not adopted (D-017 stands) |
| Availability proxy, 28 days | unchanged in all 5 weeks evaluated | — | accepted (serving rule) |
| Diversity: max 2 colourways per style | −2.4% to −2.9% | +0.4 product types per list | off by default |
| Diversity: max 4 items per product type | −1.5% to −1.8% | +0.8 product types per list | off by default |
| Diversity: max 1 colourway per style | −9% to −11% | | rejected |

Early stopping picked 249–438 trees across weeks; the old fixed 200 under-fitted the larger
candidate sets. LambdaRank optimises an NDCG-based surrogate; MAP@12 is only the reported and
early-stopping metric.

## 6. Errors and incidents

| incident | fix |
| --- | --- |
| DuckDB parse error: `returning` is a keyword (segment column) | Renamed to `is_returning` etc. |
| Candidate sets differed between identical runs: multi-threaded float sums change the last bits, flipping near-ties at caps | Scores rounded to 9 decimals before ranking (merge and allocator); test asserts identical candidates across runs |
| First allocator run used the old variant seed window (16 weeks) | Parameters moved into `budget.params`; allocator re-run (run 1 kept as `budget_frontier_run1.json`) |
| Age-banded department channel took 20–110 s (join planned before the age filter) | Age band attached inside the affinity CTE; 0.2 s |
| Merge at 250 candidates took 417 s (window over the full union) | Cap per channel before the union; 11 s |
| 200-candidate run swapped (25.6M × 88 float32 + `pd.concat` copies) | Per-week float32 matrices passed to LightGBM as a list, chunked scoring, 50% negatives; 250 run cancelled |
| INT32 overflow in the downsampling hash (`customer_idx * 1000003`) | `hash(customer_idx, article_id)`; test with production-sized ids |
| Experiment runner logged `exit=0` for a failed run, and a waiter matched that line and started a backtest concurrently | Runner records the real exit code; the concurrent backtest was aborted and re-run alone |
| One backtest week took 2,224 s instead of ~630 s (memory-compressor state after the swapping run) | Diagnosed with `sample` and `ps`; the next weeks ran normally; no code change |
| Full test suite segfaulted: LightGBM training after torch loaded (two OpenMP runtimes) | LightGBM training tests run in a subprocess |
| New old-fixture leakage test lacked department/section columns for the new default channels | Fixture extended; assertion unchanged |
| I overwrote the tracked `reports/m2_retrieval_val.json` early on | Restored from git; it now holds the final configuration's report |
| Frozen code is not deterministic (two runs differ by up to 0.4% per week) | Recorded; the new trainer is deterministic (test) |
| New-customer reason chips were hard-coded "Popular with customers your age" | Derived from evidence |

## 7. Leakage and reproducibility checks

- **Point-in-time.** Every retrieval channel is tested by corrupting the label week and asserting
  identical output (`tests/test_retrieval.py`, 11 channels; `tests/test_leakage.py`). Features: same
  corruption test plus labels-from-label-week-only (`tests/test_features.py`). Feature builders
  have no default cutoff (test).
- **Selection never touched the test week.** Exploration and the allocator used weeks before
  validation; budgets, features and objectives were chosen on validation and the 4-week backtest
  (which ends at validation). The test week was run once, after `default.yaml` was frozen
  (`reports/phase2/logs/queue.log`: "FINAL TEST WEEK RUN").
- **Early stopping** uses the most recent *training* week only.
- **Availability proxy** reads sales before the cutoff only; vocabularies come from static
  catalogue/customer metadata.
- **Labels in training only.** Positives-only grouping (D-014) and negative downsampling use the
  label week of *training* weeks, as before; evaluation weeks score all candidates of all buyers.
- **Sanity.** Final test MAP@12 0.039 is well below the 0.06 leak alarm in CLAUDE.md; recall
  bounds MAP (Recall@12 6.6% ≤ candidate recall 20.1%).
- **Reproducibility.** Seeds + `deterministic` (two fits give identical predictions, test);
  deterministic candidate sets (test); each report embeds a manifest: git commit, config hash
  (`049a011dfbad` for the final val and test runs), data fingerprint, vocabulary version,
  feature-schema hash (`ad12512a7cbc`, 88 features). All runs are logged to MLflow (`mlruns/`).

## 8. Tests and reproduction

`make test`: **76 passed** (34 before; the untracked `* 2.py` duplicates are excluded by `addopts`).
New: `test_retrieval.py` (contract and leakage per channel, merge ties, determinism, allocator),
`test_features.py` (leakage, labels, lifetime vs recency values, vocabularies, required cutoff),
`test_ranker.py` (early stopping, reproducibility, tie-breaking, re-ranking rules, downsampling),
`test_explain.py` (mapping, evidence gates, unsupported claims dropped). Track B, visual search,
assistant and API tests pass unchanged.

```bash
make test
PYTHONPATH=src .venv/bin/python -m ensemble.candidates.budget 3,2,1 0.25      # frontier (≈1 min with cached ALS)
PYTHONPATH=src .venv/bin/python -m ensemble.candidates.evaluate val           # reports/m2_retrieval_val.json
PYTHONPATH=src .venv/bin/python -m ensemble.candidates.evaluate backtest 4    # reports/m2_retrieval_backtest4.json
PYTHONPATH=src .venv/bin/python -m ensemble.ranking.train val                 # ≈16 min
make backtest                                                                 # ≈58 min
scripts/run_experiments.sh val p2_b116 p2_b160_neg50 p2_b160_xendcg           # any experiment in configs/experiments/
PYTHONPATH=src .venv/bin/python -m ensemble.evaluation.compare backtest_p2_b116 backtest_p2_b160_neg50
PYTHONPATH=src .venv/bin/python -m ensemble.ranking.rerank val_p2_b160_neg50
PYTHONPATH=src .venv/bin/python -m ensemble.ranking.explain_audit ranker_val_p2_b160_neg50
PYTHONPATH=src .venv/bin/python -m ensemble.ranking.train test               # the test week: run once
make submission serving                                                       # rebuild serving with the new model
```

Validation-week re-ranking for the chosen model: `reports/phase2/rerank_val_p2_b160_neg50.json`
(availability 28 days: 0.03772 → 0.03772).

Evidence files: `reports/phase2/` (frozen baselines, budget frontier, per-config retrieval, ranker and
backtest JSONs, re-ranking, explanation audit, paired comparisons, logs), `reports/m2_retrieval_*`,
`reports/m3_ranker_{val,test}.json`, `reports/backtest_track_a.json`.

## 9. Limitations, risks and next steps

- **Offline only.** The +7.5% is offline MAP@12. The MVP's local test ran ~14% above the Kaggle
  private LB; the regenerated `reports/submission.csv` has **not been submitted** (needs the owner).
- **Recall beyond 160 does not convert.** 21–26% recall at 200–300 candidates gives no MAP gain;
  next steps are features that separate the many category-popular candidates (e.g. customer-level
  normalised channel scores, similarity of the candidate to the customer's last basket), then
  re-test the frontier.
- **Training cost tripled.** Options with measurable trade-offs: skip the refit (use the early-stopped
  3-week model), a larger learning rate, or `deterministic: false` for exploration runs only.
- **Item cold start** (4–5% of purchases) stays at 0: needs a launch calendar or "coming soon" feed
  before content retrieval over unsold articles is legitimate.
- **Availability is a proxy.** Replace with a stock feed when available.
- **New customers** improved on the backtest (+26%) but not on the test week (+1%); small segment
  (~8% of buyers), high variance.
- **Confidence intervals** resample customers within a week; week-to-week variation is shown
  separately (4 weeks), not folded into one interval.
- **Serving was rebuilt with the new model:** `make submission` (1,371,980 customers, 49 min,
  peak 6.3 GB) then `make serving` (7 s); all API tests pass on the new store. Served chips all
  carry an evidence type; 15 of 3,960 demo recommendations show no chip rather than an
  unsupported one. The MVP outputs were kept: `data/interim/submission_mvp_private_0.03189.csv`,
  `track_a_recs_mvp.parquet`, `ranker_submission_mvp.txt`, `serving_mvp.sqlite`.

## 10. Git commits

See the commit list appended below. **Nothing was pushed**: all commits are on the local branch
`phase2-retrieval-ranking-upgrade`; `main` and `origin` are unchanged. The untracked `* 2` files were not
touched, staged or committed.
