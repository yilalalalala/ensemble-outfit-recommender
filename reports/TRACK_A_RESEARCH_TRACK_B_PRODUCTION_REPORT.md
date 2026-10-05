# Track A research benchmark and Track B production prototype

**Offline evaluation and engineering readout.** 2026-10-03 → 2026-10-05 · branch
`phase2-retrieval-ranking-upgrade` · local only: nothing was pushed, merged, published or
submitted (§16). Every number in this report is offline and comes from a stored artifact;
`scripts/verify_final_report.py` asserts the headline figures against those artifacts.

How to read it: **measured** results are stated plainly with their source; **inference** is
marked "reading"; **proposals** are in §15 only.

---

## 1. Executive summary

**Track A (research).** Under one frozen, leakage-safe protocol — six rolling weeks
2020-08-05 … 2020-09-09, an eligible catalogue knowable at each cutoff, one shared fallback,
customer-cluster bootstrap across folds (D-038) — the existing Phase-2 ensemble was compared with
faithfully reproduced **BPR-MF**, **LightGCN** and **SASRec** (gSASRec loss), each tuned only on
earlier tuning weeks and run with three seeds.

| | pooled MAP@12 (six folds) |
| --- | ---: |
| global popularity | 0.00563 |
| BPR-MF / LightGCN (3 seeds) | 0.00924 / 0.00937 |
| SASRec, the strongest reproduced baseline (3 seeds) | 0.01432 |
| repeat purchase + age-band popularity, the strongest baseline of any kind | 0.02212 |
| Phase-2 ensemble, reproduced (3 seeds) | 0.03409 |
| **Final: Phase-2 + three neural score features (3 seeds)** | **0.03525** |

- **Target met.** The final system beats the strongest reproduced baseline (SASRec) by
  **+147.29% [+143.48%, +151.09%]** MAP@12 (paired customer-cluster bootstrap, seed 42), in
  6/6 folds — far above the +5% target. It also beats the strongest baseline of any kind, the
  repeat-purchase rule, by +59.30%.
- **The ensemble improved, by evidence.** Adding the three reproduced models' customer–candidate
  dot products as ranker features gives **+3.55% [+3.10%, +3.97%]** over the reproduced Phase-2
  ensemble, 6/6 folds, at no measurable training-time or memory cost; it holds for every ranker
  seed (+3.55%, +3.50%, +3.14%). Using the same models as retrieval channels adds a further
  +0.72% but triples training time, so the predeclared adoption rule rejected it (D-040).
- **What the gain costs.** Recently launched items +13.1% and non-repeat purchases +10.4%
  Recall@12, but **tail items −2.6%** and **repeat purchases −1.9%** (both intervals below zero).
- **What did not work.** All three reproduced baselines lose to a two-line repeat-purchase rule
  on H&M (−36% to −58% MAP@12); the honest claim is narrow (§6).

**Track B (production-oriented prototype).** Complete the Look is now served **at request time**,
personalized for known customers: the personalized ranker re-orders a bounded 100-candidate
compatibility pool per (anchor, slot) using a versioned point-in-time profile snapshot.

- **Quality, offline, six folds.** The served personalized pipeline beats the previously served
  table by **+7.06% Recall@12** (CI of the difference [+0.0083, +0.0097], 6/6 folds). Bounding
  the pool to 100 keeps 98.9% of the unrestricted ranker's recall — 0.1 point short of the 99%
  bar fixed beforehand, reported as such (D-041).
- **Correctness.** A build-time gate scores 400 real requests offline (SQL features + LightGBM)
  and online (profile + NumPy runtime): **400/400 identical orderings**; the online
  personalization features equal the production SQL exactly in a unit test.
- **Engineering.** Versioned bundle with sha256 manifest and atomic swap; fallback ladder;
  availability filtering before top-k; evidence-gated reasons and provenance on every row;
  health/readiness/metrics; structured logs without raw identifiers; validated uploads;
  explicit degraded modes; Dockerfile (unverified: no Docker here), pinned serving
  requirements verified in a clean environment; runbook; 162 passing tests.
- **Performance (local, one worker).** Uncached requests **p95 26 ms**, **68 rps** at
  concurrency 4, no errors at 16; warm cache p95 2.9 ms; startup < 1 s; RSS ≤ 1 GB. Every
  predeclared threshold (D-043) met after one profiling-driven optimisation that changed no
  recommendation (re-proven by the gate).
- It is a **prototype**: no inventory feed, no live traffic, no online A/B test.

---

## 2. Starting state

| | |
| --- | --- |
| Start commit / branch | `a790a27` on `phase2-retrieval-ranking-upgrade` |
| Baseline test run | 117 passed (`make test`, before any change) |
| Python / key packages | 3.11.15; torch 2.14.0, lightgbm 4.7.0, duckdb 1.5.5, numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.1, fastapi 0.141.1 |
| Hardware / OS | Apple M5, 10 cores, 16 GB RAM; macOS 26.6.2 |
| Data fingerprint | 31,788,324 transactions, 2018-09-20 … 2020-09-22, 105,542 articles, 1,371,980 customers; article_id sum 22131896419443968, row-hash sum 15896463904003242 |
| `git status` at start | clean apart from 27 untracked `* 2.*` sync duplicates (user-owned) and this round's plan file |
| Reused artifacts | Phase-2 4-week backtest JSON (reproduction check); Track B Round-3 cache `7ad0aa24ad` (towers, training matrices, serving retrieval lists) — no tower was retrained |

---

## 3. Track A protocol and leakage controls (D-038)

- **Folds.** Six reporting folds 2020-08-05 … 2020-09-09; tuning folds 2020-07-22 and 2020-07-29;
  allocator weeks 2020-07-15/22/29 — all ending before the first reporting fold
  (`protocol.check_protocol`, tested).
- **Point in time.** Every interaction, sequence, graph, vocabulary, retrieval statistic and
  feature for week *w* reads only `t_dat < w.start`; every per-week model is trained per label
  week (10 weeks × 3 models at seed 0, plus seeds 1–2 on the reporting folds). The ranker trains
  on the four label weeks before each fold, early-stops on the most recent of them.
- **Label-week corruption tests.** Retrieval channels, ranker features, neural interactions,
  sequences and vocabularies, and the eligible catalogue are unchanged when every label-week row
  is replaced or multiplied (`tests/test_research_track_a.py`, `test_retrieval.py`,
  `test_features.py`, `test_leakage.py`).
- **Eligible catalogue.** Articles sold in the 28 days before the cutoff; lists are restricted to
  it and back-filled with the age-band best sellers; ties break on `article_id`.
- **Evaluation population.** All label-week buyers; no customer sampling for any metric.
- **Bootstrap.** 1,000 resamples, seed 0, customers resampled with all their rows across folds;
  relative intervals resample the ratio.
- **The confirmation week 2020-09-16 was not touched.** It and the Kaggle scores were observed in
  Phase 2, so they are confirmation evidence, not a holdout. No Track A choice in this round read
  it, and no confirmation run of the new system was made: the plan prefers not re-reading the
  week, and a fresh score on an already-observed week would not be an untouched test. No Kaggle
  submission was made.

---

## 4. Baselines and fidelity checks (D-039)

| model | objective | key settings (tuning folds only) | fallback |
| --- | --- | --- | --- |
| BPR-MF | BPR, uniform negatives rejecting own positives | 26-week window, dim 128, lr 0.005, reg 1e-5, 24 epochs, CPU | shared age-band list for customers outside the window |
| LightGCN | BPR + L2 on ego embeddings, mean of layers 0..3 | 26 weeks, dim 64, lr 0.02, batch 131,072, 27 epochs, MPS | same |
| SASRec | gSASRec gBCE, 256 negatives, t = 0.75 | last 50 purchases, 2 blocks, dim 64, lr 0.002, 24 epochs, MPS | same |

- **Tuning** was a declared, bounded grid (Stage A, two refinements, Stage B on the second
  tuning fold, Stage C convergence guard): `reports/track_a_research/tuning.json` holds every
  learning curve. gBCE beat the original one-negative BCE by +63% raw MAP@12; not masking
  previously bought items won for all three models. SASRec's curve was still rising slowly at the
  doubled 24-epoch budget; no further extension is in the declared budget.
- **Why gBCE, not full softmax.** Full-softmax CE measured 178 ms per step (~9 min/epoch) on the
  41k-item vocabulary; gBCE, the published SASRec objective for sampled negatives, 18 ms.
- **Toy checks.** BPR-MF and LightGCN recover planted communities and rank held-out items above
  the other community (LightGCN AUC ≥ 0.95, BPR-MF ≥ 0.85); SASRec learns a deterministic
  transition under all three losses (`tests/test_research_track_a.py`).
- **Reproducibility.** Identical CPU seed/config runs are bit-identical on toy data (tests) and on
  real data (BPR-MF, 2 epochs, fold 2020-07-22: max embedding difference 0.0, identical top-100
  lists; `reports/track_a_research/cpu_reproducibility.json`). MPS runs are not bit-reproducible;
  stochastic spread is reported over three seeds: BPR-MF ± 0.00010, LightGCN ± 0.00008, SASRec
  ± 0.00018 MAP@12.
- **The Phase-2 ensemble reproduces.** Same configuration, this protocol, seed 42, against the
  authoritative Phase-2 backtest: within 0.05%–0.34% on the four shared weeks (table in §5).

---

## 5. Track A results

### Pooled six-fold results (primary seed; ± = SD over seeds where run)

| system | MAP@12 | ± seeds | NDCG@12 | Recall@12 | coverage | novelty | seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Global popularity | 0.00563 | — | 0.01090 | 0.01572 | 0.000 | 10.11 | 1 |
| Age-band popularity | 0.00715 | — | 0.01343 | 0.01870 | 0.001 | 10.24 | 1 |
| Repeat purchase + age-band popularity | 0.02212 | — | 0.03274 | 0.03480 | 0.640 | 10.75 | 1 |
| Item-to-item CF (cosine) | 0.01321 | — | 0.02198 | 0.02739 | 0.467 | 12.47 | 1 |
| Co-visitation | 0.01502 | — | 0.02494 | 0.03105 | 0.400 | 12.01 | 1 |
| BPR-MF | 0.00924 | 0.00010 | 0.01470 | 0.01630 | 0.636 | 12.87 | 3 |
| LightGCN | 0.00937 | 0.00008 | 0.01555 | 0.01864 | 0.468 | 12.37 | 3 |
| SASRec (gBCE) | 0.01432 | 0.00018 | 0.02231 | 0.02435 | 0.470 | 12.58 | 3 |
| Phase-2 ensemble (reproduced) | 0.03409 | 0.00006 | 0.05182 | 0.05794 | 0.482 | 11.49 | 3 |
| **Final: Phase-2 + neural scores** | 0.03525 | 0.00004 | 0.05397 | 0.06091 | 0.466 | 11.43 | 3 |
| Re-allocated candidates (control) | 0.03428 | — | 0.05223 | 0.05854 | 0.474 | 11.47 | 1 |
| Neural channels + scores | 0.03549 | — | 0.05437 | 0.06141 | 0.460 | 11.43 | 1 |

### Per fold (MAP@12, mean over seeds)

| system | 08-05 | 08-12 | 08-19 | 08-26 | 09-02 | 09-09 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global popularity | 0.00248 | 0.00317 | 0.00787 | 0.00683 | 0.00673 | 0.00660 |
| Age-band popularity | 0.00383 | 0.00507 | 0.00887 | 0.00862 | 0.00880 | 0.00756 |
| Repeat purchase + age-band popularity | 0.01859 | 0.02168 | 0.02226 | 0.02083 | 0.02429 | 0.02524 |
| Item-to-item CF (cosine) | 0.01320 | 0.01444 | 0.01344 | 0.01188 | 0.01303 | 0.01346 |
| Co-visitation | 0.01456 | 0.01606 | 0.01470 | 0.01383 | 0.01574 | 0.01538 |
| BPR-MF | 0.00899 | 0.00947 | 0.00945 | 0.00832 | 0.00941 | 0.00990 |
| LightGCN | 0.00886 | 0.00981 | 0.00972 | 0.00868 | 0.00966 | 0.00960 |
| SASRec (gBCE) | 0.01368 | 0.01479 | 0.01414 | 0.01275 | 0.01486 | 0.01589 |
| Phase-2 ensemble (reproduced) | 0.02956 | 0.03329 | 0.03315 | 0.03311 | 0.03792 | 0.03760 |
| **Final: Phase-2 + neural scores** | 0.03055 | 0.03429 | 0.03439 | 0.03434 | 0.03912 | 0.03887 |
| Re-allocated candidates (control) | 0.02990 | 0.03356 | 0.03328 | 0.03332 | 0.03782 | 0.03790 |
| Neural channels + scores | 0.03071 | 0.03463 | 0.03468 | 0.03472 | 0.03928 | 0.03900 |

### Paired customer-cluster bootstrap (MAP@12; 1,000 resamples, customers across folds)

| a → b | MAP@12 a | MAP@12 b | relative | 95% CI (%) | NDCG@12 rel. | Recall@12 rel. | folds won by b |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Global popularity → BPR-MF | 0.00563 | 0.00934 | +65.85% | [+60.83, +70.62] | +34.79% | +3.70% | 6/6 |
| Repeat purchase + age-band popularity → BPR-MF | 0.02212 | 0.00934 | -57.77% | [-58.65, -56.94] | -55.11% | -53.17% | 0/6 |
| Global popularity → LightGCN | 0.00563 | 0.00939 | +66.69% | [+61.79, +71.60] | +42.63% | +18.59% | 6/6 |
| Repeat purchase + age-band popularity → LightGCN | 0.02212 | 0.00939 | -57.56% | [-58.44, -56.62] | -52.50% | -46.44% | 0/6 |
| Global popularity → SASRec (gBCE) | 0.00563 | 0.01425 | +153.00% | [+145.49, +159.79] | +104.58% | +54.93% | 6/6 |
| Repeat purchase + age-band popularity → SASRec (gBCE) | 0.02212 | 0.01425 | -35.58% | [-36.75, -34.39] | -31.87% | -30.03% | 0/6 |
| Phase-2 ensemble (reproduced) → Neural channels + scores | 0.03403 | 0.03549 | +4.30% | [+3.86, +4.78] | +4.92% | +5.98% | 6/6 |
| SASRec (gBCE) → Neural channels + scores | 0.01425 | 0.03549 | +149.08% | [+145.59, +152.90] | +143.77% | +152.19% | 6/6 |
| Repeat purchase + age-band popularity → Neural channels + scores | 0.02212 | 0.03549 | +60.46% | [+58.75, +62.26] | +66.07% | +76.46% | 6/6 |
| SASRec (gBCE) → Phase-2 ensemble (reproduced) | 0.01425 | 0.03403 | +138.82% | [+135.28, +142.62] | +132.33% | +137.96% | 6/6 |
| Repeat purchase + age-band popularity → Phase-2 ensemble (reproduced) | 0.02212 | 0.03403 | +53.85% | [+52.20, +55.53] | +58.28% | +66.51% | 6/6 |
| Phase-2 ensemble (reproduced) → **Final: Phase-2 + neural scores** | 0.03403 | 0.03524 | +3.55% | [+3.10, +3.97] | +4.14% | +5.12% | 6/6 |
| SASRec (gBCE) → **Final: Phase-2 + neural scores** | 0.01425 | 0.03524 | +147.29% | [+143.48, +151.09] | +141.95% | +150.14% | 6/6 |
| Repeat purchase + age-band popularity → **Final: Phase-2 + neural scores** | 0.02212 | 0.03524 | +59.30% | [+57.66, +61.04] | +64.83% | +75.02% | 6/6 |
| Phase-2 ensemble (reproduced) → Re-allocated candidates (control) | 0.03403 | 0.03428 | +0.74% | [+0.38, +1.09] | +0.79% | +1.03% | 5/6 |
| SASRec (gBCE) → Re-allocated candidates (control) | 0.01425 | 0.03428 | +140.58% | [+136.79, +144.50] | +134.17% | +140.41% | 6/6 |
| Repeat purchase + age-band popularity → Re-allocated candidates (control) | 0.02212 | 0.03428 | +54.98% | [+53.39, +56.65] | +59.53% | +68.22% | 6/6 |
| **Final: Phase-2 + neural scores** → Neural channels + scores | 0.03524 | 0.03549 | +0.72% | [+0.37, +1.11] | +0.75% | +0.82% | 6/6 |

### Segments (pooled, primary seed)

| segment | Repeat purchase + age-band popularity | SASRec (gBCE) | Phase-2 ensemble (reproduced) | **Final: Phase-2 + neural scores** |
| --- | ---: | ---: | ---: | ---: |
| MAP@12 returning | 0.02322 | 0.01475 | 0.03593 | 0.03722 |
| MAP@12 new customers | 0.00775 | 0.00775 | 0.00918 | 0.00940 |
| MAP@12 low activity | 0.02149 | 0.01589 | 0.03579 | 0.03690 |
| MAP@12 medium | 0.02177 | 0.01376 | 0.03513 | 0.03644 |
| MAP@12 high | 0.02632 | 0.01457 | 0.03684 | 0.03827 |
| Recall@12 head items | 0.04714 | 0.03311 | 0.08119 | 0.08613 |
| Recall@12 tail items | 0.01511 | 0.01033 | 0.01988 | 0.01936 |
| Recall@12 recently launched | 0.03735 | 0.02387 | 0.05979 | 0.06762 |
| Recall@12 repeat truth | 0.53139 | 0.22565 | 0.65310 | 0.64045 |
| Recall@12 non-repeat truth | 0.01525 | 0.01642 | 0.03451 | 0.03809 |

Truth shares: returning customers 0.929, ineligible (no sale in 28 days) 0.041, cold 0.035, recently launched 0.260, repeat 0.038, head 0.634.

### Candidate recall ceiling and conversion (primary seed)

| system | candidates / customer | candidate recall | Recall@12 | conversion |
| --- | ---: | ---: | ---: | ---: |
| Phase-2 ensemble (reproduced) | 159.8 | 0.1757 | 0.0579 | 0.330 |
| **Final: Phase-2 + neural scores** | 159.8 | 0.1757 | 0.0609 | 0.347 |
| Re-allocated candidates (control) | 159.8 | 0.1719 | 0.0585 | 0.340 |
| Neural channels + scores | 162.7 | 0.1741 | 0.0614 | 0.353 |

### Resource cost (mean per fold)

| system | train s | score s | peak RSS GB | parameters / trees | artifact MB | device |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| BPR-MF | 1439 | 30 | 2.48 | 99.7 M params | 94 | cpu |
| LightGCN | 2614 | 20 | 2.38 | 49.8 M params | 68 | mps |
| SASRec (gBCE) | 2429 | 187 | 3.19 | 2.6 M params | 74 | mps |
| Phase-2 ensemble (reproduced) | 847 | 129 | 3.94 | 355 trees, 88 features | 30.5 | cpu |
| **Final: Phase-2 + neural scores** | 842 | 174 | 3.89 | 329 trees, 91 features | 27.9 | cpu |
| Re-allocated candidates (control) | 1271 | 712 | 3.81 | 358 trees, 86 features | 31.9 | cpu |
| Neural channels + scores | 2514 | 204 | 4.01 | 386 trees, 93 features | 31.6 | cpu |

### Ablations of the final system (each refitted on all six folds, seed 42)

| removed | MAP@12 | relative to final | 95% CI (%) | folds where removal wins | features |
| --- | ---: | ---: | --- | ---: | ---: |
| collaborative | 0.03508 | -0.44% | [-0.76, -0.12] | 1/6 | 89 |
| provenance | 0.03492 | -0.91% | [-1.26, -0.56] | 1/6 | 72 |
| recency_velocity | 0.03416 | -3.06% | [-3.44, -2.68] | 0/6 | 72 |
| repeat | 0.03466 | -1.64% | [-2.01, -1.25] | 0/6 | 86 |
| sequential | 0.03458 | -1.87% | [-2.25, -1.48] | 0/6 | 90 |

### Neural models as retrievers (seed 0, mean over folds)

| model | Recall@12 | Recall@50 | Recall@100 | buyers represented | distinct articles in top-12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BPR-MF | 0.0132 | 0.0289 | 0.0429 | 0.838 | 18,735 |
| LightGCN | 0.0155 | 0.0368 | 0.0554 | 0.838 | 13,802 |
| SASRec (gBCE) | 0.0229 | 0.0526 | 0.0797 | 0.924 | 13,858 |

### Reproduction of the authoritative Phase-2 backtest

| week | Phase-2 backtest | this protocol | relative |
| --- | ---: | ---: | ---: |
| 2020-08-19 | 0.03289 | 0.03288 | -0.05% |
| 2020-08-26 | 0.03304 | 0.03302 | -0.06% |
| 2020-09-02 | 0.03813 | 0.03801 | -0.30% |
| 2020-09-09 | 0.03760 | 0.03747 | -0.34% |

**Ranker seeds.** Pooled MAP@12 per seed (42 / 43 / 44): Phase 2 0.03403 / 0.03410 / 0.03414;
final 0.03524 / 0.03529 / 0.03522 (`segments_and_seeds.json`). Pairwise comparisons in this
report use seed 42 for rankers and seed 0 for the neural models; three-seed means and SDs are in
the first table.

**Resource notes.** Neural training times are per model and week, measured while the other lane
ran. Two fold-09-09 ranker fits (`realloc`, `neural_channels`) and the `realloc` scoring time were
inflated by contention with a GPU job; D-040 shows the adoption verdict does not depend on them.

---

## 6. Final vs baselines, and the claim

| comparison (MAP@12) | relative | 95% CI | folds |
| --- | ---: | --- | ---: |
| final vs SASRec (strongest reproduced baseline) | +147.29% | [+143.48%, +151.09%] | 6/6 |
| final vs repeat purchase + age-band popularity | +59.30% | [+57.66%, +61.04%] | 6/6 |
| final vs reproduced Phase-2 ensemble | +3.55% | [+3.10%, +3.97%] | 6/6 |

**Claim, worded to the evidence.** *On H&M, over six rolling weeks ending 2020-09-09, with an
eligible catalogue of articles sold in the 28 days before each cutoff and MAP@12 over all buyers,
the Phase-2 retrieval + LambdaRank ensemble with three neural score features outperforms our
reproductions of BPR-MF, LightGCN and SASRec (gSASRec loss) by +147% or more, and the reproduced
Phase-2 ensemble by +3.55%.* Not claimed: anything about other datasets, other protocols, the
published numbers of those papers, or online behaviour.

**Reading.** The reproduced baselines are weak here because H&M purchases are dominated by
repurchase and short-term recency, which pure ID models capture poorly (SASRec, the only
sequence-aware one, is the best of the three); the ensemble's advantage is its retrieval channels
and point-in-time features, and the neural models help it most as one more signal.

---

## 7. Segments, ablations, failure analysis, rejected experiments

**Segment trade-offs of the final system vs Phase 2** (`segments_and_seeds.json`, Recall@12
unless stated): recently launched **+13.10%** [+12.18, +14.01]; non-repeat purchases **+10.37%**;
head items +6.09%; returning customers MAP@12 +3.57%; new customers MAP@12 +2.39% [−1.24, +6.69]
(not significant); **repeat purchases −1.94%** [−2.21, −1.68]; **tail items −2.63%**
[−3.46, −1.84]. The gain is concentrated in head and new-to-the-customer items; the tail and the
buy-again case lose a little.

**Ablations** (table in §5): every removal hurts with an interval below zero — recency/velocity
−3.06%, SASRec score −1.87%, repeat features −1.64%, retrieval provenance −0.91%, BPR-MF +
LightGCN scores −0.44%. No content/FashionCLIP group exists in the final configuration (D-023
keeps it off), so that ablation does not apply. Each ablation was **refitted** on all six folds.

**Failure analysis** (final system, seed 42, all 446,056 (customer, fold) rows): of all purchased
(customer, article) pairs, **6.09%** are top-12 hits, **11.48%** are retrieved but ranked below
12, and **82.43%** are never retrieved; 4.1% are outside the eligible catalogue (3.5% are cold,
first sold inside the label week). The 24 deterministic, anonymised no-hit examples
(`comparison.json` → `failure_analysis.examples`) group into: new-to-customer items outside every
channel (15), retrieved but ranked below 12 (7), new customer with only the fallback list (2).
Reading: retrieval, not ranking, is the binding limit — the candidate set covers 17.6% of truth
at ~160 candidates and the ranker converts 34.7% of that.

**Rejected or not adopted.**

| experiment | evidence | decision |
| --- | --- | --- |
| neural models as retrieval channels (allocator caps + features) | +4.30% vs Phase 2, +0.72% vs final, but +197% training time (median +63%) | rejected by the D-038 cost rule (D-040) |
| allocator re-allocation of the Phase-2 channels alone | +0.74% [+0.38, +1.09], 5/6 folds; training time +50.01% vs a +50% limit | near-miss, not adopted |
| full-softmax SASRec | ~9 min/epoch | replaced by gBCE before any result was read |
| original one-negative SASRec loss | −39% vs gBCE on the tuning fold | rejected in tuning |
| masking previously bought items (all three models) | lower on every tuning curve | rejected in tuning |
| two concurrent MPS training lanes | 158 s + 155 s per epoch vs 50 s + 65 s alone | rejected (engineering) |

---

## 8. Track B architecture: offline / online boundary

```
OFFLINE  (python -m ensemble.serving.bundle build; DuckDB + LightGBM + one PyTorch subprocess)
  mined association evidence, eligible catalogue,  ──►  compatibility ranker scores every
  two-tower + FashionCLIP candidate union (Round 3)      (live anchor, other slot) key
  personalized + compatibility rankers, trained   ──►  top-100 pool per key with its
  point-in-time on the last two label weeks              compatibility features + exact aux
  customer history (104 weeks before cutoff)      ──►  profile snapshot, one row per customer
  FashionCLIP catalogue + DeepFashion2 adapter    ──►  visual index (live) + all-article vectors
  manifest (sha256 per file), equivalence gate    ──►  .tmp-* → <version>/ → CURRENT (atomic)
─────────────────────────────────────────────────────────────────────────────────────────────
ONLINE  (uvicorn ensemble.api.app; NumPy + SQLite, PyTorch only for visual endpoints)
  request → validate → resolve anchor / slots → pool (or fallback ladder) → drop unavailable
          → profile? personal features + NumPy forest : stored compatibility order
          → diversity re-order + back-fill → evidence, provenance, reasons → versions
```

No request retrains anything or reads a training table; the API process imports neither
LightGBM nor DuckDB (verified), so the PyTorch/LightGBM OpenMP clash cannot occur.

---

## 9. API, personalization, ANN/cache choice, availability, fallbacks, provenance

- **Endpoints** (typed Pydantic schemas; v1 endpoints unchanged): `GET /healthz`, `GET /readyz`,
  `GET /api/v2/meta`, `GET /api/v2/complete-the-look?anchor&slots&customer&k&diversity`,
  `POST /api/v2/visual-search`, `POST /api/v2/outfit`, `GET /metrics`, `GET /api/v2/metrics`,
  local-only `POST /admin/reload` and `POST /admin/availability`. Errors use one schema
  `{"error": {"code", "message", "request_id"}}` and every response carries `X-Request-ID`.
- **Rows** carry `article_id`, `rank`, `target_slot`, `score`, `provenance`, `evidence_types`,
  `reasons` (each with its `evidence_type`), `backfill`, the raw `evidence` values, and the
  module carries `fallback_level`, `personalized`, `source_anchor`; the response carries model,
  catalogue, profile, availability and bundle versions.
- **Personalization** happens at request time when the customer has a profile (1,356,683
  customers with a purchase in the 104 weeks before the cutoff); otherwise the stored
  compatibility order is used. The 21 features mirror the production SQL exactly, including NULL
  and `coalesce` semantics (unit test on a toy database; gate on real requests).
- **Pool size P = 100** by D-041's predeclared rule (table in §11).
- **Fallback ladder** (evidence order): anchor pool → live colourway of the same style →
  nearest live article of the same slot in FashionCLIP space → slot popularity; short modules
  are back-filled from slot popularity and marked `backfill`.
- **Availability**: the bundle's eligible-catalogue snapshot plus runtime overrides
  (`/admin/availability`), applied before the top-k cut; part of every cache key.
- **Cache**: in-process LRU keyed by bundle version, availability version, anchor, slot,
  customer (only when personalized), k and diversity; a bundle swap starts a fresh cache.
- **ANN**: not adopted. Exact top-8 over the 26,127 × 512 live visual index takes 1.26 ms per
  query (`bench_serving_final.json`); an ANN index would add build and recall risk for no
  measurable gain (D-042).
- **Reasons** are gated on the row's own evidence (D-030, D-035): a co-purchase chip needs a
  co-count ≥ 1 and quotes a lift multiplier only at co-count ≥ 5; personal chips need the
  customer's own counts. Tests check every chip against its row.
- **Served provenance mix** under the benchmark workload (items): visual compatibility
  60.5%, popular in slot 30.3%, style co-purchase 6.7%,
  article co-purchase 2.6%; 52.3% of modules were personalized.
- **Minimal UI change**: the product page now calls v2 with the demo customer and shows
  personalized vs compatibility order, the candidate source and each card's evidence.

---

## 10. Reliability, packaging, security and privacy

- **Startup validation**: required files, sizes (and optionally sha256), manifest format,
  feature-schema hash, model/schema consistency; readiness stays false with the reason otherwise.
- **Reload** loads and validates the new bundle completely before swapping; a failed reload keeps
  the old bundle serving (409 `bundle_invalid`).
- **Degraded modes**: no visual index or FashionCLIP → `visual_model_unavailable` (503), Complete
  the Look by id unaffected; adapter absent → raw-crop search flagged `degraded`; garment detector
  unavailable or a paid backend configured → `detector_unavailable` (503) unless the client sends
  the garments; a complete outfit returns `outfit_complete: true` and no invented gap.
- **Bounded work**: model inference on one worker thread with a timeout (504 `visual_timeout`).
- **Uploads**: JPEG/PNG/WebP only, ≤ 8 MB, ≤ 40 MP, sides 32–8000 px, full decode before use;
  never written to disk or logged.
- **Logs**: one JSON line per request with request id, endpoint, status, latency, versions,
  result count, fallback levels, personalization, cache outcome, error class and a salted hash of
  the customer id — never the id, history or image bytes; unhandled errors are logged with their
  traceback (a formatter bug that dropped tracebacks was found and fixed).
- **Tests** (`tests/test_serving.py`, `test_serving_gbdt.py`): contract, error schema, fallback
  levels, availability + cache invalidation, reason gating, upload validation, degraded visual and
  detector paths, missing / corrupt / truncated / schema-mismatched bundles, same-size corruption
  caught by sha256, readiness false → reload → failed reload keeps serving, version swap
  invalidates the cache, metrics and log privacy, local-only admin, SQL ≡ online features, NumPy
  forest ≡ LightGBM.
- **Packaging**: `requirements-serve.txt` (pinned; no LightGBM, no DuckDB) and
  `requirements-serve-visual.txt`; `Dockerfile` and `docker-compose.yml` mount the bundle and
  images read-only and bake no data in; `make serving-bundle | serve-v2 | serve-smoke |
  bench-serving`; runbook `docs/RUNBOOK_SERVING.md`. **Verified**: a clean venv with only the
  pinned serving requirements runs the end-to-end smoke test (LightGBM confirmed absent).
  **Not verified**: `docker build` — Docker is not installed on this machine.
- **Synthetic fixture** (`ensemble.serving.fixture`) exercises every online path without the
  restricted H&M or DeepFashion2 data; those stay outside git and outside the image.

---

## 11. Track B quality regression and performance

### Serving regression on six rolling folds (D-041)

| comparison | Recall@12 a | Recall@12 b | relative | Δ 95% CI | NDCG@12 rel. |
| --- | ---: | ---: | ---: | --- | ---: |
| lgbm_compatibility+shipped → lgbm_personalized@pool100+shipped | 0.1275 | 0.1365 | +7.06% | [+0.0083, +0.0097] | +11.77% |
| lgbm_compatibility+shipped → lgbm_personalized@pool48+shipped | 0.1275 | 0.1379 | +8.15% | [+0.0098, +0.0110] | +11.36% |
| lgbm_compatibility+shipped → lgbm_personalized@pool24+shipped | 0.1275 | 0.1401 | +9.89% | [+0.0120, +0.0132] | +11.70% |
| lgbm_personalized → lgbm_personalized@pool100 | 0.1560 | 0.1543 | -1.09% | [-0.0019, -0.0015] | -1.13% |
| lgbm_personalized → lgbm_personalized@pool48 | 0.1560 | 0.1502 | -3.75% | [-0.0063, -0.0055] | -3.83% |
| lgbm_personalized → lgbm_personalized@pool24 | 0.1560 | 0.1452 | -6.97% | [-0.0114, -0.0103] | -6.68% |
| lgbm_compatibility → lgbm_personalized@pool100 | 0.1368 | 0.1543 | +12.83% | [+0.0168, +0.0183] | +15.68% |

### Serving bundle

Version `2020-09-23_4edf27eb8e_df69b1f77f`, serving week 2020-09-23 (cutoff 2020-09-22), 156,990 keys, 15,699,000 pool rows (P = 100), 26,165 live articles, 1,356,683 customer profiles, equivalence 400/400 identical orderings.

### Benchmark `bench_serving_baseline` (bundle 6,697 MB)

| mode | concurrency | p50 ms | p95 ms | p99 ms | throughput rps | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold | 1 | 196.4 | 219.2 | 233.1 | 9 | 0 |
| cold | 4 | 692.5 | 992.9 | 1069.6 | 8 | 0 |
| cold | 16 | 2797.1 | 4144.3 | 7531.7 | 7 | 37 |
| warm | 1 | 2.1 | 2.8 | 4.0 | 437 | 0 |
| warm | 4 | 7.7 | 10.7 | 13.1 | 497 | 0 |
| warm | 16 | 35.5 | 73.1 | 99.2 | 441 | 0 |

cold: startup→ready 0.68 s, first request 15 ms, RSS after ready 351 MB, steady 470 MB, peak 481 MB, cache {'hits': 0, 'misses': 36006, 'size': 0, 'capacity': 0}

warm: startup→ready 0.65 s, first request 9 ms, RSS after ready 350 MB, steady 1084 MB, peak 1084 MB, cache {'hits': 39144, 'misses': 8862, 'size': 8862, 'capacity': 200000}

visual search (crop_adapter): first call 5192 ms; warm c1 p50 25 / p95 28 ms; c4 p50 85 / p95 156 ms

exact top-8 over 26,127 × 512 vectors: 1.35 ms/query (54 MB)

### Benchmark `bench_serving_final` (bundle 6,697 MB)

| mode | concurrency | p50 ms | p95 ms | p99 ms | throughput rps | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold | 1 | 19.0 | 26.1 | 48.4 | 72 | 0 |
| cold | 4 | 91.0 | 121.3 | 139.0 | 68 | 0 |
| cold | 16 | 559.7 | 650.8 | 692.4 | 49 | 0 |
| warm | 1 | 2.1 | 2.9 | 4.6 | 434 | 0 |
| warm | 4 | 7.9 | 11.1 | 13.3 | 492 | 0 |
| warm | 16 | 35.4 | 78.7 | 99.4 | 434 | 0 |

cold: startup→ready 0.45 s, first request 10 ms, RSS after ready 360 MB, steady 439 MB, peak 439 MB, cache {'hits': 0, 'misses': 36006, 'size': 0, 'capacity': 0}

warm: startup→ready 0.66 s, first request 9 ms, RSS after ready 358 MB, steady 851 MB, peak 963 MB, cache {'hits': 39144, 'misses': 8862, 'size': 8862, 'capacity': 200000}

visual search (crop_adapter): first call 4874 ms; warm c1 p50 27 / p95 51 ms; c4 p50 154 / p95 263 ms

exact top-8 over 26,127 × 512 vectors: 1.26 ms/query (54 MB)

**Thresholds (D-043, fixed after the baseline run, before optimisation) and outcome:**

| measure | threshold | baseline | final |
| --- | --- | ---: | ---: |
| uncached, c1, p95 | ≤ 60 ms | 219.2 ms | 26.1 ms |
| uncached, c4, throughput | ≥ 40 rps | 8 | 68 |
| uncached, c16, errors + timeouts | 0 | 37 | 0 |
| warm cache, c1, p95 | ≤ 10 ms | 2.8 ms | 2.9 ms |
| startup to ready | ≤ 5 s | 0.68 s | 0.45 s |
| steady RSS | ≤ 2 GB | 1.08 GB | 0.85 GB |
| visual search warm, c1, p95 | ≤ 100 ms | 28 ms | 51 ms |

The optimisation (vectorised forest over all trees, NumPy lookups instead of pandas indexing)
changes no ordering: the gate re-run on the published bundle gives 400/400 identical orderings
with a maximum score difference of 2.7e-15 (`equivalence_after_optimization.json`). Bundle size
on disk 6.7 GB, of which the memory-mapped pool features are 4.6 GB and the profile store 1.25 GB;
the first visual call loads FashionCLIP (~5 s).

---

## 12. Tests and reproduction

`make test`: **162 passed** (117 at the start; 45 new: 15 Track A research, 29 serving, 1 NumPy
forest). Expected runtimes on this laptop are measured wall times from this round.

```bash
make test                                    # ~15 s
make serve-smoke                             # synthetic bundle, real server, ~30 s
# Track A research benchmark (needs the ingested data; restartable, caches everything)
ENSEMBLE_CONFIG=track_a_research make research-a   # tuning ~7 h; per-week models ~37 h on CPU+MPS;
                                                   # matrices ~25 min; each ranker fit 10–20 min
ENSEMBLE_CONFIG=track_a_research PYTHONPATH=src .venv/bin/python -m ensemble.research.queue \
    phase2,phase2_nscores,realloc,neural_channels 42          # the comparison (24 fits)
ENSEMBLE_CONFIG=track_a_research PYTHONPATH=src .venv/bin/python -m ensemble.research.queue \
    phase2_nscores 42 collaborative,sequential,repeat,recency_velocity,provenance   # ablations
ENSEMBLE_CONFIG=track_a_research PYTHONPATH=src .venv/bin/python -m ensemble.research.report
# Track B
make tb-serving-regression                   # ~53 min (reuses the Round-3 cache)
make serving-bundle                          # ~30 min end to end (rankers, pools, profiles, gate)
make bench-serving                           # ~10 min
.venv/bin/python scripts/final_report_tables.py   # every table in this report
make report-verify                           # asserts the headline figures
```

---

## 13. Incidents and the real fixes

| incident | fix |
| --- | --- |
| SASRec full-softmax CE: 178 ms/step, batch 1024 thrashed memory (48 s/step) | gSASRec gBCE (18 ms/step), decided before any result |
| BPR toy test: MF memorised the toy graph at reg 1e-4 | toy-sized regularisation; also recorded LightGCN generalises better (AUC 0.997 vs 0.90) |
| NumPy forest disagreed with LightGBM on NaN categorical values | LightGBM 4 sends NaN/negative categories right; fixed, edge rows added to the test |
| JSON log formatter dropped exception tracebacks | tracebacks now logged; found while debugging a 500 |
| NaN evidence values broke JSON responses (outfit endpoint) | evidence dicts carry only present values |
| `on_event` startup deprecated | lifespan handler |
| Benchmark server blocked: stderr pipe never drained | server logs go to a file |
| Two concurrent MPS lanes slower than one | reverted to CPU + one GPU lane |
| Bundle build: key index assumed slot-id order, pools are in slot-name order | key index sorted by (anchor, slot id); offsets kept |
| Bundle build: one profile query over 1.36M customers hit DuckDB's 9 GB limit | 16 customer partitions; resumable build (models and pools reused; one training-metadata file reconstructed from the saved models and marked as such) |
| Bundle build: stray `del` after the equivalence gate | removed |
| Gate: `cand_price_tier` differed on 3 of 40,000 rows (no score effect) | `ntile()` tie order; ties now break on `article_id` in `completion/features.py` and `pipeline.py` (Round-1 `data.py` untouched). The published bundle and the Round-3 cache predate the fix |
| Uncached serving 219 ms p95, timeouts at c16 | profiled: 92% in the per-tree loop; vectorised evaluator (§11) |
| Mac idle sleep stalled jobs for ~4 h; later a battery hibernation 18:53–21:08 | `caffeinate` tied to the job processes; jobs survived hibernation; machine kept on AC |
| Two ranker fits slowed 4–10× by contention | verdict checked by mean and median (D-040) |

---

## 14. ADRs and material files

ADRs: **D-038** protocol and adoption rule · **D-039** baseline reproductions · **D-040** adopt
neural score features, reject channels · **D-041** Track B pool re-ranking and its outcome ·
**D-042** serving runtime · **D-043** performance thresholds and outcome. Glossary: 17 new terms.

| area | files |
| --- | --- |
| Track A research | `src/ensemble/research/{protocol,data,models,neural,tune,evaluate,channels,ensemble,benchmark,pipeline,queue,report}.py`, `configs/track_a_research.yaml`, neural channels registered in `candidates/retrieval.py` |
| Track B serving | `src/ensemble/serving/{gbdt,runtime,bundle,visual,visual_index,service,observability,fixture}.py`, `src/ensemble/api/app.py` (v2), `api/static/index.html`, `completion/backtest.py` (pool variants), `configs/default.yaml` (`serving.bundle`), `configs/experiments/tb_serving_pools.yaml` |
| Packaging / ops | `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements-serve*.txt`, `Makefile`, `docs/RUNBOOK_SERVING.md`, `scripts/{serve_smoke,bench_serving,final_report_tables,verify_final_report}.py` |
| Tests | `tests/test_research_track_a.py`, `tests/test_serving.py`, `tests/test_serving_gbdt.py` |
| Artifacts | `reports/track_a_research/` (comparison, tuning, budget frontier, segments and seeds, CPU reproducibility, logs), `reports/track_b_production/` (serving regression, bundle manifest and equivalence, benchmarks, logs), `reports/track_b_round3/backtest_val_serving_pools.json` |
| Docs | `docs/DECISIONS.md`, `docs/GLOSSARY.md`, `reports/MODEL_CARD_track_a.md`, `reports/MODEL_CARD_track_b.md` |

The previous authoritative reports (`PHASE2_UPGRADE_REPORT.md`, `TRACK_B_ROUND3_UPGRADE_REPORT.md`
and their artifacts) were not modified; model cards were appended to with dated sections.

---

## 15. Limitations and the next three actions

**Limitations (measured or structural).**
- Everything is offline; no online A/B test, no live traffic, no inventory feed. Track B is a
  production-oriented prototype, not production-proven.
- The confirmation week was not re-evaluated for the new Track A system (§3); the Phase-2
  test-week and Kaggle numbers remain the only external evidence and describe the old model.
- **The new Track A features are not served**: shipping them needs the three neural models
  retrained on all data and scored for every customer.
- The reproduced baselines are faithful but bounded by a laptop budget (SASRec still improving at
  24 epochs; no full-softmax SASRec); claims name only these reproductions.
- Retrieval is the ceiling: 82% of purchased pairs are never retrieved.
- Cost measurements ran next to GPU jobs; the `neural_channels` cost verdict should be re-measured
  on an idle machine.
- Track B pool P = 100 keeps 98.9% of recall (bar 99%); with diversity on, a smaller pool served
  better — not acted on (D-041).
- Docker build unverified locally.
- The published bundle and the Round-3 cache predate the price-tier tie-break fix.

**Next three actions (proposals, highest value first).**
1. **Raise the retrieval ceiling for Track A**: an ANN-backed SASRec channel over a larger pool,
   re-measured on an idle machine with the D-038 rule — the ranker already converts a third of
   what is retrieved.
2. **Ship the Track A neural features**: full-data retraining of the three models, then
   `make submission serving`, gated on the same reproduction checks.
3. **An online A/B test design for Complete the Look personalization**, plus selecting P on the
   served (diversity-applied) metric — the one decision where offline evidence points both ways.

---

## 16. Git and status

Commits made in this round, all local on `phase2-retrieval-ranking-upgrade` (newest first), plus the
final commit that adds this report, its verifier and the run logs:

```
43ba0ba test(research): real-data CPU reproducibility check; segment trade-off and per-seed artifact
dcdb8b6 docs: glossary terms for the research benchmark and production serving; Track B model card serving update
e603984 perf(serving): vectorised forest evaluation and array lookups; thresholds set and met (D-043)
45f90a3 feat(research): authoritative Track A comparison: three seeds, ablations, failure analysis
5d21141 docs(report): render every final-report table from stored artifacts
d1106f7 feat(research): neural retrieval coverage, Phase-2 reproduction check, direct channel comparison in the report
d327889 feat(research): adopt neural dot-product ranker features (D-040); seed-42 six-fold comparison
a23b378 feat(research): allocator with neural channels on allocator weeks; neural_channels and realloc configs
b6cfb67 fix(serving): bundle key order, partitioned profiles, resumable build; deterministic price tiers
b9e342d docs(adr): D-039 baseline reproductions, D-042 serving runtime; two-lane production runs
e78180e feat: tuned baseline configuration; serving pool P = 100 by the predeclared D-041 rule
3a0b608 feat: product page shows v2 personalization, fallback and provenance; Track A report aggregator
2dbd4fe build(serving): pinned serving requirements, Dockerfile, smoke test, benchmark harness, runbook
00b1b0d feat(serving): request-time personalized Complete the Look over versioned bundles
f2b8bf4 feat(research): frozen Track A protocol, BPR-MF/LightGCN/SASRec reproductions, NumPy GBDT runtime
```

- **Nothing left the machine.** No push, no merge, no pull request, no Kaggle submission, no
  published artifact, no contact with anyone, no paid API. `origin/main` is still at `2d78709`;
  this branch has no upstream.
- **Staging discipline.** Every commit used explicit paths; `git add -A` / `git add .` were never
  used.
- **User-owned sync duplicates.** All 27 `* 2.*` files are on disk, untracked
  (`git ls-files | grep ' 2\.'` is empty), and unmodified — the newest was last modified
  2026-09-28, before this round started.
- **Working tree after the final commit**: only those 27 untracked duplicates remain. Large
  generated artifacts (per-week models, matrices, per-customer evaluations, the 6.7 GB serving
  bundle) live under the gitignored `data/` tree.
- **Verification at the final commit**: `make test` 162 passed; `scripts/verify_final_report.py`
  78/78 headline figures verified.

