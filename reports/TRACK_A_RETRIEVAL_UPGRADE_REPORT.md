# Track A retrieval-ceiling upgrade

**Offline evaluation readout.** 2026-10-06 · branch `track-a-retrieval-upgrade` · local only
(nothing pushed, merged, published or submitted). Plan fixed before any result: `d94d237`; decision
thresholds committed before any computation: `6312c96`. Every figure below comes from
`reports/track_a_retrieval/comparison.json`; `scripts/verify_retrieval_report.py` asserts the
headline figures. Measured facts are stated plainly; *reading* marks inference; proposals are in §8.

## 1. Result

**Adopted (D-044): append the customer's SASRec top list to the current candidate union, up to 300
candidates per customer (`sasrec_A300`), with the ranker unchanged.** Over the six reporting folds,
against the current final system (same per-fold ranker):

- candidate recall **0.1757 → 0.2247 (+27.90% [+27.64, +28.15])**, gain in 6/6 folds;
- MAP@12 **+0.33% [+0.18, +0.48]**, Recall@12 **+0.43% [+0.23, +0.64]**, NDCG@12 +0.41%;
- rank + score time **1.63×** and peak RSS **+17%** (limits 2× and +25%);
- every gate of the predeclared rule passes.

**The honest headline is that retrieval is now cheap to widen but hard to convert.** Candidate
recall rises 28% while MAP@12 rises 0.33%: the ranker, trained on ~160 candidates, turns only 0.5%
of the added candidate recall into Recall@12. No 500- or 1,000-candidate configuration passes the
cost gate, and none beats the best 300-candidate point on MAP@12.

## 2. Protocol and controls

- The frozen D-038 protocol: six reporting folds 2020-08-05 … 2020-09-09, eligible catalogue =
  sold in the 28 days before the cutoff, customer-cluster bootstrap across folds (1,000 resamples).
  The observed week 2020-09-16 was neither read nor scored.
- **Fixed ranker.** Every configuration is scored by the same per-fold final ranker (seed 42); no
  configuration gets credit for a different model.
- **Strict supersets.** Channel parameters are unchanged, so each channel's internal order is
  identical to the final system's; a larger budget only raises caps (frontier) or appends a neural
  list after the whole current union (append-only). Tests prove the pool ranks equal the production
  merge and that appending never evicts or duplicates a candidate.
- **Point in time.** Pools are built by the production retrieval channels and ignore the label week
  (corruption test); neural lists are exact top-1000 dot products from the models trained for each
  fold's cutoff, and reproduce the cached production top-100 lists exactly.
- **Reproduction.** Re-running the final configuration through the new pipeline gives identical AP
  for 99.9998% of 446,056 (customer, fold) rows (MAP@12 0.03524 vs 0.03524); the
  difference is one customer, from the co-visitation fix below.
- **Choices on earlier weeks only.** The frontier caps come from a greedy recall-per-candidate
  allocator seeded at the current caps on allocator weeks 2020-07-15/22/29; pool depths, budgets, the
  SASRec gate and the adoption rule were fixed in config before any result.

## 3. Stage 1 — retrieval diagnostics

### Channels in the current union (pooled over six folds; recall = share of all purchased pairs)

| channel | cap | recall alone | unique | in pool beyond cap |
| --- | ---: | ---: | ---: | ---: |
| repeat | 10 | 0.0257 | 0.0111 | 0.0057 |
| pop | 5 | 0.0074 | 0.0001 | 0.0850 |
| pop_age | 65 | 0.0700 | 0.0294 | 0.1185 |
| cf | 5 | 0.0132 | 0.0002 | 0.0441 |
| variant | 20 | 0.0322 | 0.0095 | 0.0146 |
| dept_pop | 40 | 0.0536 | 0.0097 | 0.0324 |
| section_pop | 65 | 0.0741 | 0.0161 | 0.0249 |
| covis | 25 | 0.0432 | 0.0096 | 0.0571 |
| new_arrival_pers | 10 | 0.0087 | 0.0061 | 0.0061 |

Redundant candidate rate (rows removed by de-duplication): 0.291.

### Miss taxonomy (pooled share of all purchased pairs, mutually exclusive)

| outcome | share |
| --- | ---: |
| hit top12 | 0.0609 |
| in union not top12 | 0.1148 |
| proposed then cut by cap | 0.2261 |
| lost by merge or dedup | 0.0000 |
| ineligible not proposed | 0.0408 |
| no channel proposed eligible | 0.5573 |

Of the eligible pairs no current channel proposes, the share also absent from all three neural top-1000 lists is 0.4082 of all pairs.

### Marginal recall of each neural list appended to the current union (pooled)

| list | top-100 | top-300 | top-1000 |
| --- | ---: | ---: | ---: |
| sasrec | 0.0334 | 0.0806 | 0.1916 |
| lightgcn | 0.0206 | 0.0507 | 0.1220 |
| bpr | 0.0139 | 0.0354 | 0.0912 |

### Union recall by segment (current union, pooled)

| segment | candidate recall |
| --- | ---: |
| returning | 0.1829 |
| new | 0.0773 |
| head | 0.2541 |
| tail | 0.0449 |
| recent | 0.1982 |
| repeat | 0.7992 |
| nonrepeat | 0.1512 |
| activity_low | 0.1783 |
| activity_medium | 0.1890 |
| activity_high | 0.1817 |
| eligible | 0.1832 |

Highest pairwise Jaccard overlaps: dept_pop|section_pop 0.267, pop_age|section_pop 0.212, cf|covis 0.159, covis|variant 0.115, dept_pop|pop_age 0.087.

**Reconciliation.** Recomputed candidate hits 254,553 = stored 254,553 over 1,448,653 purchased pairs → candidate recall 0.1757; 7 customers differ in candidate *count* by a few rows (max relative difference 3.4e-07).

*Reading.* The current union is not too short; it is pointed at the wrong items for most purchases.
Over half of all purchased pairs are eligible but proposed by no channel at any depth, and 22.6% are
proposed but sit below a cap. Among the deep lists, SASRec's covers the most purchases outside the
union.

## 4. Stages 2–3 — frontier, append-only neural retrieval, and the decision

### All configurations, six folds pooled (fixed seed-42 ranker per fold)

| config | cand. mean / p50 / p95 | candidate recall | Recall@12 | MAP@12 | NDCG@12 | marginal conversion | rank+score s (×final) | peak RSS GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final | 160 / 166 / 194 | 0.1757 | 0.0609 | 0.03524 | 0.05397 | — | 593 (1.00×) | 4.30 |
| sasrec_A300 | 282 / 300 / 300 | 0.2247 | 0.0612 | 0.03535 | 0.05419 | 0.005 | 970 (1.63×) | 5.03 |
| lightgcn_A300 | 270 / 300 / 300 | 0.2055 | 0.0610 | 0.03528 | 0.05405 | 0.003 | 892 (1.50×) | 5.41 |
| bpr_A300 | 270 / 300 / 300 | 0.1955 | 0.0610 | 0.03527 | 0.05402 | 0.003 | 912 (1.54×) | 4.98 |
| neural3_A300 | 282 / 300 / 300 | 0.2163 | 0.0611 | 0.03534 | 0.05416 | 0.006 | 927 (1.56×) | 5.24 |
| F300 | 308 / 337 / 391 | 0.2431 | 0.0615 | 0.03548 | 0.05438 | 0.009 | 1146 (1.93×) | 5.23 |
| sasrec_A500 | 467 / 500 / 500 | 0.2718 | 0.0613 | 0.03540 | 0.05428 | 0.004 | 1911 (3.22×) | 6.02 |
| lightgcn_A500 | 437 / 500 / 500 | 0.2348 | 0.0611 | 0.03532 | 0.05413 | 0.004 | 1362 (2.30×) | 5.90 |
| bpr_A500 | 437 / 500 / 500 | 0.2169 | 0.0611 | 0.03531 | 0.05409 | 0.004 | 2980 (5.02×) | 6.31 |
| neural3_A500 | 467 / 500 / 500 | 0.2570 | 0.0612 | 0.03537 | 0.05422 | 0.004 | 1301 (2.19×) | 6.63 |
| F500 | 501 / 539 / 627 | 0.3147 | 0.0616 | 0.03547 | 0.05439 | 0.005 | 1567 (2.64×) | 6.74 |
| sasrec_A1000 | 929 / 1000 / 1000 | 0.3550 | 0.0614 | 0.03544 | 0.05433 | 0.003 | 2718 (4.58×) | 6.98 |
| lightgcn_A1000 | 856 / 1000 / 1000 | 0.2876 | 0.0611 | 0.03535 | 0.05415 | 0.002 | 2359 (3.98×) | 7.13 |
| bpr_A1000 | 856 / 1000 / 1000 | 0.2583 | 0.0612 | 0.03533 | 0.05414 | 0.003 | 2317 (3.91×) | 7.93 |
| neural3_A1000 | 929 / 1000 / 1000 | 0.3290 | 0.0613 | 0.03543 | 0.05431 | 0.003 | 2506 (4.23×) | 7.70 |
| F1000 | 852 / 897 / 1087 | 0.4019 | 0.0614 | 0.03538 | 0.05425 | 0.002 | 2908 (4.90×) | 7.35 |

### Bootstrap versus the current final system (relative change, 95% CI; customers across folds)

| config | candidate recall | MAP@12 | Recall@12 | NDCG@12 | tail cand. recall | new-cust. cand. recall | tail Recall@12 | new-cust. MAP@12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sasrec_A300 | +27.90% [+27.64, +28.15] | +0.33% [+0.18, +0.48] | +0.43% [+0.23, +0.64] | +0.41% [+0.25, +0.56] | +71.39% [+69.88, +73.16] | +0.00% [+0.00, +0.00] | +0.45% [-0.12, +1.03] | +0.00% [+0.00, +0.00] |
| lightgcn_A300 | +16.94% [+16.75, +17.14] | +0.11% [+0.03, +0.21] | +0.17% [+0.03, +0.30] | +0.15% [+0.05, +0.25] | +62.09% [+60.61, +63.73] | +0.00% [+0.00, +0.00] | +0.56% [+0.13, +1.00] | +0.00% [+0.00, +0.00] |
| bpr_A300 | +11.27% [+11.13, +11.42] | +0.09% [+0.02, +0.17] | +0.10% [-0.01, +0.21] | +0.09% [+0.02, +0.17] | +37.95% [+36.91, +39.10] | +0.00% [+0.00, +0.00] | +0.43% [+0.09, +0.75] | +0.00% [+0.00, +0.00] |
| neural3_A300 | +23.11% [+22.90, +23.35] | +0.28% [+0.15, +0.42] | +0.39% [+0.21, +0.57] | +0.36% [+0.22, +0.51] | +67.93% [+66.50, +69.57] | +0.00% [+0.00, +0.00] | +0.73% [+0.20, +1.24] | +0.00% [+0.00, +0.00] |
| F300 | +38.36% [+38.04, +38.71] | +0.68% [+0.48, +0.87] | +1.04% [+0.80, +1.28] | +0.77% [+0.58, +0.95] | +82.92% [+81.12, +84.88] | +26.73% [+25.44, +27.92] | +3.69% [+2.96, +4.46] | -1.83% [-3.43, -0.11] |
| sasrec_A500 | +54.65% [+54.24, +55.04] | +0.47% [+0.30, +0.65] | +0.63% [+0.38, +0.88] | +0.58% [+0.40, +0.77] | +142.33% [+139.77, +145.44] | +0.00% [+0.00, +0.00] | +0.47% [-0.17, +1.15] | +0.00% [+0.00, +0.00] |
| lightgcn_A500 | +33.64% [+33.33, +33.95] | +0.24% [+0.13, +0.37] | +0.40% [+0.23, +0.58] | +0.31% [+0.18, +0.45] | +131.78% [+129.23, +134.64] | +0.00% [+0.00, +0.00] | +0.80% [+0.30, +1.35] | +0.00% [+0.00, +0.00] |
| bpr_A500 | +23.42% [+23.20, +23.68] | +0.20% [+0.11, +0.30] | +0.27% [+0.13, +0.42] | +0.22% [+0.13, +0.32] | +81.21% [+79.50, +83.17] | +0.00% [+0.00, +0.00] | +0.61% [+0.17, +1.06] | +0.00% [+0.00, +0.00] |
| neural3_A500 | +46.26% [+45.93, +46.62] | +0.38% [+0.22, +0.55] | +0.50% [+0.28, +0.73] | +0.46% [+0.29, +0.63] | +141.73% [+139.07, +144.81] | +0.00% [+0.00, +0.00] | +0.69% [+0.08, +1.31] | +0.00% [+0.00, +0.00] |
| F500 | +79.09% [+78.58, +79.62] | +0.66% [+0.40, +0.89] | +1.13% [+0.85, +1.43] | +0.78% [+0.56, +1.00] | +121.77% [+119.41, +124.34] | +144.17% [+139.88, +148.69] | +3.33% [+2.50, +4.26] | -6.37% [-8.64, -4.22] |
| sasrec_A1000 | +102.04% [+101.36, +102.71] | +0.56% [+0.37, +0.77] | +0.74% [+0.47, +1.02] | +0.66% [+0.47, +0.87] | +274.04% [+269.33, +279.30] | +0.00% [+0.00, +0.00] | +0.97% [+0.28, +1.70] | +0.00% [+0.00, +0.00] |
| lightgcn_A1000 | +63.69% [+63.21, +64.18] | +0.30% [+0.16, +0.45] | +0.40% [+0.20, +0.62] | +0.34% [+0.19, +0.49] | +260.94% [+256.58, +265.98] | +0.00% [+0.00, +0.00] | +1.14% [+0.56, +1.81] | +0.00% [+0.00, +0.00] |
| bpr_A1000 | +46.98% [+46.62, +47.38] | +0.25% [+0.13, +0.37] | +0.47% [+0.30, +0.66] | +0.32% [+0.20, +0.46] | +167.36% [+164.32, +170.82] | +0.00% [+0.00, +0.00] | +1.17% [+0.65, +1.73] | +0.00% [+0.00, +0.00] |
| neural3_A1000 | +87.25% [+86.68, +87.81] | +0.53% [+0.35, +0.72] | +0.72% [+0.45, +0.98] | +0.64% [+0.45, +0.83] | +275.46% [+270.51, +280.63] | +0.00% [+0.00, +0.00] | +0.99% [+0.31, +1.77] | +0.00% [+0.00, +0.00] |
| F1000 | +128.69% [+127.97, +129.50] | +0.40% [+0.11, +0.68] | +0.81% [+0.50, +1.12] | +0.53% [+0.27, +0.76] | +212.90% [+209.34, +216.91] | +260.51% [+253.35, +268.12] | +4.18% [+3.27, +5.13] | -7.09% [-10.09, -4.00] |

### The predeclared adoption rule, applied mechanically

| config | cand. recall ≥ 0.2021 | MAP CI > 0 | R@12 CI > 0 | folds gain / worst fold | tail / new LCB | time ×, RSS | pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sasrec_A300 | yes | yes | yes | 6/6, +22.31% | +69.88% / +0.00% | 1.63×, +17% | **yes** |
| lightgcn_A300 | yes | yes | yes | 6/6, +12.66% | +60.61% / +0.00% | 1.50×, +26% | **no** |
| bpr_A300 | no | yes | no | 6/6, +8.33% | +36.91% / +0.00% | 1.54×, +16% | **no** |
| neural3_A300 | yes | yes | yes | 6/6, +18.18% | +66.50% / +0.00% | 1.56×, +22% | **yes** |
| F300 | yes | yes | yes | 6/6, +33.66% | +81.12% / +25.44% | 1.93×, +22% | **yes** |
| sasrec_A500 | yes | yes | yes | 6/6, +44.42% | +139.77% / +0.00% | 3.22×, +40% | **no** |
| lightgcn_A500 | yes | yes | yes | 6/6, +25.46% | +129.23% / +0.00% | 2.30×, +37% | **no** |
| bpr_A500 | yes | yes | yes | 6/6, +17.53% | +79.50% / +0.00% | 5.02×, +47% | **no** |
| neural3_A500 | yes | yes | yes | 6/6, +37.19% | +139.07% / +0.00% | 2.19×, +54% | **no** |
| F500 | yes | yes | yes | 6/6, +73.34% | +119.41% / +139.88% | 2.64×, +57% | **no** |
| sasrec_A1000 | yes | yes | yes | 6/6, +85.49% | +269.33% / +0.00% | 4.58×, +62% | **no** |
| lightgcn_A1000 | yes | yes | yes | 6/6, +49.56% | +256.58% / +0.00% | 3.98×, +66% | **no** |
| bpr_A1000 | yes | yes | yes | 6/6, +35.96% | +164.32% / +0.00% | 3.91×, +84% | **no** |
| neural3_A1000 | yes | yes | yes | 6/6, +71.47% | +270.51% / +0.00% | 4.23×, +79% | **no** |
| F1000 | yes | yes | yes | 6/6, +118.74% | +209.34% / +253.35% | 4.90×, +71% | **no** |

### Per fold (candidate recall / MAP@12)

| config | 08-05 | 08-12 | 08-19 | 08-26 | 09-02 | 09-09 |
| --- | --- | --- | --- | --- | --- | --- |
| final | 0.155 / 0.03048 | 0.164 / 0.03437 | 0.171 / 0.03421 | 0.187 / 0.03436 | 0.192 / 0.03923 | 0.186 / 0.03884 |
| sasrec_A300 | 0.213 / 0.03069 | 0.217 / 0.03440 | 0.217 / 0.03429 | 0.229 / 0.03444 | 0.238 / 0.03938 | 0.236 / 0.03899 |
| lightgcn_A300 | 0.193 / 0.03054 | 0.198 / 0.03444 | 0.203 / 0.03425 | 0.213 / 0.03436 | 0.217 / 0.03929 | 0.211 / 0.03886 |
| bpr_A300 | 0.180 / 0.03051 | 0.186 / 0.03441 | 0.193 / 0.03424 | 0.204 / 0.03439 | 0.208 / 0.03928 | 0.202 / 0.03887 |
| neural3_A300 | 0.205 / 0.03061 | 0.209 / 0.03439 | 0.213 / 0.03432 | 0.222 / 0.03445 | 0.227 / 0.03938 | 0.223 / 0.03894 |
| F300 | 0.227 / 0.03062 | 0.235 / 0.03459 | 0.237 / 0.03429 | 0.252 / 0.03464 | 0.257 / 0.03960 | 0.252 / 0.03920 |
| sasrec_A500 | 0.266 / 0.03070 | 0.267 / 0.03438 | 0.261 / 0.03437 | 0.271 / 0.03453 | 0.282 / 0.03947 | 0.284 / 0.03904 |
| lightgcn_A500 | 0.229 / 0.03061 | 0.232 / 0.03447 | 0.234 / 0.03430 | 0.238 / 0.03437 | 0.241 / 0.03937 | 0.235 / 0.03891 |
| bpr_A500 | 0.207 / 0.03054 | 0.211 / 0.03439 | 0.215 / 0.03425 | 0.223 / 0.03444 | 0.226 / 0.03935 | 0.220 / 0.03892 |
| neural3_A500 | 0.253 / 0.03065 | 0.254 / 0.03446 | 0.253 / 0.03431 | 0.258 / 0.03450 | 0.264 / 0.03938 | 0.261 / 0.03899 |
| F500 | 0.298 / 0.03056 | 0.300 / 0.03455 | 0.301 / 0.03431 | 0.327 / 0.03464 | 0.333 / 0.03965 | 0.330 / 0.03916 |
| sasrec_A1000 | 0.358 / 0.03064 | 0.354 / 0.03441 | 0.344 / 0.03444 | 0.348 / 0.03459 | 0.360 / 0.03950 | 0.367 / 0.03908 |
| lightgcn_A1000 | 0.291 / 0.03060 | 0.291 / 0.03440 | 0.288 / 0.03430 | 0.285 / 0.03444 | 0.288 / 0.03945 | 0.282 / 0.03896 |
| bpr_A1000 | 0.257 / 0.03058 | 0.257 / 0.03445 | 0.258 / 0.03423 | 0.259 / 0.03446 | 0.262 / 0.03935 | 0.257 / 0.03895 |
| neural3_A1000 | 0.336 / 0.03067 | 0.332 / 0.03442 | 0.325 / 0.03442 | 0.321 / 0.03458 | 0.330 / 0.03949 | 0.330 / 0.03904 |
| F1000 | 0.389 / 0.03046 | 0.381 / 0.03441 | 0.384 / 0.03430 | 0.415 / 0.03470 | 0.421 / 0.03945 | 0.422 / 0.03901 |

**Exact search, not ANN.** Exact top-1000 dot products cost about 0.28 ms per customer
(about 18 s per fold for ~70k customers on the CPU), against about 100 s of feature
building and scoring per fold for the current system. An ANN index could not materially lower the pipeline cost, so by the
plan's rule it was not adopted.

**The SASRec gate passed** (pooled candidate-recall gain +4.90 points at 300, 6/6 folds), so LightGCN,
BPR-MF and their union were run as the plan requires. **Stage 4** (a learned fixed-budget merge)
was not run: it is warranted only when a wider union improves quality at excessive cost, and no
configuration above 300 candidates beats `F300`'s MAP@12, which already passes the cost gate.

**Decision.** Three configurations pass: `sasrec_A300`, `neural3_A300` (both 282 candidates on
average) and `F300` (308). The rule selects the smallest; the 282-candidate tie was not anticipated
by the plan and is broken for `sasrec_A300`, which is better on every quality metric and needs one
neural model in production instead of three. `F300` is the best quality-only point (MAP@12
+0.68% [+0.48, +0.87]; it converts 0.9% of its added candidate recall, against 0.5% for
`sasrec_A300`) but lowers **new-customer MAP@12 by 1.83% [−3.43, −0.11]** and runs at 1.93×.

## 5. Segments and trade-offs of the adopted configuration

- New customers: unchanged (SASRec has no vector for a customer with no history; candidate recall and
  MAP@12 differences are exactly 0).
- Tail items: candidate recall **+71.39%**, Recall@12 +0.45% [−0.12, +1.03] (not significant).
- Candidate recall rises in every fold, by +22.3% to +36.8%.
- Candidates per customer: mean 282, median and 95th percentile 300 (customers SASRec cannot
  represent keep the ~160-candidate union).

## 6. Cost (idle machine)

All evaluations ran one at a time on the Apple M5 / 16 GB laptop with no other job of this project
running; each configuration's figure is the sum over six folds of candidate construction, feature
building and scoring, against the same code path for `final` (593 s, 4.30 GB). Three fold timings
are outliers attributable to the machine, not the configuration (`sasrec_A500` 2020-08-05 755 s,
`bpr_A500` 2020-09-02 1,162 s, `F1000` 2020-08-12 900 s, against ~250–450 s for their other folds);
none of those configurations would pass the 2× limit without them either. The adopted
configuration's six folds took 165–189 s of wall time each (about 95–103 s for `final`'s rank + score).

## 7. Incidents and fixes

| incident | fix |
| --- | --- |
| Reconciliation found 4 of 11.9M candidates differing on one customer | production co-visitation kept its 50 neighbours ordered by a multi-threaded float sum; now ordered on `round(sim, 9)` (same fix as the D-027 merge). Hits always reconciled exactly |
| `DataFrame.merge` dropped `attrs` (coverage, novelty), crashing the first eval's metadata write | attrs preserved across the merge; queue made to stop on the first failure |
| A queue hand-off that polled for a process could have started extensions concurrently | the extension queue waits on the main loop's PID |
| The 1,000-candidate frontier target is unreachable within the declared pool depths | `F1000` is the full pool (852 candidates on average), reported as such |
| Usage limit paused the session once | queues and caches kept running; no compute was repeated |

## 8. Limitations and next experiment

- Offline only; the served Track A recommendations still come from the Phase-2 model.
- Fixed ranker: deeper channel ranks and appended items (no Phase-2 provenance, `n_channels = 0`)
  lie outside its training distribution, which is the likely reason for the low conversion.
- One ranker seed (42) per configuration; the seed spread of the final system was ±0.00004 MAP@12.
- The 282-candidate tie-break was a post-hoc, documented choice.

**Next recommended experiment.** Retrain the ranker on the adopted 300-candidate distribution with
the SASRec provenance (source rank, raw and calibrated score) as features, on the same rolling
folds, to raise the 0.5% conversion. Second: the diagnostics show 55.7% of purchases proposed by no
channel and 40.8% absent from all three neural top-1000 lists, so a new content-based / two-tower
retrieval channel is now justified as a follow-up (it was out of scope for this round).

## 9. Reproduction

```bash
ENSEMBLE_CONFIG=track_a_research PYTHONPATH=src .venv/bin/python -m ensemble.research.retrieval_upgrade pools <fold>        # ~1.5 min/fold
ENSEMBLE_CONFIG=track_a_research PYTHONPATH=src .venv/bin/python -m ensemble.research.retrieval_upgrade diagnostics <fold>  # ~3 min/fold
ENSEMBLE_CONFIG=track_a_research PYTHONPATH=src .venv/bin/python -m ensemble.research.retrieval_upgrade allocate            # ~1 min
ENSEMBLE_CONFIG=track_a_research PYTHONPATH=src .venv/bin/python -m ensemble.research.retrieval_upgrade queue final,sasrec_A300,F300   # ~2–8 min per config-fold
ENSEMBLE_CONFIG=track_a_research PYTHONPATH=src .venv/bin/python -m ensemble.research.retrieval_upgrade report
.venv/bin/python scripts/retrieval_report_tables.py
.venv/bin/python scripts/verify_retrieval_report.py
```

## 10. Git

Commits on `track-a-retrieval-upgrade` since `main` (newest first), plus the commit that adds this
report:

```
98345c1 feat(retrieval): six-fold frontier and append-only results; adopt SASRec append to 300 (D-044)
b130a9a feat(retrieval): six-fold diagnostics (all reconciled) and the seeded budget frontier
59c4f47 feat(retrieval): deep point-in-time pools, exact neural top-1000, diagnostics, seeded frontier, append-only eval
6312c96 config(retrieval): predeclare pool depths, budgets, SASRec usefulness gate and adoption rule
d94d237 docs(retrieval): fix Track A ceiling upgrade protocol
```

- Nothing was pushed, merged, published or submitted; `main` and `origin` are unchanged.
- Every commit staged explicit paths; the user-owned `* 2.*` sync duplicates remain untracked and
  unmodified.
- Per-customer diagnostics (`reports/track_a_retrieval/diagnostics/customers_*.parquet`, 8.2 MB)
  are gitignored by the project's `*.parquet` rule and stay local; pools, deep lists and evaluation
  rows live under the gitignored `data/` tree.
- At the final commit: `make test` 166 passed; `scripts/verify_retrieval_report.py` 22/22.

