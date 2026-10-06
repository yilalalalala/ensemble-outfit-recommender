# Model card: Track B Complete the Look (outfit completion)

| | |
| --- | --- |
| **Task** | Given an anchor article and a missing slot (bottoms, shoes, accessories…), rank articles in that slot that complete the outfit (D-002) |
| **Architecture** | Weighted reciprocal rank fusion (w = 0.5) of association rules (support ≥ 3, NPMI-ranked) and a two-tower model; diversity re-ranking (≤ 2 per product type) at serving (D-015) |
| **Two-tower** | Metadata embeddings + price tier + popularity decile + article-ID embedding (50% ID dropout); slot-conditioned query tower; sampled softmax with in-batch negatives + logQ correction, popularity-based negatives (∝ pop^0.75), hard negatives; 3 epochs |
| **Training data** | Baskets (customer × day, 2–6 articles, ≥ 2 slots) from the 16 weeks before the target week |
| **Evaluation** | Held-out baskets in the target week: one query per (basket, anchor, missing slot); universe = live catalogue (D-013) |
| **Artifacts** | `data/processed/models/two_tower.pt`, `two_tower_encoder.pkl`, `completion_pairs.parquet`, `completion_universe.parquet` |

## Offline metrics (Recall@12; relative lift vs popularity first, D-004)

| model | validation lift | validation R@12 | test lift | test R@12 |
| --- | ---: | ---: | ---: | ---: |
| popularity (per slot) | — | 0.0750 | — | 0.0837 |
| association rules (NPMI, support ≥ 3) | +47.1% | 0.1103 | +36.1% | 0.1140 |
| two-tower | +50.9% | 0.1131 | +45.0% | 0.1214 |
| **hybrid RRF (shipped)** | **+61.4%** | **0.1210** | **+54.2%** | **0.1291** |

Beyond-accuracy, test, hybrid: tail-item recall 4.4% (popularity 1.2%), catalog
coverage 20.0% (popularity 0.3%), jewellery recall 1.3%, item cold-start recall
0%.

## Ablations (validation)

| question | result |
| --- | --- |
| Lift/PMI ranking vs raw co-count | raw count +50.4% vs NPMI +47.1%; NPMI has higher tail recall (4.95% vs 4.68%) |
| Minimum support | 3 is best (2: +42%, 5: +43%, 10: +34%) |
| Negative sampling (1 epoch) | in-batch only −11.6%; + logQ +36.3%; + popularity-based +36.3%; + hard +34.8% |
| Content only vs with ID embedding | +21.6% vs +50.9% |
| Association vs two-tower vs hybrid | +47% / +51% / +61% |

## Limitations

- Co-purchase is not co-wear: labels are items bought together, not outfits
  worn together (DESIGN §5.2). The optional M8 Polyvore experiment tests this.
- Jewellery (≈ 1.3% recall) and item cold start (≈ 0%) are weak. Metadata alone
  cannot distinguish near-identical new accessories; CLIP image features
  (M7a) are the planned remedy.
- Not personalised yet. Re-ranking by the customer's Track A profile is part of
  D-011.
- The cross-feature ranker in DESIGN §5.3 is not built; RRF stands in for it.
- Offline only.

## Update 2026-09-28 — FashionCLIP towers (D-022)

The shipped model now feeds frozen FashionCLIP image vectors (projected to 64 dims) into both
towers.

| model | relative lift, validation | relative lift, test |
| --- | ---: | ---: |
| hybrid RRF, previous (MVP) | +58.6% | +51.6% |
| **hybrid RRF, with CLIP** | **+62.0%** | **+55.6%** |
| two-tower content-only, with CLIP | +38.8% (was +17.3%) | +37.7% (was +15.2%) |

The larger content-only gain means the model now leans less on article IDs, which matters for
new items. Item cold-start recall is still ≈ 0% because never-sold items rarely enter the live
catalogue in time (D-013). Jewellery recall on validation rose from 1.6% to 2.0%.

---

## Update 2026-10-03 — Round 3: leakage-free protocol, learned fusion, personalization

Full readout: [`TRACK_B_ROUND3_UPGRADE_REPORT.md`](TRACK_B_ROUND3_UPGRADE_REPORT.md).
ADRs D-031 … D-037.

| | |
| --- | --- |
| **Architecture** | Candidate union (article association by co-count and by NPMI, product-code association both ways, two-tower with and without the article-ID embedding, slot popularity) → **LightGBM LambdaRank**, one query group per (basket, anchor, target slot). The fixed RRF(w = 0.5) of D-015 is now a reported baseline, not the model. |
| **Features** | 95 columns in three groups: *compatibility* (co-count, 14 d / 56 d decayed co-counts, support, lift, PMI, NPMI at article and style level, backoff level, tower and FashionCLIP similarity, category / colour / price agreement), *candidate* (pre-cutoff popularity, recency, days sold, catalogue age, price), *personalization* (recency-weighted customer affinities for article, style, product type, department, section, garment group, colour and slot, usual price point, and compatibility × preference interactions). |
| **Two-tower** | in-batch negatives + logQ correction only; six epochs with held-out early stopping. Popularity-sampled and (product type, price tier) hard negatives are gone (D-016); redesigned retrieval-informed hard negatives were tested and left off (D-036). |
| **Protocol** | Eligible catalogue from sales strictly **before** the cutoff (D-031); six rolling label weeks for selection; the test week scored once; intervals from a bootstrap that resamples **customers** (D-032). |
| **Shipped at serving** | `lgbm_compatibility` (no customer features) — the Complete-the-Look table is keyed by (anchor, target slot) and must answer for anonymous visitors. `lgbm_personalized` is the measured upper bound (D-033). |
| **Artifacts** | `data/interim/track_b/<cache-key>/` (per-week towers, pair similarities, training matrices), `data/processed/models/complete_the_look.parquet` |

### Offline metrics

Recall@12 and NDCG@12 on held-out baskets, leakage-free catalogue. Six rolling
validation folds (2020-08-05 … 2020-09-09) and the single test week (2020-09-16).
**These are not comparable to the Round-1 numbers above** — the catalogue
definition, the query set and the truth denominator all changed (D-031).

| system | val R@12 | val NDCG@12 | val vs shipped | test R@12 | test NDCG@12 | test vs shipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| slot popularity | 0.0752 | 0.0382 | −29.8% | 0.0884 | 0.0461 | −29.9% |
| association rules (NPMI) | 0.1007 | 0.0609 | −6.3% | 0.1202 | 0.0747 | −4.7% |
| shipped RRF hybrid (Round 1) | 0.1066 | 0.0622 | — | 0.1261 | 0.0752 | — |
| RRF over the full union | 0.1088 | 0.0642 | +2.0% | 0.1322 | 0.0781 | +4.9% |
| **learned ranker, compatibility** | **0.1379** | **0.0822** | **+29.3%** | **0.1596** | **0.0956** | **+26.6%** |
| **learned ranker, personalized** | **0.1573** | **0.0961** | **+47.7%** | **0.1820** | **0.1124** | **+44.3%** |

Pooled customer-cluster bootstrap on validation (145,485 customers, 622,989
queries): personalized **+47.58% Recall@12 [+46.65%, +48.47%]**, compatibility
**+29.37% [+28.57%, +30.15%]**; both win 6/6 folds.

Beyond accuracy, test week, personalized vs the shipped hybrid: tail-item recall
0.1080 (+147%), recently-launched recall 0.1765 (+147%), jewellery 0.0287 (+363%),
catalogue coverage 0.2789 (+24%), novelty 12.84 (+0.95 bits). Intra-list diversity
is 11% lower before the serving rules and 40% higher after them.

### Ablations (six rolling folds, each group removed and the ranker refitted)

| feature group removed | Δ Recall@12 |
| --- | ---: |
| personalization (21 columns) | −12.33% |
| candidate popularity / recency / age (10) | −10.50% |
| two-tower (7) | −3.97% |
| style backoff (14) | −0.77% |
| exact-article repeat (2) | −0.74% |
| price + colour agreement (10) | −0.70% |
| FashionCLIP similarity (1) | −0.66% |
| time-decayed co-counts (6) | −0.62% |
| article association (17) | −0.42% |

| other ablation | result |
| --- | --- |
| two-tower negatives (2 weeks × 2 seeds) | in-batch only −26.5%; retrieval-informed hard negatives +0.94% against a 1.0–1.5% seed spread for +61% training time → off |
| serving diversity caps, as a re-ordering | −6.8% Recall@12, +49% distinct product types per list |
| serving diversity caps, as hard filters | −37.5% Recall@12 (modules come out part-empty) → changed to a re-ordering (D-035) |
| what the Round-1 catalogue leak was worth | −5.1% Recall@12 for every fixed-fusion system (it added un-retrievable truth), 5.9% of truth excluded under the honest catalogue |

### Limitations (Round 3)

- **Offline only.** No online A/B test; nothing here establishes engagement effects.
- **Co-purchase is not co-wear.** 15.6–17.2% of mined baskets contain a repeated
  quantity and 14.3–15.4% more than one colourway of one style. M8 (Polyvore) is
  the only way to separate the two in this project.
- **The test week is a confirmation, not a fresh holdout** — it was scored in Round 1.
- **The candidate union is the ceiling**: union recall 0.4285, of which the ranker
  converts 36.7%. More than half of all truth pairs are unreachable by any ranking.
- **Item cold start is unmeasurable** (no inventory or launch feed); replaced by a
  recently-launched segment, and 5.9% of truth articles are excluded as ineligible.
- **Serving ships the weaker of the two rankers**; closing the gap needs an online
  feature store, not a model change.
- **Jewellery is still the weakest segment** in absolute terms (0.0287 on test).
- Relative confidence intervals divide the paired difference interval by the
  observed baseline mean rather than resampling the denominator.

---

## Update 2026-10-05 — request-time personalized serving (D-041 … D-043)

Readout: [`TRACK_A_RESEARCH_TRACK_B_PRODUCTION_REPORT.md`](TRACK_A_RESEARCH_TRACK_B_PRODUCTION_REPORT.md),
runbook [`docs/RUNBOOK_SERVING.md`](../docs/RUNBOOK_SERVING.md).

| | |
| --- | --- |
| **Served model** | `lgbm_personalized` now ships, at request time: it re-orders the compatibility ranker's top-100 pool per (anchor, target slot) with the customer's point-in-time profile; anonymous and unknown customers get the compatibility order. |
| **Offline regression (six folds)** | P = 100 keeps 98.9% of unrestricted personalized Recall@12 (99% bar missed by 1.1 pt, reported); served pipeline +7.06% Recall@12 over the previously served compatibility + diversity table, CI above zero, 6/6 folds. |
| **Serving bundle** | 156,990 keys × 100 candidates, 1,356,683 customer profiles, 26,127-article visual index; published only after a 400-request offline/online equivalence gate (identical orderings). |
| **Performance (local, one worker)** | uncached p95 26 ms, 68 rps at concurrency 4; warm p95 2.9 ms; startup < 1 s; RSS ≤ 1 GB. |
| **Status** | A production-oriented prototype: no inventory feed (availability is a sales proxy plus manual overrides), no live traffic, no online A/B test. |
