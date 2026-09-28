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
