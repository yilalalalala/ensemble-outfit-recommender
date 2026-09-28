# Ensemble MVP: offline evaluation readout

**Date:** 2026-09-27 · **Scope:** M0–M6 (DESIGN §8) · **Data:** H&M, 31.8M transactions,
2018-09-20 → 2020-09-22 · **Weeks:** validation 2020-09-09..15 (model selection),
test 2020-09-16..22 (evaluated once)

All results are **offline**. A launch decision would need an online A/B test
(CTR, add-to-cart, conversion).

---

## 1. Summary

1. **Track A (next purchase): MAP@12 0.0370 on the test week, +37.5% relative
   lift over the best baseline** (repeat purchase + age-band popularity, 0.0269).
   Validation: 0.0355, +40.5%.
   **Kaggle late submission (2026-09-28): private MAP@12 0.03189, public 0.03102.**
   That is above the ~0.0300 score of about 45th place of 3,006 (silver), and
   16% below 1st place (0.0379). Late submissions are scored but not ranked.
2. **Track B (Complete the Look): Recall@12 0.129 on the test week, +54.1%
   relative lift over popularity.** The shipped model fuses association rules
   and a two-tower model. Each component alone is +36% to +45%.
3. **The co-purchase signal is real but sparse.** Of 1.33M cross-slot pairs,
   38k have support ≥ 3, and 99% of those have association lift > 1. The
   binding constraint is support, not popularity (D-016).
4. **Weak spots are specific:** item cold start (≈ 0% in both tracks),
   jewellery (1.3% recall in Track B), and new customers in Track A (the ranker
   is not better than age-band popularity on the test week).
5. **One command reproduces everything:** `make mvp` ran clean in 99 minutes
   with 5.7 GB peak memory on an M-series laptop with 16 GB of RAM (§8).

---

## 2. Track A: next-purchase ranking

| system | MAP@12 val | MAP@12 test | relative lift, test |
| --- | ---: | ---: | ---: |
| popularity (global, last 7 days) | 0.00660 | 0.00875 | |
| popularity by age band | 0.00756 | 0.00960 | |
| repeat purchase + popularity | 0.02459 | 0.02628 | |
| repeat purchase + age-band popularity (**baseline**) | 0.02524 | 0.02689 | — |
| **retrieval → LightGBM LambdaRank** | **0.03546** | **0.03699** | **+37.6%** |

**Segment analysis (test).**

| segment | share of buyers | ranker | best baseline |
| --- | ---: | ---: | ---: |
| returning customers | 91.9% | 0.03951 | 0.02845 |
| new customers (user cold start) | 8.1% | 0.00831 | 0.00909 |
| purchases of never-sold articles (item cold start) | 3.9% of purchases | Recall@12 0 | 0 |

**Retrieval (validation).** Six channels, about 116 candidates per customer,
**merged recall 13.1%**. That is the ceiling on what the ranker can find.

| channel | recall | candidates/customer | recall unique to channel |
| --- | ---: | ---: | ---: |
| repeat purchase (52 weeks) | 3.5% | 21.1 | 2.1% |
| popularity (7 days) | 4.7% | 40.0 | 1.2% |
| age-band popularity | 4.3% | 30.0 | 0.9% |
| new arrivals | 0.9% | 10.0 | 0.4% |
| item-to-item CF | 4.8% | 37.3 | 2.0% |
| variants (same style, other colourway) | 2.5% | 12.1 | 0.5% |

**Ranker.** 200 trees (validation MAP plateaus there: 0.0351 at 100 trees,
0.0355 at 200 and 300; an exploratory 600-tree run gave 0.0356). Run-to-run
variation from multithreaded training is about ±0.0002. Top features by gain: days since the customer
last bought the article, product type, last-week sales, days since the
customer's last purchase, repeat-purchase score and rank, and item-CF rank.
Model card: [MODEL_CARD_track_a.md](MODEL_CARD_track_a.md).

---

## 3. Track B: Complete the Look (outfit completion)

Task: given an anchor article and a missing slot, rank the slot's live
catalogue. Evaluation: one query per (basket, anchor, missing slot) from
held-out baskets; 97,651 validation queries and 90,571 test queries.

**Relative lift vs popularity is reported first (D-004).**

| model | lift, val | lift, test | R@12 test | NDCG@12 test | tail-item R@12 | catalog coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| popularity per slot | — | — | 0.0837 | 0.0437 | 1.2% | 0.3% |
| association rules (support ≥ 3, NPMI) | +47.1% | +36.1% | 0.1140 | 0.0709 | 4.1% | 20.8% |
| two-tower | +50.9% | +45.0% | 0.1214 | 0.0731 | 4.8% | 19.2% |
| **hybrid: RRF of both (shipped)** | **+61.4%** | **+54.2%** | **0.1291** | **0.0786** | 4.4% | 20.0% |

**Pair-mining funnel (16 weeks before the test week).**

| stage | count |
| --- | ---: |
| baskets (customer × day) | 1,517,034 |
| 2–6 articles | 870,270 |
| ≥ 2 slots | 489,233 |
| distinct cross-slot pairs | 1,325,563 |
| support ≥ 3 | 38,275 |
| support ≥ 3 and lift > 1 | 37,971 |
| support ≥ 10 | 1,599 |
| support ≥ 10 and lift ≥ 2 | 1,564 |

**Ablation studies (validation).**

| question | result | takeaway |
| --- | --- | --- |
| Negative sampling (1 epoch each) | in-batch only **−11.6%** · + logQ correction **+36.3%** · + popularity-based +36.3% · + hard +34.8% | logQ correction is what makes the two-tower beat popularity; the extra negatives add nothing measurable |
| Lift/PMI vs raw co-count | NPMI +47.1% · raw count +50.4% (tail recall 4.95% vs 4.68%) | raw count is slightly better overall; NPMI finds more tail items |
| Minimum support | 2: +42% · **3: +47%** · 5: +43% · 10: +34% | 3 is optimal; the design's default of 10 discards most signal |
| Content-only vs content + ID embedding | +21.6% vs +50.9% | behavioural ID signal doubles the gain; content-only still beats popularity |
| Association vs two-tower vs hybrid | +47% / +51% / +61% | the two methods are complementary |

Model card: [MODEL_CARD_track_b.md](MODEL_CARD_track_b.md).

---

## 4. Explainability and the product (M5, M6)

- **Reasons.** Every Track A recommendation carries up to two reason chips,
  derived from exact TreeSHAP values (LightGBM `pred_contrib`) grouped by
  feature family, e.g. "You bought this before", "Often bought with items you
  purchased". The *Why?* dialog shows the top eight feature contributions.
- **Complete the Look reasons** cite the evidence directly: "Bought together
  39.6× more often than chance" (association lift) or "Style match" (two-tower
  only).
- **Web app** (`make serve` → http://localhost:8010):
  - Home: *Recommended for you*, *Buy it again*, *Trending this week*.
  - Product page: *Complete the Look* by slot, *Other colours*, *Similar items*.
  - DS view with the tables above.
  - A demo customer picker with 300 returning and 30 new customers.
  - Impressions, clicks and "Not for me" are logged to an `events` table.
- **Diversity re-ranking.** At most two items per product type and one colourway
  per style in each module. A slot is shown only when co-purchase evidence
  exists, except shoes and accessories, which are always shown.

---

## 5. Problems found and fixed during the build

| problem | root cause | fix |
| --- | --- | --- |
| Kaggle download failed | CLI not installed; script required the legacy `kaggle.json` | installed CLI via `uv tool`; the script now probes the API instead |
| LightGBM import error | macOS needs the OpenMP runtime | `brew install libomp` (README) |
| `import ensemble` failed mid-project | macOS set the "hidden" flag on the venv's `.pth` files; Python skips hidden `.pth` | cleared the flag; Makefile and pytest use `PYTHONPATH=src` |
| MLflow refused to start | MLflow 3 put the file store into maintenance mode | SQLite backend store (`mlruns/mlflow.db`) |
| Feature frame of 3.5 GB per week | int64/object columns | float32/int32 compaction and positives-only groups (D-014): about 0.7 GB per training week |
| Serving build hung at 0% CPU | LightGBM and PyTorch load two OpenMP runtimes, which deadlocked in one process | Complete the Look precomputed in a separate PyTorch-only process |
| Two-tower below popularity | the towers had no way to represent popularity | recent-sales decile as an item feature: sample Recall@12 0.060 → 0.114 (D-015) |
| Track B crash on NaN categories | pandas 3 keeps NaN when casting to string | explicit "NA" token |
| Swimwear suggested for a jacket; duplicate colourways | no slot evidence gate, no style dedupe | evidence-gated slots and one colourway per style |
| New customers saw no "For you" items | `explode` left NumPy objects that SQLite stored as blobs | cast to int64 |

**Approaches tried and rejected (validation):** time-proximity and directed
item-CF pairs (no recall gain over co-occurrence); index-group popularity
channel (0.16% unique recall).

---

## 6. Deviations from the design

| design | MVP | reason / record |
| --- | --- | --- |
| Track B as accessory completion | outfit completion across seven slots, with jewellery as a segment | jewellery is too sparse (D-002) |
| Content-only towers | towers include a popularity decile and an ID embedding with 50% dropout | measured gain (D-015) |
| Cross-feature ranker for Track B | weighted RRF of association and two-tower | MVP scope; revisit trigger in D-015 |
| Negative mix 50/30/20 % | in-batch (all), 64 popularity-based per slot per batch, 4 hard per positive | expressed as counts; ablation shows the mix matters little once logQ is on |
| Min pair support 10 | 3 for the recommender (10 still reported in the funnel) | validation (§3) |
| CLIP image features | not in MVP | D-009/D-011: M7a |

---

## 7. Risks and next steps

1. **Done: Kaggle private 0.03189.** Local-to-leaderboard ratio is 0.86
   (0.03189 / 0.03699 local test). Use it to translate future local gains, and
   re-check it with every submission.
2. **Item cold start and jewellery:** CLIP image embeddings in both tracks
   (M7a). This is also a content-based retrieval channel for Track A.
3. **New customers:** a segment-level fallback (age-band popularity) is a
   candidate policy; the validation and test weeks disagree, so it needs
   more weeks or an A/B test.
4. **Retrieval ceiling (13%):** a content channel and more candidates, measured
   against ranker cost.
5. **Personalised Complete the Look:** re-rank Track B results with the
   customer's Track A profile (D-011).
6. **M7a visual search → M7b assistant (required, D-012).** M7b needs an
   Anthropic API key.

---

## 8. Reproducing

```bash
brew install libomp && make setup && make data
make mvp        # 99 min, 5.7 GB peak (reports/mvp_run.log)
make serve      # http://localhost:8010
make mlflow     # every run's params and metrics
```

Every number in this report is in `reports/*.json`, with the matching MLflow run.

## 9. Where to look first when checking

- **Leakage:** `tests/test_leakage.py` corrupts every row inside the label week
  and asserts that baselines, retrieval and Track B mining are unchanged.
  Features read only the `_past` view (`src/ensemble/features/track_a.py`).
- **The Track B universe assumption** (D-013): the live catalogue includes
  articles first sold during the target week.
- **Validation vs test consistency:** the ranker scores higher on test (0.0370)
  than on validation (0.0355). Baselines also rose (0.0252 → 0.0269), so the
  test week looks easier, not leaked.
- **Two-tower runs on MPS are not bit-reproducible**; single-run differences
  under about one point are noise (D-016).
