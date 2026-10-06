# Decisions

Architecture Decision Records (ADRs): numbered, dated, with the reasoning and the revisit condition. A decision whose
justification is not written down gets relitigated every time someone new reads
the code.

---

### D-001 — The competition task is adopted unchanged, on the full catalogue
**Date:** 2026-09-27 · **Status:** active

Track A solves the published task — 12 articles, seven-day horizon, MAP@12 — on
the complete catalogue rather than a filtered subset.

**Why.** The value of this dataset over a bespoke one is that 3,006 teams have
already published scores on it. That comparison only holds if the task and the
data are the same. A MAP@12 computed on a self-chosen subset is a number nobody
can check, which is the failure mode the external benchmark exists to prevent.

A second reason is statistical: cross-category signal ("customers who bought
this dress also bought that necklace") is destroyed by filtering to accessories
before training.

**Revisit if:** compute cost makes the full catalogue impractical, in which case
subsetting must be reported as a deviation and the leaderboard comparison
dropped rather than quietly retained.

---

### D-002 — Track B is outfit completion on the full catalogue; jewellery is a showcase slice
**Date:** 2026-09-27 · **Status:** active (revised 2026-09-27)

Track B trains on the full catalogue and recommends complementary articles for
any missing slot (bottoms, shoes, bag, accessories…). Jewellery is demoed and
reported as its own segment.

**Why.** The original version presented only accessories. Measured on the
16-week window, only 26k of 898k baskets (2.9%) pair a garment with jewellery,
spread over ~2,000 jewellery articles. After support and lift filtering almost
no jewellery pairs would survive, which would make Track B fail for lack of data
rather than because the idea is wrong. Broadening the slots uses the same model
and pipeline; only the complementary-pair rule changes. It also matches the
industry product (Complete the Look covers all categories) and the photo feature,
where a user's outfit can be missing any slot.

**Revisit if:** jewellery-segment metrics are strong enough to justify a
jewellery-specific model.

---

### D-003 — Temporal splits only
**Date:** 2026-09-27 · **Status:** active

Validation is the week before the test week; features for a given week are built
strictly from earlier data.

**Why.** Random splits let a model see the future. With repeat purchases, which
are common in this dataset, this inflates every metric to the point of fiction.

**How enforced.** Feature builders take a cutoff date as a required argument
rather than a defaulted one, and a test asserts no feature reads beyond it.

---

### D-004 — Popularity is the baseline everywhere
**Date:** 2026-09-27 · **Status:** active

Every result is reported alongside a popularity baseline, and Track B reports
relative lift vs. the popularity baseline before any other number.

**Why.** In recommender systems a popularity ranker is far stronger than people
expect, and the most common way to produce a convincing-looking result is to
build a model that has learned popularity and nothing else. Making it the
explicit null hypothesis is the only reliable defence.

---

### D-005 — Co-purchase pairs are filtered by PMI, not raw frequency
**Date:** 2026-09-27 · **Status:** active

Track B's pairs are filtered by Pointwise Mutual Information above a support
threshold, not by co-occurrence count.

**Why.** Raw co-occurrence mostly measures joint popularity: two best-sellers
appear together often because each appears often, not because they belong
together. PMI divides by exactly that expectation, so it is positive only when a
pair co-occurs more than independence predicts.

**Consequence worth stating in advance:** if very few pairs survive PMI
filtering, the co-purchase signal was popularity all along. That is a real
possible outcome of this project and it will be reported as a finding rather
than worked around.

---

### D-006 — Time-windowed sampling, as configuration
**Date:** 2026-09-27 · **Status:** active

The working set is the most recent ~16 weeks. The window is a parameter.

**Why.** Fashion is seasonal, and purchases from two years ago are weak evidence
about next week. The window also cuts the working set by roughly an order of
magnitude, which is what makes iteration on a laptop viable.

Customers are *not* subsampled: customer-level sampling distorts the popularity
distribution that every baseline depends on.

**Revisit at:** the final run before publication, where the full window should be
attempted and the difference reported.

---

### D-007 — DuckDB for analytics, SQLite for serving
**Date:** 2026-09-27 · **Status:** active

**Why.** ~30M rows across three related tables, aggregated repeatedly under
different date cutoffs, is past the point where file scanning is reasonable.
DuckDB is columnar, runs in-process, reads Parquet directly and handles
multi-GB joins without a server. The serving path has the opposite shape — a few
indexed point lookups per request — which SQLite does with no operational cost.

---

### D-008 — Industry-standard vocabulary and methods
**Date:** 2026-09-27 · **Status:** active

All docs, code identifiers, reports and plots use the terms recommender teams
use in industry, defined in `docs/GLOSSARY.md`. Where the original design used
a home-made term, it is replaced:

| was | now | why |
| --- | --- | --- |
| candidate generation strategies | retrieval channels | Standard two-stage vocabulary (retrieval → ranking) |
| candidate provenance | retrieval-source features | Common feature-group name |
| repurchase / same product, other colour | repeat purchase ("buy it again") / variant | Product-facing names |
| popularity-matched negatives | popularity-based negative sampling (∝ pop^0.75) + logQ correction on in-batch negatives | Published, widely deployed methods (word2vec; Yi et al., RecSys 2019) instead of an ad-hoc scheme |
| lift over popularity (ratio) | relative lift vs. popularity baseline (%) | How improvements are reported in reviews and A/B readouts |
| de-popularised Recall@K | tail-item recall | Standard popularity-bias diagnostic |
| PMI (alone) | association lift / PMI, scored as NPMI | "Lift" is the analytics/PM term; NPMI is the standard rare-pair-robust variant |

**Why.** The project doubles as training for industry work. A term that only
exists in this repository cannot be discussed with a colleague or an
interviewer, and an ad-hoc method cannot be compared with published results.

**Rule going forward.** Prefer the method an industry team would reach for
first; deviate only with a measured reason, recorded as a new ADR. A term with
no industry equivalent is marked *(project-specific)*.

---

### D-009 — LLM as interface layer, not ranker; MVP ships without it
**Date:** 2026-09-27 · **Status:** active

The MVP (through M6) contains no LLM. An LLM-based conversational assistant is
added afterwards (M7b) as an orchestration layer that calls the trained models
as tools. It never ranks or invents products.

**Why.**
- Accuracy on this task comes from behavioural data. The top competition
  solutions used GBDT rankers, and LLM rankers have not been shown to beat a
  tuned GBDT on large-catalog purchase prediction.
- Scoring 1.37M customers × 105k articles with an LLM is infeasible in cost and
  latency.
- The industry pattern (e.g. Amazon Rufus, Zalando's assistant) puts the LLM in
  front of an existing recommender for query understanding, explanation and
  conversation.
- The one place an LLM may raise accuracy is **content enrichment**
  (LLM-generated attributes such as style or occasion) for item cold start. It
  is treated as a hypothesis and tested by ablation, not assumed.

**Revisit if:** the enrichment ablation shows a large relative lift, which would
justify moving LLM features into the core pipeline.

---

### D-010 — Three entry points on one backend; photo input via visual search
**Date:** 2026-09-27 · **Status:** active

Passive modules (M6), visual search from a photo (M7a) and a conversational
assistant (M7b) all call the same Track A / Track B services. A photo is first
mapped to catalog articles, with the user confirming the match, before Track B
runs.

**Why.** Track B is trained on catalog articles, so a photo is only useful once
it is grounded to one. User garment selection plus a confirmation step is more
reliable than automatic detection at MVP scale, and keeps every
recommendation traceable to a real article.

**Amendment (2026-09-27).** Visual search and the assistant are combined:
visual search is both a standalone, LLM-free feature and a tool the assistant
calls, with shared session state and hand-off in both directions (DESIGN §7.5).
Visual search is built first (M7a) because the assistant depends on it.

---

### D-011 — Flagship journey: "snap your outfit, fill the gap"
**Date:** 2026-09-27 · **Status:** active

The post-MVP product is built around one journey. The user photographs an
outfit. A vision-language model (VLM) lists the pieces, visual search maps each
piece to catalogue articles, and gap detection finds the missing slots. Track B
fills each gap, re-ranked by the customer's Track A profile. The explanation
cites association lift from basket data.

**How vision and LLM combine:** hybrid (pattern C). Embedding retrieval handles
visual similarity; VLM-extracted attributes (colour, occasion) become filters.
Offline VLM enrichment of the catalogue is tested by ablation.

**Evaluation:** leave-one-out on held-out multi-slot baskets (fill-in-the-blank,
FITB) for gap filling; match accuracy on a hand-labelled photo set for visual
search.

---

### D-012 — The conversational assistant (M7b) is required
**Date:** 2026-09-27 · **Status:** active

M7b is in scope as a must-do milestone after M7a, as decided by the project
owner. It remains an interface and orchestration layer (D-009) and is
evaluated on grounding (hallucination rate), tool-call accuracy, LLM-as-judge
quality, latency and cost. The Polyvore co-wear vs co-purchase experiment is
optional (M8).


---

### D-013 — Track B is evaluated against the live catalogue
**Date:** 2026-09-27 · **Status:** active

For each target week, the candidate set for a slot is every article sold in the
4 weeks before the week or during it, i.e. the assortment a retailer knows is
on sale. Popularity and all model inputs are still computed strictly before
the week.

**Why.** Scoring against all 105k articles would include years of discontinued
stock that no real surface would show. Knowing the current assortment is not a
leak about *pairs*: every model, including the popularity baseline, gets the
same universe.

**Revisit if:** the project gains real stock data.

---

### D-014 — Track A evaluation and training population
**Date:** 2026-09-27 · **Status:** active

MAP@12 is averaged over customers who purchase in the label week (the metric
is undefined for the rest). Retrieval and features are built for those
customers during evaluation. The ranker is trained on (week, customer) groups
that contain at least one retrieved positive, because a group with no
positive gives LambdaRank no gradient. Submission inference ranks every
customer.

**Why.** Standard practice for this benchmark. It keeps a training week at
about 2.8M rows instead of 8.3M, which fits in 16 GB of RAM with four
training weeks.

---

### D-015 — Two-tower towers see a popularity bucket; hybrid is the Track B model
**Date:** 2026-09-27 · **Status:** active

The item encoder includes the article's recent-sales decile within its slot.
The shipped Track B model is weighted reciprocal rank fusion (RRF) of
association rules (support ≥ 3, NPMI) and the two-tower model, weight 0.5.

**Why (validation week, 97,651 queries).** Two-tower Recall@12 on a sample
rose from 0.060 to 0.114 with the popularity bucket. On the full validation
set the relative lift vs popularity was +51% for the two-tower, +47% for
association, and +61% for the RRF hybrid (`reports/m4_track_b_val.json`).

This departs from the design's content-only towers. Popularity is a
legitimate item attribute known at serving time. logQ correction still stops
the loss from rewarding popularity for its own sake.

**Revisit if:** the cross-feature ranker (DESIGN §5.3) is built; it would
replace RRF.

---

### D-016 — Findings that change the design's expectations
**Date:** 2026-09-27 · **Status:** active

Recorded as findings (D-005), not worked around:

- **Support, not lift, is the binding filter.** 1.36M cross-slot pairs; 39.8k
  with support ≥ 3; 1,562 with support ≥ 10. Among pairs with support ≥ 10,
  99.9% have lift > 1 and 98% have lift ≥ 2. The co-purchase signal is real,
  but sparse.
- **Raw co-count ranks slightly better than NPMI** (Recall@12 +50% vs +47%
  relative lift). NPMI recovers more tail items. NPMI is kept, because the
  hybrid adds its tail coverage and the difference is within noise.
- **logQ correction is what makes the two-tower work.** In-batch negatives
  alone score −12% vs popularity; adding logQ gives +36% (1-epoch ablation,
  final run). Popularity-based and hard negatives add nothing measurable on
  top. An apparent jewellery gain from hard negatives in a first run (1.0% →
  1.5%) did not replicate: MPS training is not bit-reproducible, and
  single-run differences of a few points of relative lift are noise.
- **Jewellery and item cold start remain weak** (jewellery Recall@12 1.3% for
  the hybrid on test; cold ≈ 0%). Metadata-only towers cannot tell one new earring from another.
  CLIP image features (M7a) are the planned remedy.

---

### D-017 — Rolling backtest; no new-customer fallback
**Date:** 2026-09-28 · **Status:** active

Track A is evaluated on 4 consecutive label weeks before the test week
(2020-08-19 … 2020-09-09), retraining the ranker for each, and reported as
mean ± std (`reports/backtest_track_a.json`).

**Results.** Baseline MAP@12 0.0232 ± 0.0020; ranker 0.0332 ± 0.0032; relative
lift +43% ± 5% (every week between +37% and +48%). For new customers, the
ranker scored 0.0083 ± 0.0024 and age-band popularity 0.0085 ± 0.0005. The
fallback was better in 1 of 4 weeks.

**Decision.** No fallback policy. The single-week result that suggested one
(test week, MVP) was noise. The fallback's lower variance is recorded for an
online A/B test.

---

### D-018 — FashionCLIP; local LLMs for development, Claude for comparison
**Date:** 2026-09-28 · **Status:** active

- **Image model:** FashionCLIP (`patrickjohncyh/fashion-clip`), run locally.
  Product images are padded to a white square, not centre-cropped. Image and
  text share one space, so one index serves image search and text search.
- **LLMs behind one interface** (`ensemble.llm.client`):
  - Qwen3-VL 8B **instruct** for the assistant (vision + tools). Ollama's
    Qwen2.5-VL has no tool support, and the default Qwen3-VL tag (thinking)
    ignores `think=false` and runs at about 9 tokens/s.
  - Qwen2.5-VL 7B for garment detection. Its pixel boxes are only correct if the
    image is first resized to Qwen's internal grid (≈0.9 MP, sides multiples of
    28); see `outfit._resize_bytes`.
  - Claude Opus 5 for the paid comparison; Claude Haiku 4.5 + Batch API for
    relevance judging.
- **Budget guard:** $8 on cumulative paid spend (`reports/llm_spend.json`),
  below the owner's $10 prepaid cap.

---

### D-019 — Visual search evaluation protocol
**Date:** 2026-09-28 · **Status:** active

79 photos:
- 36 openly licensed Wikimedia Commons photos, publishable with attribution;
- 43 owner photos, local testing only.

The VLM detects garments; each garment is matched in every query mode; the top
5 are judged relevant or not (same type, similar colour and style) by Claude
Haiku 4.5. The metric is Precision@5.

**Leak correction.** The first judging run showed the judge the VLM's text
description. The "text" mode searches with that description, so the judge
favoured it. The judge now sees only pixels and the coarse category. The first
run is kept in `reports/m7a/*_v1_description_leak.json`.

**Spot check.** 120 judgements were relabelled by Claude (not a human): 71%
agreement with the judge. Human gold labels are still pending.

---

### D-020 — DeepFashion2 consumer-to-shop adapters (full training set)
**Date:** 2026-09-28 · **Status:** active

Street↔shop domain adaptation, the step that makes industry visual search
work (Alibaba Pailitao, Pinterest, Meta GrokNet), trained on DeepFashion2's
paired data:
- **Model:** two small residual adapters (user side, shop side) on frozen
  FashionCLIP vectors, symmetric InfoNCE loss, in-category batches.
- **Data:** the full train split (169,584 items, 18,159 identities). The dataset
  is research-only and stays in `data/raw` (gitignored). Pipeline by Codex,
  optimised to decode each image once for both views.

**Results (DeepFashion2 validation, 12,117 user queries, exact identity match,
category-filtered gallery):**

| | Recall@1 | Recall@10 |
| --- | ---: | ---: |
| FashionCLIP, crop | 0.378 | 0.670 |
| + adapter, crop | **0.585** | **0.827** |
| + adapter, background removed (mask) | 0.545 | 0.793 |
| both adapters, summed | 0.607 | 0.835 |

**Data-scaling curve** (background-removed view, Recall@1): 10% of identities
0.482 · 25% 0.494 · 50% 0.526 · 100% 0.545. Still rising, about +2 points per
doubling.

**Finding.** Removing the background hurts, both on DeepFashion2 (mask) and on
our photos (rembg). The adapters use the plain crop.

---

### D-021 — Image search defaults
**Date:** 2026-09-28 · **Status:** active

On the 79-photo set (Precision@5 / Precision@1):

| mode | P@5 | P@1 |
| --- | ---: | ---: |
| crop | 0.357 | 0.356 |
| **crop + DeepFashion2 adapter** | **0.477** | **0.540** |
| rembg crop | 0.269 | 0.302 |
| rembg crop + adapter | 0.412 | 0.428 |
| text | 0.642 | 0.714 |
| text retrieve → image rerank | 0.635 | 0.711 |

**Decision.**
- **Standalone image search (no LLM)** uses crop + adapter (+34% P@5 over raw
  CLIP).
- **The snap flow** uses text retrieve → image rerank. It ties with text alone
  (within noise) and is slightly better on jewellery (0.48 vs 0.46) and product
  photos. It is chosen because the owner requires image similarity to take part.

**Revisit** with human gold labels and the owner's street↔product pair set
(exact-match benchmark).

---

### D-022 — Track B uses FashionCLIP in both towers
**Date:** 2026-09-28 · **Status:** active

Validation week (97,651 queries), relative lift in Recall@12 vs popularity:

| model | without CLIP | with CLIP |
| --- | ---: | ---: |
| two-tower | +46.6% | +50.8% |
| two-tower, content only (no ID) | +17.3% | +38.8% |
| hybrid RRF w=0.5 | +58.6% | **+62.0%** |

Jewellery Recall@12 for the hybrid: 1.64% → 1.95%.

**Other results.**
- **Style-level backoff:** raises tail recall (5.2% vs 4.9%) and catalog
  coverage (26% vs 22%) at equal recall (+61.4%), so it is not selected; it is
  kept as an option.
- **Negative-sampling ablation** (3 seeds, 1 epoch):
  - in-batch only −12.2% ± 1%;
  - + logQ correction +32.7%;
  - + popularity-based negatives +31.8%;
  - + hard negatives +31.9%.

  This confirms D-016 with seed variance: logQ correction matters; the extra
  negatives do not.

**Test week** (90,571 queries, evaluated once, both configurations in the same run):
- hybrid with CLIP +55.6% vs without CLIP +51.6% relative lift;
- two-tower with CLIP +48.6% vs without +42.6%.

Selected configuration: `reports/m4_selected.json`.

---

### D-023 — Track A keeps FashionCLIP off
**Date:** 2026-09-28 · **Status:** active

With the visual retrieval channel and the CLIP ranker features
(`configs/exp_clip_track_a.yaml`), validation MAP@12 was 0.03573 vs 0.03546
without, +0.8%. That is within run-to-run variance (±0.6%).

- **Retrieval:** the visual channel added 0.27% unique recall (merged
  13.06% → 13.33%) for 14.6 extra candidates per customer, the lowest-precision
  channel.
- **Ranker:** `ca_clip_sim_max` ranked 15th by gain (2%).

**Decision.** Flags stay off; the gain is not proven.

**Finding.** Next purchases at H&M are rarely look-alikes of past purchases.
Visual similarity helps completion (Track B, D-022), not next-purchase ranking.

**Revisit if:** a 4-week backtest shows a consistent gain, or cold-item
retrieval is attempted with a launch calendar.

---

### D-024 — Assistant guardrails; local model as default, Claude as the paid option
**Date:** 2026-09-28 · **Status:** active

**Guardrail.** An end-to-end test found the local model inventing an anchor
article id ("12345") when the customer described an item without one; it
answered "no data".
- `complete_the_look` now rejects ids that no tool returned and the customer
  did not type, and tells the model to call `search_catalog` first.
- The system prompt says so too.
- A new eval case (`ctl_described`) covers it; both models now search first,
  then complete the look.

**Comparison** (24 turns, same serving store):

| | Qwen3-VL 8B instruct (local) | Claude Opus 5 |
| --- | ---: | ---: |
| tool / argument accuracy | 100% / 100% | 100% / 100% |
| constraint pass | 91.7% | 91.7% |
| hallucination rate | 0% | 0% |
| turns with products | 83% | 92% |
| latency median / p90 | 17 s / 100 s | 7.5 s / 18 s |
| judge helpfulness / faithfulness (1–5) | 3.54 / 3.54 | 4.46 / 4.25 |
| cost | $0 | ≈ $0.02 per turn |

The judge is Claude, so self-preference bias is possible. The objective
metrics are tied.

**Garment detection** (30 photos): Claude found 27% more garments (195 vs 154),
with valid boxes 85% vs 80%, at 6 s vs 42 s per photo. Slot-level agreement
(Jaccard) was 0.83.

**Decision.** The local model stays the default: free, private, and tied on
the objective metrics. Claude is one environment variable away
(`ENSEMBLE_LLM=claude`) for better answer quality and latency at about $0.02
per turn. That trade-off is the owner's product call.

---

### D-025 — Human gold labels and judge calibration
**Date:** 2026-09-28 · **Status:** active (confirmed on held-out round 2)

The owner labelled 100 judgements (10 garments × 2 modes × 5 results) on the blind Label page,
following a written guideline: same type, similar colour, similar style; all three required; if
unsure, 0.
- **Self-consistency:** 94% (15 of 16 repeated pairs). The one conflict was resolved by the owner.

**Result.** The Haiku batch judge used for all visual-search numbers agrees with the human on only
**58% (κ 0.16)** of judgements. The earlier 71% came from a Claude reviewer, not a human, and
overstated reliability. Human and judge agree on the ranking of modes:

| mode | human Precision@5 | Haiku judge Precision@5 |
| --- | ---: | ---: |
| text retrieve → image rerank | 0.54 [0.34, 0.74] | 0.64 |
| crop + adapter | 0.34 [0.18, 0.50] | 0.48 |

**Judge calibration** (the same 100 human labels; garment-level bootstrap 95% intervals):

| judge | agreement | κ |
| --- | ---: | ---: |
| Haiku, short prompt (rerun) | 0.67 [0.53, 0.80] | 0.34 |
| Haiku + guideline | 0.61 [0.52, 0.70] | 0.21 |
| Sonnet 5 + guideline | 0.69 [0.54, 0.85] | 0.34 |
| **Opus 5 + guideline** | **0.75 [0.64, 0.85]** | **0.48** |

Rerunning the same Haiku judge moved agreement from 0.58 to 0.67, so single judge runs are noisy.

**Decisions.**
- **Conclusions about modes** rest on human labels where available, and on relative (not absolute)
  judge numbers otherwise.
- **Default judge:** Opus 5 + guideline, subject to confirmation on round 2 (a stratified, disjoint,
  held-out set: 3 jewellery, 2 bag and 1 each of shoes, outerwear, bottom, top, sunglasses). Round 1
  is the calibration set, used to choose; round 2 is only for reporting.
- **No full re-judge:** a full Opus re-judge (≈$15 with Batch) exceeds the budget.

**Round 2 (held-out)** — 100 judgements, stratified by category (3 jewellery, 2 bag and 1 each of
shoes, outerwear, bottom, top, sunglasses), disjoint from round 1. Owner self-consistency: 87.5%
(14 of 16).

| judge | agreement | κ |
| --- | ---: | ---: |
| Haiku batch (stored) | 0.64 [0.55, 0.75] | 0.30 |
| Haiku, short prompt (rerun) | 0.72 [0.62, 0.82] | 0.43 |
| **Opus 5 + guideline** | **0.81 [0.72, 0.90]** | **0.57** |

Human Precision@5: text retrieve → image rerank 0.44; crop + adapter 0.28. The ranking matches
round 1, with lower absolute scores on this jewellery- and bag-heavy set.

**Confirmed:** Opus 5 + the written guideline is the default judge for future visual-search
evaluations. It costs about $0.007 per judged list.


---

### D-026 — Exact-match benchmark from the owner's street↔product composites
**Date:** 2026-09-28 · **Status:** active

**Data.** 113 owner images (local only). A curated `MANIFEST.csv` types each one:
- 53 left|right outfit-breakdown composites;
- 6 top|bottom earring posts;
- 5 collages (skipped);
- 49 jewellery close-ups without a product.

**Extraction.** Composites are split by a non-white pixel-profile cut, with the axis taken from the
manifest.
- Qwen2.5-VL boxes garments in the photo and products in the panel.
- A pair is kept only when its category occurs exactly once on both sides.
- Earring posts use the panel's non-white box plus a targeted earring prompt; 1 of 6 was dropped
  after a visual check.
- Result: **154 pairs** (bottom 35, shoes 34, bag 22, top 22, outerwear 16, jewellery 6, …).
- A visual check of the first 10 pairs was all correct.

**Benchmark.** For each street crop, the rank of its true product among same-slot candidates;
Recall@K with bootstrap 95% intervals. No judge: the composite is the ground truth.

| mode | R@1, full gallery (H&M live + extracted; median 7,276) | R@1, extracted-only gallery (median 35) |
| --- | ---: | ---: |
| crop | 0.34 | 0.40 |
| crop, background removed | 0.25 | 0.29 |
| **crop + DeepFashion2 adapter** | **0.42 [0.34, 0.49]** | **0.59 [0.51, 0.66]** |
| text | 0.08 | 0.49 |
| **text retrieve → image rerank** | 0.36 | **0.62 [0.54, 0.69]** |

The extracted-only gallery removes a style shortcut: the owner's product shots share a white
background and a watermark, unlike H&M images, which can make the full-gallery numbers optimistic.

**Findings.**
1. The DeepFashion2 adapter improves exact-item retrieval in both settings: +23% R@1 (full) and
   +47% (extracted-only), with non-overlapping intervals in the fair setting.
2. Exact-item search and substitute search are different tasks. Text alone finds substitutes
   (D-021) but not the exact item in a large gallery (0.08). Image + adapter finds the exact item.
   This supports the two product surfaces: standalone image search (crop + adapter, "find this")
   and the snap flow (text → image rerank, "find something like this").
3. Background removal hurts in every evaluation (DeepFashion2, judged relevance, exact match).
4. Jewellery exact match (n = 6): R@10 0.17 raw vs 0.50 with the adapter. Directional only.

---

### D-027 — Track A retrieval: channel contract and a frontier-chosen candidate budget
**Date:** 2026-10-02 · **Status:** active

**Context.** Merged candidate recall was 13.06% at 115.9 candidates/customer on the
validation week (13.2% ± 0.3% over 4 weeks), and 92.7% of purchased articles had sold
in the 7 days before the cutoff. The ceiling was personalisation over ~19k live
articles, not more global popularity.

**Decision.**
1. Every channel obeys one contract, `(customer_idx, article_id, score)`, with
   deterministic ties (score rounded to 9 decimals, then `article_id`) and per-channel caps;
   per-channel score and rank stay as provenance and ranker features.
2. New channels: department- and section-conditioned popularity (affinity × sales, age-banded),
   directional time-weighted co-visitation (seeds from 52 weeks), colourway variants seeded
   from 104 weeks, personalised new arrivals (launch proxy × department affinity), fine
   (10-band) age popularity.
3. Caps come from a greedy recall-per-candidate allocator on the three weeks before
   validation (25% customer sample; `ensemble.candidates.budget`), then are confirmed on the
   full validation week, which the allocator never saw.
4. Operating point: **~160 candidates/customer**:
   `repeat 10, pop 5, pop_age 65, cf 5, variant 20, dept_pop 40, section_pop 65, covis 25, new_arrival_pers 10`.

**Evidence (validation week unless stated).**

| candidates/customer | merged recall | ranker MAP@12 (same ranker) | training time |
| ---: | ---: | ---: | ---: |
| 115.9 (old channel mix) | 13.06% | 0.03582 | 435 s |
| 116.3 (frontier) | 15.64% | 0.03736 (+4.3% [+3.1, +5.5]) | 531 s |
| 159.7 (frontier) | 18.58% | 0.03743 | 730 s |
| 200.7 (frontier) | 21.03% | 0.03724 | 1,594 s (swapping) |

4-week backtest: recall 15.56% ± 0.63% at 116 and **18.41% ± 0.94% at 160**; MAP@12 at
160 (with 50% negative downsampling, D-028) beat 116 by +0.55% [+0.14, +0.98] pooled, and
in every week.

**Rejected.**
- *Implicit ALS channel:* +0.06 pt recall at 160 for +75 s per week (single-threaded fit);
  the allocator gave it ≤ 5 slots.
- *FashionCLIP visual channel:* the allocator gave it no budget at any target (D-023 stands).
- *Global new arrivals, more repeat depth (all history), cosine CF beyond 5:* below the
  frontier's marginal recall per candidate.
- *Budgets ≥ 200:* recall keeps rising (21–26%) but MAP@12 does not, and the training
  matrix exceeds 16 GB RAM.

**Known gap.** Item cold start (first sale inside the label week, ~5% of purchases) stays
at 0% recall: no channel may read the label week, and the dataset has no launch calendar.

**Revisit if:** the ranker gains features that convert recall above 160 into MAP, memory
grows, or a launch calendar becomes available.

---

### D-028 — Ranker training: temporal early stopping, seeds, vocabularies, downsampled negatives
**Date:** 2026-10-02 · **Status:** active

**Decision.**
- Number of trees from early stopping on the **most recent training week** (MAP@12),
  then a refit on all training weeks with that count. The validation/test week is never
  used to pick trees. The old diagnostic reported a "300-round" MAP from a 200-tree model;
  it now evaluates only trees that exist.
- LightGBM `seed`, `deterministic`, `force_col_wise`; stable row order; ties in the
  predicted score break on `article_id`. Two fits on the same data give identical predictions
  (test).
- Stable, versioned vocabularies (`data/processed/vocab/vocab.json`, version hash in the run
  manifest) replace `hash(value) % N`; NULL → 0, unseen → missing.
- Lifetime features are suffixed `_life`; new groups `recency_affinity` (recency-weighted
  customer × article / style / type / colour / index group / garment group / department
  affinities, days since last purchase in the type and department) and `short_velocity`
  (1- and 3-day sales, 3-day trend, 1-week vs 4-week price change).
- **50% deterministic negative downsampling** in training weeks.
- LambdaRank stays: it is an NDCG-based surrogate; MAP@12 is the reported metric.

**Evidence.**
- Feature groups: +0.56% [−0.29, +1.41] on validation with the old retrieval (n.s.), but
  **+3.39% [+2.93, +3.89] pooled over the 4-week backtest at 160 candidates**, positive every week.
- Negative downsampling at 160: validation 0.03772 vs 0.03743 with all negatives, half the
  training rows (9.5M vs 18.8M).
- Infrastructure alone (seeds, early stopping, vocabularies; no new features): 0.03562 vs
  0.03555 frozen; neutral, as intended.

**Rejected.**
- `rank_xendcg` objective: −2.61% [−3.63, −1.59] vs LambdaRank on validation.
- 6 training weeks (with 50% negatives): 0.03762 vs 0.03772 for 4 weeks; no gain for +50% data.

**Cost.** Training is slower than before: early stopping plus refit fit ~2.3× the trees,
and `deterministic` adds overhead (≈ 10–14 min per week at 160 candidates vs ≈ 4.5 min
before at 116).

---

### D-029 — Re-ranking stage: availability proxy on, diversity caps off by default
**Date:** 2026-10-02 · **Status:** active

**Decision.** After the ranker, drop articles whose last observed sale is more than
28 days before the cutoff (*availability proxy*, no stock feed exists; reads only
pre-cutoff sales). Diversity caps (colourways per style, items per product type) are
implemented and reported but not applied by default.

**Evidence (4-week backtest, chosen model; validation agrees).** The 28-day rule leaves
MAP@12 unchanged in every week (e.g. 0.03289 → 0.03289) while removing likely
unavailable articles; 14 days is also neutral but cuts coverage more. Diversity costs accuracy in every week:
max 2 colourways per style −2.4% to −2.9%, max 4 items per product type −1.5% to −1.8%,
max 1 colourway per style −9% to −11%; they raise distinct product types per list from
5.1–5.5 to 5.5–6.8 (`reports/phase2/rerank_backtest_p2_b160_neg50.json`).

**Revisit if:** a stock feed exists (replace the proxy), or an online test values
diversity above the offline MAP cost.

---

### D-030 — Reason chips need evidence, not only SHAP attribution
**Date:** 2026-10-02 · **Status:** active

**Decision.** A reason is shown only if its features push the score up (positive summed
SHAP) **and** the raw feature values support the sentence (e.g. "You bought this before"
needs `ca_n_article_life ≥ 1`; "in another colour" needs a purchase of a *different*
article of the same style). Every reason carries an evidence type: `personal_history`,
`similarity` or `trending`. New-customer chips are derived from evidence (age-band or
global best seller), not a fixed label.

**Evidence.** Audit on 36,000 validation recommendations (3,000 customers) of the chosen
model: **16.2% of SHAP-only chips were unsupported** by the data: "New arrival" 39%,
"In your usual price range" 31%, "A style you bought, in another colour" 30%, "Trending"
27%, "In a colour you often choose" 13%. With the gate, every chip is supported and 99.74%
of recommendations still carry one (`reports/phase2/explain_audit.json`; the
baseline-retrieval model gave 14.2%).

**Note.** SHAP is model attribution, not causation; the gate makes chips truthful
statements about the data, not claims about why a customer buys.

---

### D-031 — Track B catalogue eligibility is decided before the cutoff; the Round-1 universe is kept as an oracle
**Date:** 2026-10-03 · **Status:** active · **supersedes the protocol in** D-013

**Decision.** The eligible catalogue for label week *w* is "sold at least once in the
`universe_weeks` before *w*'s cutoff" — a *proxy* for availability, because the dataset has
no inventory or launch feed, but one that uses no target-week information. Truth articles
outside it are dropped from the primary protocol and counted (`n_truth_dropped`), not
scored as guaranteed misses. The Round-1 definition (sales between four weeks before *w*
and the **end** of *w*, D-013) stays reachable as `eligible_universe(..., oracle=True)`, for
measurement only.

**Evidence** (`reports/track_b_round3/protocol_leak.json`, two weeks, towers trained once per
week and both catalogues scored). The oracle catalogue adds ~1,600 articles (27,064 → 28,903
and 27,070 → 28,611) and ~4,000 queries, and it **lowers** Recall@12 by 5.07–5.11% for every
fixed-fusion system, because the extra articles contribute 5.87% more truth pairs with no
pre-cutoff footprint — nothing can retrieve them. Under the leakage-free catalogue those
5.87% are excluded and counted instead.

**Why it still matters.** Under the Round-1 universe an article could be a legal
recommendation purely because of sales inside the week being predicted, so eligibility was
not knowable at prediction time and "item cold-start recall" was a property of the
definition, not of a model. The honest protocol cannot measure true item cold start at all
(an article with no pre-cutoff sale is never eligible), so that metric is replaced by
*recently launched*: recall on articles whose first observed sale is within `new_item_days`
of the cutoff (`metrics.segments`).

**Consequence.** Round-1 and Round-3 Track B recall levels are **not comparable** — different
query sets, different truth denominators. Comparisons are only meaningful inside one
protocol. `reports/track_b_round3/baseline_frozen.json` keeps the Round-1 numbers with their
caveats so they are not quietly restated.

**Revisit if:** an inventory or launch feed becomes available (replace the proxy and restore a
true cold-start metric).

---

### D-032 — Track B selects on rolling folds with customer-cluster intervals; the test week is a single confirmation
**Date:** 2026-10-03 · **Status:** active

**Decision.** Every Track B modelling decision is made on six consecutive label weeks ending
at the validation week. For fold *w*, mining, the catalogue, the towers, every feature and
the ranker's training labels come from weeks strictly before *w*, and the ranker trains on the
`train_weeks` label weeks before *w*. The test week is evaluated **once**, after selection,
with the ablation table switched off (`ENSEMBLE_CONFIG=experiments/tb3_final … backtest
test`). Intervals come from a paired bootstrap that resamples **customers** with all of their
queries.

**Evidence.** One basket produces several (anchor, target slot) queries and one customer
several baskets, so queries are not independent; on correlated synthetic data the clustered
interval is more than 3× wider than the naive one
(`test_cluster_bootstrap_is_wider_than_ignoring_clusters`). Across the six folds the same
system moves by a factor of ~1.4 in Recall@12 (`shipped_rrf_hybrid` 0.0864 → 0.1234), which is
why one week cannot support a model choice. The six-fold run also reproduced the two shipping
models exactly from cached matrices (same tree counts, same metrics), so fold-to-fold
differences are data, not training noise.

**Honest caveats.** The test week was already scored in Round 1 and those numbers have been
read, so it is a confirmation, not a fresh holdout. The relative intervals divide the paired
difference interval by the observed baseline mean instead of resampling the denominator, so
the relative bounds are slightly too narrow. The "±" in per-fold tables is the between-fold
standard deviation, not a standard error — adjacent folds share most of their mining window.

**Revisit if:** a later week of data arrives (then there is a genuinely untouched week).

---

### D-033 — Learned LambdaRank fusion replaces fixed RRF for Track B; serving ships the compatibility variant
**Date:** 2026-10-03 · **Status:** active · **supersedes** D-015

**Decision.** Track B ranks a candidate **union** — article association by raw co-count and by
NPMI, product-code (style) association both ways, two-tower retrieval with and without the
article-ID embedding, and slot popularity — with a LightGBM LambdaRank model, one query group
per (basket, anchor, target slot). Trees are chosen by temporal early stopping on the most
recent training week. The fixed RRF(w = 0.5) of D-015 is kept as a reported baseline. The
precomputed serving table ships `lgbm_compatibility` (no customer features); `lgbm_personalized`
is the measured upper bound.

**Evidence** (six rolling folds, `reports/track_b_round3/backtest_val_ablations.json`; pooled
customer-cluster bootstrap over 145,485 customers and 622,989 queries):

| system | Recall@12 | NDCG@12 | vs shipped RRF (R@12) | folds won |
| --- | ---: | ---: | ---: | ---: |
| `shipped_rrf_hybrid` | 0.1066 | 0.0622 | — | — |
| `rrf_all_sources_union` (same union, fixed fusion) | 0.1088 | 0.0642 | +1.98% [+1.36, +2.56] | 6/6 |
| `lgbm_compatibility` | 0.1379 | 0.0822 | **+29.37% [+28.57, +30.15]** | 6/6 |
| `lgbm_personalized` | 0.1573 | 0.0961 | **+47.58% [+46.65, +48.47]** | 6/6 |

**Why it is not a retrieval-budget effect.** `rrf_all_sources_union` fuses the *identical*
189-candidate union with equal-weight RRF and gains 2%. Candidate-set size and union recall
(0.4285) are reported with every result.

**Why the compatibility variant ships.** The table is keyed by (anchor, target slot) and must
answer for anonymous visitors on any product page, so a row per customer is not
precomputable; request-time personalized scoring needs an online feature store and a model
server. The gap is reported (+15.11% Recall@12 on returning customers), not claimed.

**Revisit if:** a candidate-budget change moves the union's ceiling materially, a feature
store exists, or an online test contradicts the offline ordering.

---

### D-034 — Track B personalization is point-in-time, and it is taste rather than repurchase
**Date:** 2026-10-03 · **Status:** active

**Decision.** The ranker gets customer features computed only from purchases before the label
week: history size and recency, exact-article and style affinity, recency-weighted
product-type / department / section / garment-group / colour / slot shares, the customer's
usual price point and distance from it, and explicit interactions between compatibility
evidence and customer preference. Customers with no history get `c_has_history = 0` and null
affinities, so the compatibility path answers unchanged — an explicit no-history path, not a
fallback model.

**Evidence.** Personalization is the largest single feature group: removing all 21 customer
features costs **−12.33% Recall@12** and **−14.48% NDCG@12** on the six folds, more than any
other group. It is concentrated where history exists: **+15.11%** over the compatibility
ranker on 582,992 returning-customer queries, **+0.66%** on 39,997 new-customer queries.

**Not a buy-it-again effect.** Over the six folds only **3.4%** of truth articles had already
been bought by that customer (7.9% for the style —
`reports/track_b_round3/label_audit.json`), and removing the two exact-article repeat features
costs **0.74%**. The gain is category, section, colour and price taste.

**Revisit if:** an online feature store exists (then ship the personalized model and
re-measure), or `c_has_history` coverage changes materially.

---

### D-035 — Track B serving: evidence stored per row, chips gated on it, diversity caps as a re-ordering
**Date:** 2026-10-03 · **Status:** active · **extends D-030 to Track B**

**Decision.** Every row of the precomputed Complete-the-Look table carries the evidence it was
ranked on — co-count, lift, NPMI, style co-count / NPMI / lift, backoff level, source flags,
two-tower and FashionCLIP similarity, colour agreement, price-tier distance, model score — plus
one provenance label (`co_purchase`, `style_co_purchase`, `visual_compatibility`,
`popular_in_slot`, `other`). A reason chip is emitted only when the column it quotes is
non-null for *that* pair. The diversity rules (one colourway per style, at most
`serving.max_per_product_type` per product type) are applied as a **re-ordering** and the
module is then back-filled to `serving.ctl_per_slot`, instead of dropping the items the caps
reject.

**Evidence** (`reports/track_b_round3/backtest_val_serving_rules.json`, six folds, re-ranked
from the shipping model's top-48 pool, relative to the unrestricted ranker):

| rule reading | Δ Recall@12 | Δ NDCG@12 | Δ distinct product types / list | mean list length |
| --- | ---: | ---: | ---: | ---: |
| caps as a re-ordering, back-filled | **−6.81%** | −6.27% | **+49.10%** | 12.00 |
| caps as hard filters (Round-1 behaviour) | **−37.45%** | −23.60% | +110.54% | **8.32** |

The hard filter costs 5.5× more accuracy, and the last column says why: modules come out
part-empty, and some fall below five items, so even Recall@5 drops (0.0861 → 0.0712). Both
readings still beat the shipped RRF hybrid except the hard filter (−18.9%).

**Consequence.** `tests/test_api.py::test_product_page_and_diversity` moved from "no module
breaks the cap" to structural assertions plus variety; the cap arithmetic is unit-tested in
`test_apply_diversity_*`. This is a deliberate change to pre-existing serving behaviour.

**Revisit if:** an online test prefers a strictly diverse but part-empty module, or the
accuracy cost of the caps grows with a stronger ranker.

---

### D-037 — Association evidence stays plural; style backoff is a feature, not a switch
**Date:** 2026-10-03 · **Status:** active · **refines** D-005

**Decision.** Mining keeps raw co-count, time-decayed co-counts (14 d and 56 d half-lives),
marginal supports, lift, PMI and NPMI as **separate** columns at article level and at
product-code level, with a mining floor of `min_support = 2`. The ranker chooses among them.
Style-level backoff is not a global on/off choice: both levels are always joined and
`backoff_level` (0 = article evidence, 1 = style only, 2 = neither) is a feature.

**Evidence.** The funnel shows why no single gate is right: of ~290,000 cross-slot pairs with
support ≥ 2, ~82,000 reach support ≥ 3 and only ~2,900 reach ≥ 10, while lift > 1 removes
almost nothing once support ≥ 2 (293,130 of 298,182 on the first fold) — D-005 stands. The
ablation shows the evidence is highly redundant rather than individually critical: removing
all 17 article-level association features costs **0.42%** Recall@12, removing the 14
style-backoff features costs **0.77%**, removing the six time-decayed columns costs **0.62%**.
Association still carries 12.8% of the model's split gain and `a_co_share` is its fifth most
important feature, so it is doing work — the style level and the towers simply reconstruct
most of it.

**Consequence.** Effort is better spent on the towers and the customer than on more
association statistics.

**Revisit if:** other half-lives or a basket-noise filter are tested on rolling folds. Both
are part of the artifact cache key, so each variant costs a full rebuild (~2 h).

---

### D-036 — Two-tower negatives: in-batch + logQ, and nothing else earns its place
**Date:** 2026-10-03 · **Status:** active · **refines** D-016 and D-022

**Decision.** The Track B towers train with in-batch negatives and the logQ correction only
(`two_tower.negatives: [in_batch, logq]`, `hard_negatives: 0`), six epochs with held-out
early stopping (patience 2) on baskets from the tail of the mining window.

**Evidence** (`reports/track_b_round3/ablation_towers.json`; two label weeks × two seeds per
configuration, two-tower list scored alone on the target week over the leakage-free
catalogue):

| negatives | fold Recall@12 | SD | content-only | vs in-batch + logQ | mean s |
| --- | ---: | ---: | ---: | ---: | ---: |
| in-batch only | 0.0706 | 0.0015 | 0.0670 | **−26.48%** | 147 |
| in-batch + logQ | 0.0961 | 0.0015 | 0.0919 | — | 166 |
| + 4 retrieval-informed hard negatives | 0.0970 | 0.0010 | 0.0928 | +0.94% | 267 |

**logQ is essential** — removing it costs 26.5%. **The redesigned hard negatives are not.**
Round 1 tested popularity-sampled and (product type, price tier) hard negatives and found
nothing; Round 3 replaced them with retrieval-informed mining (re-score a popularity-sampled
pool with the model being trained, mask the positive, its colourways and basket co-occurring
articles as likely false negatives) and gets +0.94% against a per-run spread of 1.0–1.5% —
paired by (week, seed) that is +0.0009 with a paired SD of ~0.0018 and one of four pairs
negative — for +61% training time. Off by default. The reading is that the constraint is not
the negative-sampling design but that in-batch + logQ already saturates this architecture on
3.1 M pairs.

**The epoch budget is adequate.** In all eight weeks the 6-epoch run picks epoch 6 and the
held-out curve still looks like it is rising, which suggests a binding budget. It is not:
allowing 12 epochs with patience 2 on the validation week ran 8 and selected epoch 6, with a
fold Recall@12 (0.0985) inside the seed spread of the 6-epoch runs. One week, one seed.

**Determinism.** CPU is bit-identical across repeated runs with the same seed; MPS is not
(0.08017826 vs 0.08020096 at one epoch). That spread is two orders of magnitude smaller than
the seed spread, so every tower conclusion here is a mean over two weeks × two seeds and no
single-run difference is reported as a result.

**Revisit if:** the architecture changes (then re-test hard negatives), or a decisive tower
comparison is needed — run it on CPU, which is reproducible.

---

### D-038 — Track A research protocol: six rolling folds, a sales-proxy eligible catalogue, one fallback, customer-cluster intervals
**Date:** 2026-10-03 · **Status:** active · **frozen before any comparison was run**

**Decision.** Track A's research comparison (`configs/track_a_research.yaml`, `research.protocol`,
`src/ensemble/research/protocol.py`) uses:

- **Six reporting folds**, the consecutive label weeks 2020-08-05 … 2020-09-09 (ending at
  validation). Every per-week model, retrieval statistic, vocabulary, graph, sequence and
  feature reads only `t_dat < week.start`; the ranker trains on the four label weeks before
  each fold. Six are feasible: the earliest ranker training week (2020-07-08) still has 22
  months of history behind it.
- **Tuning folds** 2020-07-22 and 2020-07-29, and **allocator weeks** 2020-07-15/22/29, all
  ending before the first reporting fold. Neural hyperparameters and any new candidate budget
  are chosen there, never on a reporting fold.
- **The test week 2020-09-16 is confirmation evidence only.** It and the Kaggle scores have
  been observed (Phase 2); they are not used for selection, debugging, thresholds or stopping,
  and no Kaggle submission is made.
- **Evaluation customers:** every label-week buyer (D-014), no sampling.
- **Eligible catalogue** `E(w)`: articles sold at least once in the 28 days before `w.start`
  (the D-029 availability proxy; 94.3% of validation-week purchase pairs fall inside it).
  Every system's list is restricted to `E(w)` and back-filled with the age-band best sellers
  of the 7 days before the cutoff, which are always in `E(w)`. This is also every system's
  new-user fallback. Ties break on `article_id`.
- **Metrics:** MAP@12 (selection), Recall@12, NDCG@12, candidate recall and conversion,
  catalogue coverage over `E(w)`, novelty (mean −log₂ of 28-day sales share).
- **Segments, declared before scoring:** returning vs new customers; activity bands from
  52-week transaction counts (low ≤ 12, medium 13–32, high ≥ 33 — the tertiles among buyers
  of tuning week 2020-07-22); head = top 10% of `E(w)` by 7-day sales; recently launched =
  first sale within 28 days before the cutoff; repeat vs non-repeat truth (bought before the
  cutoff or not); cold and ineligible truth reported as ceilings.
- **Uncertainty:** paired bootstrap, 1,000 resamples, seed 0, resampling **customers across
  folds** (a customer who buys in three folds is one cluster carrying three rows); the relative
  difference is computed inside every resample, fixing the fixed-denominator approximation
  noted in D-032.
- **Seeds:** neural baselines 0, 1, 2; ranker 42, 43, 44.
- **Adoption rule for any change to the ensemble:** pooled ΔMAP@12 95% interval entirely above
  zero, a gain in at least 4 of 6 folds, and at most +50% ranker training time and +25% peak
  memory. The target (+5% over the strongest reproduced baseline with a CI above zero) is
  reported as met or not met; the protocol is not changed to meet it.

**Why.** One week cannot carry a model choice (D-032). A catalogue restriction that is
knowable at the cutoff keeps every system from winning or losing on stale stock, and a single
shared fallback means a model is never credited with a popularity list it did not produce.

**Revisit if:** a later week of data arrives (a genuinely untouched holdout), or an inventory
feed replaces the sales proxy.

---

### D-041 — Track B request-time personalization re-ranks a bounded compatibility pool; tolerance fixed before measuring
**Date:** 2026-10-03 · **Status:** active (thresholds fixed before the regression run)

**Decision.** The serving bundle stores, per (anchor, target slot), the compatibility ranker's
top-P candidates with their compatibility features. At request time the personalized ranker
(D-033, D-034) re-orders that pool with point-in-time customer features read from a versioned
profile snapshot; the diversity rules (D-035) then re-order and back-fill. P is the smallest of
{24, 48, 100} meeting both predeclared tolerances on the six rolling validation folds
(`configs/experiments/tb_serving_pools.yaml`):

1. `lgbm_personalized@poolP` keeps at least **99%** of unrestricted `lgbm_personalized`
   pooled Recall@12, and
2. `lgbm_personalized@poolP+shipped` (what serving shows) is **not worse** than the currently
   served `lgbm_compatibility+shipped`: pooled customer-cluster bootstrap 95% interval of the
   Recall@12 difference entirely above zero.

If no P meets (1), the largest P is used and the shortfall is reported as a product trade-off.

**Outcome (2026-10-03, `reports/track_b_production/serving_regression.json`; six folds, 622,989
queries, customer-cluster bootstrap).** Retention of unrestricted personalized Recall@12: P = 24
93.1%, P = 48 96.3%, **P = 100 98.9%** — no P meets the 99% bar, so by the rule **P = 100** ships
and the 1.1% shortfall is reported. Tolerance (2) holds: served `lgbm_personalized@pool100+shipped`
vs the current `lgbm_compatibility+shipped` = **+7.06% Recall@12**, interval of the difference
[+0.0083, +0.0097] (6/6 folds). Observed but not acted on (it was not the predeclared criterion):
with the diversity rules applied a smaller pool serves *better* (P = 24: +9.89%), because the caps
then draw replacements from a shorter, stronger list. Revisit with an online test or a rule that
selects P on the served (+shipped) metric.

---

### D-039 — Reproduced baselines: BPR-MF, LightGCN and SASRec (gSASRec loss), tuned on tuning folds only
**Date:** 2026-10-03 · **Status:** active

**Decision.** Track A is benchmarked against three faithful, minimal PyTorch reproductions
(`src/ensemble/research/models.py`), each trained per label week on data before its cutoff and
scored exactly over the eligible catalogue (D-038): BPR-MF (Rendle et al. 2009; uniform negatives
rejecting the customer's own training positives, SparseAdam, CPU), LightGCN (He et al. 2020;
symmetric-normalised bipartite graph, mean of layers 0..L, BPR with L2 on the ego embeddings,
N(0, 0.1) init, MPS) and SASRec (Kang & McAuley 2018; causal self-attention over the last 50
purchases, tied embeddings). SASRec is trained with **gSASRec's gBCE** (Petrov & Macdonald,
RecSys 2023; 256 uniform negatives per sequence, t = 0.75): full-softmax cross-entropy
(Klenitskiy & Vasilev 2023) costs ~9 min per epoch on a 41k-item vocabulary here, and gBCE is
the published SASRec objective designed to recover most of that quality with sampled negatives.
The original one-negative BCE was kept in the grid.

**Selection** (`reports/track_a_research/tuning.json`; bounded grid declared before running:
Stage A, two refinements, Stage B on the second tuning fold, Stage C convergence guard):
BPR-MF 26-week window, dim 128, lr 0.005, reg 1e-5, 24 epochs; LightGCN 26 weeks, 3 layers,
dim 64, lr 0.02, batch 131,072, 27 epochs; SASRec gBCE, lr 0.002, dropout 0.2, 24 epochs (Stage
C doubled the budget because epoch 12 was the curve's last point; at 24 it was still rising
slowly, +3.4% from epoch 12 to 24, and no further extension is in the declared budget).
Not masking the customer's own purchases won for every model (repurchase is a large share of
H&M purchases). gBCE beat BCE by +63% raw MAP@12 on the tuning fold.

**Repeat handling and time.** Sequences keep repeats; same-day purchases are ordered by
`article_id` (no time of day in the data); time gaps are not modelled (standard SASRec).
**Fallback:** customers a model cannot represent (no interaction in its window or vocabulary)
get the shared age-band fallback (D-038), and that share is reported per model.

**Reproducibility.** Toy tests check the expected ranking behaviour and that identical CPU
seed/config runs reproduce exactly (`tests/test_research_track_a.py`); MPS runs are not
bit-reproducible (D-036), so stochastic spread is reported over three seeds.

**Revisit if:** more compute allows full-softmax SASRec or longer LightGCN schedules.

---

### D-042 — Track B serving runtime: versioned bundles, NumPy GBDT, evidence-ordered fallbacks, exact vector search
**Date:** 2026-10-03 · **Status:** active

**Decision.**
- **Offline/online split.** `ensemble.serving.bundle` builds a versioned, immutable bundle (pools
  with compatibility features, profile snapshot, models, catalogue/availability snapshot, visual
  index, sha256 manifest) and publishes it atomically (`CURRENT` pointer). No request retrains or
  reads training tables.
- **Ranker runtime.** The personalized LightGBM model is exported to a NumPy tree evaluator
  (`ensemble.serving.gbdt`), equal to `Booster.predict` within 1e-9 including NaN and categorical
  edge cases, so the API never loads LightGBM next to PyTorch (the OpenMP clash). A build-time
  gate compares offline SQL + LightGBM with the online path on sampled real requests and refuses
  to publish on any ordering difference.
- **Fallback ladder** in evidence order: the anchor's own pool → a live colourway of the same
  style → the nearest live article of the same slot in FashionCLIP space → slot popularity;
  ordering is personalized whenever the customer has a profile. Every row carries
  `provenance`, `evidence_types` and chips gated on its own evidence (D-030, D-035).
- **Availability** (snapshot + runtime overrides) is applied before the top-k cut, and is part of
  every cache key together with the bundle version.
- **Vector search is exact**, no ANN: see the benchmark in the final report (exact top-k over the
  live visual index takes about a millisecond on this laptop, so an ANN dependency would add
  build and recall risk for no measurable gain).

**Revisit if:** the live catalogue grows by an order of magnitude, or latency targets tighten.

---

### D-040 — Track A adds the reproduced baselines' scores as ranker features, not as retrieval channels
**Date:** 2026-10-04 · **Status:** active · **frozen before seeds 43/44, ablations and any confirmation run**

**Decision.** The final Track A system is `phase2_nscores`: the Phase-2 candidate set and features
(D-027, D-028) plus three features, the BPR-MF, LightGCN and SASRec dot products of the customer
and the candidate (`nn_<model>_dot`), from models trained per week on data before the cutoff.
`research.ensemble.final` is set to it.

**Evidence** (six reporting folds, seed 42, pooled customer-cluster bootstrap over customers
across folds, `reports/track_a_research/comparison.json`). Under the D-038 rule, fixed before the
comparison ran:

| vs `phase2` (0.03403) | MAP@12 | relative (95% CI) | folds won | train time | peak RSS | rule |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `phase2_nscores` | 0.03524 | +3.55% [+3.10, +3.97] | 6/6 | −0.5% | −1.4% | **adopt** |
| `neural_channels` (allocator caps + neural channels + features) | 0.03549 | +4.30% [+3.86, +4.78] | 6/6 | +197% | +1.9% | reject (time) |
| `realloc` (allocator caps, no neural signal) | 0.03428 | +0.74% [+0.38, +1.09] | 5/6 | +50.01% | −3.3% | reject (time) |

**Caveat on cost.** Every fit ran next to a GPU training job; two fold-09-09 fits (`realloc`
3,129 s, `neural_channels` 8,946 s against 600–1,000 s elsewhere) were slowed by that contention.
The verdict does not depend on them: by the median over folds `neural_channels` is still +63%
(above +50%) and `phase2_nscores` still ≈ 0%. `realloc` misses the limit by 0.01 percentage
points on the mean only; it is reported as a near-miss, not adopted.

**Reading.** The neural signal helps the ranker (+3.6%, every fold) at no measurable cost; giving
it candidate slots adds a further ~0.7% MAP@12 that the allocator's re-shaping of the other
channels partly explains (`realloc` +0.7% by itself), at a cost the rule rejects.

**Revisit if:** training time is measured on an otherwise idle machine (the `neural_channels`
cost leg may then pass), or a retrieval change raises recall above 160 candidates per customer.

---

### D-043 — Serving performance thresholds, set after the baseline measurement and before optimization
**Date:** 2026-10-05 · **Status:** active

**Baseline** (`reports/track_b_production/bench_serving_baseline.json`; Apple M5 laptop, 16 GB, one
uvicorn worker, real bundle, 2,000-request seeded Zipf workload, half the requests with a known
customer): with the cache warm (82% hits) Complete the Look answers in p50 2.1 ms / p95 2.8 ms at
concurrency 1 and ~440–500 rps; **uncached, every request computing ~6 modules, p50 196 ms, ~8 rps,
and 37 timeouts at concurrency 16** — not acceptable for a service whose cache is cold after every
bundle swap. Startup to ready 0.65 s; peak RSS 1.08 GB; visual search warm p95 28 ms; exact
top-8 over the 26,127-article visual index 1.35 ms.

**Thresholds** (the final benchmark must meet all of them; none may be met by changing what is
recommended — any change to ordering has to pass the offline equivalence gate):

| measure | threshold |
| --- | --- |
| uncached Complete the Look (all slots), concurrency 1 | p95 ≤ 60 ms |
| uncached, concurrency 4 | ≥ 40 rps |
| uncached, concurrency 16 | 0 errors and 0 timeouts (10 s client timeout) |
| warm cache, concurrency 1 | p95 ≤ 10 ms |
| startup to ready | ≤ 5 s |
| steady-state RSS under the workload | ≤ 2 GB |
| visual search, warm, concurrency 1 | p95 ≤ 100 ms |

**Why these.** They are 3–4× tighter than the measured uncached baseline where it is weak and
keep the measured headroom where it is already good; all are local single-worker numbers, not
production SLOs.

**Outcome (2026-10-05, `bench_serving_final.json`; same machine, workload and bundle).** One
engineering change, no change to what is recommended: the NumPy forest now evaluates every
(row, tree) pair per depth step over flat global arrays (categorical splits as a lookup table),
and the catalogue lookups on the request path use NumPy arrays instead of pandas indexing.
Uncached p95 219 → **26 ms** (c1), throughput 8 → **68 rps** (c4), **0** errors at c16 (was 37);
warm p95 2.9 ms; startup ≤ 0.66 s; steady RSS 0.85 GB; visual warm p95 51 ms — every threshold
met. Ranking equivalence re-proven after the change: the build gate re-run on the published bundle
gives 400/400 identical orderings, max |score difference| 2.7e-15 against LightGBM
(`equivalence_after_optimization.json`), and the evaluator equals the previous per-tree one
within 1e-9 in `tests/test_serving_gbdt.py`.
