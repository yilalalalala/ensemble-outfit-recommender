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
