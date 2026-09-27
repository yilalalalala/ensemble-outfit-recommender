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
