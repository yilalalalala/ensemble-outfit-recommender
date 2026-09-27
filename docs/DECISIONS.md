# Decisions

Numbered, dated, with the reasoning and the revisit condition. A decision whose
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

### D-002 — Accessory focus is a question, not a data filter
**Date:** 2026-09-27 · **Status:** active

Track B trains on the full catalogue and only *presents* accessories.

**Why.** The interesting question — which accessory completes this garment —
requires garments in the training data by definition. Filtering to accessories
first removes the anchors.

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
lift over popularity before any other number.

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
