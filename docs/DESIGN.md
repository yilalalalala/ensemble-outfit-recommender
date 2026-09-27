# Ensemble — system design

A fashion recommender built on the H&M Group transaction dataset, in two parts:
a personalised next-purchase recommender measured against a public benchmark,
and an accessory-completion recommender that answers a question the benchmark
does not ask.

---

## 1. Objectives

### 1.1 Track A — Next-purchase recommendation (the benchmark track)

For each customer, predict up to 12 articles they will purchase in the seven
days immediately following the training window.

This is the task defined by the H&M Personalized Fashion Recommendations
competition, and it is adopted **unchanged and on the full catalogue**. That is
deliberate: solving the published task on the published data means the score is
externally checkable against 3,006 teams. Subsetting the catalogue would produce
a number nobody can verify.

**Metric: MAP@12** — Mean Average Precision at 12.

```
                U        1          min(n,12)
MAP@12 =  1/U   Σ   ───────────      Σ        P(k) · rel(k)
               u=1   min(m_u, 12)   k=1
```

where `U` is the number of customers, `m_u` is how many articles customer `u`
actually bought, `P(k)` is precision within the first `k` predictions, and
`rel(k)` is 1 if the prediction at position `k` was purchased.

The property that matters: **hits early are worth more than hits late**, and
being wrong costs nothing beyond the opportunity — so all 12 slots should always
be filled.

**Calibration of expectations.** Published private leaderboard scores:

| | private MAP@12 |
| --- | ---: |
| 1st place | 0.0379 |
| ~45th of 3,006 (silver) | 0.0300 |

Scores are small because most customers buy nothing in a given week, and what
they do buy is often unpredictable. **0.03 is a strong result, not a broken
pipeline.** Any design that seems to produce 0.5 has a leak.

### 1.2 Track B — Accessory completion (the research track)

Given an anchor garment, recommend the jewellery or accessory that completes it.

The competition task is purely sequential personalisation: *what will this person
buy next*. It carries no notion of *what goes with what*. Yet the data contains
basket structure — items bought by one customer on one day — which is real
evidence of intended pairing, paid for with real money.

Track B mines that signal. It is the part of this project that is not a
reproduction of published work.

**Scope note.** Track B is a difference in *question*, not a subset of the data.
It trains on the full catalogue, because "customers who bought this dress also
bought that necklace" is exactly the cross-category signal that disappears if
the data is cut down to accessories first. Only the *presentation* is focused on
accessories.

---

## 2. Data

| table | rows | contents |
| --- | ---: | --- |
| `transactions_train` | 31,788,324 | customer_id, article_id, t_dat, price, sales_channel_id |
| `customers` | 1,371,980 | age, club member status, fashion news frequency, postal code |
| `articles` | 105,542 | product_type_name, product_group_name, colour, department, index_group, garment_group, detail description |
| images | ~105k | one JPEG per article |

Jewellery sits inside the Accessories product group (roughly 2,160 articles
alongside bags and other accessories), so Track B's anchor→accessory pairs are
available without any external data.

### 2.1 Sampling strategy

The full dataset is ~3.5 GB of transactions and ~30 GB of images. Neither the
statistics nor the engineering require all of it.

- **Time window.** The most recent ~16 weeks of transactions. Fashion is
  strongly seasonal, and two-year-old purchases are weak evidence about next
  week. This also cuts the working set by roughly an order of magnitude.
- **Customers.** All customers active in the window. No subsampling of
  customers, because customer-level sampling distorts the popularity
  distribution that every baseline depends on.
- **Images.** Fetched only for articles that survive the window. Embedded once,
  cached, never re-read during training.

Every sampling decision is recorded with its justification in `docs/DATA.md`
and is a parameter, not a constant, so the full-data run is a config change.

### 2.2 Storage

A relational database, not flat files. With ~30M rows across three related
tables and repeated windowed aggregation, file scanning stops being viable:
candidate generation alone needs per-customer and per-article histories pulled
thousands of times per training run.

**DuckDB** for analytics and feature building — columnar, runs in-process, reads
Parquet directly, and handles multi-GB joins on a laptop. **SQLite** for the
serving path, which only ever needs indexed point lookups.

Schema, indices and the ingestion path are specified in `docs/DATA.md`.

---

## 3. Evaluation protocol

**Temporal splits, never random.** A random split lets the model see the future,
and in a dataset with repeat purchases this inflates every metric enormously.

```
  ├──────────── training window ────────────┤ val week ├ test week ┤
                                    (features stop here)
```

- **Test week** — the final 7 days. Touched once, at the end.
- **Validation week** — the 7 days before it. Used for all model selection.
- **Feature cutoff.** Every feature for a given week is computed strictly from
  data before that week. This is the single easiest place to leak, so feature
  construction takes the cutoff date as a required argument rather than a
  default.

**Cold-start reporting.** Metrics are reported overall *and* split by segment:
returning customers, new customers, and articles with no prior purchases. An
aggregate number hides the fact that most recommenders are excellent at repeat
purchases and useless at everything else.

---

## 4. Track A architecture

Two stages, which is the standard shape for a catalogue this size: scoring
105k articles for 1.37M customers is 10^11 pairs, so a cheap stage must
narrow the field before an expensive stage orders it.

### Stage 1 — Candidate generation (recall-oriented)

Several independent strategies, unioned:

| strategy | rationale |
| --- | --- |
| Repurchase | The customer's own recent articles. In fashion retail this single rule is a large share of all correct predictions. |
| Popularity | Top sellers in the recent window, globally and per customer segment. The baseline everything must beat. |
| Item-to-item collaborative filtering | Articles frequently co-purchased with the customer's history. |
| Content similarity | Nearest neighbours in metadata + image embedding space. The only strategy that can reach cold articles. |
| Same product, other colour | An H&M-specific pattern: the same garment in a different colourway. |

**Metric: Recall@K.** The fraction of truly-purchased articles that appear
anywhere in the candidate set. This is the **ceiling on the final score** — an
item the ranker never sees cannot be recommended — so it is measured and tuned
separately before any ranking work begins.

The trade-off is explicit: larger candidate sets raise Recall@K and raise
training cost. The choice of K is reported with its cost, not assumed.

### Stage 2 — Ranking

**LightGBM with LambdaRank**, a gradient-boosted tree ensemble whose objective
optimises ranking position directly rather than per-item error.

Features span four groups:

- **Customer** — age, tenure, purchase frequency, average price point, channel mix
- **Article** — recent sales velocity, price, product group, colour, age of the article
- **Customer×Article** — has this person bought it before, bought this product
  type before, bought this colour before, price relative to their usual spend
- **Candidate provenance** — which strategy proposed this candidate, and its
  rank within that strategy. Frequently one of the strongest features.

---

## 5. Track B architecture — accessory completion

This section is the design that distinguishes the project, so it is specified in
more detail.

### 5.1 Problem statement

Given an anchor garment `a` and a customer context `x`, rank accessory articles
`c` by the probability that `c` belongs in the same outfit as `a`.

### 5.2 Where the labels come from, and why they are suspect

A **basket** is the set of articles one customer purchased on one day. If a
basket contains garment `a` and accessory `c`, that is a positive pair.

**The obvious objection: co-purchase is not co-wear.** Someone buying a week of
clothing in one trip generates dozens of pairs that were never intended to go
together. Taken naively, this signal mostly measures "both items are popular".

Three filters address this, and the third is the one that matters:

1. **Basket size.** Keep baskets of 2–6 items. Larger baskets are stock-ups, not
   outfits. This is a blunt instrument but removes the worst offenders.
2. **Category structure.** Keep pairs where one side is a garment and the other
   an accessory. Two accessories together say little about completion.
3. **Association strength, not raw count.** Filter pairs by **PMI** — Pointwise
   Mutual Information:

   ```
   PMI(a, c) = log [ P(a, c) / ( P(a) · P(c) ) ]
   ```

   `P(a,c)` is how often the two appear in a basket together; `P(a)·P(c)` is how
   often they *would* co-occur if purchases were independent. **PMI is positive
   only when a pair co-occurs more than popularity alone explains.** This is
   precisely the "both items are popular" confound, removed by construction.

   Raw PMI is unstable for rare pairs, so counts below a support threshold are
   discarded and the remainder uses a smoothed variant.

The pair-mining step reports how many pairs survive each filter, because if
almost nothing survives PMI filtering then the co-purchase signal was popularity
all along — a negative result worth knowing early rather than late.

### 5.3 Model — two-tower with a cross-feature ranker

**Why two towers.** Accessory embeddings can be computed once and indexed, so
serving is a nearest-neighbour lookup rather than a scoring pass over thousands
of candidates. It also allows cold-start articles, since the tower reads
content, not an ID.

```
  garment a                                accessory c
      │                                         │
  ┌───▼─────────────────┐              ┌────────▼────────────┐
  │ metadata embeddings │              │ metadata embeddings │
  │ product_type        │              │ product_type        │
  │ colour, department  │              │ colour, department  │
  │ garment_group       │              │ garment_group       │
  │ price tier          │              │ price tier          │
  │ + CLIP image vector │              │ + CLIP image vector │
  └───┬─────────────────┘              └────────┬────────────┘
      │  MLP                                    │  MLP
      └──────────────┐              ┌───────────┘
                     ▼              ▼
                  cosine similarity  →  compatibility score
```

A second, small **cross-feature ranker** re-orders the top candidates using
features that cannot be expressed as a dot product of two independent towers:

- colour-pair harmony, computed in CIELAB (a colour space where numeric
  distance approximates perceived difference)
- price-tier compatibility — a £400 coat and a £3 hair tie rarely form an outfit
- seasonal co-occurrence in the same week of year
- the customer's own history with this accessory type

Keeping this model small and feature-named means each recommendation decomposes
into readable reasons, which the tower dot product alone cannot provide.

### 5.4 Negative sampling — the part that decides whether this works

Positives are mined pairs. Negatives determine what the model actually learns,
and the naive choice teaches it the wrong thing.

| strategy | share | what it forces the model to learn |
| --- | ---: | --- |
| In-batch | 50% | Cheap and plentiful; separates broad categories |
| **Popularity-matched** | 30% | Negatives sampled to match the positive's popularity. **Without this the model can score well by learning "recommend popular accessories" and nothing else.** |
| **Hard** | 20% | Same accessory sub-type and similar price as the positive. Forces a decision about *this necklace vs that necklace*, not *necklace vs shoes*. |

The popularity-matched slice is the defence against the most common failure in
recommender evaluation: a model that appears to work, and is actually a
popularity chart.

### 5.5 Metrics for Track B

Track B cannot borrow MAP@12, because the task is different. Its evaluation is
held-out baskets from the test week.

**Primary**

| metric | definition | why |
| --- | --- | --- |
| **Recall@K** | Of held-out (garment → accessory) pairs, how often the true accessory appears in the top K | The headline: did we find it at all |
| **NDCG@K** | Normalized Discounted Cumulative Gain — rewards the true accessory appearing early | Ordering quality, not just presence |

**The metric that decides whether Track B is real**

| metric | definition |
| --- | --- |
| **Lift over popularity** | Recall@K of the model ÷ Recall@K of "always recommend the most popular accessories" |

A popularity ranker is the null hypothesis, and in recommender systems it is a
much stronger opponent than people expect. **A lift near 1.0 means the model
learned nothing about compatibility**, whatever its absolute Recall@K looks
like. This number is reported before any other Track B result.

A stricter variant, **de-popularised Recall@K**, evaluates only on pairs the
popularity baseline gets wrong, isolating what the model adds.

**Diagnostics**

| metric | what it detects |
| --- | --- |
| Catalogue coverage | Fraction of accessories ever recommended. A model that only ever suggests twenty items is useless in production regardless of its Recall |
| Novelty | Mean inverse popularity of recommendations — quantifies the head-of-catalogue bias |
| Cold-article Recall@K | Recall restricted to accessories with no purchase history. The content towers should degrade gracefully here; an ID-based model collapses to zero |
| Colour-harmony agreement | How often the model's picks agree with classical colour theory. **Reported as a finding, not a target** — disagreement would be interesting, since the model is fitted to what people actually bought |

### 5.6 Planned ablations

Each isolates one design decision, all on identical temporal splits:

1. **Content vs collaborative vs hybrid** — measured separately on cold articles,
   where the difference should be starkest
2. **With and without CLIP image features** — does what a garment *looks like*
   add anything over its metadata? A fair test needs real behavioural labels,
   which this dataset provides
3. **Negative sampling** — random vs popularity-matched vs hard, to quantify how
   much of any apparent performance is popularity in disguise
4. **PMI filtering on and off** — how much of the co-purchase signal survives
   controlling for popularity

---

## 6. Explainability

Every recommendation carries the reasons behind it. The competition's
leaderboard rewards accuracy alone, so this is not something the published
solutions provide, and it is the difference between a score and a product.

Both ranking models support it: LightGBM through SHAP values (SHapley Additive
exPlanations, a per-prediction attribution of each feature's contribution) and
the Track B cross-feature ranker through direct decomposition, since a linear
score is a sum of `weight × feature` terms.

Reasons surface as plain statements — *"frequently bought with similar dresses"*,
*"colour complements your selection"*, *"in your usual price range"* — each
traceable to the feature that produced it.

---

## 7. Milestones

| | deliverable | exit criterion |
| --- | --- | --- |
| **M0** | Ingestion, database, temporal splits | Row counts reconcile against the source files; splits provably leak-free |
| **M1** | Popularity and repurchase baselines | A first MAP@12 on the validation week, comparable to published baselines |
| **M2** | Candidate generation | Recall@K measured per strategy and for the union |
| **M3** | LightGBM ranker | MAP@12 positioned against the public leaderboard |
| **M4** | Track B pair mining and two-tower model | Recall@K **and lift over popularity** |
| **M5** | Ablations and explainability | Each design decision supported by a measurement |
| **M6** | API and interface | The journey runs end to end in a browser |

**M1 produces the first externally comparable number.** Establishing that the
measurement is sound before building anything on top of it is the point of
sequencing it this way.

---

## 8. Risks

| risk | mitigation |
| --- | --- |
| **Popularity dominates.** Both tracks can look successful while having learned only what sells. | Popularity is the explicit baseline everywhere; Track B additionally uses popularity-matched negatives and reports lift before anything else. |
| **Temporal leakage.** The most common way recommender results become fiction. | Feature construction takes a cutoff date as a required argument; a test asserts no feature reads past it. |
| **Co-purchase is not co-wear.** Track B's premise could simply be wrong. | PMI filtering with reported survival rates at each stage. If little survives, that is the finding, and it arrives at M4 rather than at the end. |
| **Scale.** 31M transactions will not fit in naive pandas workflows. | Time-windowed sampling; DuckDB for out-of-core joins; sampling parameters, not constants. |
| **Cold start.** New customers and articles are where offline metrics quietly fail. | Reported as its own segment from M1, never folded into an aggregate. |
