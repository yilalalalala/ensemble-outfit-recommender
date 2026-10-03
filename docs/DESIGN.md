# Ensemble — system design

A fashion recommender built on the H&M Group transaction dataset, in two parts:
a personalised next-purchase recommender measured against a public benchmark,
and an outfit-completion ("Complete the Look") recommender that answers a
question the benchmark does not ask.

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

### 1.2 Track B — Outfit completion / Complete the Look (the research track)

Given an anchor garment and a missing **slot** (bottoms, shoes, bag,
accessories…), recommend the complementary article that completes the outfit.
Jewellery is the **showcase slice**: demoed and reported as its own segment, but
not the scope (D-002).

The competition task is purely sequential personalisation: *what will this person
buy next*. It carries no notion of *what goes with what*. Yet the data contains
basket structure — items bought by one customer on one day — which is real
evidence of intended pairing, paid for with real money.

Track B mines that signal. It is the part of this project that is not a
reproduction of published work.

**Scope note.** Track B is a difference in *question*, not a subset of the data.
It trains and recommends on the full catalogue. A jewellery-only version was
considered and rejected on data grounds: in the 16-week window only 26k of 898k
baskets (2.9%) pair a garment with jewellery, spread over ~2,000 jewellery
articles, too sparse to survive support and lift filtering (D-002).

---

## 2. Data

| table | rows | contents |
| --- | ---: | --- |
| `transactions_train` | 31,788,324 | customer_id, article_id, t_dat, price, sales_channel_id |
| `customers` | 1,371,980 | age, club member status, fashion news frequency, postal code |
| `articles` | 105,542 | product_type_name, product_group_name, colour, department, index_group, garment_group, detail description |
| images | ~105k | one JPEG per article |

The Accessories product group has 11,158 articles (bags, hats, scarves,
sunglasses, belts…), of which ~2,000 are jewellery (earrings, necklaces, rings,
bracelets). Track B's pairs come from basket structure, with no external data.

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

### Stage 1 — Retrieval / candidate generation (recall-oriented)

Several independent retrieval channels, merged:

| retrieval channel | rationale |
| --- | --- |
| Repeat purchase ("buy it again") | The customer's own recent articles. In fashion retail this single rule is a large share of all correct predictions. |
| Popularity | Top sellers in the recent window, globally and per customer segment. The baseline everything must beat. |
| Item-to-item collaborative filtering | Articles frequently co-purchased with the customer's history. |
| Content similarity | Nearest neighbours in metadata + image embedding space. The only strategy that can reach cold articles. |
| Variant (same style, other colorway) | An H&M-specific pattern: the same `product_code` in a different colour. |

**Channel contract (Phase 2).** Every channel writes `(customer_idx, article_id,
score)`; the merge ranks each channel per customer with deterministic ties, applies
a per-channel cap and keeps each candidate's per-channel score and rank as
provenance (and ranker features). Implemented channels: repeat purchase, global
and age-band popularity, launch-proxy new arrivals (global and personalised by
department), item-to-item CF (cosine), directional time-weighted co-visitation,
colourway variants, department- and section-conditioned popularity, implicit ALS
and FashionCLIP similarity. Caps are not hand-set: a greedy allocator extends
whichever channel adds the most new true hits per new candidate on historical
weeks, tracing a recall vs candidates-per-customer frontier
(`ensemble.candidates.budget`); the operating point is chosen on downstream
MAP@12, memory and time (D-027).

**Metric: Recall@K.** The fraction of truly-purchased articles that appear
anywhere in the candidate set. This is the **ceiling on the final score** — an
item the ranker never sees cannot be recommended — so it is measured and tuned
separately before any ranking work begins.

The trade-off is explicit: larger candidate sets raise Recall@K and raise
training cost. The choice of K is reported with its cost, not assumed.

### Stage 2 — Ranking

**LightGBM with LambdaRank** (learning to rank, listwise objective) — a GBDT whose objective
optimises ranking position directly rather than per-item error.

Features span four groups:

- **Customer** — age, tenure, purchase frequency, average price point, channel mix
- **Article** — recent sales velocity, price, product group, colour, age of the article
- **Customer×Article** — has this person bought it before, bought this product
  type before, bought this colour before, price relative to their usual spend
- **Retrieval-source features** — which retrieval channel proposed this candidate,
  and its rank and score within that channel. Frequently one of the strongest features.

The number of trees is chosen by temporal early stopping on the most recent
training week, then the model is refit on all training weeks. LambdaRank
optimises an NDCG-based surrogate; MAP@12 is the metric reported.

### Stage 3 — Re-ranking (business rules)

Applied to the ranker's ordered list and evaluated separately: an availability
eligibility rule (DATA.md, availability proxy) and optional diversity caps
(colourways per style, items per product type), reported with intra-list
diversity, catalogue coverage and novelty next to MAP@12.

---

## 5. Track B architecture — outfit completion

This section is the design that distinguishes the project, so it is specified in
more detail.

### 5.1 Problem statement

Given an anchor article `a`, a target slot `s` and a customer context `x`, rank
articles `c` in slot `s` by the probability that `c` belongs in the same outfit
as `a`.

**Slots** are product groups that can be worn together: upper body, lower
body, full body, shoes, accessories, socks & tights, swimwear. A pair is
**complementary** when its two articles are in different slots. Underwear,
nightwear and non-apparel groups are excluded.

### 5.2 Where the labels come from, and why they are suspect

A **basket** is the set of articles one customer purchased on one day. If a
basket contains articles `a` and `c` in different slots, that is a positive pair.

**The obvious objection: co-purchase is not co-wear.** Someone buying a week of
clothing in one trip generates dozens of pairs that were never intended to go
together. Taken naively, this signal mostly measures "both items are popular".

Three filters address this, and the third is the one that matters:

1. **Basket size.** Keep baskets of 2–6 items. Larger baskets are stock-ups, not
   outfits. This is a blunt instrument but removes the worst offenders.
2. **Slot structure.** Keep pairs whose articles are in different slots. Two
   tops bought together are substitutes or a stock-up, not an outfit.
3. **Association strength, not raw count.** This is standard market basket
   analysis: filter pairs by **lift** (association-rule lift), equivalently
   **PMI** (Pointwise Mutual Information, the same quantity on a log scale and
   the name used in ML papers):

   ```
   lift(a, c) = P(a, c) / ( P(a) · P(c) )          PMI(a, c) = log lift(a, c)
   ```

   `P(a,c)` is how often the two appear in a basket together; `P(a)·P(c)` is how
   often they *would* co-occur if purchases were independent. **PMI is positive
   only when a pair co-occurs more than popularity alone explains.** This is
   precisely the "both items are popular" confound, removed by construction.

   Raw PMI is unstable for rare pairs, so pairs below a minimum **support**
   threshold are discarded and the remainder is scored with **NPMI**
   (normalised PMI, bounded to [−1, 1] and less biased toward rare pairs).

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
| In-batch negatives, with **logQ correction** | 50% | Cheap and plentiful; separates broad categories. In-batch sampling over-samples popular items, so the logit is corrected by the item's sampling probability (Yi et al., Google, RecSys 2019) |
| **Popularity-based negative sampling** | 30% | Negatives drawn with probability ∝ popularity^0.75 (the word2vec scheme), so best-sellers appear as negatives about as often as they appear as positives. **Without this the model can score well by learning "recommend popular accessories" and nothing else.** |
| **Hard negatives** | 20% | Same accessory sub-type and similar price as the positive. Forces a decision about *this necklace vs that necklace*, not *necklace vs shoes*. |

The popularity-based slice, together with logQ correction, is the defence against the most common failure in
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
| **Relative lift vs. popularity baseline** | (Recall@K of the model − Recall@K of "always recommend the most popular accessories") ÷ the latter, reported as a percentage |

A popularity ranker is the null hypothesis, and in recommender systems it is a
much stronger opponent than people expect. **A relative lift near 0% means the
model learned nothing about compatibility**, whatever its absolute Recall@K looks
like. This number is reported before any other Track B result.

A stricter view, **tail-item recall** (popularity-stratified recall), evaluates
only on accessories outside the popularity head, isolating what the model adds.

**Diagnostics**

| metric | what it detects |
| --- | --- |
| Catalog coverage | Fraction of accessories ever recommended. A model that only ever suggests twenty items is useless in production regardless of its Recall |
| Novelty | Mean inverse popularity of recommendations — quantifies the head-of-catalogue bias |
| Item cold-start Recall@K | Recall restricted to accessories with no purchase history. The content towers should degrade gracefully here; an ID-based model collapses to zero |
| Colour-harmony agreement *(project-specific)* | How often the model's picks agree with classical colour theory. **Reported as a finding, not a target** — disagreement would be interesting, since the model is fitted to what people actually bought |

### 5.5a As built, after Round 3 (2026-10-03)

This section is the specification; the deviations below are recorded as ADRs and
measured in [`reports/TRACK_B_ROUND3_UPGRADE_REPORT.md`](../reports/TRACK_B_ROUND3_UPGRADE_REPORT.md).

| §5 says | what was built | why |
| --- | --- | --- |
| "a second, small cross-feature ranker re-orders the top candidates" | Built: LightGBM LambdaRank over a candidate union, one query group per (basket, anchor, target slot). It is the model now, not a re-order on top of the fused list. | The fixed RRF it replaced could not use support, lift, recency, backoff level or the customer. +29.4% Recall@12 without customer features, +47.6% with them, on six rolling folds (D-033). |
| colour-pair harmony "computed in CIELAB" | Colour agreement at the dataset's own granularity (`colour_group_code`, `perceived_colour_master_id`, `perceived_colour_value_id`) rather than a CIELAB distance | The catalogue ships categorical colour codes, not sRGB values; a CIELAB distance would need a colour-to-RGB table that is not in the data. The colour-agreement group is worth 0.2% of split gain, so the stronger version is unlikely to change much (D-037). |
| "seasonal co-occurrence in the same week of year" | Not built. Time-decayed co-counts (14 d, 56 d half-lives) are built instead. | Two years of data give at most two observations per (pair, week of year). The decay features cost 0.62% to remove, so recency matters a little and seasonality was not worth a cache rebuild to test (D-037). |
| "the customer's own history with this accessory type" | Built and extended: 21 point-in-time customer features (article, style, product type, department, section, garment group, colour, slot, price), the largest single feature group (−12.3% to remove) | D-034. Only 3.4% of truth articles are repeat purchases, so this is taste, not buy-it-again. |
| metrics: "Item cold-start Recall@K" | Replaced by *recently launched* recall (first observed sale within 28 days of the cutoff) | Under a leakage-free catalogue an article with no pre-cutoff sale can never be eligible, so item cold-start recall is identically zero by construction. The dataset has no inventory or launch feed, so true pre-launch availability is unknowable (D-031). |
| evaluation on "the test week" | Selection on six rolling validation weeks; the test week scored once, after selection | One week cannot support a model choice: the same system moves by a factor of ~1.4 across six consecutive weeks (D-032). |

### 5.6 Planned ablations

Each isolates one design decision, all on identical temporal splits:

1. **Content vs collaborative vs hybrid** — measured separately on cold articles,
   where the difference should be starkest
2. **With and without CLIP image features** — does what a garment *looks like*
   add anything over its metadata? A fair test needs real behavioural labels,
   which this dataset provides
3. **Negative sampling** — random vs popularity-based (+ logQ correction) vs hard negatives, to quantify how
   much of any apparent performance is popularity in disguise
4. **Lift/PMI filtering on and off** — how much of the co-purchase signal survives
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

## 7. User experience

### 7.1 Surfaces and entry points

Recommendations reach the user through three entry points, ordered by how much
traffic each carries in a real e-commerce product. Most recommendation traffic
is **passive**: the user asks for nothing and the page is already personalised.

| entry point | user action | backend | milestone |
| --- | --- | --- | --- |
| **Passive, personalised** | Opens the home page or a product page | Track A (*For you*, *Buy it again*); Track B (*Complete the Look*) | M6 |
| **Visual search** ("shop the look" from a photo) | Uploads an outfit photo and taps the garment to match | Crop → CLIP image embedding → ANN search over catalog → matched anchor article → Track B | M7a |
| **Conversational** | Types a request in plain language, optionally with a photo | LLM orchestrator → tool calls into Track A / Track B / visual search → grounded answer | M7b |

All three share one backend. The UI entry points differ; the models do not.

### 7.2 Pages and modules

Results are shown as **modules** (horizontal carousels, also called shelves or
rows), not as a single flat list of 12. **K in an offline metric is not the
number of UI slots**: MAP@12 is the benchmark's K, while a module typically
shows 4–8 items with more on scroll.

- **Home.** *For you* (Track A) · *Buy it again* (repeat purchase) · *Trending
  this week* (popularity, also the fallback for **user cold start**).
- **Product page.** *Complete the Look* (Track B) · *Other colours* (variants) ·
  *Similar items* (content-based).
- **Complete the Look diversity.** One item per accessory type (earrings, bag,
  belt…) rather than five near-identical necklaces. This is a **re-ranking**
  step for **diversity**, the standard third stage after retrieval and ranking.
- **Reasons.** Each card carries one or two reason chips derived from model
  features (§6).
- **Demo login.** A customer picker stands in for authentication so any
  customer's view can be reproduced.
- **Internal DS view.** Offline metrics, segment analysis and a SHAP breakdown
  for any recommendation.

### 7.3 Visual search (photo input)

The industry name for this is **visual search**, and matching a photo taken in
the wild to catalog product shots is the **street-to-shop** problem (Pinterest
Lens, Google Lens, ASOS Style Match, Amazon StyleSnap).

```
photo ─▶ user taps / boxes the garment ─▶ crop ─▶ CLIP embedding
      ─▶ ANN search over catalog image embeddings ─▶ top matched articles
      ─▶ user confirms the closest match ─▶ Track B Complete the Look
```

- **Why the user selects the garment.** An outfit photo contains several items;
  asking the user to tap one is simpler and more reliable than automatic
  **object detection**, which can be added later.
- **Why a confirmation step.** Track B is trained on catalog articles, so a photo
  must first be mapped to one. Showing the top matches and letting the user pick
  keeps the recommendation grounded when the visual match is imperfect.
- **Known gap.** H&M images are studio product shots; user photos are not. The
  dataset has no labelled street photos, so visual-search quality is judged on a
  small hand-labelled set and reported as such.

### 7.4 Conversational assistant (LLM)

The LLM is the **interface and orchestration layer**; the trained models remain
the source of every recommendation (D-009).

- **Tool calling.** Tools: `recommend_for_customer`, `complete_the_look`
  (with filters such as category, colour, price tier), `visual_search`,
  `get_article_details`.
- **Query understanding.** The LLM translates "for a wedding, nothing too
  flashy" into tool arguments.
- **Grounding.** Every article shown must come from a tool result; product cards
  are rendered from tool output, not from LLM text.
- **Multimodal input.** A photo in the chat goes through visual search (§7.3); the
  LLM may describe it but does not choose products from it.
- **Evaluation.** Hallucination rate (articles not returned by a tool), tool-call
  accuracy, **LLM-as-judge** answer quality on a fixed request set, latency and
  cost per conversation.

### 7.5 Visual search and the assistant together

Both entry points ship. Visual search is a **tool the assistant calls**, and it
is also a **standalone feature** that works without the LLM. This is the pattern
behind Google Lens **multisearch** (photo plus a text refinement such as "in
green") and image input in Amazon Rufus.

| | Visual search (camera button) | Assistant (chat, text and/or photo) |
| --- | --- | --- |
| Best for | "Find this" / "what goes with this?", one tap | Constraints and refinement: occasion, budget, colour, "not that one" |
| Input | Photo, then tap the garment | Free text, optionally with a photo |
| Turns | One shot | Multi-turn; remembers the confirmed item and earlier constraints |
| Output | Matched articles → *Complete the Look* modules | Grounded answer with product cards and reasons |
| Latency and cost | Milliseconds, no LLM cost | Seconds, per-conversation LLM cost |
| If the LLM is down | Unaffected | Falls back to visual search and modules |

**Hand-off in both directions.**

- From visual search to chat: the results page has an *Ask about this look*
  button that opens the assistant with the confirmed article already in context.
- From chat to visual search: a photo sent in chat is passed to the
  `visual_search` tool. The top matches come back as cards and the user
  confirms one inside the conversation. The assistant then calls
  `complete_the_look` with the constraints parsed from the text.

```
photo + "for a wedding, gold, under my usual budget"
   │
   ├─▶ visual_search(crop)            → top-3 matches → user confirms #1
   └─▶ query understanding            → {occasion: formal, colour: gold, price_tier: ≤ usual}
                                   ▼
            complete_the_look(article, filters) → re-rank for diversity → cards + reasons
   "cheaper?" → same tools, tighter price filter (session state reused)
```

**Shared session state.** The confirmed anchor article, parsed constraints and
the items already shown are stored per session, so both surfaces read and write
the same context. Items already shown can be excluded from later answers.

**Evaluated separately.** Visual search is measured by match accuracy (§7.3).
The assistant is measured by grounding and answer quality (§7.4). A combined
failure can then be traced to the component that caused it.

### 7.6 Feedback loop

Every **impression**, click, add-to-cart and "not for me" is logged as an event.
This is **implicit feedback** for future training and the data an **online A/B
test** would read. In this project the events are logged locally; there is no
live traffic, so every reported metric remains offline.

---

## 8. Milestones

| | deliverable | exit criterion |
| --- | --- | --- |
| **M0** | Ingestion, database, temporal splits | Row counts reconcile against the source files; splits provably leak-free |
| **M1** | Popularity and repurchase baselines | A first MAP@12 on the validation week, comparable to published baselines |
| **M2** | Retrieval | Recall@K measured per retrieval channel and for the merged set |
| **M3** | LightGBM ranker | MAP@12 positioned against the public leaderboard |
| **M4** | Track B pair mining and two-tower model | Recall@K **and relative lift vs. popularity** |
| **M5** | Ablations and explainability | Each design decision supported by a measurement |
| **M6** | API and web UI (no LLM): home and product pages, reason chips, DS view (§7.2) | The journey runs end to end in a browser |
| **M7a** | Visual search and "snap your outfit, fill the gap" (§7.3, D-011) | Top-5 match accuracy on a hand-labelled photo set |
| **M7b** | *(required, D-012)* Conversational assistant with tool calling (§7.4), using visual search as a tool with two-way hand-off (§7.5); LLM content enrichment tested as an ablation | Hallucination rate 0 on the eval set; enrichment reported as relative lift, positive or not |
| **M8** | *(optional)* Polyvore Outfits: co-wear vs co-purchase compatibility, fill-in-the-blank | Reported as a comparison, positive or not |

**M1 produces the first externally comparable number.** Establishing that the
measurement is sound before building anything on top of it is the point of
sequencing it this way.

---

## 9. Risks

| risk | mitigation |
| --- | --- |
| **Popularity dominates.** Both tracks can look successful while having learned only what sells. | Popularity is the explicit baseline everywhere; Track B additionally uses popularity-based negative sampling with logQ correction and reports relative lift vs. popularity before anything else. |
| **Temporal leakage.** The most common way recommender results become fiction. | Feature construction takes a cutoff date as a required argument; a test asserts no feature reads past it. |
| **Co-purchase is not co-wear.** Track B's premise could simply be wrong. | Lift/PMI filtering with reported survival rates at each stage. If little survives, that is the finding, and it arrives at M4 rather than at the end. |
| **Scale.** 31M transactions will not fit in naive pandas workflows. | Time-windowed sampling; DuckDB for out-of-core joins; sampling parameters, not constants. |
| **Cold start.** New customers and articles are where offline metrics quietly fail. | Reported as its own segment from M1, never folded into an aggregate. |
