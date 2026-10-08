# Glossary

The vocabulary this project uses, aligned with how recommender-system teams in
industry talk. Each entry says who uses the term, so it is clear which words
belong in a data-science review and which in a product review.

Audience key: **DS** = data scientists / ML engineers, **PM** = product managers,
**Both** = used across the table. Terms marked *(project-specific)* are not
industry standard and are named as such wherever they appear.

---

## Problem framing

| term | meaning | audience | in this project |
| --- | --- | --- | --- |
| **Next-purchase prediction** / **personalized recommendations** | Predict what a given user buys next | Both | Track A |
| **Complete the Look (CTL)** / **Shop the Look** | Given an item, recommend complementary items that form an outfit. Product name used by Pinterest, Zalando, Amazon and others | Both (PM-facing name) | Track B |
| **Complementary vs. substitute items** | Complements are bought *together* (dress + necklace); substitutes are bought *instead* (necklace A vs necklace B) | DS | Track B recommends complements |
| **Frequently Bought Together (FBT)** | The co-purchase widget on e-commerce product pages | PM | The simplest form of Track B |
| **Buy it again** / **repeat purchase** | Recommend items the user already bought | Both | Track A repurchase retrieval |

## Data

| term | meaning | audience |
| --- | --- | --- |
| **Implicit feedback** | Signals inferred from behaviour (purchases, clicks) rather than stated ratings. Only positives are observed | DS |
| **Basket** / **market basket** | Items bought in one transaction; here, one customer on one day | Both |
| **Market basket analysis** | Mining which items are bought together, using **support**, **confidence** and **lift** | Both |
| **SKU / variant / colorway** | H&M `product_code` is the style; `article_id` is a style-colour variant | Both |
| **Long tail** / **head items** | The many rarely-sold items vs the few best-sellers | Both |

## Evaluation

| term | meaning | audience |
| --- | --- | --- |
| **Offline evaluation** | Scoring on historical data. Every number in this project is offline | Both |
| **Online evaluation / A/B test** | Scoring with live users; the metric that ultimately decides launches (CTR, conversion, revenue per user). Out of scope here, but offline metrics are a proxy for it | Both |
| **Time-based split** / **out-of-time validation** | Train on the past, validate on a later window, test on the latest window | DS |
| **Data leakage** / **label leakage** | Information from the evaluation period reaching training | DS |
| **Point-in-time correctness** | Every feature is computed only from data available at prediction time. Feature-store term (Feast, Tecton) | DS |
| **Baseline** | The simple method a model must beat. In recsys, the **popularity baseline** (top-sellers) is standard and strong | Both |
| **MAP@K** | Mean Average Precision at K; rewards hits ranked early. The competition metric | DS |
| **Recall@K** / **Hit rate@K** | Fraction of true items appearing in the top K | DS |
| **NDCG@K** | Normalized Discounted Cumulative Gain; ranking quality with position discount | DS |
| **Relative lift** (a.k.a. "lift") | `(model − baseline) / baseline`, reported as a percentage, e.g. "+18% Recall@12 over popularity". The standard way PMs discuss improvements | Both |
| **Catalog coverage** | Share of catalog ever recommended | Both |
| **Novelty / diversity / popularity bias** | Beyond-accuracy metrics; how head-heavy recommendations are | DS |
| **Segment analysis** / **slicing** | Reporting metrics per user or item group | Both |
| **Tail-item recall** / **popularity-stratified recall** | Recall computed only on less popular items, isolating what the model adds beyond best-sellers | DS |

## Cold start

| term | meaning | audience |
| --- | --- | --- |
| **Cold start** | No interaction history to learn from | Both |
| **User cold start** / **item cold start** | New customers / new articles. Standard split | Both |
| **Warm users / items** | Those with history | DS |

## Architecture

| term | meaning | audience |
| --- | --- | --- |
| **Two-stage recommender: retrieval → ranking** | Cheap retrieval narrows millions of items to hundreds; an expensive ranker orders them. Also called **candidate generation → ranking**. Some stacks add a **re-ranking** stage for business rules or diversity | Both |
| **Retrieval channels / sources** | Independent candidate generators whose outputs are merged | DS |
| **Item-to-item collaborative filtering (item-item CF)** | "Customers who bought X also bought Y"; Amazon's 2003 approach | Both |
| **Content-based filtering** | Recommend by item attributes / images rather than behaviour. The standard answer to item cold start | Both |
| **Learning to rank (LTR)** | Models trained to order lists: pointwise, pairwise, **listwise**. LambdaRank / LambdaMART are the standard GBDT versions | DS |
| **GBDT (LightGBM, XGBoost, CatBoost)** | Gradient-boosted decision trees; the industry default ranker for tabular features | DS |
| **Two-tower model** / **dual encoder** | One network embeds the query, one embeds the item; score = dot product. Enables **ANN (approximate nearest neighbour)** search at serving time | DS |
| **Embedding** | A learned dense vector representing an item or user | Both |
| **CLIP** | OpenAI's image-text model; a standard off-the-shelf image embedding | DS |

## Training

| term | meaning | audience |
| --- | --- | --- |
| **Positive / negative examples** | Observed interactions / constructed non-interactions | DS |
| **Negative sampling** | Constructing negatives, since implicit data has none | DS |
| **In-batch negatives** | Use the other positives in the same mini-batch as negatives. Standard for two-tower training | DS |
| **Sampling bias correction** / **logQ correction** | In-batch negatives over-sample popular items; subtract `log(sampling probability)` from the logit to correct it (Yi et al., Google, RecSys 2019) | DS |
| **Popularity-based negative sampling** | Draw negatives with probability ∝ popularity^α (α≈0.75, from word2vec) so the model cannot win by recommending best-sellers | DS |
| **Hard negatives** / **hard negative mining** | Negatives that are similar to the positive (same type, same price), forcing fine distinctions | DS |

## Association statistics

| term | meaning | audience |
| --- | --- | --- |
| **Support** | How often a pair occurs: `P(a, c)` | Both |
| **Confidence** | `P(c \| a)`: of baskets with `a`, the share that also contain `c` | Both |
| **Lift** (association rules) | `P(a, c) / (P(a) · P(c))`. >1 means the pair co-occurs more than chance | Both |
| **PMI** (Pointwise Mutual Information) | `log(lift)`. Same quantity on a log scale; the name used in NLP/ML papers. **In a PM or analytics conversation, say "lift".** | DS |
| **NPMI** | PMI normalised to [−1, 1]; less biased toward rare pairs | DS |

> **Two meanings of "lift".** *Association lift* (above) measures a pair. *Relative
> lift* (Evaluation) measures a model vs a baseline. Always qualify which one:
> "association lift ≥ 2" vs "+18% relative lift over popularity".

## Experimentation and process

| term | meaning | audience |
| --- | --- | --- |
| **Ablation study** | Remove one component, retrain, measure the drop. Standard in ML teams and papers; PMs rarely use the word | DS |
| **Experiment tracking** | Logging params, metrics and artifacts per run (MLflow, Weights & Biases) | DS |
| **Design doc** / **RFC** | Proposal reviewed before building. `docs/DESIGN.md` | Both |
| **ADR (Architecture Decision Record)** | Numbered, dated decision with context and consequences. `docs/DECISIONS.md` | DS / Eng |
| **Model card** | Standard summary of a model's intended use, data, metrics and limitations | Both |
| **SHAP** | Per-prediction feature attributions | DS |
| **Explainability** / **recommendation reasons** | User-facing "Because you bought…" text | Both |

## Product and interface

| term | meaning | audience |
| --- | --- | --- |
| **Module / carousel / shelf** | A titled horizontal row of recommendations on a page | PM |
| **Surface** | A place in the product where recommendations appear (home, product page, cart, email) | PM |
| **Impression** | An item was shown to a user; the denominator of CTR | Both |
| **CTR, add-to-cart rate, conversion rate** | Online metrics read in A/B tests | Both |
| **Re-ranking** / **diversity** | A final stage adjusting the ranked list for variety or business rules | Both |
| **Visual search** | Search by image instead of text | Both |
| **Street-to-shop** | Matching real-world photos to catalog product shots | DS |
| **ANN (approximate nearest neighbour) index** | Fast vector search (FAISS, ScaNN, HNSW) | DS |
| **Object detection** | Locating items within an image | DS |

## LLM systems

| term | meaning | audience |
| --- | --- | --- |
| **Conversational recommender** / **shopping assistant** | Recommendations through dialogue | Both |
| **Agentic system** / **agent** | An LLM that decides which tools to call and in what order to fulfil a request | Both |
| **Tool calling** / **function calling** | The LLM emits structured calls to external functions (here, the recommender APIs) | DS |
| **Orchestration layer** | The LLM coordinating other services rather than doing the core computation | DS |
| **Query understanding** | Turning free text into structured intent and filters | Both |
| **Grounding** | Constraining outputs to verified data (here: articles returned by tools) | Both |
| **Hallucination rate** | Share of outputs containing unsupported content, here articles not returned by a tool | Both |
| **LLM-as-judge** | Using an LLM with a rubric to grade outputs at scale | DS |
| **Content enrichment** | Generating item attributes (style, occasion) from text or images | Both |
| **Guardrails** | Rules and checks constraining LLM behaviour | Both |
| **Multimodal input** | Input combining modalities, e.g. image + text | Both |
| **Multisearch** | Google's name for searching with an image plus a text refinement | Both |
| **Composed image retrieval (CIR)** | Research term for retrieval from a reference image plus a text modification ("this, but in gold") | DS |
| **Session state** / **conversation memory** | Context carried across turns (confirmed item, constraints, items already shown) | Both |
| **Graceful degradation** / **fallback** | The product still works in reduced form when one component (here, the LLM) fails | Both |


## Evaluation practice (added with the improvement plan)

| term | meaning | audience |
| --- | --- | --- |
| **Rolling backtest** / **walk-forward validation** | Re-run the same recipe on several consecutive past weeks and report mean ± std, so a decision does not rest on one week | DS |
| **Multi-seed evaluation** | Repeat a stochastic training run with different random seeds; differences smaller than the seed spread are noise | DS |
| **Offline–online alignment** | Checking that the offline metric tracks the external or online one (here: local test MAP vs Kaggle private LB) | Both |
| **Fallback policy** / **segment routing** | Serving a different model or rule to a segment where the main model is weak (e.g. new customers → age-band popularity) | Both |

## Vision

| term | meaning | audience |
| --- | --- | --- |
| **FashionCLIP** | CLIP fine-tuned on fashion product images and text (Chia et al., 2022); image and text share one embedding space | DS |
| **Frozen embeddings** | Pre-trained vectors used as fixed input features, not updated during training | DS |
| **Typographic attack** | CLIP-family models over-weight text visible in an image (a label reading "iPod" on an apple) | DS |
| **Grounding (vision)** | Localising an object in an image, usually as a bounding box | DS |
| **Precision@K (relevance-judged)** | Share of the top-K results judged relevant; used when there is no single correct answer | DS |

## LLM engineering

| term | meaning | audience |
| --- | --- | --- |
| **LLM abstraction layer** / **LLM gateway** | One internal interface over several model providers, so the model can be swapped (local for development, hosted for production) | DS / Eng |
| **Tool-call accuracy / argument accuracy** | Did the model call the right tool, with the right arguments? | DS |
| **Constraint satisfaction** | Do the recommended items meet the request's constraints (colour, category, price), checked against catalogue metadata | Both |
| **Prompt caching** | Re-using the processed prefix of a prompt (system prompt, tool definitions) across calls at a fraction of the price | DS / Eng |
| **Batch API** | Asynchronous bulk requests at a discount (50% on Anthropic), for work that need not be real time | DS / Eng |
| **Self-preference bias** | An LLM judge tends to rate outputs from its own model family higher; report judge scores next to objective metrics | DS |
| **Budget guard** | A hard spending cap enforced in code before each paid call | Eng |

## Retrieval and ranking engineering (added in Phase 2)

| term | meaning | audience |
| --- | --- | --- |
| **Co-visitation** / **co-visitation matrix** | Item-to-item counts of "bought (or viewed) B soon after A" by the same user, usually time-weighted and directional. The standard candidate source in session/e-commerce recommenders (e.g. Kaggle OTTO and H&M solutions) | DS |
| **Category-conditioned popularity** | Best sellers restricted to the categories (department, section) a user buys from, weighted by the user's affinity. A personalised popularity channel | DS |
| **Matrix factorisation (ALS)** | Learns user and item vectors whose dot product predicts interaction; ALS (alternating least squares, Hu–Koren–Volinsky 2008) is the implicit-feedback standard | DS |
| **Candidate budget** / **recall–size Pareto frontier** | The number of candidates per user the ranker can afford, and the best recall reachable at each size. Choosing caps on the frontier rather than per channel by feel | DS |
| **Unique recall** / **marginal contribution** | Hits only one channel finds, i.e. the recall lost if that channel were removed | DS |
| **Early stopping** | Stop adding trees when a held-out metric stops improving; here the held-out set is the most recent *training* week (temporal), never the evaluation week | DS |
| **Negative downsampling** | Keep every positive but only a share of negatives when training a ranker, to cut memory and time | DS |
| **Surrogate objective** | The loss a model optimises in place of the business metric (LambdaRank optimises an NDCG-based surrogate; MAP@12 is reported) | DS |
| **Availability proxy** *(project-specific name)* | Without a stock feed, treating an article as unavailable when its last observed sale before the cutoff is too old. A business rule in the re-ranking stage | Both |
| **Eligibility filter** / **business rules** | Re-ranking stage rules that remove or demote items (out of stock, duplicates) before display | Both |
| **Intra-list diversity**, **catalog coverage**, **novelty** | Beyond-accuracy metrics: distinct categories within a list, distinct items shown across users, mean −log₂(popularity share) of shown items | Both |
| **Cluster (customer-level) bootstrap** / **paired bootstrap** | Resample customers with replacement to put a confidence interval on a metric difference between two systems scored on the same customers | DS |
| **Feature attribution vs. explanation faithfulness** | SHAP says which features moved a score; a user-facing reason is *faithful* only if the underlying data supports the sentence ("you bought this before" needs a recorded purchase) | Both |
| **Point-in-time correctness** / **cutoff** | Every feature for a prediction date uses only data available before it | DS |

## Candidate fusion and point-in-time protocol (added in Track B Round 3)

| term | meaning | audience |
| --- | --- | --- |
| **Reciprocal rank fusion (RRF)** | Merge several ranked lists by summing `w / (c + rank)`, typically `c = 60` (Cormack et al., 2009). No training and no score calibration needed, which is why it is the usual first fusion a team ships | DS |
| **Learned fusion** / **learning-to-rank fusion** | Replacing a fixed fusion rule with a trained ranker over the union of the sources, using each source's presence, rank and score as features. The standard next step after RRF | DS |
| **Candidate union** | The deduplicated set of candidates from every retrieval source for one query, with each source's rank kept as a feature | DS |
| **Query group** (learning to rank) | The set of candidates that compete for one ranking decision; LambdaRank optimises within a group. Here one group is one (basket, anchor, target slot) | DS |
| **Hierarchical backoff** | When exact-key evidence is too sparse, fall back to a coarser key (here: all colourways of a `product_code` pooled), and expose *which* level was used as a feature instead of choosing globally | DS |
| **Time-decayed / recency-weighted co-occurrence** | Co-occurrence counts where an older basket contributes less, `0.5 ^ (age / half_life)`. The **half-life** is the age at which a basket counts half as much | DS |
| **Sampled softmax** / **in-batch negatives** | Train a retrieval model by scoring each positive against the other items in the batch instead of the whole catalogue | DS |
| **logQ correction** | Subtract `log` of a candidate's sampling frequency from its logit, because in-batch sampling over-samples popular items (Yi et al., 2019, "Sampling-bias-corrected neural modeling"). Without it, in-batch training collapses toward popularity | DS |
| **Hard negatives** | Wrong candidates the current model scores highly, mined deliberately so the model learns the fine distinctions. **Retrieval-informed** hard negatives are mined by re-scoring a pool with the model being trained | DS |
| **False negative** (retrieval training) | A "negative" that is actually a valid answer (another colourway of the positive, an item with basket co-occurrence evidence). Masked out rather than learned against | DS |
| **Oracle protocol** *(project-specific name)* | An evaluation variant that deliberately uses information unavailable at prediction time, kept only to measure what an earlier leak was worth. Never a reported result | DS |
| **Evidence provenance** | Recording, per recommendation, which signal produced it (co-purchase, style co-purchase, visual compatibility, popularity fallback) and the raw values behind it, so an explanation can be checked against the data | Both |
| **Feature store** | The serving-side system that computes and serves point-in-time-correct features at request time (Feast, Tecton). Personalised recommendations that cannot be precomputed per key need one | DS / Eng |
| **Recently launched recall** *(project-specific name)* | Recall restricted to articles whose first observed sale is within `new_item_days` of the cutoff. Used where true item cold start is unmeasurable because the dataset has no launch or inventory feed | DS |

## Research benchmark and production serving (added in the Track A research / Track B production round)

| term | meaning | audience |
| --- | --- | --- |
| **BPR (Bayesian Personalized Ranking)** | Pairwise loss that pushes a bought item above a sampled unbought one for the same user (Rendle et al., 2009). **BPR-MF** = BPR on a matrix-factorisation model | DS |
| **LightGCN** | Graph collaborative filtering that only averages ID embeddings over the user–item graph, no feature transforms (He et al., 2020); a standard strong CF baseline | DS |
| **SASRec** | Self-attentive sequential recommender: a causal Transformer over the user's last purchases predicts the next item (Kang & McAuley, 2018) | DS |
| **gBCE / gSASRec** | Generalised binary cross-entropy: BCE with k sampled negatives and the positive term raised to a calibration power, so a sampled loss stops over-predicting (Petrov & Macdonald, 2023) | DS |
| **Reproduced baseline** | A published model re-implemented and tuned under *your* protocol, so the comparison is apples to apples; a claim may only name baselines actually reproduced | DS |
| **Tuning fold** vs **reporting fold** | Weeks used to choose hyperparameters vs weeks used to report results; keeping them disjoint keeps the reported numbers honest | DS |
| **Adoption rule** / **ship criterion** | A predeclared test a change must pass before it is adopted (here: CI above zero, folds won, cost limits) | Both |
| **Ablation (refit)** | Remove one feature group and *retrain*; zeroing inputs of a trained model is not an ablation | DS |
| **Serving bundle** *(project-specific name)* | Versioned, immutable directory of everything the online path reads (models, candidate pools, profiles, catalogue snapshot, manifest with hashes) | DS / Eng |
| **Train–serve skew** | Online features or scores differing from what the model saw offline; checked here by an **equivalence gate** before a bundle is published | DS / Eng |
| **Request-time (online) personalization** | Scoring with the user's features when the request arrives, instead of a precomputed per-key table | Both |
| **Liveness / readiness probe** | `/healthz` (process up) vs `/readyz` (able to serve: artifacts loaded and validated) | Eng |
| **p50 / p95 / p99 latency** | Median and tail response times; tails (p95, p99) are what users feel under load | Both |
| **Cold vs warm cache** | Latency with every request computed vs with repeated requests answered from cache | Eng |
| **Cache invalidation** | Making sure no stale answer survives a model or data change; here the bundle and availability versions are part of every cache key | Eng |
| **Atomic swap / rollback** | Publishing a new version by switching one pointer after it is complete and validated; rolling back = switching it back | Eng |
| **Exact vs approximate nearest-neighbour search** | Brute-force similarity over all vectors vs an index (HNSW, FAISS) that trades a little recall for speed; only worth it when exact is too slow | DS / Eng |

## Retrieval ceiling (added in the Track A retrieval upgrade)

| term | meaning | audience |
| --- | --- | --- |
| **Candidate recall ceiling** | Share of purchased items present anywhere in the candidate set; no ranker can exceed it | DS |
| **Append-only retrieval** | Adding a new source's candidates *after* the existing set, never evicting an existing candidate, so the old system's candidates are a subset of the new one's | DS |
| **Miss taxonomy** | Classifying every missed purchase by where it was lost: never proposed, proposed but cut by a cap, lost in de-duplication, or retrieved but ranked out of the top k | DS |
| **Marginal conversion** *(project-specific name)* | Added Recall@k divided by added candidate recall: how much of the extra coverage the ranker turns into shown hits | DS |

## Ranker retrain (added in the Track A ranker retrain)

| term | meaning | audience |
| --- | --- | --- |
| **Distribution matching (train/serve candidate parity)** | Training the ranker on candidates produced by exactly the retrieval used at inference, so it never ranks a region of feature space it did not see in training; the ranking analogue of avoiding training–serving skew | DS |
| **Retrieval provenance features** | Ranker features describing *how* a candidate was retrieved (source, rank within the source, source score), e.g. YouTube's and Pinterest's rankers consume retriever ranks/scores | DS |
| **Ablation ladder** | A predeclared sequence of variants, each adding one change to the previous (here R0 → R1 retrain → R2 + features → R3 tuning), so each step's effect is measured separately | DS |

## Public demo build (added in the public demo rebuild, D-046)

| term | meaning | audience |
| --- | --- | --- |
| **Static site hosting** | Serving pre-built files only (GitHub Pages, S3 + CDN): no application process, so no database, no model inference and no server-side rendering | Eng |
| **Demo mode** *(project-specific name)* | A build-time switch that makes the application's own frontend answer its API calls from checked-in responses instead of a server; declared by a `<meta>` tag the real build never emits, never inferred from the hostname | Eng |
| **Frozen response** / **fixture** | A real API response captured once and checked in, then replayed verbatim; the demo's product, Complete-the-Look, assistant and photo data are all frozen responses, not regenerated content | Eng |
| **Golden/provenance assertion on published numbers** | A test that the figures a published page shows equal the stored evaluation artifacts exactly, unrounded and unrecomputed — the deployment analogue of `report-verify` | DS / Eng |
| **Documented no-op** | A call the public build accepts and deliberately does nothing with (here: event logging), stated as such rather than silently dropped | Eng |
| **Project subpath (base path)** | A site served below the domain root (`/<repo>/` on GitHub Pages); every asset and data URL must be document-relative or it breaks | Eng |
