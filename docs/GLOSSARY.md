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
