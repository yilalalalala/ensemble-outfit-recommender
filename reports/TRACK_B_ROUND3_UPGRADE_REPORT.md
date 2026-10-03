# Track B Round 3 — Complete the Look upgrade

**Offline evaluation readout.** 2026-10-03 · branch `phase2-retrieval-ranking-upgrade` ·
nothing pushed, merged, published or submitted (§19).

---

## 1. Executive summary

Track B ("what goes with this?") was a fixed reciprocal rank fusion of association
rules and a two-tower model, selected on one validation week, over a catalogue
that read the week being predicted. This round replaced the protocol, the
candidate generation and the fusion, and added the customer.

**The result.** On six consecutive rolling validation weeks, with every model
trained and every feature built strictly before each week:

| system | Recall@12 | NDCG@12 | vs shipped RRF hybrid | folds won |
| --- | ---: | ---: | ---: | ---: |
| slot popularity (D-004 reference) | 0.0752 | 0.0382 | −29.8% | 0/6 |
| association rules (NPMI, support ≥ 3) | 0.1007 | 0.0609 | −6.3% | 1/6 |
| **shipped RRF hybrid (Round 1)** | **0.1066** | **0.0622** | — | — |
| equal-weight RRF over the full candidate union | 0.1088 | 0.0642 | +2.0% | 6/6 |
| **learned ranker, compatibility only** | **0.1379** | **0.0822** | **+29.3%** | **6/6** |
| **learned ranker, + personalization** | **0.1573** | **0.0961** | **+47.7%** | **6/6** |

Pooled customer-cluster bootstrap against the shipped hybrid, over 145,485
customers and 622,989 queries: **+47.58% Recall@12 [+46.65%, +48.47%]** and
**+54.45% NDCG@12 [+53.45%, +55.46%]** for the personalized ranker; **+29.37%
[+28.57%, +30.15%]** and **+32.00% [+31.21%, +32.82%]** for the compatibility-only
ranker. Both win on every fold. The target was +5%.

**Why this is ranking, not a bigger net.** Equal-weight RRF over the *identical*
189-candidate union gains 2.0%. The same union, ranked by the learned model,
gains 29.3% without any customer feature. The gain is the fusion, not the
retrieval budget.

**What the model actually leans on.** Removing one feature group at a time and
refitting on every fold: personalization −12.3%, candidate popularity −10.5%,
two-tower −4.0%, style backoff −0.8%, exact-article repeat −0.7%, price + colour
−0.7%, FashionCLIP −0.7%, time decay −0.6%, article association −0.4%. No single
evidence source is load-bearing; the model wins by combining many weak ones, which
is also why it is robust.

**Three findings that change how earlier numbers should be read.**

1. The Round-3 reproduction of the shipped baseline was initially too weak — it
   fused a raw association list instead of the popularity-back-filled one Round 1
   actually fuses. Fixed; the corrected baseline is 7% stronger, and the first
   six-fold run's headline (+60.3%) was overstated. Every number in this report
   comes from the corrected run.
2. The Round-1 catalogue leak **did not inflate** Recall@12. It lowered it by
   5.1%, because it added 5.9% more truth articles that no pre-cutoff model could
   retrieve. It was still an invalid protocol — eligibility was not knowable at
   prediction time — but Round-1 and Round-3 recall levels are not comparable in
   either direction.
3. Personalization is **not** a buy-it-again effect. Only 3.4% of
   Complete-the-Look truth articles had already been bought by that customer, and
   removing the exact-article repeat features costs 0.7%. The gain is category,
   colour and price taste, and it appears only where there is history: on returning
   customers the personalized ranker adds +15.1% over the compatibility ranker, on
   new customers +0.7%.

**Not everything improved.** The shipped hybrid's own tail-item recall fell once
the baseline was corrected (popularity back-fill concentrates it on the head), the
two-tower channel is under-trained in every fold (the held-out curve is still
rising at the last epoch), and serving still ships the compatibility ranker rather
than the better personalized one, because the Complete-the-Look table is
anchor-level and a per-customer row is not precomputable.

---

## 2. What changed versus the earlier Round-3 artifact

`reports/track_b_round3/backtest_val.json` was produced before the baseline fix
and is retained for traceability, but **it is superseded**. The only system whose
numbers changed is the baseline:

| fold | `shipped_rrf_hybrid` R@12, first run | corrected | the learned ranker's lift, first run → corrected |
| --- | ---: | ---: | --- |
| 2020-08-05 | 0.0807 | 0.0864 | +60.5% → +49.9% |
| 2020-08-12 | 0.0833 | 0.0931 | +64.1% → +46.9% |
| 2020-08-19 | 0.0934 | 0.1015 | — → +52.0% |
| 2020-08-26 | 0.1026 | 0.1138 | — → +45.7% |
| 2020-09-02 | 0.1143 | 0.1234 | +57.9% → +46.4% |
| 2020-09-09 | 0.1161 | 0.1217 | +52.5% → +45.6% |
| **mean** | **0.0984** | **0.1066** | **+60.29% → +47.74%** |

Every other system reproduced to the digit, which also confirms the pipeline is
deterministic: the two shipping models were refitted from the same cached
matrices and returned the same tree counts (for example 131 and 138 trees on fold
2020-08-05) and the same metrics.

**The cause.** Round 1 builds its association channel with
`models.association(con, q, uni, k = 4·12, …)`, which back-fills each anchor's
list from the slot's 8·12 most popular articles *before*
`models.rrf([assoc, two_tower], [0.5, 0.5], 12)` fuses it. The Round-3
reproduction fused the un-back-filled list, so popularity only reached the tail of
the fused list instead of earning reciprocal-rank credit inside the association
channel. `fusion.baseline_lists` / `baseline_recs` now reproduce Round 1 exactly.

---

## 3. The protocol, and what the old one was worth

### 3.1 What changed

| | Round 1 | Round 3 |
| --- | --- | --- |
| Eligible catalogue | sold between 4 weeks before the target week and the **end** of the target week (D-013) | sold in the 4 weeks **before the cutoff** (D-031) |
| Model selection | one validation week | six consecutive rolling label weeks (D-032) |
| Test week | scored once, then read | scored once, after selection, ablation table off |
| Uncertainty | none reported; one basket's queries treated as independent | paired bootstrap resampling **customers** with all of their queries |
| Truth articles outside the catalogue | impossible by construction | dropped and counted (`n_truth_dropped`) |
| Item cold start | reported as ≈ 0 | unmeasurable; replaced by *recently launched* (first sale within 28 days of the cutoff) |

### 3.2 What the leak was worth

`protocol_check` trains the towers once per week and scores every fixed-fusion
baseline over both catalogues (`reports/track_b_round3/protocol_leak.json`, two
weeks):

| week | catalogue | eligible articles | queries | truth pairs | slot popularity | assoc NPMI | shipped hybrid | RRF all sources |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-09-02 | leakage-free | 27,064 | 96,881 | 137,818 | 0.0892 | 0.1226 | 0.1233 | 0.1218 |
| 2020-09-02 | Round-1 oracle | 28,903 | 101,282 | 144,969 | 0.0845 | 0.1162 | 0.1169 | 0.1155 |
| 2020-09-09 | leakage-free | 27,070 | 93,672 | 132,168 | 0.0788 | 0.1161 | 0.1186 | 0.1219 |
| 2020-09-09 | Round-1 oracle | 28,611 | 97,651 | 140,070 | 0.0750 | 0.1103 | 0.1128 | 0.1159 |

Mean effect of the oracle catalogue: **−5.11%** (slot popularity), **−5.11%**
(association), **−5.07%** (shipped hybrid), **−5.05%** (RRF all sources). Under
the leakage-free catalogue **5.87%** of truth articles are not eligible and are
excluded; under the oracle catalogue, 0% are, by construction.

**How to read this.** The leak was real — an article could be a legal
recommendation because of sales inside the week being predicted, which makes
"item cold-start recall" a property of the definition rather than of a model — but
it was not flattering. It added truth articles with no pre-cutoff footprint, which
nothing could retrieve, so it depressed every system by about 5%. The practical
consequence is that Round-1 numbers (validation shipped hybrid 0.1214) and Round-3
numbers (same week, leakage-free, 0.1217 in the ablation run) must not be read as
a before/after; only comparisons inside one protocol mean anything.

### 3.3 The query set and the mining funnel

Per label week, from a 16-week mining window ending at the cutoff:

| week | baskets in window | multi-slot baskets | pairs support ≥ 2 | ≥ 3 | ≥ 10 | lift > 1 | style pairs | queries | eligible articles | truth dropped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-07-22 | 1,420,184 | 461,799 | 298,182 | 86,970 | 2,972 | 293,130 | 379,500 | 126,776 | 28,252 | 4,603 |
| 2020-07-29 | 1,434,963 | 467,095 | 294,856 | 84,868 | 2,864 | 290,194 | 378,320 | 135,154 | 27,773 | 6,173 |
| 2020-08-05 | 1,447,850 | 472,001 | 289,092 | 81,994 | 2,786 | 284,988 | 375,012 | 118,124 | 27,524 | 4,202 |
| 2020-08-12 | 1,477,225 | 482,575 | 293,358 | 82,926 | 2,746 | 289,350 | 381,333 | 108,057 | 27,282 | 4,959 |
| 2020-08-19 | 1,486,520 | 485,797 | 290,056 | 81,390 | 2,734 | 286,496 | 378,488 | 98,537 | 27,151 | 6,433 |
| 2020-08-26 | 1,512,808 | 493,839 | 291,082 | 81,694 | 2,876 | 287,770 | 380,431 | 107,718 | 27,126 | 5,615 |
| 2020-09-02 | 1,532,229 | 499,413 | 289,906 | 81,220 | 2,998 | 286,886 | 379,261 | 96,881 | 27,064 | 8,964 |
| 2020-09-09 | 1,536,586 | 498,958 | 284,972 | 79,586 | 3,124 | 282,334 | 373,220 | 93,672 | 27,070 | 7,902 |

The support filter is where the evidence goes: of ~290,000 cross-slot pairs that
occur at least twice, ~82,000 occur at least three times and only ~2,900 at least
ten (D-005 stands — the long tail of pairs is thin). Lift > 1 removes almost
nothing once support ≥ 2 is imposed (293,130 of 298,182 on the first week), which
is why Round 3 keeps support, lift, PMI and NPMI as *features* instead of as a
gate (D-037).

### 3.4 What the label actually contains

`reports/track_b_round3/label_audit.json`, all six folds, 882,947 truth pairs:

| week | truth pairs | truth article already bought by that customer | same style already bought | pairs from customers with history |
| --- | ---: | ---: | ---: | ---: |
| 2020-08-05 | 167,134 | 3.03% | 7.28% | 93.77% |
| 2020-08-12 | 152,467 | 3.58% | 8.04% | 94.46% |
| 2020-08-19 | 138,727 | 3.58% | 8.43% | 93.63% |
| 2020-08-26 | 154,633 | 3.27% | 7.77% | 93.19% |
| 2020-09-02 | 137,818 | 3.51% | 7.89% | 92.49% |
| 2020-09-09 | 132,168 | 3.62% | 8.26% | 93.01% |
| **mean** | | **3.43%** | **7.95%** | **93.42%** |

Basket-noise proxies over the same mining windows: 15.6–17.2% of mined baskets
contain a repeated quantity, 14.3–15.4% contain more than one colourway of one
style, 2.7–2.9% span four or more slots, mean 3.36–3.44 articles and 2.25–2.27
slots per basket. These are reported, not acted on: the implemented filter hook is
a row-level predicate while the proxies are basket-level properties, and testing a
filter costs a full artifact rebuild (§16, §18).

---

## 4. Results on the six rolling validation folds

Source: `reports/track_b_round3/backtest_val_ablations.json`. Candidate union 189
candidates per query, union recall ceiling **0.4285**. The learned rankers can
only choose from that union, and they convert 32.2% (compatibility) and 36.7%
(personalized) of its ceiling into a top-12 hit. The fixed-fusion baselines draw
from their own channel lists rather than the union, so the ceiling bounds them
only indirectly; `rrf_all_sources_union` is the control that uses the identical
union.

### 4.1 Headline

| system | Recall@12 | Recall@5 | NDCG@12 | vs shipped (R@12) | vs shipped (NDCG@12) | vs popularity (R@12) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `slot_popularity` | 0.0752 | 0.0400 | 0.0382 | −29.76% | −38.74% | +0.00% |
| `assoc_npmi` | 0.1007 | 0.0632 | 0.0609 | −6.34% | −2.99% | +33.66% |
| `shipped_rrf_hybrid` | 0.1066 | 0.0650 | 0.0622 | +0.00% | +0.00% | +43.56% |
| `rrf_all_sources` | 0.1060 | 0.0669 | 0.0616 | −0.55% | −0.75% | +42.92% |
| `rrf_all_sources_union` | 0.1088 | 0.0693 | 0.0642 | +1.99% | +3.39% | +46.47% |
| **`lgbm_compatibility`** | **0.1379** | **0.0861** | **0.0822** | **+29.33%** | **+31.96%** | **+85.61%** |
| **`lgbm_personalized`** | **0.1573** | **0.1017** | **0.0961** | **+47.74%** | **+54.70%** | **+112.09%** |

### 4.2 Fold by fold (Recall@12)

| system | 08-05 | 08-12 | 08-19 | 08-26 | 09-02 | 09-09 | mean | between-fold SD | folds won |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `slot_popularity` | 0.0527 | 0.0629 | 0.0780 | 0.0897 | 0.0892 | 0.0788 | **0.0752** | 0.0147 | 0/6 |
| `assoc_npmi` | 0.0705 | 0.0803 | 0.0997 | 0.1150 | 0.1226 | 0.1161 | **0.1007** | 0.0212 | 1/6 |
| `shipped_rrf_hybrid` | 0.0864 | 0.0931 | 0.1015 | 0.1138 | 0.1234 | 0.1217 | **0.1066** | 0.0153 | — |
| `rrf_all_sources` | 0.0884 | 0.0914 | 0.1003 | 0.1122 | 0.1219 | 0.1216 | **0.1060** | 0.0147 | 1/6 |
| `rrf_all_sources_union` | 0.0885 | 0.0936 | 0.1044 | 0.1143 | 0.1262 | 0.1257 | **0.1088** | 0.0160 | 6/6 |
| `lgbm_compatibility` | 0.1126 | 0.1183 | 0.1318 | 0.1483 | 0.1610 | 0.1555 | **0.1379** | 0.0201 | 6/6 |
| `lgbm_personalized` | 0.1296 | 0.1367 | 0.1542 | 0.1658 | 0.1806 | 0.1771 | **0.1573** | 0.0210 | 6/6 |

### 4.3 Fold by fold (NDCG@12)

| system | 08-05 | 08-12 | 08-19 | 08-26 | 09-02 | 09-09 | mean | between-fold SD | folds won |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `slot_popularity` | 0.0267 | 0.0320 | 0.0412 | 0.0454 | 0.0436 | 0.0402 | **0.0382** | 0.0073 | 0/6 |
| `assoc_npmi` | 0.0414 | 0.0478 | 0.0618 | 0.0685 | 0.0730 | 0.0728 | **0.0609** | 0.0134 | 1/6 |
| `shipped_rrf_hybrid` | 0.0484 | 0.0536 | 0.0595 | 0.0671 | 0.0724 | 0.0724 | **0.0622** | 0.0100 | — |
| `rrf_all_sources` | 0.0493 | 0.0539 | 0.0597 | 0.0664 | 0.0703 | 0.0700 | **0.0616** | 0.0087 | 1/6 |
| `rrf_all_sources_union` | 0.0507 | 0.0557 | 0.0628 | 0.0684 | 0.0736 | 0.0742 | **0.0642** | 0.0096 | 6/6 |
| `lgbm_compatibility` | 0.0643 | 0.0684 | 0.0799 | 0.0884 | 0.0968 | 0.0951 | **0.0822** | 0.0137 | 6/6 |
| `lgbm_personalized` | 0.0763 | 0.0830 | 0.0948 | 0.1009 | 0.1112 | 0.1103 | **0.0961** | 0.0143 | 6/6 |

The "±" is the standard deviation **between folds**, not a standard error:
consecutive folds share most of their 16-week mining window and many of their
customers. The spread is real seasonality — the same system moves by a factor of
~1.4 across six weeks — and is the reason one week cannot support a model choice.

### 4.4 Uncertainty: pooled customer-cluster bootstrap

1,000 resamples, customers as the resampling unit (a customer is drawn with all of
their queries), pooled over the six folds: 145,485 customers, 622,989 queries.
Baseline is `shipped_rrf_hybrid`.

**Recall@12**

| system | Δ | 95% CI | relative | relative 95% CI | P(Δ > 0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `slot_popularity` | −0.03124 | [−0.03221, −0.03021] | −29.55% | [−30.46%, −28.58%] | 0.000 |
| `assoc_npmi` | −0.00627 | [−0.00707, −0.00547] | −5.93% | [−6.69%, −5.18%] | 0.000 |
| `rrf_all_sources` | −0.00061 | [−0.00113, −0.00009] | −0.58% | [−1.07%, −0.08%] | 0.005 |
| `rrf_all_sources_union` | +0.00209 | [+0.00143, +0.00270] | +1.98% | [+1.36%, +2.56%] | 1.000 |
| **`lgbm_compatibility`** | **+0.03105** | **[+0.03021, +0.03187]** | **+29.37%** | **[+28.57%, +30.15%]** | **1.000** |
| **`lgbm_personalized`** | **+0.05031** | **[+0.04933, +0.05125]** | **+47.58%** | **[+46.65%, +48.47%]** | **1.000** |

**NDCG@12**

| system | Δ | 95% CI | relative | relative 95% CI | P(Δ > 0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `slot_popularity` | −0.02382 | [−0.02448, −0.02317] | −38.65% | [−39.73%, −37.60%] | 0.000 |
| `assoc_npmi` | −0.00157 | [−0.00202, −0.00112] | −2.55% | [−3.28%, −1.82%] | 0.000 |
| `rrf_all_sources` | −0.00056 | [−0.00084, −0.00030] | −0.90% | [−1.36%, −0.49%] | 0.000 |
| `rrf_all_sources_union` | +0.00201 | [+0.00168, +0.00235] | +3.26% | [+2.72%, +3.81%] | 1.000 |
| **`lgbm_compatibility`** | **+0.01972** | **[+0.01923, +0.02022]** | **+32.00%** | **[+31.21%, +32.82%]** | **1.000** |
| **`lgbm_personalized`** | **+0.03355** | **[+0.03293, +0.03418]** | **+54.45%** | **[+53.45%, +55.46%]** | **1.000** |

The intervals are narrow because the sample is large and the effect is far from
zero; they are **not** a claim about online behaviour. The relative interval
divides the paired difference interval by the observed baseline mean rather than
resampling the denominator, so the relative bounds are slightly too narrow (§16).

### 4.5 Segments

| system | tail | recently launched | jewellery | returning customer | new customer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `slot_popularity` | 0.0197 | 0.0671 | 0.0125 | 0.0753 | 0.0745 |
| `assoc_npmi` | 0.0440 | 0.0927 | 0.0133 | 0.1001 | 0.1093 |
| `shipped_rrf_hybrid` | 0.0356 | 0.0603 | 0.0107 | 0.1055 | 0.1233 |
| `rrf_all_sources` | 0.0386 | 0.0507 | 0.0077 | 0.1046 | 0.1254 |
| `rrf_all_sources_union` | 0.0470 | 0.0616 | 0.0117 | 0.1074 | 0.1288 |
| `lgbm_compatibility` | 0.0799 | 0.1243 | 0.0176 | 0.1368 | 0.1546 |
| `lgbm_personalized` | **0.0912** | **0.1415** | **0.0275** | **0.1574** | **0.1557** |

- **Tail** (outside the top 10% of each slot by window sales): +156% over the
  shipped hybrid. The learned ranker is not a popularity model; it is better at
  the tail *and* at the head.
- **Recently launched** (first observed sale within 28 days of the cutoff): +135%.
  This is the honest replacement for item cold start, which is unmeasurable here.
- **Jewellery** (the separately reported segment, D-002): +157% relative, but
  still only 0.0275 absolute. This remains the weakest segment.
- **Returning vs new customer** is the cleanest internal check that
  personalization is doing what it claims: 93.4% of queries come from customers
  with history, and the personalized ranker beats the compatibility ranker by
  **+15.11%** on them and by **+0.66%** on the 39,997 new-customer queries, where
  `c_has_history = 0` and the affinity features are null. A leak would not respect
  that boundary.
- The shipped hybrid's **tail** recall (0.0356) is now *below* plain association
  rules (0.0440). That is a consequence of the baseline correction, not a
  regression introduced here: back-filling the association channel from popularity
  before fusing concentrates the fused list on the head.

### 4.6 Per slot (Recall@12)

| system | upper | lower | full | shoes | accessories | socks | swimwear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `slot_popularity` | 0.0338 | 0.0716 | 0.0723 | 0.1876 | 0.0886 | 0.3072 | 0.1038 |
| `assoc_npmi` | 0.0697 | 0.1106 | 0.0761 | 0.1881 | 0.0903 | 0.3158 | 0.1013 |
| `shipped_rrf_hybrid` | 0.0741 | 0.1236 | 0.0736 | 0.1555 | 0.0986 | 0.3331 | 0.1166 |
| `rrf_all_sources_union` | 0.0795 | 0.1304 | 0.0688 | 0.1299 | 0.0926 | 0.3519 | 0.1096 |
| `lgbm_compatibility` | 0.0991 | 0.1570 | 0.1197 | 0.2074 | 0.1056 | 0.3800 | 0.1616 |
| `lgbm_personalized` | **0.1135** | **0.1792** | **0.1445** | **0.2341** | **0.1183** | **0.4071** | **0.1905** |

The learned ranker wins in every slot. The two fixed-fusion controls *lose* to the
shipped hybrid on `shoes` and `full`, which is where popularity back-fill inside
the association channel helps most; the personalized ranker beats it there anyway
(+50.5% on shoes, +96.3% on dresses and jumpsuits).

### 4.7 Beyond accuracy

| system | catalogue coverage | novelty | distinct product types / list | distinct styles / list |
| --- | ---: | ---: | ---: | ---: |
| `slot_popularity` | 0.0031 | 11.98 | 0.2949 | 0.8317 |
| `assoc_npmi` | 0.2256 | 12.22 | 0.2994 | 0.8468 |
| `shipped_rrf_hybrid` | 0.2277 | 11.74 | 0.2800 | 0.8218 |
| `rrf_all_sources_union` | 0.4031 | 12.04 | 0.2743 | 0.8051 |
| `lgbm_compatibility` | 0.2238 | 12.73 | 0.2704 | 0.8260 |
| `lgbm_personalized` | 0.2881 | 12.75 | 0.2652 | 0.8004 |

Coverage and novelty both improve over the shipped hybrid (+26.5% coverage,
+1.01 bits of novelty — the learned ranker shows *less* popular items on average).
Intra-list diversity is slightly lower (−5.3% distinct product types, −2.6%
distinct styles), which is what the serving re-ranking rules exist to correct
(§9). `rrf_all_sources_union` has the highest coverage of all because it spreads
reciprocal-rank credit over a 189-candidate union — coverage is cheap when accuracy
is not the objective.

---

## 5. Where the gain comes from: feature-group ablation

Each row removes one feature group from the 95-column schema and **refits the
ranker on every fold** (same cached matrices, same early-stopping rule). Baseline
is the full personalized model. `features.check_groups` asserts that each group
actually removes the expected number of columns before anything is fitted, so a
renamed feature fails the run rather than producing a row that reads "no effect".

| feature group removed | columns | Recall@12 | Δ vs full | NDCG@12 | Δ vs full | folds where removal wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| *(none — full model)* | 95 | **0.1573** | — | **0.0961** | — | — |
| personalization (all `c_`, `cu_`, `x_`) | 21 | 0.1379 | **−12.33%** | 0.0822 | **−14.48%** | 0/6 |
| candidate popularity / recency / age | 10 | 0.1408 | **−10.50%** | 0.0877 | −8.75% | 0/6 |
| two-tower (ranks, similarities, interaction) | 7 | 0.1511 | −3.97% | 0.0927 | −3.49% | 0/6 |
| style backoff (all `s_`, backoff level) | 14 | 0.1561 | −0.77% | 0.0951 | −1.01% | 0/6 |
| exact-article repeat (`cu_article_*`) | 2 | 0.1562 | −0.74% | 0.0945 | −1.60% | 0/6 |
| price + colour agreement | 10 | 0.1562 | −0.70% | 0.0953 | −0.82% | 1/6 |
| FashionCLIP similarity | 1 | 0.1563 | −0.66% | 0.0954 | −0.65% | 1/6 |
| time-decayed co-counts | 6 | 0.1563 | −0.62% | 0.0956 | −0.51% | 1/6 |
| article association (all `a_`, interactions) | 17 | 0.1567 | −0.42% | 0.0949 | −1.19% | 1/6 |

### How to read it

**Two groups matter; the rest are redundant with each other.** Personalization and
candidate popularity each cost ~10–12%. Everything else costs under 4%, and six of
the nine groups cost under 1%. That is not evidence that association or the towers
are useless — the `rrf_all_sources_union` control shows the union itself is worth
only +2% without a ranker, and `assoc_npmi` alone is worth +33.7% over popularity
— it is evidence of **redundancy**: article association, style association, both
towers and CLIP all answer "do these go together", and the ranker can reconstruct
most of one from the others. Dropping any one is cheap; dropping the compatibility
evidence as a whole is not available as a single row in this table, but
`slot_popularity` (−29.8%) and `lgbm_minus_popularity` (which keeps all
compatibility evidence and loses 10.5%) bracket it.

**The ranker is not a popularity model.** Candidate popularity is its most
valuable *single* group, but the pure popularity baseline reaches less than half
its recall (0.0752 vs 0.1573), and removing popularity still leaves +32.0% over
the shipped hybrid. Popularity features let the model calibrate *how much* prior
to apply to each candidate, which a fixed fusion cannot do.

**Association is almost free to remove — and that is the most surprising row.**
Removing all 17 article-level association features costs 0.42%. The style-level
backoff (pooled over all colourways of a `product_code`) plus the towers carry
nearly the same signal. This argues against spending more effort on association
statistics and for spending it on the towers and the customer.

**The buy-it-again hypothesis is refuted twice.** The label audit says only 3.4%
of truth articles were already bought by that customer; the ablation says removing
the two exact-article repeat features costs 0.74% while removing all
personalization costs 12.33%. The personalization gain is category, colour, section
and price taste.

### Feature importance (LightGBM split gain, mean over the six folds)

| feature | mean gain share | in top 25 on |
| --- | ---: | ---: |
| `cand_product_type_no` | 9.53% | 6/6 folds |
| `anchor_product_type_no` | 8.75% | 6/6 |
| `cand_section_no` | 6.63% | 6/6 |
| `cand_colour_group_code` | 6.49% | 6/6 |
| `a_co_share` (co-count / anchor support) | 5.96% | 6/6 |
| `cu_style_n` (customer's purchases of this style) | 5.69% | 6/6 |
| `cand_slot_pop_rank` | 3.49% | 6/6 |
| `cand_pop_recent` | 3.20% | 6/6 |
| `cand_pop_window` | 3.05% | 6/6 |
| `cand_slot_pop_pct` | 2.76% | 6/6 |
| `cand_age_days` | 2.59% | 6/6 |
| `tt_rank` (two-tower rank) | 2.51% | 6/6 |
| `a_co_d14` (14-day decayed co-count) | 2.40% | 6/6 |
| `cu_article_days_ago` | 2.31% | 6/6 |
| `cu_dept_share` | 1.98% | 6/6 |

By group (a feature can belong to two groups, so these do not sum to 1):
category/slot attributes and `same_*` agreement 36.0%, personalization 19.6%,
popularity 17.6%, association 12.8%, towers 8.2%, style backoff 4.1%, decay 3.5%,
repeat 2.6%, price 1.6%, CLIP 1.2%, colour agreement 0.2%.

The largest single contributors are the **anchor and candidate category
attributes** — which product type goes with which product type, in which section,
in which colour. That is the compatibility prior the fixed RRF could not express
at all, and it is the clearest explanation of where +29% without any customer
feature comes from.

---

## 6. Personalization (Phase 3)

Point-in-time customer features, all computed from purchases strictly before the
label week (`features._customer_tables`, window `customer_weeks = 104`):

| group | features |
| --- | --- |
| history size and recency | `c_has_history`, `c_n_tx`, `c_n_days`, `c_days_since_last`, `c_mean_price` |
| exact item and style | `cu_article_n`, `cu_article_days_ago`, `cu_style_n` |
| category affinity (recency-weighted shares, weight `1/(1 + days_ago/7)`) | `cu_type_share`, `cu_dept_share`, `cu_section_share`, `cu_ggroup_share`, `cu_slot_share`, plus raw counts `cu_type_n`, `cu_dept_n` |
| colour affinity | `cu_colour_share`, `cu_cmaster_share` |
| price | `cu_price_dist` (distance from the customer's usual price) |
| compatibility × preference interactions | `x_npmi_type`, `x_tt_type`, `x_co_dept` |

Customers with no pre-cutoff purchase get `c_has_history = 0` and null affinities,
so the compatibility path answers unchanged — there is an explicit no-history path,
not a fallback model.

| segment | queries | shipped hybrid | `lgbm_compatibility` | `lgbm_personalized` | personalization gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| returning customer | 582,992 | 0.1055 | 0.1368 | **0.1574** | **+15.11%** |
| new customer (no history) | 39,997 | 0.1233 | 0.1546 | 0.1557 | +0.66% |

This is the ablation the plan asked for, and it behaves exactly as a correct
point-in-time implementation should: the gain is concentrated where history
exists, and it is ~0 where the features are null. Note that new-customer queries
are *easier* for every system (their baskets are more head-heavy), which is why
the shipped hybrid scores higher on them than on returning customers.

---

## 7. Candidate generation and the learned fusion

### 7.1 The union

Five sources, each contributing its own ranking, built at **key** level
((anchor, target slot)) rather than per query — one basket produces several queries
that share those keys, which makes building per key roughly 10× cheaper:

| source | budget per key | rows per fold (mean) |
| --- | ---: | ---: |
| article association, ranked by raw co-count | 100 | 205,000 |
| article association, ranked by NPMI | 100 | *(same table, second rank column)* |
| style (`product_code`) association, by co-count | 60 | 217,000 |
| style association, by NPMI | 60 | *(same table, second rank column)* |
| two-tower with article-ID embedding | 80 | 2,648,000 |
| two-tower, content only (ID embedding off) | 40 | 1,324,000 |
| slot popularity | 100 | 700 |

Union: **189 candidates per query**, **0.4285** union recall. Candidate-set size is
reported with every result precisely so a ranking gain cannot be confused with a
retrieval-budget gain; `rrf_all_sources_union` holds the union fixed and changes
only the fusion, and gains 2.0%.

### 7.2 The ranker

LightGBM LambdaRank, one query group per **(basket, anchor, target slot)**, trained
on the two label weeks before each fold with their own point-in-time features.
Number of trees by temporal early stopping on the most recent *training* week
(never the fold week), then refit on both weeks with that count:

| fold | trees, personalized | NDCG@12 at that count | trees, compatibility |
| --- | ---: | ---: | ---: |
| 2020-08-05 | 131 | 0.6702 | 138 |
| 2020-08-12 | 133 | 0.6487 | 69 |
| 2020-08-19 | 149 | 0.6630 | 110 |
| 2020-08-26 | 129 | 0.6756 | 92 |
| 2020-09-02 | 149 | 0.6650 | 146 |
| 2020-09-09 | 161 | 0.6425 | 86 |

Training matrices: ~4.0 M rows per fold (two weeks), positive rate 0.85–0.95%,
built from a deterministic 30,000-query customer-level subsample per week with 35%
negative downsampling (every positive kept). A simpler calibrated alternative was
not pursued: LambdaRank at 69–161 trees over 95 features is already the smallest
model in this family, it trains in ~63 s per fold, and the fixed-fusion controls
establish that the complexity is earning its place.

---

## 8. Two-tower negatives, epoch budget and determinism (Phase 5)

`reports/track_b_round3/ablation_towers.json`: two label weeks × two seeds per
configuration, trained point-in-time, evaluated three ways — held-out Recall@12 on
baskets from the tail of the mining window (the rule that picks the epoch),
`fold_recall@12` of the two-tower list alone on the target week over the
leakage-free catalogue, and the same with the article-ID embedding switched off
(what a brand-new article could be retrieved by).

| negatives | fold Recall@12 | seed/week SD | content-only Recall@12 | held-out Recall@12 | vs in-batch + logQ | mean seconds | runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| in-batch only | 0.0706 | 0.0015 | 0.0670 | 0.0529 | **−26.48%** | 147 | 4 |
| **in-batch + logQ** | **0.0961** | 0.0015 | 0.0919 | 0.0854 | — | 166 | 4 |
| in-batch + logQ + 4 retrieval-informed hard | 0.0970 | 0.0010 | 0.0928 | 0.0858 | +0.94% | 267 | 4 |

### logQ is not optional

Dropping the logQ correction costs **26.5%** of the tower channel's recall. This
reproduces D-016 on the Round-3 protocol with a faster trainer, and it is the one
negative-sampling component with an unambiguous effect: in-batch sampling
over-samples popular items, and without the correction the tower collapses toward
popularity.

### The redesigned hard negatives still do not earn their place

Round 1 tested popularity-sampled negatives and (product type, price tier) hard
negatives and found nothing (D-016). Those were removed. What is tested here is a
different design — **retrieval-informed** hard negatives: each step re-scores a
popularity-sampled pool with the model being trained and takes the
highest-scoring wrong items, with the positive, its colourways and articles that
co-occur with the anchor in a basket masked out as likely false negatives.

The result is +0.94%, against a per-run spread of 0.0010–0.0015 (1.0–1.5%). Paired
by (week, seed) the difference is +0.0009 with a paired SD of ~0.0018 and one of
four pairs negative. At n = 4 that is not a detectable effect, and it costs **+61%
training time** (267 s vs 166 s per week). They stay **off** (`hard_negatives: 0`).

The stronger reading: Round 1 concluded "these particular hard negatives do not
help"; Round 3 redesigned them and reached the same place, which suggests the
binding constraint is not the negative-sampling design but that in-batch + logQ
already saturates what this architecture can extract from 3.1 M training pairs
(D-036).

### The epoch budget is adequate — contrary to what the fold logs suggest

In all eight weeks the 6-epoch run picked epoch 6, and the held-out curve still
looked like it was rising (for example 2020-09-09: 0.0693, 0.0808, 0.0848, 0.0861,
0.0858, 0.0886). That reads like a binding budget. The probe says otherwise:
allowing up to 12 epochs with patience 2 on 2020-09-09 ran **8** epochs and
selected **epoch 6**, and its fold Recall@12 (0.0985) is inside the seed spread of
the 6-epoch runs (0.0961 ± 0.0015). The curve flattens immediately after epoch 6.
Caveat: one week, one seed — enough to withdraw the "under-trained" claim, not
enough to rule out a gain on other weeks.

### Determinism

| backend | one epoch, identical seed, repeated | identical? |
| --- | --- | --- |
| CPU | 0.08163852, 0.08163852 | yes |
| MPS | 0.08017826, 0.08020096 | no (0.028% apart) |

MPS kernels are not bit-reproducible, so no tower conclusion in this report rests
on a single run: every configuration above is a mean over two weeks × two seeds,
and the seed/week SD is reported next to it. The MPS run-to-run spread (0.028%) is
two orders of magnitude smaller than the seed spread (1.0–1.5%), so seed choice,
not backend nondeterminism, is what the comparisons have to survive.

---

## 9. Serving (Phase 7)

### 9.1 What the diversity rules cost

Serving re-ranks each Complete-the-Look module for diversity: one colourway per
style, at most `serving.max_per_product_type = 2` items of one product type.
Round 1 applied those as **hard filters** — an item the caps rejected was dropped.
`backtest_val_serving_rules.json` scores both readings as their own systems, out
of the shipping model's top-48 pool, on the same six folds:

| variant | Recall@12 | Recall@5 | NDCG@12 | distinct product types / list | distinct styles / list | mean list length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm_compatibility` (no rules) | 0.1379 | 0.0861 | 0.0822 | 0.2704 | 0.8260 | 12.00 |
| + one colourway per style | 0.1259 | 0.0807 | 0.0770 | 0.2949 | **1.0000** | 12.00 |
| + max 2 per product type | 0.1292 | 0.0783 | 0.0779 | 0.3996 | 0.8810 | 12.00 |
| **+ both (re-ordering, back-filled)** | **0.1285** | **0.0769** | **0.0770** | **0.4032** | **0.9179** | **12.00** |
| + both as hard filters | 0.0863 | 0.0712 | 0.0628 | 0.5694 | 1.0000 | **8.32** |

Relative to the unrestricted ranker:

| variant | Δ Recall@12 | Δ NDCG@12 | Δ distinct product types | Δ distinct styles |
| --- | ---: | ---: | ---: | ---: |
| one colourway per style | −8.68% | −6.32% | +9.06% | +21.06% |
| max 2 per product type | −6.33% | −5.13% | **+47.74%** | +6.65% |
| **both, re-ordering** | **−6.81%** | **−6.27%** | **+49.10%** | **+11.12%** |
| both, hard filters | **−37.45%** | −23.60% | +110.54% | +21.06% |

**The decision this forced.** Applied as a re-ordering with back-fill to the module
size, the shipped caps cost 6.8% Recall@12 and buy 49% more distinct product types
per list — a defensible trade. Applied as hard filters they cost **5.5× more
accuracy**, and the reason is visible in the last column: modules come out
part-empty (8.32 of 12 positions filled; some modules fall below five items, which
is why even Recall@5 drops). Serving therefore now applies the caps as a
re-ordering and back-fills (`serving.diversity_fill_back: true`, D-035), and
`tests/test_api.py::test_product_page_and_diversity` moved to the new contract —
the cap arithmetic itself is unit-tested in `test_apply_diversity_*`. This is a
deliberate change to pre-existing serving behaviour, justified by the table above.

All four variants beat the shipped RRF hybrid: +18.1% to +21.2% Recall@12, 6/6
folds (pooled bootstrap for the shipped re-ordering: **+20.56%** [+19.71%,
+21.33%]). The hard-filter variant is the one exception at −18.9%, which is the
measurement that changed the policy.

### 9.2 The rebuilt Complete-the-Look table

`ensemble.completion.serve_round3` replaces the Round-1 `serve_ctl`:

| | Round 1 | Round 3 |
| --- | --- | --- |
| Fusion | fixed RRF(w = 0.5) of association and the two towers | `lgbm_compatibility` over the full candidate union |
| Serving week | week after the data ends | same |
| Diversity | hard caps | caps as a re-ordering, back-filled to `ctl_per_slot` |
| Per-row evidence | `source` ∈ {co_purchase, style_match}, `lift` | co-count, lift, NPMI, style co-count / NPMI / lift, backoff level, source flags, two-tower and CLIP similarity, colour agreement, price-tier distance, model score, and one provenance label |
| Reason chips | one sentence per row, chosen by `source` | built from the stored evidence, each chip emitted only if its own column is non-null for that pair |

**Why the compatibility ranker and not the personalized one.** Complete the Look is
an anchor-level surface: it has to answer for an anonymous visitor on any product
page, so the table is keyed by (anchor, target slot) and cannot hold a row per
customer (26k anchors × 6 slots × 1.4 M customers). Scoring the personalized model
at request time needs point-in-time customer features served online and a model
server — a feature store, not a modelling change. The cost of not having one is
the measured gap between the two rankers: **+15.11% Recall@12 on returning
customers**. That is reported as a known shortfall, not hidden (D-033).

**Provenance.** Each row gets exactly one label, strongest evidence first:
`co_purchase` (article-level co-count ≥ 1), `style_co_purchase` (style-level
co-count ≥ 1), `visual_compatibility` (retrieved by either tower),
`popular_in_slot` (popularity fallback), `other`. The API builds chips from the
row's own columns, so a visually retrieved pick can no longer borrow a co-purchase
claim and a popularity fallback says so (D-035, extending D-030 to Track B).
Columns are detected at query time, so a Round-1 serving store still renders; the
Round-1 table is kept as `complete_the_look_round1.parquet` when the Round-3 build
replaces it.

### 9.3 What the rebuilt table looks like

`reports/track_b_round3/serving.json`. Serving week **2020-09-23** (the week after
the data ends, cutoff 2020-09-22, so nothing is held back), ranker trained on
2020-09-09 and 2020-09-16 with their own point-in-time features: 74 features, 132
trees. 26,165 live articles × their 6 other slots = **156,990 keys**, 26.3 M
candidate rows scored in 13 anchor batches, **1,255,920 served rows** (8 per
module), 132 s plus 234 s for the serving-week towers, inside the 16 GB budget.

| provenance of a served pick | rows | share |
| --- | ---: | ---: |
| `visual_compatibility` (retrieved by a tower) | 762,188 | **60.7%** |
| `popular_in_slot` (popularity fallback) | 394,137 | **31.4%** |
| `style_co_purchase` | 63,801 | 5.1% |
| `co_purchase` (article-level) | 35,794 | **2.9%** |

**This is the most uncomfortable number in the report, and it is the point of
storing provenance.** Only 2.9% of what Complete the Look shows has article-level
co-purchase evidence behind it, and 31.4% is a popularity fallback with no
compatibility evidence at all. The funnel in §3.3 predicts it — ~82,000 pairs reach
support ≥ 3 against 26,165 live anchors × 6 slots — but the Round-1 table labelled
every row either `co_purchase` or `style_match`, so the UI implied evidence that
was not there for most picks. Now a popularity pick says "Popular pick for this
category" and a tower pick says "Visually matches this piece".

A related gate: a lift ratio computed from four co-purchases is a small-sample
artefact (a real served row had co = 4 and lift = 193×), so a chip quotes the
multiplier only at co ≥ 5 and otherwise states the count alone.

Mean distinct product types per module: **3.86** of 8 items. An end-to-end check
through the API on a demo customer's product page returns full modules for Tops,
Shoes and Accessories with chips such as "Bought together 4× in past baskets",
"Bought with this style 3× in past baskets", "Visually matches this piece", "Same
colour family", "Same price range" — and an audit of every chip against its own
row found **no unsupported chip**.

The SQLite serving store was rebuilt from it (`python -m ensemble.api.build`,
6 s, 1,255,920 Complete-the-Look rows, 26,364 articles referenced) and the full
test suite passes against it (117 tests).

---

## 10. The single final test-week evaluation

Run **once**, after every selection decision above was made, with the ablation
table switched off so the test week never saw ten models:

```
ENSEMBLE_CONFIG=experiments/tb3_final .venv/bin/python -m ensemble.completion.backtest test
```

Test week **2020-09-16** (the last seven days of the dataset). The ranker trained
on 2020-09-02 and 2020-09-09 with their own point-in-time features; the towers for
the test week were trained from data ending 2020-09-15. 86,350 queries, 21,283
customers, 184 candidates per query, union recall 0.4719.
`reports/track_b_round3/backtest_test.json`, git commit `5978fbe`, 452 s, peak 5.4 GB.

| system | Recall@12 | Recall@5 | NDCG@12 | vs shipped (R@12) | 95% CI | vs shipped (NDCG@12) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `slot_popularity` | 0.0884 | 0.0480 | 0.0461 | −29.92% | [−32.29, −27.68] | −38.71% | [−41.34, −36.11] |
| `assoc_npmi` | 0.1202 | 0.0763 | 0.0747 | −4.67% | [−6.26, −3.00] | −0.71% | [−2.24, +1.03] |
| `shipped_rrf_hybrid` | 0.1261 | 0.0776 | 0.0752 | — | | — | |
| `rrf_all_sources` | 0.1304 | 0.0796 | 0.0747 | +3.38% | [+2.26, +4.50] | −0.73% | [−1.74, +0.30] |
| `rrf_all_sources_union` | 0.1322 | 0.0837 | 0.0781 | +4.86% | [+3.43, +6.33] | +3.88% | [+2.54, +5.31] |
| **`lgbm_compatibility`** | **0.1596** | **0.0996** | **0.0956** | **+26.57%** | **[+24.74, +28.34]** | **+27.11%** | **[+25.29, +28.81]** |
| **`lgbm_personalized`** | **0.1820** | **0.1187** | **0.1124** | **+44.33%** | **[+42.20, +46.43]** | **+49.45%** | **[+47.16, +51.78]** |
| `lgbm_compatibility` + shipped diversity rules | 0.1494 | 0.0878 | 0.0897 | +18.48% | | +19.28% | |

### Did validation predict the test week?

| quantity | six validation folds | test week | shortfall |
| --- | ---: | ---: | ---: |
| `lgbm_personalized` vs shipped, Recall@12 | +47.74% | +44.33% | −3.4 pts |
| `lgbm_compatibility` vs shipped, Recall@12 | +29.33% | +26.57% | −2.8 pts |
| `lgbm_personalized` vs shipped, NDCG@12 | +54.70% | +49.45% | −5.3 pts |
| shipped diversity rules vs shipped hybrid | +20.58% | +18.48% | −2.1 pts |

Every lift is 2–5 percentage points smaller on the test week than the validation
mean, in the same direction for every system. That is the expected sign — the
operating point was chosen on those folds — and the shrinkage is small relative to
the effect, which is what matters. The ordering of all seven systems is identical
on validation and test.

### Test-week segments

| system | tail | recently launched | jewellery | returning | new customer | coverage | novelty | div. type | div. style |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `slot_popularity` | 0.0136 | 0.0911 | 0.0000 | 0.0883 | 0.0895 | 0.0031 | 11.87 | 0.2429 | 0.8496 |
| `assoc_npmi` | 0.0455 | 0.1143 | 0.0004 | 0.1193 | 0.1312 | 0.2173 | 12.21 | 0.2595 | 0.8522 |
| `shipped_rrf_hybrid` | 0.0438 | 0.0715 | 0.0062 | 0.1248 | 0.1422 | 0.2241 | 11.89 | 0.2748 | 0.8180 |
| `rrf_all_sources_union` | 0.0577 | 0.0782 | 0.0089 | 0.1306 | 0.1514 | 0.3970 | 12.17 | 0.2767 | 0.8060 |
| `lgbm_compatibility` | 0.0927 | 0.1586 | 0.0155 | 0.1583 | 0.1751 | 0.2223 | 12.82 | 0.2404 | 0.8244 |
| `lgbm_personalized` | **0.1080** | **0.1765** | **0.0287** | **0.1826** | 0.1750 | 0.2789 | 12.84 | 0.2432 | 0.8031 |
| `lgbm_compatibility` + diversity rules | 0.0864 | 0.1445 | **0.0380** | 0.1485 | 0.1610 | 0.2270 | 12.83 | **0.3842** | **0.9074** |

Tail +147%, recently launched +147%, jewellery +363% relative to the shipped
hybrid; coverage +24%, novelty +0.95 bits. Intra-list diversity is 11% lower than
the shipped hybrid before the serving rules and 40% higher after them.
Interestingly the diversity rules are the best configuration for **jewellery**
(0.0380 vs 0.0155 unrestricted) — forcing variety inside a module surfaces
accessories that the raw score ordering buries.

### Test-week per slot (Recall@12)

| system | upper | lower | full | shoes | accessories | socks | swimwear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shipped_rrf_hybrid` | 0.0833 | 0.1454 | 0.1167 | 0.1870 | 0.1098 | 0.3081 | 0.0989 |
| `lgbm_compatibility` | 0.1118 | 0.1864 | 0.1549 | 0.2201 | 0.1178 | 0.3606 | 0.1544 |
| `lgbm_personalized` | **0.1254** | **0.2124** | **0.2025** | **0.2382** | **0.1301** | **0.3784** | **0.1867** |

Wins in every slot, as on validation.

### The honest caveat

This week was scored once in Round 1 and those numbers have been read, so it is a
**confirmation, not a fresh holdout**. What it does establish is that the
operating point chosen on six earlier weeks transfers to a seventh with a 2–5
point shrinkage and no change in ordering. It does not establish anything about
online behaviour.

---

## 11. Architecture and code changes

### What the pipeline looks like now

```
                     week w  (label week; every arrow reads only t_dat < w.start)
                        │
  association.py  ──────┤  tb_pairs / tb_style_pairs: cross-slot co-counts, time-decayed
                        │  co-counts (14 d, 56 d half-lives), support, lift, PMI, NPMI, all
                        │  as separate columns at article level and at product_code level
                        │
  protocol.py     ──────┤  eligible_universe(): sold in the 4 weeks BEFORE the cutoff
                        │  queries():  one row per (basket, anchor, target slot) in week w
                        │  restrict_truth(): truth articles outside the catalogue are dropped
                        │                    and counted, not silently scored as misses
                        │
  towers.py       ──────┤  two-tower retrieval, in-batch + logQ, held-out early stopping
                        │  (own process: PyTorch and LightGBM ship clashing OpenMP runtimes)
                        │
  candidates.py   ──────┤  _tb_assoc / _tb_style / _tb_tt / _tb_ttc / _tb_pop, each with
                        │  its own rank, built per (anchor, target slot) KEY, not per query
                        │
  features.py     ──────┤  one row per (query, candidate): compatibility + candidate +
                        │  point-in-time personalization features (95 columns)
                        │
  ranker.py       ──────┤  LightGBM LambdaRank, one group per (basket, anchor, target slot),
                        │  trees chosen by temporal early stopping on the last training week
                        │
  metrics.py      ──────┤  Recall@5/@12, NDCG@12, per slot, tail, recently launched,
                        │  jewellery, returning vs new customer, coverage, novelty,
                        │  list diversity, customer-cluster bootstrap
                        │
  backtest.py     ──────┘  the rolling backtest that drives all of it, with per-week
                           artifacts cached under a configuration hash
```

### Files added

| file | what it does |
| --- | --- |
| `src/ensemble/completion/protocol.py` | leakage-free eligible catalogue (and the Round-1 oracle, for measurement only), rolling folds, truth restriction, customer-level resampling units, run manifest |
| `src/ensemble/completion/association.py` | point-in-time cross-slot mining with raw co-count, time-decayed co-counts, support, lift, PMI and NPMI kept separate; style-level pooling; basket-noise audit |
| `src/ensemble/completion/candidates.py` | the candidate union and its per-source ranks, built at key level; `register_keys` makes the key space sliceable for serving |
| `src/ensemble/completion/features.py` | the 95-column feature matrix, the feature groups the ablation table switches off, and `check_groups` as the schema contract |
| `src/ensemble/completion/towers.py` | two-tower retrieval rebuilt for speed, logQ, retrieval-informed hard negatives, held-out early stopping, pair-similarity export |
| `src/ensemble/completion/ranker.py` | LightGBM LambdaRank fit, temporal early stopping, deterministic top-k, gain importance by feature group |
| `src/ensemble/completion/metrics.py` | the Round-3 metric set and the customer-cluster bootstrap |
| `src/ensemble/completion/fusion.py` | the fixed-fusion baselines (importable without LightGBM or PyTorch) and the serving diversity rules |
| `src/ensemble/completion/pipeline.py` | artifacts shared by the two processes; cache key; the serving key space and its batching |
| `src/ensemble/completion/backtest.py` | the rolling backtest, the ablation pass, the per-fold reports |
| `src/ensemble/completion/protocol_check.py` | scores the fixed-fusion baselines over the leakage-free catalogue and the Round-1 oracle catalogue |
| `src/ensemble/completion/ablate_towers.py` | two-tower negative-sampling ablation, epoch-budget probe, backend-determinism probe |
| `src/ensemble/completion/label_audit.py` | what the Complete-the-Look label actually contains |
| `src/ensemble/completion/serve_round3.py` | the serving table, scored by the learned ranker, with per-row evidence |
| `tests/test_track_b_round3.py` | 41 tests: leakage, fold boundaries, metrics, ablation groups, serving key space, reason chips |
| `scripts/tb3_tables.py` | renders every table in this report from the stored JSON |
| `scripts/tb3_verify_report.py` | asserts every hand-typed figure in this report against those artifacts |
| `configs/experiments/tb3_final.yaml` | same cache key as `default`, ablation table off: the serving-rule folds and the single test-week run |
| `configs/experiments/tb3_smoke.yaml`, `tb3_light.yaml`, `tb3_smoke_light.yaml` | fast end-to-end smoke runs; smaller DuckDB budget for audits that run next to a backtest |

### Files changed

| file | change |
| --- | --- |
| `configs/default.yaml` | the whole `track_b:` block: folds, training weeks, candidate budget per source, ranker and two-tower hyperparameters, `customer_weeks`, `feature_ablations`, `serving_rules` |
| `src/ensemble/api/app.py` | Complete-the-Look reason chips are built from the evidence stored on each served row and cite only what is non-null for that pair; slot ordering counts any association-backed pick; serving columns are detected at query time so a Round-1 store still renders |
| `docs/DECISIONS.md` | D-031 … D-037 |
| `reports/MODEL_CARD_track_b.md` | Round-3 section |

Nothing in the Round-1 Track B path (`completion/data.py`, `models.py`, `run.py`,
`serve_ctl.py`) was modified, so `make track-b` still reproduces the Round-1
numbers exactly.

---

## 12. Leakage and reproducibility checks

### Protocol-level

| check | how | result |
| --- | --- | --- |
| Mining ignores the label week | swap every label-week article for a different one; mine again | identical `tb_pairs` (co, lift, NPMI) — `test_mining_ignores_label_week` |
| The eligible catalogue ignores the label week | same swap | identical catalogue; the swapped article is never eligible — `test_eligible_universe_ignores_label_week` |
| Every feature column ignores the label week | same swap, compare all 95 columns of every candidate row | `assert_frame_equal` passes — `test_features_ignore_label_week` |
| The *volume* of label-week activity cannot move a feature | keep the queried baskets, multiply the other label-week baskets ×10 | identical features for the shared customers, identical mining, identical catalogue — `test_label_week_volume_does_not_move_any_feature` |
| Customer history stops at the cutoff | customers whose only purchases are inside the label week | `c_has_history == 0` for all of them — `test_customer_history_excludes_label_week` |
| Truth stays inside the basket | every truth article must be in that customer's same-day purchases | passes — `test_query_truth_stays_inside_the_basket` |
| Folds never touch the test week | fold list + the test-mode week list | all fold weeks end before the test week starts; the test-mode training weeks are the two before it — `test_folds_end_at_validation_and_never_touch_test`, `test_test_mode_training_weeks_precede_the_test_week` |
| Training weeks precede each fold | fold-by-fold assertion | passes — `test_training_weeks_precede_each_fold` |
| The oracle catalogue is only reachable on purpose | `oracle=True` must differ from the default | passes — `test_oracle_universe_does_see_label_week` |

Every feature-building entry point takes `week` as a **required** argument with no
default, so a call site cannot forget the cutoff (`association.mine`,
`protocol.eligible_universe`, `protocol.queries`, `features.build_context`,
`pipeline.prepare_sql`, `pipeline.price_tiers`).

### Ablation-level

`features.check_groups` asserts that every feature group still removes at least
its expected number of columns from the live schema, and the backtest calls it
before fitting anything. A renamed feature therefore fails the run instead of
producing an ablation row that silently reads "no effect". `subset` additionally
refuses a group that removes nothing.

### Determinism

| component | result |
| --- | --- |
| LightGBM ranker | `deterministic: true`, fixed seed, `force_col_wise`. The ablation run refitted the two shipping models on the same folds as the first run and reproduced the tree counts and the metrics exactly (for example fold 2020-08-05: 131 trees and Recall@12 0.12958 both times). |
| Top-k tie-breaking | `np.lexsort` on (−score, article_id): a tie always resolves to the smaller article id — `test_groups_and_top_k_are_deterministic` |
| Customer subsampling for training weeks | hashed on `customer_idx`, so a basket is never split and the sample is stable across runs — `test_sample_queries_is_deterministic_and_customer_level` |
| Bootstrap | fixed seed (0), 1,000 resamples, customers as the resampling unit |
| Two-tower on CPU | bit-identical across repeated runs with the same seed |
| Two-tower on MPS | **not** bit-identical; the spread is quantified in §9 and tower comparisons are made across seeds, not from single runs |

### Traceability

Every report JSON carries a manifest: git commit, whether the tree was dirty,
config name, a SHA-1 of the `track_b` configuration, and a data fingerprint (row
count, date range, and a checksum over `article_id`). Per-week artifacts are
cached under `data/interim/track_b/<cache-key>/`, where the key is a SHA-1 of
every setting that can change the mined evidence, the catalogue, the candidate
union or the towers — so a configuration change can never pick up a stale
artifact. `feature_ablations` and `serving_rules` are deliberately *not* part of
the key: they change which models are fitted, not the matrices, which is what
makes the ablation pass cheap.


---

## 13. Runtime and memory

Everything ran on one Apple Silicon laptop with 16 GB of RAM. Peak RSS is the
`ru_maxrss` of the driving process.

| run | what it does | wall time | peak RSS |
| --- | --- | ---: | ---: |
| `backtest val 6` (cold cache) | 8 weeks of towers + 7 training matrices + 6 folds × 2 models | 53 min | 5.9 GB |
| **`backtest val 6 ablations`** | 6 folds × **10** models, artifact cache warm | **111 min** | **6.7 GB** |
| `backtest val 6 serving_rules` | 6 folds × 2 models + 4 diversity variants, cache warm | 26 min | 5.5 GB |
| `backtest test` | 1 new week of towers + 1 fold × 2 models + 4 variants | 7.5 min | 5.4 GB |
| `protocol_check 2` | 2 weeks of towers, both catalogues scored | 6 min | — |
| `label_audit 6` | 6 weeks of DuckDB aggregation | 4 min | — |
| `ablate_towers 2 2` | 12 tower trainings + a 12-epoch probe + 4 determinism runs (17 runs, 2,679 s of training) | 55 min | — |
| `serve_round3 towers` | serving-week towers + 157 k keys of pair similarities in 13 batches | 4 min | — |
| `serve_round3 build` | ranker fit + 26.3 M candidate rows scored in 13 batches | 2 min | — |
| `api.build` | SQLite serving store | 6 s | — |
| `make test` | 117 tests | 6 s | — |

Per fold of the ablation run: ~5 s to mine and register a week (towers cached),
~63 s per model fit (10 models), ~442 s to score 19.6 M candidate rows with all
ten models plus the RRF control. The caches on disk are 1.9 GB of per-week tower
artifacts, pair similarities and training matrices, plus 224 MB of serving
retrieval lists.

**What keeps it inside 16 GB:** feature matrices are built in 8,000-query chunks
and downcast to float32/int32; training matrices are written to Parquet and
evicted as soon as no later fold needs them; the towers run in a separate process
that exits; serving works in batches of 2,000 anchors and sub-chunks each batch.
The two-tower rewrite (27 s per epoch instead of 370 s) is what made a rolling
backtest affordable at all.

---

## 14. Problems encountered, and what was done about them

**1. The Round-3 reproduction of the shipped baseline was weaker than the real
Round-1 model.** `shipped_rrf_hybrid` fused the raw association list with the
two-tower list. Round 1 (`completion/models.py::association` called with
`k = 4·12`, then `models.rrf`) first back-fills the association channel from the
slot's 8·12 most popular articles and *then* fuses, so popularity earns
reciprocal-rank credit inside the association list rather than only filling the
tail. Fixed in `fusion.baseline_lists` / `baseline_recs`. The effect is large
enough to matter: on fold 2020-08-05 the baseline moves from 0.0807 to 0.0864
Recall@12, and the headline lift of the learned ranker drops accordingly. The
first six-fold run (`backtest_val.json`) was produced by the pre-fix code and is
retained but superseded; §2 and §4 use the corrected run.

**2. PyTorch and LightGBM crash in one process on macOS** (two OpenMP runtimes).
The two-tower stage runs as its own process and hands results to the ranker
process through the artifact cache; the serving build does the same; the
LightGBM tests run in a fresh interpreter.

**3. Two-tower training was too slow for a rolling backtest.** The Round-1
trainer re-encoded each batch through a Python dict twice per step and drew hard
negatives with one `rng.choice` per positive: ~370 s per epoch over 3.2 M pairs.
Article ids are now mapped to rows once with `np.searchsorted` and every
per-batch tensor is a gather on a precomputed matrix: ~27 s per epoch. Eight
weeks of towers became affordable (~190 s per week including retrieval and pair
similarities).

**4. Later epochs slowed down on MPS.** The Metal allocator keeps cached blocks;
`torch.mps.empty_cache()` after each epoch and after each retrieval batch keeps
epoch time flat.

**5. Memory.** A fold's feature matrix is ~20 M rows × 95 columns. Everything is
built in query-id chunks, downcast to float32/int32, written to a per-week
Parquet cache, and training matrices are evicted as soon as no later fold needs
them. Peak RSS for the full six-fold ablation run stayed around 6 GB on a 16 GB
machine. Serving covers ~10× more keys than a fold, so it is built in batches of
2,000 anchors and each batch is scored in 8,000-query sub-chunks.

**6. Truth articles outside the leakage-free catalogue.** 5.6% of truth articles
have no sale before the cutoff, so they cannot be in a legal catalogue. Scoring
them as guaranteed misses would have buried an unreachable ceiling inside every
number, so `restrict_truth` drops them and reports the count per week. This is
also why Round-1 and Round-3 recall levels are not comparable (§3).

**7. Feature-group ablations can silently become no-ops.** A renamed feature
would make "minus towers" identical to the full model, and the table would read
"no effect". `features.check_groups` now asserts the expected number of removed
columns per group against the live schema, and the backtest calls it before
fitting; `x_tt_type` (a tower × customer interaction) was in fact being left
behind by the `towers` group and is now removed with it.

**8. The serving diversity rules change list length, not just order.** Measuring
them with a fill-back would have hidden that. Both are reported: `shipped` (order
cost only, list stays 12 long) and `shipped_strict` (the hard caps serving
actually applies, which can shorten a module).

**9. A provenance label that fell through to an unsupported chip.** The first
serving build labelled 9,131 rows `other` — candidates retrieved *only* by the
content-only tower, because the label checked `src_two_tower` and not
`src_two_tower_content`. The API then fell through to a generic "Style match"
chip, which is precisely the unsupported claim the evidence gate exists to
prevent. Both flags are now stored and both count as visual compatibility;
`other` is 0 rows.

**10. A reason chip quoting a 193× lift off four co-purchases.** A real served row
had `a_co = 4` and `lift = 193.4`, and the chip said "193.4× more often than
chance". The ratio is arithmetically correct and statistically meaningless. The
chip now quotes the multiplier only at `co ≥ 5` and states the count alone below
that. This was found by reading the rendered chips, not by a test — which is why
the end-to-end chip audit is now part of the verification.

**11. `DataFrame.rank` shadows a column named `rank`.** `out.rank.tolist()` in a
test resolved to the *method*. Test-only, caught immediately, fixed with
`out["rank"]` — recorded because the serving table has a `rank` column and the
same trap is one attribute access away in any code that touches it.

## 15. Rejected approaches, and why

| approach | why it was not kept |
| --- | --- |
| Shipping `lgbm_personalized` in the precomputed Complete-the-Look table | The table is keyed by (anchor, target slot) and has to answer for anonymous visitors on any product page. A row per customer is not precomputable (26 k anchors × 6 slots × 1.4 M customers), and request-time scoring needs an online feature store and a model server that this local SQLite demo does not have. The gain it would buy is measured and reported instead of being claimed (§6, §9.2). |
| Round-1 popularity-sampled negatives and (product type, price tier) hard negatives | D-016 already found no measurable gain; they were removed rather than carried forward, and replaced by retrieval-informed hard negatives, which are judged on §9's evidence. |
| Keeping the Round-1 target-week catalogue as the primary universe | Eligibility was not knowable at prediction time, which makes "item cold-start recall" an artefact. It is retained only as an explicit `oracle=True` comparison (§3). |
| A constructed "pre-launch availability" proxy for true item cold start | The dataset has no inventory or launch feed. Any proxy would be invented, not measured, so the metric is replaced by *recently launched* (first observed sale within 28 days of the cutoff) and the limitation is stated instead. |
| A single global association score (NPMI only, or raw co-count only) | Round 1 had to choose one. Raw co-count, decayed co-counts, support, lift, PMI and NPMI are now separate features and the ranker picks; §8 shows what dropping each group costs. |
| Treating (anchor, slot) queries as independent for confidence intervals | One basket produces several queries and one customer several baskets. All intervals resample **customers** with all of their queries; a test asserts the clustered interval is more than 3× wider than the naive one on correlated data. |

---

## 16. Limitations

**Everything here is offline.** Recall@12 and NDCG@12 on held-out baskets are
proxies for a surface nobody has interacted with. A real decision to ship needs
an online A/B test with engagement and add-to-basket metrics; nothing in this
report establishes that.

**The label is co-purchase, not co-wear.** A Complete-the-Look truth item is
another article the same customer bought on the same day. The basket audit says
what that proxy costs: 15.6–17.2% of mined baskets contain a repeated quantity,
14.3–15.4% contain more than one colourway of one style, and 2.7–2.9% span four
or more slots — all signs of a shopping trip rather than an outfit. The optional
Polyvore experiment (M8) is the only way in this project to separate the two.

**The test week is a confirmation, not a fresh holdout.** It was scored once in
Round 1 and those numbers have been read. Model selection in this round used the
six rolling validation folds only, and the test week was touched once, after
selection, with the ablation table switched off. It still cannot be presented as
an untouched holdout, and it is not.

**Catalogue eligibility is a sales proxy.** With no inventory or launch feed,
"could a merchandiser have offered this article in this week" is approximated by
"it sold at least once in the four weeks before the cutoff". That is why true
item cold start is not measurable here and is replaced by a recently-launched
segment; it also means 5.6% of truth articles are excluded from the primary
protocol (§3).

**The relative confidence intervals are an approximation.** The customer-cluster
bootstrap resamples the paired *difference*; the relative interval divides that
interval by the observed baseline mean rather than resampling the denominator
too. For gains this far from zero the difference is cosmetic, but the relative
bounds are slightly too narrow. Resampling the ratio is a one-line change and is
listed as a next step.

**The between-fold spread is not a standard error.** Consecutive folds share most
of their 16-week mining window and many of their customers, so the per-fold
numbers are positively correlated. The "±" in the per-fold tables is the
between-fold standard deviation — a description of how the six weeks differ, not
an uncertainty on the mean. The pooled customer-cluster bootstrap is the
uncertainty statement.

**`rrf_all_sources_union` is a reach control, not a tuned baseline.** It sums
reciprocal ranks over the union's rank columns, and the slot-popularity rank is
defined for *every* candidate, so the control carries more popularity signal than
a hand-tuned fixed fusion would. Its purpose is to show that the learned
ranker's gain does not come from a larger candidate pool, and it does that; it is
not evidence about the best achievable fixed fusion.

**The shipped serving model is not the best model measured.** Serving ships
`lgbm_compatibility`. The personalized variant is better on every fold but needs
an online feature store (§15).

**The candidate union is the ceiling, and it is low.** Union recall is 0.4285: more
than half of all truth pairs are not in the 189 candidates any system may choose
from, so no amount of ranking can reach them. The learned ranker converts 36.7% of
that ceiling into a top-12 hit. Raising the ceiling is a separate problem from
ranking it, and it was not attempted this round.

**The two-tower conclusions rest on two weeks and two seeds.** The negative-sampling
table is four runs per configuration, the epoch-budget probe is one week and one
seed, and MPS is not bit-reproducible. The logQ result (−26.5%) is far outside that
noise; the hard-negative result (+0.94%) is inside it, which is exactly why hard
negatives are reported as "not detectable at this sample size" rather than as "no
effect".

**MPS is not bit-reproducible.** Tower comparisons are therefore made across
seeds and the seed spread is reported; single-run tower differences smaller than
that spread carry no information.

**Jewellery is still weak** in absolute terms, even though it improves a lot in
relative terms. Metadata and a global image vector do not separate
near-identical accessories; region-level visual features are the obvious next
attempt and were out of scope here.

**Not tested on rolling folds:** the basket-noise filter (`basket_filter`) and
alternative time-decay half-lives. Both are part of the cache key, so each
variant means rebuilding eight weeks of tower artifacts and training matrices
(~2 h per variant). The audit that motivates the filter is reported, the decay
features are kept as features and their value is measured by the `decay`
ablation, but the filter itself and other half-lives remain untested. They are
items 4 and 5 of §18.

---

## 17. Reproduction

All commands run from the repository root with the project venv (Python 3.11).
The dataset must already be ingested (`make ingest`).

```bash
# 0. environment (once)
make setup

# 1. the frozen Round-1 baseline is already stored:
#    reports/track_b_round3/baseline_frozen.json

# 2. how much the Round-1 catalogue definition was worth  (~25 min)
PYTHONPATH=src .venv/bin/python -m ensemble.completion.protocol_check 2

# 3. what the label contains  (~4 min)
PYTHONPATH=src .venv/bin/python -m ensemble.completion.label_audit 6

# 4. the six-fold rolling backtest with the full feature-group ablation table
#    (~2 h; the first run also trains eight weeks of towers, ~25 min of that)
PYTHONPATH=src .venv/bin/python -m ensemble.completion.backtest val 6 ablations

# 5. the serving diversity rules on the same folds  (~35 min, reuses the cache)
ENSEMBLE_CONFIG=experiments/tb3_final PYTHONPATH=src .venv/bin/python \
    -m ensemble.completion.backtest val 6 serving_rules

# 6. two-tower negatives, epoch budget and backend determinism  (~75 min)
PYTHONPATH=src .venv/bin/python -m ensemble.completion.ablate_towers 2 2

# 7. THE SINGLE FINAL TEST-WEEK EVALUATION — run once, after selection  (~12 min)
ENSEMBLE_CONFIG=experiments/tb3_final PYTHONPATH=src .venv/bin/python \
    -m ensemble.completion.backtest test

# 8. serving artifacts and store  (~45 min)
PYTHONPATH=src .venv/bin/python -m ensemble.completion.serve_round3 build
PYTHONPATH=src .venv/bin/python -m ensemble.api.build

# 9. tests
make test

# 10. regenerate every table in this report from the stored JSON
.venv/bin/python scripts/tb3_tables.py backtest_val_ablations \
    backtest_val_serving_rules backtest_test

# 11. check every hand-typed number in this report against those artifacts
.venv/bin/python scripts/tb3_verify_report.py
```

`scripts/tb3_verify_report.py` re-reads the JSON and asserts each of the 56
headline figures quoted in this report, exiting non-zero on the first mismatch.
It passes at the commit this report was written at.

A fast end-to-end smoke run of the whole backtest (one fold, one tower epoch,
small budgets, ~3 min) is:

```bash
ENSEMBLE_CONFIG=experiments/tb3_smoke PYTHONPATH=src .venv/bin/python \
    -m ensemble.completion.backtest val 1 smoke
```

### Artifacts

| file | what |
| --- | --- |
| `reports/track_b_round3/baseline_frozen.json` | the Round-1 Track B results as observed before this round, with their protocol caveats |
| `reports/track_b_round3/protocol_leak.json` | leakage-free catalogue vs the Round-1 oracle catalogue, per week and per system |
| `reports/track_b_round3/label_audit.json` | label composition and basket-noise proxies per week |
| `reports/track_b_round3/backtest_val.json` + `.log` | the first six-fold run (superseded; see §2) |
| `reports/track_b_round3/backtest_val_ablations.json` + `.log` | **the authoritative validation result**: all systems, the feature-group ablation table, per-fold and pooled bootstrap, feature importance |
| `reports/track_b_round3/backtest_val_serving_rules.json` + `.log` | the serving diversity rules on the same folds |
| `reports/track_b_round3/ablation_towers.json` + `.log` | two-tower negatives, epoch budget, backend determinism |
| `reports/track_b_round3/backtest_test.json` + `.log` | the single final test-week evaluation |
| `reports/track_b_round3/serving.json` | what the serving build produced, with provenance mix and feature importance |
| `data/interim/track_b/<cache-key>/` | per-week tower artifacts, pair similarities and training matrices (gitignored) |
| `data/interim/track_b/per_query/` | per-query Recall@5/@12 and NDCG@12 for every system and fold, for re-bootstrapping without rerunning (gitignored) |
| `mlruns/` | MLflow run per backtest, one metric set per system (gitignored) |

---

## 18. What to do next, in order of expected value per hour

1. **A candidate-budget frontier for the union.** Union recall is 0.4285 and the
   ranker converts 36.7% of it; more than half of all truth pairs are simply not
   reachable. Ranking is now the strong part of the funnel and retrieval is the
   weak one. Track A already has the method (recall-vs-candidates Pareto frontier,
   D-027) and it transfers directly: sweep the per-source budgets, plot union
   recall against candidates per query, and pick an operating point. Cost: the
   budget is part of the cache key, so each point is a full rebuild (~2 h), but the
   first three points would say whether the ceiling is cheap or expensive to lift.
2. **An online feature store for personalized Complete the Look.** The prize is
   measured: +15.11% Recall@12 on returning customers, who are 93.4% of queries.
   The work is request-time point-in-time feature serving plus a model server, not
   modelling.
3. **Resample the ratio in the bootstrap** so the relative intervals are exact
   rather than divided by a fixed denominator. Minutes of work; the per-query
   scores are already cached under `data/interim/track_b/per_query/`, so no model
   has to be refitted.
4. **Test the basket-noise filter on rolling folds.** The audit says 15.6–17.2% of
   mined baskets contain a repeated quantity and 14.3–15.4% more than one
   colourway of one style. The implemented hook (`track_b.basket_filter`) is a
   row-level predicate on `tb_bk`, while those proxies are basket-level
   properties, so the filter has to be expressed basket-level first. It is part of
   the cache key, so each variant costs a full artifact rebuild (~2 h).
5. **Test other time-decay half-lives.** Currently 14 and 56 days, both exposed as
   features. The `decay` ablation says what the pair is worth; it does not say the
   pair is optimal. Same cost structure as (4), and the ablation suggests the
   expected return is small.
6. **Region-level visual features for jewellery and accessories.** A single global
   FashionCLIP vector per article cannot separate near-identical accessories, and
   jewellery is still the weakest segment in absolute terms (0.0275). A crop-level
   or multi-region embedding is the next honest attempt, and the M7a DeepFashion2
   adapters already produce crops. Note the ablation's warning: removing the single
   CLIP feature costs 0.66%, so visual signal has to get much better to matter.
7. **Co-wear labels (M8, Polyvore).** The only way to separate "bought together"
   from "worn together" in this project.

## 19. Repository status

### Commits made in this round (local only)

```
bcb1f04 feat(serving): rebuild Complete the Look with the learned ranker; honest provenance
dbd226e docs: Track B model card and README for Round 3
86e0069 docs(report): Round-3 upgrade report; single final test-week evaluation
5978fbe docs(adr): D-031..D-037 for the Round-3 Track B protocol, fusion and serving
46b70d4 feat(track-b): six-fold ablation results; diversity caps become a re-ordering
bd7c6ee docs(glossary): candidate fusion and point-in-time protocol terms (D-008)
26b4a7e fix(serving): keep the Round-1 Complete-the-Look table when the Round-3 build replaces it
010668c test: evidence rows, label-audit point-in-time history; report table renderer
ce3ab80 feat(track-b): label-composition audit; strict serving caps; batch sub-chunking
fd3de84 feat(track-b): Round-3 serving path, evidence-gated reasons, diversity trade-off
4868d5b feat(track-b): ablation harness, tower/protocol probes, stronger leakage tests
24b57da feat(track-b): leakage-free rolling protocol, candidate union, learned fusion
```

Branch: `phase2-retrieval-ranking-upgrade`. `24b57da` was already present at the
start of this session (it is the Phase 1–4 scaffolding, listed last above); every
commit above it is this round's work.

### Nothing left the machine

Nothing was pushed, merged, rebased onto `main`, or opened as a pull request. No
Kaggle submission was made, no artifact was published, no external service was
contacted, and no paid API was called. Every number in this report came from a
local run against the local DuckDB database.

### Pre-existing files that were left alone

The working tree contains Finder/iCloud sync duplicates (`* 2.py`, `* 2.json`,
`docs/HANDOFF 2.md`). None of them was deleted, renamed, staged or modified; the
pytest configuration already ignores `* 2.py` so they stay out of the suite.
`docs/CLAUDECODE_TRACK_B_ROUND3_PLAN.md` and `docs/HANDOFF.md` are unchanged
apart from the plan file being committed as-is.
