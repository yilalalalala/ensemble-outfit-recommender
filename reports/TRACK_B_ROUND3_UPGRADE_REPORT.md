# Track B Round 3 — Complete the Look upgrade

**Offline evaluation readout.** 2026-10-03 · branch `phase2-retrieval-ranking-upgrade` ·
nothing pushed, merged, published or submitted (§18).

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
filter costs a full artifact rebuild (§15, §17).

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
resampling the denominator, so the relative bounds are slightly too narrow (§15).

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
co-count ≥ 1), `visual_compatibility` (retrieved by a tower), `popular_in_slot`
(popularity fallback), `other`. The API builds chips from the row's own columns, so
a visually retrieved pick can no longer borrow a co-purchase claim and a popularity
fallback says so (D-035, extending D-030 to Track B). Columns are detected at query
time, so a Round-1 serving store still renders; the Round-1 table is also kept as
`complete_the_look_round1.parquet` when the Round-3 build replaces it.

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
