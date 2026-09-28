# Ensemble post-MVP improvements: offline evaluation readout

**Date:** 2026-09-28 · **Scope:** improvement plan after the MVP: rolling backtest, M7a
(FashionCLIP, visual search, DeepFashion2 adaptation), M7b (conversational assistant) ·
**Decisions:** D-017 … D-024 · **Paid LLM spend:** $4.12 of an $8 budget guard

All results are offline. Launch decisions would need online A/B tests.

---

## 1. Summary

1. **The MVP ranker's gain is stable across weeks.** 4-week backtest: MAP@12 0.0332 ± 0.0032
   vs baseline 0.0232 ± 0.0020, +43% ± 5% relative lift. A new-customer fallback does not help
   (better in 1 of 4 weeks), so it is not adopted (D-017).
2. **Image search now works through street↔shop domain adaptation.**
   - Adapters trained on the full DeepFashion2 train split raise exact-match Recall@1 on
     DeepFashion2 validation from 0.378 to 0.585 (0.607 with both views).
   - On our 79 outfit photos, image-only Precision@5 rises from 0.36 to 0.48 (D-020, D-021).
3. **Complete the Look improves with image features:** +62.0% vs +58.6% relative lift on the
   validation week and +55.6% vs +51.6% on the test week, and content-only cold-start ability more
   than doubles (+17% → +39%) (D-022).
4. **Track A does not benefit from image similarity** (+0.8% MAP@12, within noise), so it stays off.
   Customers rarely buy look-alikes of past purchases (D-023).
5. **The assistant is grounded and accurate on both back ends:** 100% tool and argument accuracy,
   0% hallucination. Claude Opus 5 answers better and faster; the local Qwen model is free (D-024).

## 2. Track A: rolling backtest (D-017)

| label week | baseline MAP@12 | ranker MAP@12 | relative lift |
| --- | ---: | ---: | ---: |
| 2020-08-19 | 0.0223 | 0.0306 | +37% |
| 2020-08-26 | 0.0208 | 0.0303 | +46% |
| 2020-09-02 | 0.0243 | 0.0360 | +48% |
| 2020-09-09 (validation) | 0.0252 | 0.0358 | +42% |
| **mean ± std** | **0.0232 ± 0.0020** | **0.0332 ± 0.0032** | **+43% ± 5%** |

**New customers:** ranker 0.0083 ± 0.0024 vs age-band popularity 0.0085 ± 0.0005. No
fallback is adopted.

**FashionCLIP in Track A** (D-023):
- validation MAP@12 0.03573 vs 0.03546;
- the visual channel adds 0.27% unique recall for 14.6 extra candidates per customer.

Kept off.

## 3. Image search (M7a)

### 3.1 DeepFashion2 consumer-to-shop, exact identity match (D-020)

Validation split: 12,117 user queries, category-filtered shop gallery. Adapters are trained on
169,584 items (18,159 identities).

| | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| FashionCLIP, crop | 0.378 | 0.581 | 0.670 |
| FashionCLIP, background removed | 0.381 | 0.562 | 0.643 |
| **adapter, crop** | **0.585** | **0.760** | **0.827** |
| adapter, background removed | 0.545 | 0.722 | 0.793 |
| adapters combined | 0.607 | 0.772 | 0.835 |

**Data-scaling curve** (background-removed adapter, Recall@1):

| share of training identities | Recall@1 |
| --- | ---: |
| 10% | 0.482 |
| 25% | 0.494 |
| 50% | 0.526 |
| 100% | 0.545 |

The curve is still rising at about +2 points per doubling, so more paired data would help, with
diminishing returns.

### 3.2 Our outfit photos (D-019, D-021)

79 photos (36 Commons, 43 owner) and 374 garments detected by Qwen2.5-VL. Top-5 results are judged
by Claude Haiku 4.5, which sees pixels and the category only.

| query mode | Precision@5 | Precision@1 |
| --- | ---: | ---: |
| whole photo | 0.332 | 0.366 |
| crop | 0.357 | 0.356 |
| crop, background removed (rembg) | 0.269 | 0.302 |
| **crop + DeepFashion2 adapter** *(standalone image search)* | **0.477** | **0.540** |
| crop, rembg + adapter | 0.412 | 0.428 |
| text (VLM description) | 0.642 | 0.714 |
| crop + text (equal weights) | 0.547 | 0.591 |
| text + 0.3 crop | 0.600 | 0.652 |
| **text retrieve → image rerank** *(snap flow)* | **0.635** | **0.711** |

- **Why raw crops fail.** FashionCLIP matches the close-up framing of fabric detail shots.
  Domain adaptation is what fixes it, and it is also what Pailitao, Pinterest and GrokNet rely on.
- **Background removal hurts** on both datasets.
- **Hardest category:** jewellery (0.46–0.48 at best).
- **Text in photos does not hurt** (0.65 with text vs 0.63 without).
- **Judge reliability (D-025).** The Haiku batch judge agrees with the owner's 100 human labels
  on only 58% of judgements (κ 0.16). The earlier 71% came from a Claude reviewer, not a human,
  and overstated reliability. The ranking of modes holds under human labels:
  - text retrieve → image rerank: 0.54 [0.34, 0.74];
  - crop + adapter: 0.34 [0.18, 0.50].

  Read the absolute judge numbers above as relative. Opus 5 with the written guideline agrees best:
  0.75 (κ 0.48) on the calibration round and **0.81 (κ 0.57) on the held-out round 2**, so it is
  the default judge from now on. Round 2 human P@5 (jewellery- and bag-heavy): text → image rerank
  0.44, crop + adapter 0.28, the same ranking.
- **95% intervals** (photo-level bootstrap) for judged Precision@5:
  - crop 0.357 [0.33, 0.39];
  - crop + adapter 0.477 [0.45, 0.51];
  - text 0.642 [0.61, 0.67];
  - text → image rerank 0.635 [0.60, 0.67].

**Garment detection** (30 photos):

| | Claude Opus 5 | Qwen2.5-VL |
| --- | ---: | ---: |
| garments found | 195 | 154 |
| valid boxes | 85% | 80% |
| seconds per photo | 6 | 42 |

Slot agreement (Jaccard) is 0.83.

## 4. Complete the Look with FashionCLIP (D-022)

| model | validation lift | test lift |
| --- | ---: | ---: |
| two-tower, without CLIP | +46.6% | +42.6% |
| two-tower, with CLIP | +50.8% | +48.6% |
| two-tower content-only, without CLIP | +17.3% | +15.2% |
| two-tower content-only, with CLIP | +38.8% | +37.7% |
| hybrid RRF, without CLIP | +58.6% | +51.6% |
| **hybrid RRF, with CLIP (shipped)** | **+62.0%** | **+55.6%** |

**Style-level backoff:** raises catalog coverage (26% vs 22%) and tail recall, at equal recall.
Not selected.

**Negative-sampling ablation** (3 seeds):

| recipe | relative lift |
| --- | ---: |
| in-batch only | −12.2% |
| + logQ correction | +32.7% |
| + popularity-based negatives | +31.8% |
| + hard negatives | +31.9% |

## 5. Conversational assistant (M7b, D-024)

| | Qwen3-VL 8B instruct (local, default) | Claude Opus 5 |
| --- | ---: | ---: |
| tool / argument accuracy | 100% / 100% | 100% / 100% |
| constraint pass | 91.7% | 91.7% |
| hallucination rate | 0% | 0% |
| turns with products | 83% | 92% |
| latency median / p90 | 17 s / 100 s | 7.5 s / 18 s |
| judge helpfulness / faithfulness | 3.54 / 3.54 | 4.46 / 4.25 |
| cost | $0 | ≈ $0.02 per turn |

The judge is Claude, so self-preference bias is possible.

**Problems found and fixed on the way:**
- the default Qwen3-VL tag ignores `think=false`;
- the model added unnecessary filters (tools now relax them);
- citations in free-form formats (the parser now accepts any grounded id);
- fabricated tool arguments (the anchor-id guard).

## 6. Product

The web app (`make serve`, http://localhost:8010) now has:
- **Style assistant:** chat with photo upload; product cards come only from tools, and ungrounded
  ids are removed.
- **Visual search:** drag a box, pick a category, find similar (crop + adapter, no LLM).
- **Analyse whole outfit:** detection → text retrieve and image rerank → Complete the Look for the
  missing slots.
- **Ask about this look:** hands a visual-search result over to the chat.
- **Label:** a blind human-labelling page for judge validation.

## 7. Problems found and fixed

| problem | fix |
| --- | --- |
| Qwen2.5-VL boxes overflowed the image | resize to Qwen's internal grid (≈0.9 MP, sides multiples of 28) before sending |
| the judge saw the text description (evaluation leak) | the judge sees pixels and category only; the leaky run is kept for the record |
| the reviewer spot check was mislabelled "human" | renamed; labeller recorded as Claude |
| CLIP ranker features needed a 20 GB matrix (OOM) | per-customer gather |
| DeepFashion2 embedding decoded each image twice | one pass for both views |
| the platform permission checker blocked tools for hours | handed off to Codex, which built the DeepFashion2 pipeline and fixed two bugs |

## 8. Next steps

1. Human labels are done: 2 rounds, 200 judgements; the judge choice is confirmed on held-out data
   (D-025).
2. **Exact-match benchmark from the owner's street↔product pairs:** 3 composite images so far;
   about 30 more gives about 150 pairs, with jewellery close-ups prioritised.
3. **Jewellery-specific adaptation:** DeepFashion2 has no jewellery. A jewellery pair set would
   extend the adapters to jewellery.
4. **Online:**
   - A/B test the assistant back end (cost vs answer quality);
   - A/B test the new-customer fallback (variance vs mean).

## 9. Reproducing

```bash
make backtest          # 4-week Track A backtest
make clip              # FashionCLIP catalogue embeddings
make df2               # DeepFashion2 manifest, embeddings, adapters, evaluation (needs data/raw/deepfashion2)
make track-b           # validation (model selection), test once, serving artifacts
make visual-eval       # detection, matching, judging, report (paid judge: ~$0.6 per 4 modes)
make assistant-eval    # local assistant eval (free); ENSEMBLE_LLM=claude for the paid comparison
make serving serve     # rebuild the store and run the app on :8010
```
