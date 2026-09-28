# Handoff: improvement plan (post-MVP) — state as of 2026-09-28

Written by Claude Code for the next agent (Codex). Claude stopped because its tool-permission
checker kept failing (a platform issue, not a project issue). Read `CLAUDE.md` (working rules,
industry conventions, autonomous-until-done workflow) and `docs/DESIGN.md`, `docs/DECISIONS.md`,
`docs/GLOSSARY.md` first. If you read `AGENTS.md` rather than `CLAUDE.md`, copy it across.

## 1. Repository state
- `main` holds the MVP (M0–M6), merged and committed. MVP readout: `reports/MVP_REPORT.md`.
  Kaggle late submission: private MAP@12 0.03189.
- **Branch `improve-backtest` holds ALL post-MVP work, UNCOMMITTED.** Commit it first, in logical
  commits (backtest / CLIP / Track B CLIP+backoff / visual search / LLM layer / assistant / UI),
  then merge `--no-ff` with a PR-style message (see earlier merge commits for the format).
- Environment: `.venv` (Python 3.11, uv). Always run with `PYTHONPATH=src` (macOS hides the
  editable-install `.pth`). `brew install libomp` is already done.
- **Never import LightGBM and PyTorch in the same process** (two OpenMP runtimes deadlock on macOS).
  Track A uses numpy only for CLIP maths; PyTorch lives in `completion/`, `vision/`, `assistant/`,
  and `api/app.py` (loaded lazily).
- Ollama is running locally (`ollama serve`) with `qwen3-vl:8b-instruct` (assistant: vision + tools)
  and `qwen2.5vl:7b` (garment detection). `qwen3-vl:8b` (thinking variant) is pulled but unused:
  it ignores `think=false` and is slow.
- The Anthropic API key is in `.env` (gitignored). **Budget guard: $8 cumulative**, tracked in
  `reports/llm_spend.json` (currently **$1.34**, all spent on visual judging). The owner prepaid $10
  with no auto-reload.
- Port 8000 belongs to another app of the owner. Serve on 8010 (`make serve`).

## 2. Done in this phase
| step | status | outputs |
| --- | --- | --- |
| Rolling backtest (4 weeks before the test week) | done | `src/ensemble/evaluation/backtest.py`, `reports/backtest_track_a.json` |
| FashionCLIP catalogue embeddings (105,100 images) | done | `src/ensemble/vision/clip.py`, `data/processed/clip/*.npy` |
| Visual search + "snap your outfit" pipeline | done | `src/ensemble/vision/outfit.py` |
| Garment detection with Qwen2.5-VL, 79 photos | done | `reports/m7a/detections_ollama.json` (374 garments, 80% valid boxes) |
| Visual search eval: 5 query modes, Haiku judge via Batch API | done | `src/ensemble/vision/eval_visual_search.py`, `reports/m7a_visual_search.json` |
| LLM abstraction layer (Ollama / Claude) with budget guard | done | `src/ensemble/llm/client.py` |
| Assistant: tools, agent loop, grounding, eval harness | done | `src/ensemble/assistant/{tools,agent,eval}.py` |
| Assistant eval with local Qwen3-VL instruct | done | `reports/m7b_assistant_ollama.json` |
| API endpoints + "Style assistant" page (chat + visual search) | written, not browser-tested | `src/ensemble/api/app.py`, `static/index.html` |
| Track A visual channel + CLIP ranker features | written, flags OFF, **not evaluated** | `src/ensemble/candidates/visual.py`; `configs/default.yaml`: `retrieval.use_visual`, `features.use_clip` |
| Track B CLIP towers + hierarchical backoff + multi-seed ablation | written, **not run** | `src/ensemble/completion/{models,run,data,serve_ctl}.py` |
| Tests | `tests/test_assistant.py` added (5 pass) | run `make test` |

### Key results so far
- **Backtest:** ranker MAP@12 0.0332 ± 0.0032 vs baseline 0.0232 ± 0.0020; relative lift +43% ± 5%
  (every week between +37% and +48%). New-customer fallback (age-band popularity) scored
  0.0085 ± 0.0005 vs the ranker's 0.0083 ± 0.0024, and was better in only 1 of 4 weeks → **not adopted**.
- **Visual search Precision@5** (the judge sees pixels + category only):
  photo 0.33 · crop 0.36 · **text 0.64** · crop+text 0.55 · text+0.3crop 0.60.
  Crops fail because FashionCLIP matches close-up *framing* (fabric detail shots in the catalogue):
  the street-to-shop domain gap. Jewellery is hardest (text mode 0.46). Text visible in the photo
  does not hurt. A first judging run leaked the VLM's description to the judge (favouring "text");
  it is kept as `reports/m7a/*_v1_description_leak.json`.
- **Judge agreement:** 71% against a spot check of 120 judgements. **Correction needed:** that spot
  check was labelled by Claude, not a human. The file is misnamed `reports/m7a/human_spot_check.json`.
  Rename it to `reviewer_spot_check_claude.json`, rename `judge_vs_human` → `judge_vs_reviewer` in
  `eval_visual_search.py::stage_report`, and state this in the reports.
- **Assistant (local Qwen3-VL 8B instruct, 23 turns):** tool accuracy 100%, argument accuracy 100%,
  constraint pass 91%, hallucination 0%, turns with products 87%, median latency 28 s.

## 3. Open problems
1. **Pending one-line change:** `snap(..., mode="crop+text")` in `src/ensemble/vision/outfit.py`
   should default to the best measured mode. Decide after item 2.
2. **Image search must stay a first-class feature** (the owner's requirement). Implement and evaluate:
   - `crop_nobg`: remove the crop's background (e.g. `rembg`) and paste it on white before CLIP, to
     mimic product shots. Use this mode for the standalone, LLM-free visual search.
   - `text_then_image_rerank`: retrieve the top 50 by text, then rerank by image similarity (the
     industry multi-stage pattern). Candidate default for the snap flow.
   Add both to `MODES`, then run `match <modes>`, `judge` (incremental, ~$0.15 per mode), and `report`.
3. **Owner-supplied street↔product pairs:** 3 new images were added to `data/raw/outfit_photos/`
   (check the newest files; Claude could not list the folder). More will follow as
   `pair_XXX_street.jpg` / `pair_XXX_product.jpg` plus `pairs.csv`. Use them as:
   (a) an **exact-match benchmark**: inject the product images into the gallery and report
   Recall@1/5/10 of the true product for each mode (objective, no LLM judge);
   (b) once ≥300 pairs exist, train a small **adapter** (a linear/MLP projection on frozen FashionCLIP,
   InfoNCE loss, held-out pairs for evaluation). Do not fine-tune CLIP itself on so few pairs.
   Owner photos are local-only (not publishable); record them in `MANIFEST.csv`.
4. **Human gold labels:** add a small labelling page (one garment + 5 results, relevant yes/no) so the
   owner can label 60–100 judgements, and report judge–human agreement from those labels.
5. Qwen sometimes adds unnecessary filters. Tools now relax filters when a search comes back empty
   (`tools.py`). Two turns still violate constraints; inspect `reports/m7b_assistant_ollama.json`.

## 4. Next steps, in order
1. Commit the branch (section 1). Run `make test`.
2. Open problems 1, 2 and 3(a); record the decision in a new ADR.
3. **Track B validation:** `python -m ensemble.completion.run val` (≈1 h: two-tower with and without
   CLIP, backoff, hybrid grid, negative-sampling ablation × 3 seeds), then `test`, then `serve`.
   The selection is written to `reports/m4_selected.json`.
4. **Track A validation with CLIP:** run `python -m ensemble.ranking.train val` with
   `retrieval.use_visual: true` and `features.use_clip: true`, and compare against 0.03546. Enable them
   in the config only if they help; then run `test` once, and re-run the 4-week backtest for the chosen
   config if time allows.
5. Rebuild serving: `make submission` (only if Track A changed), `python -m ensemble.completion.serve_ctl`,
   `make serving`. Browser-test the Style assistant page (headless Chrome screenshots work).
6. **Claude comparison (paid, ≈$4):** `python -m ensemble.assistant.eval claude --judge` (Claude Opus 5 as
   the assistant, rubric judge; note self-preference bias), and
   `python -m ensemble.vision.eval_visual_search detect claude 30` (detection agreement).
   Stay under the budget guard.
7. Write the ADRs (drafts below). `docs/GLOSSARY.md` already has the new terms. Add a
   "Post-MVP improvements" readout to `reports/MVP_REPORT.md` and update the model cards. Commit and merge.

## 5. ADR drafts to add to docs/DECISIONS.md (Claude could not write them)
- **D-017 Rolling backtest; no new-customer fallback:** numbers as in section 2.
- **D-018 FashionCLIP; local LLMs for development, Claude for comparison:**
  - Qwen3-VL 8B instruct for the assistant. Ollama's Qwen2.5-VL has no tool support, and the default
    Qwen3-VL tag ignores think=false.
  - Qwen2.5-VL 7B for garment detection. Its boxes are only correct if the image is first resized the
    way Qwen resizes internally (area ≈0.9 MP, sides multiples of 28); see `outfit._resize_bytes`.
  - Claude Opus 5 for the paid comparison; Haiku 4.5 + Batch API for judging.
  - Budget guard $8.
- **D-019 Visual search evaluation protocol:** 79 photos (36 from Wikimedia Commons, publishable;
  43 from the owner, local only), five query modes, and a Haiku judge that sees pixels + category only
  (the leak correction). Metric: Precision@5. Reviewer spot check by Claude: 71% agreement.
  Human gold labels are pending.

## 6. Watch-outs learned the hard way
- `pgrep -f "<text>"` in a wait loop matches the loop's own shell; match `python -m ensemble...` instead.
- pandas 3: `.values` can be read-only; use `.to_numpy(copy=True)` before in-place operations.
- transformers 5: `get_image_features` returns an output object; use `.pooler_output`.
- MLflow 3 refuses the file store; the SQLite backend is configured in `tracking.py`.
- MPS training is not bit-reproducible; judge ablations on multi-seed means.
- Running Ollama models while LightGBM trains slows both a lot; avoid overlapping heavy jobs.
