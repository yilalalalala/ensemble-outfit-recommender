# Claude Code Plan — Publish the Real Ensemble UI

## Objective

Replace the current simplified GitHub Pages portfolio with a public, static-demo build of the **same
Shopper / Style Assistant / DS Studio interface served by `make serve` at `http://localhost:8010`**.
The public URL must remain:

`https://yilalalalala.github.io/ensemble-outfit-recommender/`

The public build must look and behave like the real application, while using frozen, verified local API
responses because GitHub Pages cannot run FastAPI, SQLite, model inference, Ollama, or Claude.

Work autonomously through every definition-of-done item below. Inspect the current repository, the real
local app, existing tests, reports, workflows, and current Pages implementation before changing code.
Fix ordinary failures yourself. Do not change model training or evaluation results. Do not push, merge,
publish, or delete anything; leave a clean, verified commit on the current branch and write the report.

Preserve all user-owned untracked `* 2.*` files and `reports/track_a_retrain/logs/` exactly. Never stage them.

## Product decision

Do **not** maintain a second, visually different frontend. The real frontend under
`src/ensemble/api/static/` is the source of truth. The public demo should reuse that HTML, CSS, JavaScript,
components, routes, typography, product cards, cart, responsive behavior, shopper/studio switch, and copy.

GitHub Pages is static, so add a narrowly scoped **public demo data adapter** that supplies frozen outputs
captured from the current serving bundle. Do not pretend that browser-side clicks run the recommendation
models. Clearly describe these as stored outputs from the verified local application, but keep technical
provenance in DS Studio rather than consumer pages.

## 1. Architecture and implementation

1. Inspect all frontend modules in `src/ensemble/api/static/`, all API routes in
   `src/ensemble/api/app.py`, and the existing `portfolio/` site.
2. Establish one-source-of-truth reuse. Prefer a deterministic build/sync script that assembles
   `portfolio/` from the real frontend rather than manually duplicating files. The generated public build
   must use relative, repository-project-safe URLs so it works below
   `/ensemble-outfit-recommender/` on GitHub Pages.
3. Add an explicit demo-mode boundary, selected by a checked-in config or the public build—not by brittle
   hostname guesses alone. The production FastAPI behavior must remain unchanged.
4. In demo mode, route read requests to checked-in fixtures generated from the real serving store. No
   network request may target `/api` on github.io. Mutating shopper actions such as cart, swatches, profile
   choice, and “Not for me” should work locally in the browser. Event logging may be a documented no-op.
5. Keep the public code dependency-light and compatible with GitHub Pages. Do not publish the full SQLite
   database, model files, secrets, private photos, or raw datasets.

## 2. Public routes and behaviors

The following must render using the same components and styling as localhost:

- `#/` — Discover shopper page, real catalogue photos, family deduplication, colour swatches, prices,
  persistent cart, and “Not for me”.
- `#/collections` — existing collection navigation and category behavior.
- `#/assistant` — the exact Style Assistant / photo layout. Provide one clearly labelled stored example
  conversation and stored recommendations so the interaction is demonstrable without an LLM backend.
  Text submission may select from documented stored examples; never fabricate a live model response.
- `#/studio` and its existing subviews — the same DS Studio navigation, cards, tables, and result provenance.
  Inspect/label controls that need a backend should either use a faithful stored example or be visibly
  disabled with concise explanation.
- product routes and all navigation reachable from the public landing page must not 404 or hang.

Responsive layouts must match the real app at 1440×900, 1024×768, and 390×844. No horizontal overflow,
overlap, empty loading state, broken image, console error, or inaccessible keyboard trap.

## 3. Frozen data and result integrity

1. Export the minimum fixture set from the current verified `data/processed/serving.sqlite` and reports.
   Use an existing returning shopper as the default and include enough families/variants to exercise every
   visible interaction.
2. Reuse the local catalogue photographs already approved for this non-commercial portfolio. Do not use
   AI-generated product images.
3. Preserve the latest verified numbers exactly. At minimum, the DS Studio overview must still show:
   - Track A ranker test MAP@12: **0.03918**
   - Test lift over the best baseline: **+45.7%**
   - Track A validation MAP@12: **0.03772**
   - the same Track B comparison count and all detailed Track A/Track B/visual/assistant values currently
     shown by localhost and backed by stored reports
4. Add an automated assertion that the public fixtures’ headline metrics equal the canonical local DS
   Studio/report values. Do not round differently, recompute, or replace the last verified results.
5. Label all evaluation values as offline. Do not imply production traffic, online A/B testing, live model
   inference, or a functioning checkout.

## 4. README and AI-interaction image

Update every sentence introduced for the old simplified portfolio. Remove wording such as “lightweight
presentation layer” where it implies a different UI. Explain instead:

- the live link is the same frontend as the local FastAPI application;
- GitHub Pages uses frozen, verified API outputs for public browsing;
- local `make serve` enables the real database, model-backed visual search, outfit analysis, and assistant;
- the public demo does not run models or store shopper events.

Keep the live-demo link prominent. Ensure README links and screenshots use repository-relative paths and
render on GitHub.

Add the clearest supplied AI interaction screenshot to the README, with an accurate caption such as
“Style Assistant — grounded recommendation and outfit-photo analysis in the local model-backed app.” Use:

`/var/folders/ys/svn53sds1h74m41rmdcs3bd80000gn/T/codex-clipboard-2f073c70-1dd4-4d43-a433-75da5ebf07e9.png`

Copy it into a sensible tracked path such as `docs/assets/style_assistant_interaction.png`. Do not crop out
the assistant answer, detected photo region, or returned products. Optimize file size losslessly if useful.
Do not use the second supplied image unless it is needed for comparison; the first is more legible.

Replace other README screenshots only if necessary to prevent stale or contradictory visuals. All visible
screenshots must be from the real app, not AI-generated mockups.

## 5. Tests and verification

Add automated coverage for:

- deterministic public build/sync and relative asset URLs;
- every fixture-backed route and required response shape;
- exact headline metric equality with canonical results;
- no external `/api` calls in public demo mode;
- cart persistence, colour changes, deduplication, navigation, stored assistant example, and DS Studio;
- missing-fixture and unavailable-backend states;
- accessibility labels and reduced-motion behavior.

Run the full Python and JavaScript test suites. Run browser journeys against both:

1. the real FastAPI app; and
2. the built `portfolio/` site under a subpath-equivalent local server.

Capture desktop, tablet, and mobile screenshots of Discover, Style Assistant, and DS Studio. Programmatically
check console errors, failed requests, image failures, overflow, and the exact displayed metrics.

## 6. GitHub Pages and backward compatibility

- Keep `.github/workflows/pages.yml` deploying `portfolio/`.
- Validate the workflow and generated site without publishing it.
- Do not break `make serve`, existing API contracts, model pipelines, tests, or the local 8010 UI.
- Remove obsolete custom-portfolio code/assets only when the replacement is verified and no longer uses them.

## Definition of done

- Public build visually and structurally matches the current localhost application.
- Shopper, Style Assistant, and DS Studio are publicly browsable with frozen real outputs.
- Public demo has no broken routes/assets/API calls and works at the GitHub project subpath.
- All displayed results match the latest verified local results exactly.
- README copy accurately describes the new architecture and includes the supplied AI interaction screenshot.
- No AI-generated product images remain.
- Full tests and two-surface browser checks pass.
- Only intended files are committed; all user-owned untracked duplicates/logs remain untouched.
- Write `reports/PUBLIC_DEMO_FULL_UI_REPORT.md` containing architecture, files changed, fixture provenance,
  exact metrics checked, tests/browser evidence, limitations, git status, commit hash, and deployment steps.
