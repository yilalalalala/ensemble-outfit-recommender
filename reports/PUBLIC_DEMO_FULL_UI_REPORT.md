# Public demo: publish the real Ensemble UI

**Date:** 2026-10-08 · **Branch:** `public-demo-full-ui` · **Plan:** `docs/CLAUDECODE_PUBLIC_DEMO_FULL_UI_PLAN.md`
· **ADR:** [D-046](../docs/DECISIONS.md) · **Live URL (unchanged):** <https://yilalalalala.github.io/ensemble-outfit-recommender/>

## Conclusion

The GitHub Pages site is no longer a second, simplified frontend. `portfolio/` is now generated from
`src/ensemble/api/static/` — the same HTML, CSS, JavaScript, routes, components, typography, product
cards, cart and copy that `make serve` serves — and answers the application's own request paths from
frozen responses captured over HTTP from the running local app. Shopper, Style Assistant and DS Studio
are publicly browsable. Every evaluation number shown is the stored report, published unchanged.

| | before | after |
| --- | --- | --- |
| frontend | hand-written `portfolio/app.js` + `styles.css`, 4 hard-coded products | the real frontend, generated (`make public-demo`) |
| data | typed into the page | 663 articles, 701 Complete-the-Look captures, 5 profiles, all captured from the local app |
| headline numbers | `0.0333` / `+56%` / `0.59` / `100%`, typed by hand | the serving store's `reports` table, published byte-identical |
| product photography | 4 images | 663 catalogue images (downscaled), closed set, nothing AI-generated |
| assistant | a canned sentence assembled in the browser | 8 real captured turns, 7 with the products the answer cites |
| photo services | absent | one openly licensed photo with the real `/api/visual-search` and `/api/snap` responses |

## 1. Architecture

**One source of truth.** `src/ensemble/api/static/` is the frontend. `portfolio/` holds no frontend code
of its own; `python -m ensemble.api.public_demo sync` copies it and removes anything the frontend no
longer has.

**Demo mode is declared, not guessed.** The generated `index.html` carries
`<meta name="ensemble-demo" content="demo/">`. The real `index.html` never emits it, so the production
FastAPI behaviour is unchanged. No hostname sniffing (asserted in `tests/test_public_demo.py`).

**One adapter at the API boundary.** `static/js/demo.js` (new, ~150 lines) routes the frontend's own
paths to fixtures. `core.js` consults it in exactly three places:

| call site | demo behaviour |
| --- | --- |
| `api(path, opts)` | answered from a fixture; errors keep the real `ApiError` status/code, so the UI's existing states apply unchanged |
| `probe(path)` (new; `/readyz`, used by `app.js` and `studio.js`) | answered from the captured `/readyz` |
| `logEvent(...)` | returns immediately — a documented no-op; the public demo stores no shopper events |

Nothing above that boundary knows which mode it is in: routing, grouping, swatches, cart, prices,
reason chips, DS Studio tables and the responsive layout are the same code in both.

**Relative URLs.** `index.html` now references `static/…` instead of `/static/…`; at the server root
those resolve identically, and under `/<repo>/` they resolve correctly. `core.js` resolves the
placeholder image with `new URL("../placeholder.svg", import.meta.url)`. Captured responses have their
`"/images/<id>.jpg"` rewritten to `"images/<id>.jpg"`. Fixture requests are built from
`location.pathname.replace(/[^/]*$/, "")`, which is `/` locally and `/ensemble-outfit-recommender/` on
Pages.

**No speculative requests.** `demo/ctl/index.json` lists every Complete-the-Look capture, so the page
asks only for a file that exists. (Without it, the personalized-then-guest fallback produced visible
404s in the network log.)

### Surfaces that need a backend
Per the plan, each is either a faithful stored example or visibly withdrawn with a short explanation —
never a fabricated live result.

| surface | published build |
| --- | --- |
| Style Assistant text | 8 saved exchanges; the suggestion chips are the saved questions; a saved two-turn exchange offers its follow-up as a chip. An unsaved question gets *"That one isn't among the saved questions"* and the list — never a generated answer. |
| Photo upload | withdrawn (the composer's attach button and the dropzone), replaced by one saved, openly licensed photo with its framed garment; both services return the real captured responses. Credit shown on the page. |
| Label (gold labels for the visual-search judge) | removed from the DS Studio nav; the route still resolves and says it runs locally only (it writes labels to disk and shows crops of unpublished evaluation photos). |
| Serving telemetry | shown, tagged **captured snapshot**, with copy stating it is the traffic that server process had seen when the build was made — "not an evaluation and not public traffic". |

## 2. Files changed

| file | change |
| --- | --- |
| `src/ensemble/api/static/js/demo.js` | **new** — the demo data adapter and its router |
| `src/ensemble/api/static/js/core.js` | demo switch in `api()`; new `probe()`; `logEvent()` no-op; module-relative placeholder |
| `src/ensemble/api/static/js/app.js` | `probe("/readyz")`; Label dropped from the demo nav; demo provenance in the utility bar |
| `src/ensemble/api/static/js/studio.js` | `probe()`; demo provenance on Evaluation and Serving; Label explains itself |
| `src/ensemble/api/static/js/stylist.js` | saved exchanges, saved-answer label, follow-up chips, unsaved-question notice, saved photo example |
| `src/ensemble/api/static/css/ensemble.css` | `.saved-note`, `.footer__note` |
| `src/ensemble/api/static/index.html` | document-relative asset URLs |
| `src/ensemble/api/public_demo.py` | **new** — `sync` / `capture` / `build` |
| `configs/public_demo.yaml` | **new** — every parameter of the build |
| `Makefile` | `public-demo`, `public-demo-sync`, `public-demo-preview` |
| `tests/test_public_demo.py`, `tests/js/demo.test.mjs`, `tests/test_ui_demo_js.py` | **new** — 19 Python tests, 13 node assertions |
| `scripts/ui/public_journeys.js` | **new** — browser journeys against the built site at the project subpath |
| `portfolio/` | regenerated; `app.js`, `styles.css` and `assets/*.jpg` deleted |
| `README.md` | live-demo wording, the Style Assistant screenshot, licensing, repository map, ADR count |
| `docs/DECISIONS.md`, `docs/GLOSSARY.md` | D-046; six new terms |
| `docs/assets/style_assistant_interaction.png` | **new** — the supplied screenshot, flattened onto white and losslessly re-encoded (554 KB → 413 KB, 874×1670, uncropped) |

## 3. Fixture provenance

Captured on 2026-10-08 from `uvicorn ensemble.api.app:app --port 8031` against
`data/processed/serving.sqlite`, serving bundle `2020-09-23_4edf27eb8e_df69b1f77f`
(model `4edf27eb8e`, catalog `df69b1f77f`, serving week 2020-09-23, cutoff 2020-09-22).
1,463 HTTP calls. The serving store is read **only** to pick which articles to publish — the published
values all come back over HTTP, unchanged.

| fixture | source endpoint | count |
| --- | --- | --- |
| `customers.json` | `/api/customers` | 5 profiles (3 returning, 2 new); default `168058` (age 26, 621 items), the same profile the local app falls back to |
| `home/<c>.json` | `/api/home/{c}` | 5 × (for you / buy it again / trending) |
| `product/<a>.json` | `/api/product/{a}` | 663 |
| `ctl/<a>.json`, `ctl/<a>-<c>.json` | `/api/v2/complete-the-look?k=8` | 701 (guest + each profile's own for-you anchors, captured personalized) |
| `explain/<c>.json` | `/api/explain/{c}/{a}` | 5 × 12, with SHAP contributions |
| `families.json` | `/api/catalog/families` | 219 families, ≤5 colourways each |
| `metrics.json` | `/api/metrics` | the 7 stored evaluation reports, verbatim |
| `serving.json` | `/readyz`, `/api/v2/meta`, `/api/v2/metrics` | one snapshot |
| `assistant.json` | `/api/assistant/session` + `/message` (Ollama, `qwen3-vl:8b-instruct`) | 8 turns |
| `photo.json` + `photo.jpg` | `/api/visual-search`, `/api/snap` | 1 photo, 2 responses |
| `images/<a>.jpg` | `data/raw/images` | 663, downscaled to 760 px wide, q80 |

**How the published set was chosen** (`configs/public_demo.yaml`): the 90 articles in the five profiles'
home modules; every product a saved assistant answer or photo response cites (27 + 27); the 140
most-used Complete-the-Look items across the captured modules; and the colourways of all of those
(≤5 per family) — 663 articles in total. Each captured Complete-the-Look module is then the **live
response filtered to published articles, in its own order**, re-ranked 1..n, and dropped below two
items — never padded, never re-scored. `test_published_articles_are_a_closed_set` proves nothing a
visitor can click leads outside the set.

**Language-model responses are captured once and reused.** They are not reproducible across model
versions, so a rebuild never silently replaces a published answer; `--force` / `--recapture <id>`
re-captures deliberately. Each saved exchange is its own conversation — sharing one conversation across
eight unrelated questions made the model stop emitting `[[article_id]]` citations entirely, so all eight
turns rendered with no product cards. Catalogue questions are captured as a guest and personalized
questions signed in as `168058`, matching how `src/ensemble/assistant/eval.py` asks each kind; this is
recorded per turn in the fixture. Seven of the eight turns cite products (`hallucinated: []` on all
eight); the eighth is the off-topic scope guard, which correctly cites none.

**Photo.** `data/raw/outfit_photos/full_03.jpg` — *New York Street Fashion (Unsplash)* by Yegide
Matthews, **CC0**, from Wikimedia Commons; the credit is rendered on the page. No private evaluation
photo is in the build.

## 4. Metrics checked

The four values the DS Studio overview shows, derived exactly as `studio.js` derives them:

| value | published | source |
| --- | --- | --- |
| Track A ranker test MAP@12 | **0.03918** | `m3_ranker_test.metrics["map@12"]` = `0.03917510232956311` |
| Test lift vs best baseline | **+45.7%** | vs `repeat_purchase+popularity_by_age` test `map@12` |
| Track A validation MAP@12 | **0.03772** | `m3_ranker_val.metrics["map@12"]` = `0.03772064562082999` |
| Track B models compared (test week) | **10** | `m4_track_b_test.results` |
| Stored reports | **7** | `/api/metrics` |

Asserted three ways:

1. `test_published_reports_are_the_serving_stores_reports_unchanged` — `demo/metrics.json` equals the
   `reports` table of `data/processed/serving.sqlite` exactly (skips where that store is absent);
2. `test_published_headline_metrics_are_the_last_verified_results` — the derived values and the two
   unrounded floats, hard-coded, so the check holds without the store;
3. the browser journey reads the four rendered KPI tiles and the `title` attribute carrying
   `0.03917510232956311`, so what a visitor sees is checked, not only what is on disk.

Nothing was rounded differently, recomputed or replaced. Every page labels the results **offline**;
`"All numbers are offline (historical data). A launch decision would need an online A/B test."` is
asserted present. There is no claim of production traffic, online A/B evidence, live inference or a
working checkout (Checkout stays disabled).

## 5. Tests and browser evidence

**Python — `PYTHONPATH=src .venv/bin/python -m pytest -q`: `210 passed, 1 warning in 14.55s`**
(191 before this work, 19 new). The new tests cover: deterministic `sync` (same frontend → byte-identical
build, twice; stale files removed), relative URLs with every referenced asset present, demo mode declared
not guessed, the real app unchanged, every fixture-backed route and response shape, the closed article
set, relative image references, Complete-the-Look filtering (never padded, ranks renumbered, k respected),
grounded assistant turns, the photo example's licence and both responses, no absolute URL or stray
`fetch(` in the build, the three metric assertions above, and that `pages.yml` still deploys `portfolio/`.

**Node — `tests/js/demo.test.mjs` (run by `tests/test_ui_demo_js.py`): 13/13.** Fixture routing under a
project subpath, families slicing, `k` truncation and slot filtering, guest fallback, 404 on an
unpublished anchor, the event no-op, 503 + `no_backend` for upload/label, exact and fuzzy saved-question
matching with `null` for anything unsaved, the photo example, DS Studio fixtures, and per-profile
explanations.

**Browser (Playwright + Chrome).** Both surfaces, at 1440×900, 1024×768 and 390×844.

| surface | command | result |
| --- | --- | --- |
| real FastAPI app | `node scripts/ui/journeys.js center,copy,home,swatch,cart,switch,product,collections,tabs,studio,label,keyboard,motion,offline http://localhost:8031/` | **110/110** |
| real app, model-backed | `node scripts/ui/journeys.js assistant,visual http://localhost:8031/` | **10/10** (live Ollama assistant and FashionCLIP visual search) |
| published build, project subpath | `node scripts/ui/public_journeys.js` against `http://localhost:8041/ensemble-outfit-recommender/` | **109/109** |

The public run asserts, programmatically: every route renders with content and no empty loading state
(`#/`, `#/collections`, two collection tabs, `#/assistant`, `#/studio`, `#/studio/serving`,
`#/studio/inspect`, `#/label`, legacy `#/ds`, an unknown route) at all three viewports; **no console
error, no failed request, no request leaving the published site** (any `/api` or off-origin request
fails the run); **no horizontal overflow**; **no broken image** (the page is scrolled so every
`loading="lazy"` card is given its chance, and the neutral placeholder counts as broken); one card per
product family with a USD price, a selected swatch and a black Add to cart; swatch, "Not for me",
profile switch, collection tabs and category filter; ten reachable product links all resolving;
cart add/persist/remove; the saved conversation, its follow-up chip, the unsaved-question notice, and
both saved photo services; the four DS Studio values above plus the serving snapshot and Inspect's
SHAP; Label withdrawn from the nav and explaining itself; keyboard operation (skip link, swatch, cart
dialog, Escape returns focus, no trap) and accessible names on cards, swatches, images and the live
region; reduced-motion honoured both ways; and a missing fixture producing a recoverable message with a
working retry rather than a dead page.

Screenshots (regenerate with the command above; `data/interim/` is gitignored):
`data/interim/public_demo_screenshots/` — `discover_{1440,1024,390}.png`,
`assistant_{1440,1024,390}.png`, `studio_{1440,1024,390}.png`, `assistant_saved_1440.png`,
`studio_serving_1440.png`, `product_1440.png`, and `public_journeys_result.json` with all 109 outcomes.

## 6. Reproducing and deploying

```bash
make serve PORT=8031            # the local app the build captures from (needs the serving store)
make public-demo                # capture + sync -> portfolio/   (~8 min: 1,463 calls, 663 images, 8 LLM turns)
make public-demo-sync           # frontend -> portfolio/ only; no data, no network
make public-demo-preview        # http://localhost:8041/ensemble-outfit-recommender/
PYTHONPATH=src .venv/bin/python -m pytest -q
PLAYWRIGHT=<playwright> node scripts/ui/public_journeys.js
```

Deployment is unchanged: `.github/workflows/pages.yml` uploads `portfolio/` on a push to `main` that
touches `portfolio/**`. It was validated (parsed, artifact path and trigger asserted in the test suite)
but **not run**: nothing was pushed, merged or published by this work.

## 7. Limitations

- **The published catalogue is a subset.** 663 of 105,542 articles and 5 of 330 demo profiles.
  A Complete-the-Look module is the live response filtered to that subset, so a module can be shorter
  than it is locally, and a module left with fewer than two published items is dropped. This is stated
  in D-046 and enforced by the tests; it is a publishing bound, not a model change.
- **No live inference.** Typing an unsaved question, uploading a photo or labelling needs
  `make serve`. The build says so on each surface rather than approximating a result.
- **The saved assistant turns are one model's output on one day** (`qwen3-vl:8b-instruct` via Ollama),
  captured once. They are unedited, but they are a selection of eight questions, not an evaluation —
  the evaluation is `reports/m7b_assistant_ollama.json` and the README table.
- **The serving snapshot is frozen.** `/readyz`, `/api/v2/meta` and the counters are from build time.
  The page says so; it would be misleading if it did not.
- **Size.** The build is 72 MB (39 MB photography, 33 MB JSON). `configs/public_demo.yaml` holds the
  knobs (`ctl_pool`, `max_variants_per_family`, `image_width`, `image_quality`) if that needs to come
  down. Well inside the GitHub Pages 1 GB limit, but it is a real addition to the repository.
- **Photography is downscaled** to 760 px wide (q80) from 1166×1750, so the product page is slightly
  softer on a high-DPI display than it is locally.
- **Not addressed:** the README's other screenshots were left as they are (they are current and from
  the real app); the `#/label` route and the two upload services remain local-only by design.

## 8. Working tree

**Preserved untouched and never staged**, as required: `reports/track_a_retrain/logs/`, `docs/HANDOFF 2.md`,
and every pre-existing `* 2.*` file under `reports/`, `src/ensemble/` and `tests/`.

> **Worth your attention.** During this work a process on this machine (not this build) copied the newly
> written files repeatedly: `portfolio/demo/ctl/`, `portfolio/demo/product/` and `portfolio/images/`
> now also contain ~6,700 untracked `"<name> 2.json"` … `"<name> 5.json"` / `.jpg` duplicates,
> taking `portfolio/` from 72 MB to 111 MB on disk. The same pattern produced the `* 2.*` files that
> were already in the tree before this work. **Nothing of the sort is staged or committed**, and
> GitHub Pages deploys from the repository checkout, so none of it reaches the published site. They were
> left in place because the task said to preserve untracked `* 2.*` files. They also made the fixture
> tests hang, so the tests and `tests/js/demo.test.mjs` now read only the filenames the build writes
> (`<id>.json`, `<id>-<customer>.json`, `index.json`) and `ctl/index.json`, and ignore anything else in
> those directories. To remove them:
> `find portfolio -regex '.* [0-9]\.\(json\|jpg\)' -delete` — your call, not run here.

Staged for this work: 2,085 files (2,059 of them the generated `portfolio/` build), 0 matching a
duplicate pattern, 0 under `reports/`.

```
 M Makefile                              A  configs/public_demo.yaml
 M README.md                             A  docs/CLAUDECODE_PUBLIC_DEMO_FULL_UI_PLAN.md
 M docs/DECISIONS.md                     A  docs/assets/style_assistant_interaction.png
 M docs/GLOSSARY.md                      A  scripts/ui/public_journeys.js
 M src/ensemble/api/static/{index.html,css/ensemble.css,js/{app,core,studio,stylist}.js}
 A  src/ensemble/api/{public_demo.py,static/js/demo.js}
 A  tests/{test_public_demo.py,test_ui_demo_js.py,js/demo.test.mjs}
 M portfolio/{index.html,favicon.svg,.nojekyll}   D portfolio/{app.js,styles.css,assets/*.jpg}
 A  portfolio/{static/**,demo/**,images/**,photo.jpg}
```

Nothing was pushed, merged, published or deleted outside `portfolio/`'s own generated contents.

**Commit:** `dfe4c337780646f113d5ad18f8f36e758ad4feca` on `public-demo-full-ui` (this report follows in the next commit;
`git log --oneline -2` shows both).
