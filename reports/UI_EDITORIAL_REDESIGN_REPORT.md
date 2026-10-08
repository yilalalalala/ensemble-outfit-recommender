# Editorial retail UI redesign: implementation report

2026-10-08 · branch `ui-editorial-redesign` (from `1315337`, plan `ecafe72`) · local only: nothing was pushed,
merged, published, deployed or opened as a PR. Plan: `docs/CLAUDECODE_UI_EDITORIAL_REDESIGN_PLAN.md`.

## 1. Before and after

**Before.** The UI was one 321-line `index.html`:
- a flat white header with an uppercase "ENSEMBLE" label;
- a raw `#customer · age · purchases` `<select>` sitting next to the Shop / Style assistant / DS view / Label links;
- every surface built from identical horizontal card rows, with reason chips printed on each card;
- the DS view's tables overflowed by 427 px at 390 px width.

**After.** The same FastAPI static app is now an image-led editorial boutique:
- a centred lowercase serif `ensemble` masthead on warm ivory;
- **Shop / DS Studio** is an explicit experience switch, kept apart from a **demo-profile control** ("Shopping as Shopper, 26");
- an asymmetric Discover page built from real recommendation photography;
- a retailer-style product page;
- a calm Style Assistant with two clearly separated photo services;
- an information-dense DS Studio that shares the same type and token system.

No API contract, model, metric, ranking output or research artefact was changed. No dependency was added.

## 2. Design system and information architecture

**Tokens** (`:root` in `css/ensemble.css`):
- **Colour.** Canvas ivory `#f6f1e8`, surface `#fbf8f3`, ink `#1b1916`, muted `#655e54` (5.6:1 on the canvas), stone hairlines `#ddd4c6`, and one oxblood accent `#6b1f25`. There are also success, warning and error colours, and a photo backdrop matched to the catalogue's studio grey.
- **Type.** Display serif `Didot, "Bodoni 72", "Bodoni MT", …, serif` and utility sans `"Helvetica Neue", -apple-system, …`. Both are local or system fonts; nothing is loaded remotely.
- **Scale.** Fluid `clamp()` sizes for the masthead, display, title, section, product, metadata and label roles. Fluid spacing, gutters and gaps; a 1,360 px content width; a 38 rem reading measure.
- **Images and motion.** Image ratio 2:3, matching the 1166×1750 catalogue photos, plus 4:5 feature crops shown uncropped with `contain`. Motion uses durations of 160/320/700 ms on one easing curve.
- **Shape.** A 2 px radius, a two-ring focus style, and a single elevation used only for dialogs.
- **Dark mode.** A dark variant follows `prefers-color-scheme`. Product photography keeps its light backdrop, like a gallery print.

**Control model.** The three controls are deliberately separate:

| control | where | what it does |
| --- | --- | --- |
| Experience switch | utility bar above the masthead; repeated in the mobile drawer | `Shop` (`#/`) ↔ `DS Studio` (`#/studio`), with `aria-current="page"` |
| Demo profile | masthead right: monogram (age) plus "Shopping as / Inspecting profile" | opens a dialog of the 330 real demo customers, labelled "Shopper, 26" or "New shopper, 56" with purchase count and last-order date. The raw ID appears only as a secondary detail. The choice persists in `localStorage` (`ensemble.profile`) and falls back to the previous default, the first returning customer. |
| Primary navigation | masthead left (a drawer below 860 px) | Shop: Discover · Collections · Style Assistant. Studio: Evaluation · Serving · Inspect · Label. |

**Routes.** Every pre-redesign deep link still works:
- `#/`, `#/product/<id>`, `#/assistant?anchor=<id>`, `#/label?round=<n>`;
- `#/ds`, now an alias of `#/studio`.

New routes are `#/collections[/<module>]` and `#/studio/serving|inspect`. Unknown routes show a not-found state.

## 3. File-by-file changes

| file | change |
| --- | --- |
| `src/ensemble/api/static/index.html` | Rewritten as a semantic shell: skip link; utility bar with the experience switch; 3-column masthead; `main`; footer; native `<dialog>`s for the drawer, the profile picker and Why this; a polite announcer. |
| `src/ensemble/api/static/css/ensemble.css` *(new)* | The whole design system, components, per-surface layouts, motion and responsive rules. |
| `src/ensemble/api/static/js/core.js` *(new)* | API client with customer-safe errors, a per-customer home cache, event logging, the shared product tile, Why this (reasons plus a SHAP disclosure), Not for me, dialogs, loading and error states, reveal motion, image fallback. |
| `src/ensemble/api/static/js/app.js` *(new)* | Router with a stale-render guard, navigation state, experience switch, profile picker and persistence, mobile drawer, one delegated click handler, `/readyz` degraded-mode note. |
| `src/ensemble/api/static/js/shop.js` *(new)* | Discover, Collections and the product page. |
| `src/ensemble/api/static/js/stylist.js` *(new)* | Style Assistant chat, photo upload and preview, visual search, outfit analysis. |
| `src/ensemble/api/static/js/studio.js` *(new)* | DS Studio pages: Evaluation, Serving, Inspect, plus Label. |
| `src/ensemble/api/static/favicon.svg` *(new)* | Serif "e" favicon; this also removes the old `/favicon.ico` 404. |
| `tests/test_ui_static.py` *(new)* | 7 tests: static routing and MIME types, module-import resolution, landmarks, a centred lowercase wordmark grid rule, switch/profile separation, no remote dependencies, every API path and event type still used, motion gated on reduced-motion. |
| `scripts/ui/journeys.js`, `scripts/ui/shots.js` *(new)* | Reproducible browser journeys and viewport screenshots. Playwright comes from `$PLAYWRIGHT`; no repo dependency. |
| `src/ensemble/api/app.py` | **Unchanged.** `/` still serves `index.html` and the new assets use the existing `/static` mount. |

## 4. Surfaces

**Discover** is built only from `/api/home/{customer}`.

| module | how it is shown |
| --- | --- |
| Hero | Greeting, one editorial line and the top three *Selected for you* pieces as an offset photographic collage, numbered by real rank |
| *Selected for you* (ranks 4–12) | Deterministic asymmetric grid: feature rows give the row's highest-ranked piece a 6/12 tile, alternating with staggered 4/4/4 rows; 2 columns on mobile |
| *Complete the look* | Built around No. 01 from the live `/api/v2/complete-the-look`. Returning customers only; omitted if unavailable |
| *Buy it again* | Real last-bought date and count |
| *Trending this week* | "Most bought this week by shoppers in your age group", as a compact catalogue grid |

- Modules the API does not return are omitted. New customers get honest copy ("No purchase history yet…") and no Buy it again.
- No campaign image, price, name or metric is invented. The articles table has no price, so none is shown anywhere.
- **Collections:** one tab per real module, plus category filters computed from the items.
- **Product page:**
  - two-column image and information layout (stacked on mobile);
  - breadcrumb plus a Back control: history when navigated in-app, otherwise Discover;
  - catalogue facts;
  - **Add to bag, labelled *Demo*.** It records the existing `add_to_cart` event and says that nothing is purchased.
  - *Why this is in your edit* appears only when the piece is in the profile's ranked edit.
  - *Complete the look* comes from v2, with personalised/compatibility and fallback level in plain words and a "How this look was assembled" disclosure (model, catalogue, request id). It falls back to the stored v1 table with a visible notice.
  - *Other colours* and *Similar items*.
- **Why this:** a dialog with plain-language reasons and their evidence type. For ranked items, the SHAP contributions sit behind a "Model detail" disclosure, with values exactly as returned.
- **Not for me:** dims the tile, disables the button, shows "Feedback noted" and announces it.
- **Style Assistant:**
  - **States:** empty (with three suggested prompts), thinking ("may call the recommender tools"), answer (with a tool trace, latency, and removed-unverified-product warnings), and error (with retry; the session is kept and an expired session is recreated once).
  - **Grounding:** answer text is escaped first, then only `**bold**` is rendered. Cited cards are numbered to match `(1)`, `(2)` in the answer.
  - **Photo input:** preview, optional pointer/touch framing (whole photo by default, so a keyboard is never required), and replace/remove controls.
  - **Two separate services:** *Find visually similar products* (`/api/visual-search`) and *Complete my outfit* (`/api/snap`, with an elapsed-time progress label).
  - **Privacy copy:** it is based on the backend the server reports, local (`ollama`) or a named remote provider.
- **DS Studio:**
  - **Evaluation:** the same four tables and the same formatting as the old DS view, now with row headers, the exact value on hover, an "Offline" tag, KPI tiles and horizontally scrollable focusable table regions.
  - **Serving:** `/readyz`, bundle versions and rankers from `/api/v2/meta`, and live counters and latency from `/api/v2/metrics`, explicitly tagged *live · this process*.
  - **Inspect:** the profile's ranked list plus reasons and SHAP for any item, as DESIGN §7.2 asks.
  - **Label:** the same workflow as a utility layout. Buttons use `aria-pressed` with a "✓ Relevant" text state, so state is not shown by colour alone. There is a progress bar, Save is disabled while saving, and errors keep the selection.

## 5. Compatibility matrix (plan §18)

| workflow | endpoint(s) | verified by |
| --- | --- | --- |
| Customer list and switching | `GET /api/customers`, `GET /api/home/{id}` | journeys: switch to 477553 updates the hero, removes Buy it again, impressions carry the new id, persists across reload |
| Personalised home modules | `GET /api/home/{id}` | journeys + `test_api` |
| Product deep links and back | `#/product/<id>`, `GET /api/product/{id}` | journeys: tile → product, CTL item → product, Back, other-colour and similar navigation, legacy `#/ds` |
| Impression and click events | `POST /api/events` (`impression`, `click`, `not_for_me`, `add_to_cart`) | journeys inspect the posted payloads: surface, article, anchor, customer |
| Why this | `GET /api/explain/{c}/{a}` | journeys: home tile and product page; dialog, Escape, focus return |
| Not for me | `POST /api/events` | journeys |
| Complete the Look / similar / other colours | `GET /api/v2/complete-the-look`, v1 fallback in `/api/product` | journeys (live, plus forced 503 → stored suggestions with notice) |
| Assistant session and messaging | `POST /api/assistant/session`, `POST /api/assistant/{sid}/message` | journeys: real local-model answer in 37 s, tool trace, session kept across navigation |
| Visual search | `POST /api/visual-search` | journeys: framed upload → 8 catalogue matches |
| Outfit photo analysis | `POST /api/snap` | journeys: local vision model detected 5 garments with catalogue matches |
| DS metrics and tables | `GET /api/metrics`, `/readyz`, `/api/v2/meta`, `/api/v2/metrics` | journeys: 4 tables / 40 rows; 4 tables / 28 rows |
| Label workflow | `GET /api/label/tasks`, `POST /api/label` | journeys: toggle → payload `[1,0,1,0,0]` (save intercepted, see §8) |
| Health / degraded | `/readyz`, v2 failure → v1 | utility-bar note when not ready; forced-503 journey |

## 6. Accessibility, responsiveness, motion, performance

- **Accessibility:**
  - landmarks (header, nav ×2 labelled, main, footer), heading order, real buttons and links, labelled form controls;
  - focusable, scrollable table regions with `scope` headers;
  - skip link; a visible two-ring focus style;
  - native modal dialogs for the profile picker, Why this and the drawer, with Escape and focus return;
  - polite announcements for profile changes, feedback and assistant replies;
  - validation messages tied to their control with `aria-describedby` and `role="alert"`;
  - alt text from real product name, colour and type only;
  - state never shown by colour alone.
- **Responsive:** designed and checked at 1440×900, 1024×768 and 390×844.
  - Editorial grids step from 12 to 6 to 2 columns, and catalogue grids from 6/4 to 3/4 to 2.
  - The masthead keeps the wordmark centred, with a drawer and monogram on mobile.
  - Touch targets are at least 44 px.
  - Horizontal overflow is 0 px on every captured route; the old DS view overflowed by 427 px.
- **Motion:**
  - first-view fade-up reveals via `IntersectionObserver`, staggered over at most three tiles;
  - a 2.5% image scale on hover-capable devices;
  - underline and dialog transitions, plus a 320 ms view-enter that never delays navigation;
  - no parallax, carousel or scroll-jacking.
  - All of it sits behind `prefers-reduced-motion: no-preference`; under `reduce`, no element is hidden and no view animation runs (verified).
- **Performance and privacy:**
  - images lazy-load below the fold, with width/height and aspect-ratio reserved;
  - no remote fonts, trackers or libraries;
  - one delegated click listener;
  - the home response is cached per customer, so content never crosses profiles, and a render token discards stale responses;
  - uploads go only to the existing local endpoints.

## 7. Automated tests

```text
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider
  baseline (before any edit, ecafe72): 171 passed, 1 warning
  final (after 2601fd7):               178 passed, 1 warning   (171 + 7 new in tests/test_ui_static.py)
```

The 1 warning is pre-existing and unrelated. There were no failures before or after.

## 8. Browser journeys and screenshots

**How the journeys were run.**
- Server: `uvicorn ensemble.api.app:app --port 8031` on the real serving store and bundle `2020-09-23_4edf27eb8e_df69b1f77f`. An older server the user had left on :8010 was not touched.
- Browser: installed Chrome via Playwright, launched with `PLAYWRIGHT=… caffeinate -i -s node scripts/ui/journeys.js`.
- **74/74 checks passed** (61 fast, 13 model-backed), covering:
  1. Shop default customer and all modules;
  2. profile switching with no stale cross-customer results;
  3. product, explanation, feedback, related modules, demo add-to-bag;
  4. a Style Assistant text turn;
  5. visual search and outfit analysis with a local fixture photo (`data/raw/outfit_photos/IMG_1503.jpg`, CC-licensed per its `CREDITS.csv`);
  6. DS Studio (all three pages);
  7. Label;
  8. keyboard: skip link, experience switch, profile dialog, mobile drawer;
  9. reduced motion versus normal motion;
  10. backend-unavailable (product request aborted → friendly error, retry recovers, navigation intact) and degraded live CTL.
- **Wordmark centring, measured:** the centre falls at 719.99 / 720, 511.99 / 512 and 195.00 / 195 px at the three widths, on both Shop and Studio.
- **Label save:** intercepted, so the user's gold-label file was not modified. All 20 real round-2 tasks are already labelled, so the journey marks one task pending in the response.
- **Side effect:** the journeys logged ordinary demo events to the gitignored `serving.sqlite` events table, as normal use does.

**Screenshots inspected.** Each of the following was captured at all three viewports, viewport and full page:
- Shop home, product detail, Collections;
- Style Assistant (empty, thinking, answer, visual search, outfit);
- DS Studio Evaluation, Serving and Inspect;
- Label;
- Why dialog, mobile drawer, degraded product, and dark mode.

Defects found during review and fixed in `2601fd7`:
- tablet hero tiles overlapping the lead image, and the headline pushed below the fold;
- Inspect and Label thumbnails ignoring `aspect-ratio`, because the HTML `height` attribute won;
- a focus ring drawn around `<main>` after navigation;
- Studio narrower than the masthead grid, and partial table borders;
- product names in Complete the Look slots inheriting the uppercase heading style;
- a photo-backdrop seam around product photos;
- "Unknown" catalogue placeholders shown in tile metadata;
- raw `**markdown**` in assistant answers.

The final set (48 viewport images, 13 journey images, plus the "before" set) is in `data/interim/ui_editorial_screenshots/`. That folder is gitignored because the images contain dataset photography.

## 9. Console and network findings

- **Final runs:** zero uncaught errors, zero console errors and zero failed requests in every checked journey.
- **Before:** one 404 on every page load (`/favicon.ico`); it is gone now that the favicon is declared.
- **Incident:** during one run the Mac slept, and Chrome reported `ERR_NETWORK_IO_SUSPENDED` on `/api/snap`. The UI showed its "couldn't reach the server" state with retry, as designed. Re-run under `caffeinate`, the check passed. It was a host problem, not an app defect.

## 10. Known limitations and deferred work

- **No prices and no real cart.** The dataset has no price column; Add to bag is labelled *Demo*.
- **One photo per article.** There is no gallery and no hover crossfade, by design.
- **Profile labels.** There are no customer names, so labels are derived from segment, age and history. Several profiles can share a label ("Shopper, 26"); the ID is in the secondary detail.
- **Box framing needs a pointer.** Keyboard users search the whole photo. A keyboard-adjustable box is a possible follow-up.
- **DS Studio values.** It shows the stored M1–M4 reports that `/api/metrics` serves. The later research results live in `reports/` and are not served by the API, so they are not shown. Adding them would need an API change, which was out of scope.
- **Impressions.** They are still logged at render time, as before, rather than on visibility. Changing that would change analytics semantics.
- **Not done:**
  - no hero parallax (optional in the plan);
  - no automated colour-contrast audit tool — contrast was checked from token values;
  - no screen-reader session — accessibility was checked through semantics and keyboard journeys.

## 11. Commits on `ui-editorial-redesign`

```text
ecafe72 docs(ui): define editorial retail redesign                                  (pre-existing plan)
c2eb31d feat(ui): editorial retail redesign of the no-build frontend
deeffce test(ui): static routing, page semantics and API-contract checks for the editorial UI
2601fd7 fix(ui): visual QA fixes from the three-viewport review
da96175 test(ui): reproducible browser journeys and viewport screenshot scripts
<this report> docs(report): editorial UI redesign implementation report
```

## 12. `git status --short` audit

Recorded after the report commit. **No tracked file is modified or staged.** The only untracked entries are user-owned and were preserved untouched:
- 33 `* 2.*` sync duplicates (for example `docs/HANDOFF 2.md` and `src/ensemble/vision/clip 2.py`);
- `reports/track_a_retrain/logs/`.

Task outputs outside git are under the gitignored `data/interim/ui_editorial_screenshots/`.
