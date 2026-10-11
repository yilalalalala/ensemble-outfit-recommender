# AI Stylist product flow and consumer UI upgrade

2026-10-11 · released to `main` as **`57c72b2`** (branch `stylist-product-flow`, merged `--no-ff`) ·
public site <https://yilalalalala.github.io/ensemble-outfit-recommender/> (GitHub Pages run succeeded on
`57c72b2`) · backend `https://yilalalalala--ensemble-recommender-web.modal.run` (Modal, redeployed with the
new endpoints) · catalogue photos on Cloudflare R2. Spec written before any UI change:
[`docs/AI_STYLIST_PRODUCT_FLOW.md`](../docs/AI_STYLIST_PRODUCT_FLOW.md).

## 1. Result

Shoppers now start by choosing a goal, and the AI Stylist is three focused tasks instead of one crowded page.
The core journey is **choose a goal → provide text or a photo → confirm what we understood → see
recommendations → refine with the stylist**:

- **Gateway (`#/`).** Two editorial choices built from the shopper's own recommendations, *Shop your edit* and
  *Style with AI*, plus a quiet *Explore the system* link to DS Studio.
- **AI Stylist hub (`#/stylist`).** Three tasks, each on its own route with one shared shell.
- **Complete my look.** Four steps: Photo → Review → Direction → Your edit. On the Review step the shopper sees
  what the vision model detected, on the photo, and can correct, remove or add pieces before anything is
  recommended. *Refine this edit* opens a conversation preloaded with the look.

Track A ranking, Track B Complete the Look, visual search, the grounded assistant, product-family grouping with
real colourway swatches, prototype USD prices, the persistent cart, product pages and DS Studio are unchanged.
No model was retrained, and no frozen data was reintroduced.

## 2. Route map

| route | page | notes |
| --- | --- | --- |
| `#/` | Gateway | new; the wordmark target; offers "continue where you left off" without redirecting |
| `#/shop` | Your edit (ranked, personalised) | was `#/`; content unchanged |
| `#/collections[/:edit]`, `#/product/:id` | unchanged | breadcrumb and Back fallback now point to `#/shop` |
| `#/stylist` | AI Stylist hub | new |
| `#/stylist/ask[?anchor=][&context=look]` | Ask the stylist / Refine your edit | replaces the chat column |
| `#/stylist/find[?step=photo\|frame\|results]` | Find this piece | replaces the visual-search panel |
| `#/stylist/look[?step=photo\|review\|direction\|edit]` | Complete my look | replaces one-shot outfit analysis |
| `#/assistant[?anchor=]` | → `#/stylist/ask[?anchor=]` | `history.replaceState`, so Back never bounces |
| `#/studio…`, `#/ds`, `#/label` | DS Studio | unchanged (Label stays local-only) |

## 3. Decisions

| decision | why |
| --- | --- |
| Hash routes with `?step=` and per-tab `sessionStorage` | Back, Forward and refresh restore any step with no server state. A deep link to an unreachable step falls back to the furthest valid one. *Start over* clears the state. |
| Additive staged endpoints rather than changing `/api/snap` | Existing consumers keep working. The UI moves to the staged contract so detection always lands on Review. |
| Occasion = Complete the Look pool reordered by text fit | Grounded and cheap: products come only from model output, with no extra LLM call. |
| Refinement context sent with the first message (not as an automatic turn) | No unsolicited model call. The context card shows the shopper exactly what the stylist knows. |
| Session recovery replays the context and recent turns, keeping product ids | After a Modal restart the stylist can still refer to the products it suggested. An earlier version stripped the ids, and the model then asked for them. |
| Fixed Feature / Trio / Pair presets on a 12-column grid | Varied sizes that read as art-directed. Every count from 1 to 40 fills whole rows with nothing dropped (unit-tested). Rows align on their bottom edge. |
| 4:3 only for compact pieces (shoes, accessories) | A 4:3 crop of a 2:3 garment photo cuts away most of it. Garments in wide slots use 1:1. |
| One product per family in an edit | Two colourways of one product would otherwise take two slots, and page grouping could shrink an edit below four (found on the live backend). |
| Gateway imagery = the shopper's own recommendations | No campaign imagery is invented. |
| README screenshots with a CC0 photo | The previous screenshot showed a private test photo. The new photo is credited in the caption. |

## 4. API changes (additive)

- **`POST /api/stylist/detect`** (photo) → `{garments: [{id, category, colour, description, box}], categories}`.
  - Boxes are fractions of the image.
  - It uses the same budgeted vision client as `/api/snap` (Claude Haiku when hosted, the local VLM locally).
  - Detector failure returns a clean `503`; a reached cap returns `503 ai_budget_reached`; a non-image returns `415`.
- **`POST /api/stylist/edit`** (photo, `garments` JSON, `goal ∈ complete|occasion|similar`, `occasion?`,
  `customer?`) → `{goal, summary, anchor, matched, picks[4–6]: {card, reason, group}, alternatives}`.
  - Confirmed colour, description and category feed the existing text-rerank-image matching.
  - Complete the Look comes from the serving bundle, with the stored table as fallback.
  - A complete outfit gets pairings rather than an invented gap.
  - One product per family; alternatives never repeat a pick's family.
  - Input is validated (1–8 pieces, known categories, an occasion when that goal is chosen).
- `/api/snap`, `/api/v2/outfit`, `/api/visual-search` and every other endpoint are **unchanged**; a test asserts
  they still exist.

## 5. Files and commits

| file | change |
| --- | --- |
| `docs/AI_STYLIST_PRODUCT_FLOW.md` | new spec: audit, route map, state machine, contracts, copy, states, a11y, responsive, grid, acceptance criteria |
| `src/ensemble/api/app.py` | the two stylist endpoints (+185 lines) |
| `static/js/gateway.js`, `photo.js`, `look.js`, `find.js` | new: gateway; shared photo handling and staging; the two photo flows |
| `static/js/stylist.js` | hub, shared task shell, Ask with context and session recovery (rewritten) |
| `static/js/app.js`, `shop.js`, `index.html` | routes and redirect; editorial grid presets; links to the new routes |
| `static/css/ensemble.css` | token refactor and the components for the new pages |
| `tests/test_stylist_api.py`, `tests/js/layout.test.mjs`, `tests/test_ui_static.py`, `tests/test_ui_catalog_js.py` | API, layout and route contracts |
| `scripts/ui/journeys.js`, `scripts/ui/readme_shots.js` | stylist journeys, six-width checks, README capture |
| `README.md`, `docs/assets/*.png` | route map and final screenshots |
| `requirements-modal.txt` | was used by the Modal deploy but never committed |

```text
4afefe4 docs(stylist): product-flow spec for the AI Stylist and gateway; track requirements-modal.txt
006d824 feat(api): staged Complete my look contract — /api/stylist/detect and /api/stylist/edit
6166bc6 feat(ui): gateway, AI Stylist hub and staged task flows; controlled editorial grid and token refactor
79252dc test(ui): stylist flow journeys, six-width overflow checks, layout presets and route contracts
d7e6c3e fix(stylist): one product per family in picks; natural garment labels; robust journey checks
be284bc docs(readme): shopper route map and final screenshots of the live AI Stylist flows
0376af3 build(pages): rebuild the hosted frontend with the AI Stylist flows
57c72b2 merge: AI Stylist product flow and consumer UI architecture
```

## 6. Visual design system

- **Colour.** White canvas; soft ivory surfaces (`#faf8f4`, `#f7f5f1`); ink `#111`; warm greys `#5d5853`
  (6.6:1) and `#756f69` (4.6:1). One deep-burgundy accent, `#611c2a` (13.6:1), used for step numbers,
  detection boxes and focus only.
- **Primary actions** are black with white text and a dark-grey hover, never burgundy.
- **Type.** Two families: Didone display serif and a neutral sans. The monospace font was removed; Studio feature
  names use tabular sans.
- **Motion.** Tokens of 180 / 220 / 240 ms, reveals of 8 px, and no parallax. All of it is removed under
  `prefers-reduced-motion`.
- **Images.** Only 3:4, 1:1 and 4:3, with `object-fit: cover` around a focal point. This applies to cards,
  product pages, the cart, Studio and Label thumbnails.
- **Wordmark.** Unchanged and exactly centred (720.00 of 720 at 1440 px).

## 7. Tests and journeys

```text
PYTHONPATH=src .venv/bin/python -m pytest -q        217 passed, 1 warning (pre-existing)
  new: test_stylist_api.py (16), layout presets 1–40 items (node), route/endpoint contracts
Browser journeys (Chrome via Playwright; scripts/ui/journeys.js)
  local app (port 8031), all journeys, 1440 px ............................ 176/176
  Pages build → live Modal backend, all journeys, 1440 px ................. 173/173
  Pages build → live Modal backend, stylist journeys, 375 px .............. 54/55, then 10/10 on re-runs (see §9)
  public URL after deploy, 1440 px (center, copy, home, swatch, product,
    studio, gateway, widths, look, find, ask) .............................. 143/143
  public URL after deploy, 375 px (gateway, look, find, ask) ............... 44/44
```

The journeys cover the following:
- **Gateway:** both choices, one-click edit, DS Studio link, legacy redirect, resume without redirect.
- **Complete my look:**
  - invalid-file rejection and detection landing on Review, with overlays;
  - deep-link fallback, recolour / remove / add corrections, and refresh restoring them;
  - the edit request carrying exactly the confirmed pieces, and the occasion context requirement;
  - 4–6 priced, swatched, black-CTA picks with one family each, and alternatives behind a control;
  - the refinement context sent, and Start over.
- **Find this piece:** dragging a frame, the keyboard whole-photo switch, real results, one next action.
- **Ask:** grounded cards, and a forced lost session that recovers with history replayed and no raw error.
- **Widths:** 320 / 375 / 768 / 1024 / 1280 / 1440 px × 9 routes with no horizontal overflow.
- **Existing journeys still pass:** cart, swatches, grouping, product, Studio, Label, keyboard, reduced
  motion, offline.

Live Claude cost this month during all testing: **$0.37** of the $10 cap.

## 8. Screenshots

- **Before:** `data/interim/ui_stylist_upgrade/before/` (main @ `21daeeb`: Discover, the single-page assistant,
  product, Studio at 1440 / 1024 / 390).
- **After:** `data/interim/ui_stylist_upgrade/final/` (Review, Your edit, Refine and Find results at 1440 and
  375; gateway, Your edit, hub and Studio at three widths) and `…/public*/` (captured from the public URL).
  These folders are gitignored because they contain catalogue photography.
- **README:** `docs/assets/` is the real app on the live backend. The stylist strip uses the CC0 photo
  *New York Street Fashion* (Yegide Matthews), credited in the caption. No AI-generated imagery.

## 9. Accessibility, performance and flakiness

- **Accessibility:**
  - focus moves to the step heading on every step and route change (no visible ring on headings, rings kept on
    controls);
  - the progress header is an ordered list with `aria-current="step"`;
  - garment editing uses labelled controls;
  - the whole-photo option makes Find this piece fully keyboard-operable;
  - async updates use the polite live region;
  - touch targets are ≥ 44 px, except swatches at 24 px, which meets WCAG 2.2 AA;
  - colour is never the only signal (a ✓ on the chosen direction, `aria-pressed` on chips).
- **Performance:**
  - photos are downscaled in the browser to ≤ 1600 px JPEG before upload or storage (a few hundred KB);
  - gateway images use real recommendations, with lazy loading below the fold;
  - no new dependencies, still no build step.
- **Separating product bugs from flakiness:**
  - *Product bugs found and fixed:* the step header collapsed through a `.progress` class collision with
    Label; a 12-column gap made the gateway overflow; edits could drop to 3 picks because of same-family
    colourways; garment labels repeated the colour; the burgundy hover competed with the primary.
  - *Test or measurement issues:* Playwright exposes no request body for Blob/FormData uploads (fixed with route
    interception or in-page recording); measurements taken mid-transition; a hero read before the remote API
    finished at 375 px (the URL was correct, and it passed twice on re-run with an element wait).

## 10. Known limits and next steps

- **Detection quality is the model's.** The CC0 test photo's shirt dress came back as a "top" until corrected,
  which is what Review is for. A next step would be offering the description's own noun as a suggested category.
- **Occasion styling** reorders Complete the Look by text fit. It does not add new categories (for example a
  blazer for "the office") unless the stylist conversation does.
- **Assistant sessions** are process-local on one Modal container. Recovery is client-side; persistent server
  sessions would remove the need.
- **The free-text occasion** is limited to 60 characters and not otherwise interpreted.
- **Screenshots of the review step** depend on live detection, so README images are reproducible but not
  byte-identical.
- **Not done:** no screen-reader session (verified through semantics, focus and keyboard journeys) and no
  automated contrast tool (ratios computed from the tokens).
