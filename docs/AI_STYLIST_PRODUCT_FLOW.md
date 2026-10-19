# AI Stylist product flow and consumer UI architecture

**Status:** specification, written before implementation (2026-10-11). **Branch:** `stylist-product-flow`.
**Supersedes** the single-page Style Assistant of `CLAUDECODE_UI_EDITORIAL_REDESIGN_PLAN.md` §10. Where this
document and the earlier UI plans disagree on stylist flow, routes or tokens, this document controls.

Core journey: **choose a goal → provide text or a photo → confirm what we understood → see recommendations →
refine with the stylist.** Pages are organised by the decision the shopper is making, never by which model or
endpoint answers.

---

## 1. Current-state audit (main @ `21daeeb`)

| area | state |
| --- | --- |
| Hosting | GitHub Pages serves `portfolio/` (real frontend, `ensemble-api-base` meta) → FastAPI on Modal (`deploy/modal_app.py`, CPU, one container, scale-to-zero) → catalogue photos on Cloudflare R2. Claude Haiku capped at $10/month (Modal Dict counter). Label refused when hosted. |
| Routes (hash) | `#/` Discover (ranked home) · `#/collections[/id]` · `#/product/:id` · `#/assistant[?anchor=]` (chat + visual search + outfit analysis on one page) · `#/studio`, `#/studio/serving`, `#/studio/inspect`, `#/ds` (alias), `#/label` |
| Stylist page | Three jobs on one screen: a chat column, a photo slot with crop, a "find similar" form and an "analyse outfit" button, plus every result set below. Outfit analysis (`/api/snap`) detects and completes in one call — the shopper never sees or corrects what was detected. |
| APIs used by the UI | `/api/customers`, `/api/home/{c}`, `/api/product/{a}`, `/api/catalog/families`, `/api/v2/complete-the-look`, `/api/events`, `/api/visual-search`, `/api/snap`, `/api/assistant/session`, `/api/assistant/{sid}/message`, `/api/metrics`, `/readyz`, `/api/v2/meta`, `/api/v2/metrics`, `/api/explain/{c}/{a}` (Studio only), `/api/label*` (local only) |
| Strengths to keep | Track A ranked edit, product-family grouping with real colourway swatches, prototype USD prices, persistent per-profile cart, product pages, Track B Complete the Look, FashionCLIP + adapter visual search, grounded tool-calling assistant, DS Studio separation |
| Gaps | no entry guidance; stylist tasks compete on one page; no review/correction of detected garments; session loss only partially recovered (no context replay); editorial grid uses free vertical "drops" and 2:3 `contain` images rather than a fixed composition system |

## 2. Route map

| route | page | notes |
| --- | --- | --- |
| `#/` | **Gateway** — "Welcome back" and two solid choices (DS Studio via the masthead switch) | new; wordmark target in Shop |
| `#/shop` | **Your edit** (ranked, personalised home) | was `#/`; content unchanged |
| `#/collections[/:edit]`, `#/product/:id` | **Collections** (tabs: Selected for you · Buy it again · Trending); the "Your edit" page is removed and `#/shop[/:edit]` redirects here (D-048) | |
| `#/stylist` | **AI Stylist hub** — three tasks | new |
| `#/stylist/ask[?anchor=:id][&context=look]` | **Ask the stylist** — conversation | replaces the chat column |
| `#/stylist/find[?step=photo\|frame\|results]` | **Find this piece** | replaces the visual-search panel |
| `#/stylist/look[?step=photo\|review\|direction\|edit]` | **Complete my look** — four steps | replaces one-shot outfit analysis |
| `#/assistant[?anchor=]` | → redirects (`location.replace`) to `#/stylist/ask[?anchor=]` | legacy deep link kept |
| `#/studio…`, `#/ds`, `#/label` | DS Studio | unchanged |

Unknown routes keep the not-found state. Redirects use `history.replaceState`, so Back never bounces.

## 3. User goals and page responsibilities

| page | user goal | one primary action |
| --- | --- | --- |
| Gateway | decide how to shop today | **View your collections** or **Style with AI** (two equal editorial choices) |
| Your edit | browse what was picked for me | product card → Add to cart / open product |
| Stylist hub | pick the kind of help I need | one card per task |
| Ask the stylist | get advice in my words | Send |
| Find this piece | find a product like one in my photo | Photo → **tap the pieces** (spotted for you; or drag around one) → the four closest per piece |
| Complete my look | finish an outfit I am wearing | Photo → **Review** → **Your edit** (four picks of mixed kinds; D-050) |

## 4. Complete my look — state machine

```
          upload ok                  confirm (≥1 garment)        choose goal + minimal context
[photo] ───────────▶ [detecting] ───▶ [review] ───────────────▶ [direction] ───────────────▶ [building] ─▶ [edit]
   ▲  replace/remove      │ fail            ▲  edit/add/delete            ▲ Back                     │ fail      │
   │                      ▼                 │                             │                          ▼           │
   └──────────────── [detect-error: retry | add pieces yourself]          └──────────────── [edit-error: retry]  │
   └────────────────────────────── Start over (clears sessionStorage) ─────────────────────────────────────────── ┘
                                                                     [edit] ──Refine this edit──▶ #/stylist/ask?context=look
```

- **Never skipped:** detection always lands on Review; the edit is built only after the shopper confirms.
- **Persistence:** `sessionStorage["ensemble.look.v1"]` keeps `{photo (downscaled JPEG data URL ≤ 1600 px), garments,
  goal, occasion, result, step}` per tab. Back/forward/refresh restore the step; **Start over** clears it.
- **Step URL:** `?step=` mirrors the state; a step that is not yet reachable (e.g. `edit` without a result) falls back
  to the furthest valid step.

Find this piece follows the same pattern with `ensemble.find.v2` and steps `photo → frame (Choose) → results`.

## 5. Data and API contracts

Existing endpoints are unchanged. Two **additive** endpoints implement the staged outfit flow:

### `POST /api/stylist/detect` (multipart: `photo`)

Detect garments only, with the same vision client and budget cap as `/api/snap`.
```json
{"garments": [{"id": "g1", "category": "knitwear", "colour": "beige", "description": "oversized sweatshirt",
               "box": [0.21, 0.18, 0.79, 0.55]}],        // fractions of the image, or null
 "categories": ["top", "outerwear", …]}
```
Errors: `413/415/400` invalid upload; `503 detector_unavailable` (consumer copy: "We couldn't look at that photo
just now"); `503 ai_budget_reached`.

### `POST /api/stylist/edit` (multipart: `photo`, `garments` JSON, `goal`, `occasion?`, `customer?`)

`garments`: the shopper-confirmed list `[{category, colour?, description?, box?}]` (1–8 items; categories from
`CATEGORIES`; boxes as fractions). `goal ∈ {complete, occasion, similar}`; `occasion` free text ≤ 60 chars.
```json
{"goal": "complete", "summary": "We see a beige knitwear and light blue bottom. Here is what finishes it.",
 "matched": [{"category": "knitwear", "article_id": 907903001}],
 "picks": [{"card": {…article card…}, "reason": "Completes the look: shoes", "group": "shoes"}],     // 4–6
 "alternatives": [{"title": "Shoes", "items": [{…card…}]}],
 "anchor": 907903001}
```
- `similar`: per garment, catalogue matches by the existing text-rerank-image search (colour/description edits feed
  the text query); picks = best match per garment.
- `complete`: anchor = best match of the main garment; Complete the Look for the missing slots (serving bundle if
  loaded, stored table otherwise); picks = round-robin across slots.
- `occasion`: as `complete`, then the pool is reordered by FashionCLIP text similarity to the occasion; reasons name it.
No product appears that a model did not return. Reasons are consumer phrases, never scores or model names.

### Assistant session recovery
The client keeps a replayable context (`{context_message, last_user_message}`). On `404` from
`/api/assistant/{sid}/message` it creates a new session, replays the context message once, then resends the
last user message. The shopper sees a quiet note, never a raw error.

## 6. Navigation rules

- Every stylist step shows **Back**, the **progress header**, **Change task** (→ hub), and exactly **one dominant
  next action** (black button). Secondary actions are quiet links.
- The masthead's experience switch, profile and cart stay; Shop navigation: *Collections · AI Stylist*; shoppers see no footer (D-048).
- The wordmark returns to the gateway; nothing auto-redirects, so the gateway and task switch are always one
  click away. The gateway is deliberately plain ("Welcome back", two black buttons, one line each, centred; the question
  below is the page heading for screen readers only). A touch
  sends a soft ripple from the point of contact, and choosing a path fades the page away (≈ 0.5 s, the chosen
  button last) before the next view enters. Both are off under `prefers-reduced-motion`, and modified clicks
  (open in new tab) are left to the browser.

## 7. Consumer microcopy

| place | copy |
| --- | --- |
| Gateway | "Welcome back" (heading, screen readers only: "How would you like to shop today?") · **View your collections** — "Pieces chosen for you, from what you love." · **Style with AI** — "Ask a stylist, find a piece from a photo, or finish an outfit." |
| Hub | "What can we help with?" · Ask the stylist · Find this piece · Complete my look |
| Look step 1 | "Show us your outfit" · "A full-length, well-lit photo works best. Your photo is only used to find pieces for you." |
| Detecting | "Looking at your outfit…" → "Picking out each piece…" |
| Look step 2 | "Is this right?" · "Fix anything we missed before we style it." |
| Look step 3 | "What would you like?" · Complete the look · Dress for an occasion · Find similar pieces |
| Building | "Finding pieces that go with it…" → "Putting your edit together…" |
| Look step 4 | "Your edit" · summary sentence · "More options" · "Refine this edit" |
| Errors | "We couldn't look at that photo just now. Please try again." · "That file is not a photo we can read. Please choose a JPEG, PNG or WebP." · "This feature is resting for now. Please try again later." |

Banned on consumer pages: offline, ranker, FashionCLIP, DeepFashion2, tool calling, model, API, raw ids, scores.

## 8. Loading / error / empty / success states

Every async step has: a staged status line (`role="status"`), disabled primary action while pending, a retry button
on failure that keeps the shopper's inputs, an empty state with a next step ("We couldn't find any pieces. Add or
fix a piece above."), and a success state that moves focus to the new heading.

## 9. Accessibility

Semantic headings per step; the progress header is an ordered list with `aria-current="step"`; focus moves to the
step heading after every step/route change; async results are announced through the polite live region; garment
editing uses labelled `<select>`/`<input>` controls with visible focus; the framing tool has a keyboard
"Use the whole photo" option and arrow-key nudging is not required; all controls ≥ 44 px; motion 180–240 ms and
removed under `prefers-reduced-motion`.

## 10. Responsive behaviour

Breakpoints tested at 320, 375, 768, 1024, 1280, 1440. Desktop: expressive 12-column editorial; tablet (≤ 1024):
disciplined two columns; mobile (≤ 600): single column for stylist steps and gateway, two columns for product
grids. Progress header collapses to "Step 2 of 4 · Review" under 480 px. No horizontal scroll at any width.

## 11. Editorial grid (controlled asymmetry)

A 12-column grid with three preset compositions, chosen deterministically by row index:
- **Feature** — one 3:4 image across 6 columns spanning two rows; two 3:4 images (3 + 3) in the first row; a text or
  product-detail block filling the second row beside it.
- **Trio** — three 3:4 images across 4 + 4 + 4, tops aligned.
- **Pair** — 4:3 image across 7 columns with a 3:4 image across 5, baselines aligned.
Images use only 3:4, 1:1 or 4:3 with `object-fit: cover` and a focal point (`--focus-y`, default 42 %). Rows never
leave holes: an incomplete last row falls back to the Trio preset. Product-family grouping applies across the page.

## 12. Acceptance criteria (measurable)

1. `#/` shows the gateway; **View your collections** reaches Collections in one click (after the fade); the masthead's
   DS Studio switch reaches DS Studio; `#/assistant?anchor=X` lands on `#/stylist/ask?anchor=X`.
2. The hub offers exactly three tasks, each on its own route.
3. Complete my look cannot reach step 4 without passing Review; a deleted/edited/added garment changes the request
   payload; refresh on any step restores it; **Start over** clears it.
4. Step 4 shows 4–6 picks with price, swatches and Add to cart; alternatives are behind one control.
5. **Refine this edit** opens Ask with the look's context; a forced session loss recovers without a visible error.
6. Find this piece: spotted pieces are chosen by tapping (photo or list, keyboard operable), each chosen piece shows an editable category, a framed area takes the category it covers, and results are the four closest real visual matches per piece.
7. No banned term on any consumer route (automated check); one family per page (automated check).
8. At 320–1440 px: no horizontal overflow; one dominant CTA per stylist step; focus lands on the step heading.
9. Reduced motion removes transitions; all journeys pass keyboard-only.
10. Full test suite and all browser journeys pass locally and on the public URL before release.
