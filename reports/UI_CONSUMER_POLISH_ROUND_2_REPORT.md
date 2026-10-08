# Consumer UI polish, round two: implementation report

2026-10-08 · branch `ui-editorial-redesign` (from the plan commit `e19a079`) · local only: nothing pushed,
merged, deployed, published or opened as a PR. Plan: `docs/CLAUDECODE_UI_CONSUMER_POLISH_ROUND_2.md`, which
takes precedence over the first UI plan where they conflict.

**Result:**
- **What shoppers see:** they shop a store. Product cards carry real colourway swatches, a stable USD price
  and a black/white **Add to cart** that feeds a working, persistent cart. Each product family appears once
  per page.
- **What moved to DS Studio:** every recommendation explanation, technical note and demo disclaimer. Studio
  remains fully functional.
- **Verification:**
  - 191/191 tests pass (178 baseline + 13 new).
  - 116/116 real-browser checks pass.
  - 54 screenshots were reviewed at 1440×900, 1024×768 and 390×844.

## 1. The 16 requests: implementation and evidence

The checks named in the evidence column are in `scripts/ui/journeys.js`.

| # | request | implementation | evidence |
| --- | --- | --- | --- |
| 1 | Remove *Why this* from shopper surfaces | All Why controls and the Why dialog are gone, and shopper modules never call `/api/explain`. *Not for me* becomes **Hide** (×) on the image, keeping the `not_for_me` event and never sitting in the cart slot. | `copy` and `home` journeys; `test_shopper_modules_never_explain_or_show_why_this`; the `no /api/explain` journey check |
| 2 | Black **Add to cart** with white text on every card and the product page | `.btn-cart`: `#000` background and `#fff` text, restrained hover, press and focus states, an accessible name with product and colour, disabled while pending | `auditCards` checks the computed colours, label and text on every visible card; the product-page check |
| 3 | Functional persistent cart | Per-profile `localStorage` cart, masthead count, accessible drawer (§5) | `cart` journey (14 checks); `tests/js` cart tests |
| 4 | Page-wide deduplication | One card per product family per page, first occurrence wins (§2) | `home`, `product`, `collections`, `assistant` and `visual` checks report `dupFamilies: 0`; 6 JS grouping tests; 3 API tests |
| 5 | Colour swatches under the name | Real colourways as labelled swatch buttons that update the card in place (§3) | `swatch` journey: all variants exposed; image, link, colour, price and cart target update |
| 6 | Consistent USD prices | Prototype merchandising-price layer (§4) | `test_merch.py` (5 tests); every card shows `$NN.NN`; the same price in the cart |
| 7 | New home introduction | "A personal edit, made with you in mind." | `copy` journey (exact string) |
| 8 | Purchased-items copy | "Pieces you have bought." | `copy` journey |
| 9 | Remove counts and profile suffixes | No "· 12 pieces", "12 pieces · for Shopper, 26", tab, filter or colourway counts, rank numbers, or "most recent first" | `copy` journey on 6 shopper routes, with a regex blocklist |
| 10 | Remove the footer disclaimer | The shopper footer is the wordmark plus shop links. The provenance note shows only in DS Studio. | `copy` (footer) and `studio` (`.footer__studio` displayed) checks |
| 11 | Remove the service status | The shopper utility bar is empty. The bundle-readiness note appears only in DS Studio. Live Complete the Look failures fall back quietly. | `copy`, `studio`, and `offline` (forced 503: no service wording) checks |
| 12 | Assistant copy | "Tell us what you're looking for, or bring a photo. We'll help you find pieces that feel right." No tool, model, backend, trace or "never invents" copy. | `copy` and `assistant` checks |
| 13 | Time estimates | "It may take a few seconds." and "It may take about a minute."; no "laptop" | `copy` journey |
| 14 | Wordmark typography | Didone rhythm: an oversized italic initial *e*, a roman body and an italic *em* echo. `aria-label="ensemble"`; letter spans `aria-hidden`. Local serif stack only. | `center` journey: centre 720.00/720, 512.00/512, 194.99/195; no collision with menu, profile or cart; `test_landmarks_and_centered_lowercase_wordmark` |
| 15 | White shopper background | `--canvas: #ffffff`; automatic dark mode removed; photos keep their studio backdrop inside the frame | `center` journey with OS dark mode emulated: body `rgb(255,255,255)`; static test |
| 16 | Category tabs grey to black | Inactive `#767676`, selected `#111` with an underline and `aria-current` / `aria-pressed`, never a pill. Works in scrollable rows and by keyboard. | `collections` journey (computed colours, transparent backgrounds, keyboard Enter) |

The plan's §10 separation audit is enforced by the same blocklist on every shopper route: ranking, model,
tool, API, SHAP, evidence, offline, telemetry, dataset, demo and service wording. In DS Studio, Inspect still
shows the evidence-gated reasons and SHAP, now labelled "DS only; not shown to shoppers". The Evaluation and
Serving pages and the technical footer are unchanged.

## 2. Product-family grouping

- **Key.** The catalogue's own `product_code`. Verified on the local data: `article_id // 1000 ==
  int(product_code)` for all 105,542 articles (asserted by `test_product_code_convention_holds_for_the_whole_catalogue`).
  The UI does not derive it: the additive endpoint `GET /api/catalog/families?articles=…` returns each
  article's real `product_code` from the `articles` table.
- **Never grouped by name.** 3,072 product names are shared by more than one family, so names are ignored
  (JS test and `test_same_name_products_from_different_families_are_not_merged`).
- **Rules** (`createPageGroups` in `static/js/catalog.js`):
  1. Exact duplicate articles are removed page-wide.
  2. Colourways of one family collapse into the first card.
  3. The first occurrence in display order owns the card's position and module; later occurrences only
     enrich it, as swatches.
  4. A module emptied by grouping shows its remaining unique items or is omitted. Nothing is cloned to fill
     space; for example, on the default profile Buy it again shows 2 cards because its other purchases are
     already on the page.
  5. If the family lookup omits an article, only its exact id is used.
- **Scope.**
  - Discover groups the hero, Selected for you, Complete the look, Buy it again and Trending together. The
    live Complete the Look is fetched before rendering so display order decides ownership.
  - Collections, and the product page. The product page claims its own family first, so its colourways are
    swatches, never cards.
  - Each assistant answer, visual-search result and outfit analysis is grouped on its own. Earlier chat
    messages keep their historical products.
- **Analytics.** Impressions are now logged once per *shown* card (`impression` count equals the number of
  visible cards in the `home` check). The event types and payloads are unchanged.

## 3. Swatches

- **Data.** The family's real articles from the catalogue, in `article_id` order, limited to colourways with a
  local product photo. The requested article is always included. No colourway is invented.
- **Cards** show up to 6 swatches. The selected colour comes first and is never cut off. A "+N" link to the
  product page reveals the rest; families reach 75 colourways. **The product page shows every colourway.**
- **Labels.**
  - Shopper-friendly colour names: "Other Pink" becomes "Pink (other shade)", "Unknown" becomes "Assorted",
    "Transparent" becomes "Clear".
  - Two articles of one family can share a colour group: 8,791 such pairs, differing by print or shade, a
    column the serving store does not carry. These are labelled "Black, option 1/2" rather than with a
    guessed name.
- **Mapping.** An explicit table covers all 47 catalogue colour groups. "Other", "Unknown" and multicolour
  values get a labelled neutral hatched swatch; "Transparent" gets a checker.
- **Accessibility.**
  - Every swatch is a `<button>` with `aria-pressed` and a label such as "Black, selected", plus a black ring
    and underline, so selection is not shown by colour alone.
  - The hit area is 24×24 px, which meets the WCAG 2.2 AA minimum target size.
- **Selecting a swatch** updates, in place, the image, alt text, name, links (`#/product/<id>`), colour label,
  price, Hide label and Add to cart target. On the product page it also updates the title, the facts and the
  address, using `history.replaceState`.

## 4. Prototype USD prices

**These are portfolio/demo merchandising prices, not historical H&M or source-dataset prices.** The source
transactions only carry a normalised, unit-less `price`. The note lives in code comments
(`src/ensemble/api/merch.py`, `configs/merch_pricing.yaml`) and in this report, not on shopper pages.

- **Policy** (checked in):
  1. A band by `product_type_name` (55 types), else by `product_group_name`, else a default.
  2. Scaling by index group (Baby/Children ×0.6, Divided ×0.8) and ×1.6 for "Premium" departments.
  3. A choice among standard US price points inside the band (`…9.99`, `…4.99`, `99.00`, `119.00`, …) using
     `sha256(product_code)`.
- **Family-stable.** The band uses the family's canonical, lowest-`article_id` article, so every colourway
  shares one price.
- **Deterministic** across reloads, routes, customers and processes (a test runs a fresh interpreter with a
  different `PYTHONHASHSEED`).
- **Coverage.** Across all 47,224 families: 46,851 end in `.99` and 373 in `.00`. Prices range from $3.99 to
  $199.00, with a median of $24.99. Example: Skye NW flared trs (Trousers) is $34.99 in every colour.
- **One source.** Prices are computed server-side and sent with the family data. Every surface (home,
  collections, product, cart, assistant, visual search, outfit results, related modules) reads the same value
  and formats it with `Intl.NumberFormat("en-US", {style: "currency", currency: "USD"})`.
- **Isolation.** Nothing feeds a model, a feature or an evaluation.

## 5. Cart

- **Model.** The pure `Cart` class in `catalog.js` holds one line per article, so the same article *and*
  colour increments rather than duplicating. Lines record name, colour label, image and price; quantity is
  capped at 10.
- **Persistence.** `localStorage` key `ensemble.cart.<customer>`, one cart per demo profile, so switching
  profiles never shows another shopper's cart. Storage failures (blocked or full) fall back to memory, and
  the drawer says "Your cart couldn't be saved on this device…" without breaking shopping (tested in the
  browser and in JS).
- **Masthead.** A bag icon with a black count badge and an accessible name such as "Cart, 2 items". It is
  hidden in DS Studio.
- **Drawer.** A native modal `<dialog>` that slides in from the right, with Escape and focus return. Each line
  shows image, name, colour, unit price, − / + quantity, line total and Remove. Below the lines are the
  subtotal and a disabled **Checkout** (payment is out of scope). The empty state links to Continue shopping.
- **Add to cart.**
  - Disabled while pending, with an "Added" confirmation and a live announcement.
  - It logs the existing `add_to_cart` event with the **selected colourway's** `article_id`, its surface
    (`for_you`, `complete_the_look`, …, or `product_page`) and the anchor where present.

## 6. Copy changes (exact)

| where | before | after |
| --- | --- | --- |
| Home lede (returning) | A personal edit, ranked from your own purchases and what is selling now. | A personal edit, made with you in mind. |
| Home lede (new) | No purchase history yet, so this first edit is drawn from what shoppers your age are buying this week. | A first edit, chosen from what shoppers your age love this week. |
| Home eyebrow | Selected for you · 12 pieces | Selected for you |
| Hero meta | Shopping as Shopper, 26 · 621 items bought · last order … | *(removed; the links remain: Shop the edit · Ask the stylist)* |
| Edit aside | Larger pieces rank higher in your edit | *(removed)* |
| Complete the look | Built around No. 01 / What our outfit model pairs with … | Styled with the {product} |
| Buy it again | Pieces you have bought, most recent first. · "Last bought … · 2 times" | Pieces you have bought. · "Last bought Sep 22, 2020" |
| Trending | Most bought this week by shoppers in your age group. | Popular this week with shoppers your age. |
| Collections | "12 pieces · for Shopper, 26", tab counts, "All 12" / "Trousers 7" | *(no counts)* |
| Product page | Demo badge; "Demo shop: there is no checkout or price…"; "Why this is in your edit"; personalised/compatibility tags; "How this look was assembled"; "Showing stored suggestions…" | *(removed)*; Complete the look lede "Pieces that go with this one." |
| Footer | "…there are no prices and no checkout. Every metric shown is offline unless labelled as live telemetry of this demo server." | *(shopper footer: links only; the technical sentence shows only in DS Studio)* |
| Utility bar | Offline demo on the H&M dataset · no checkout / Live outfit service unavailable · showing stored suggestions | *(empty in Shop; DS Studio shows bundle readiness and the offline-metrics note)* |
| Assistant intro | Ask in plain language, or bring a photo. Every piece shown comes from the recommender's own tools; the assistant never invents products. | Tell us what you're looking for, or bring a photo. We'll help you find pieces that feel right. |
| Assistant status | Local model · Thinking · may call the recommender tools · tool trace, latency, cost, "Removed N unverified products" | Finding pieces for you… *(no trace)* |
| Visual search | …Takes a few seconds. | …It may take a few seconds. |
| Outfit | A vision model identifies… Takes about a minute on this laptop. | We'll recognise each piece you're wearing… It may take about a minute. |
| Photo privacy | Photos go only to this demo's own server… on this machine… model provider… | Your photo is only used to find pieces for you. |
| Errors | "…temporarily unavailable on the demo server", raw messages | "We can't connect right now…", "We couldn't look at that photo just now. Please try again." |
| Profile dialog | Choose a demo profile / …customers from the dataset… | Choose a profile / Each profile is a real, anonymised shopper… |

## 7. Visual system

- **Background.** Pure white canvas. The near-black ink is `#111`, muted text `#5e5e5e` (6.5:1) and faint
  text `#767676` (4.5:1, used for inactive tabs).
- **Dark mode.** No automatic dark mode: the OS dark preference is ignored (verified by emulation).
- **Fonts.** Product names, section and page titles use the local Didone serif in a responsive hierarchy.
  Prices, swatch labels, buttons and navigation use the sans-serif.

## 8. Screenshots and viewport review

**Final set:** 54 images (27 viewport plus 27 full-page) in the gitignored `data/interim/ui_editorial_screenshots/round2/viewports/`. Captured with `scripts/ui/shots.js` at 1440×900, 1024×768 and 390×844 for:
- Shop: home, product, collections, trending;
- the Style Assistant;
- DS Studio: Evaluation, Serving, Inspect;
- Label.

Every capture has 0 px horizontal overflow and no console errors. Journey captures, including the cart
drawer, wordmark crops, tabs, visual search and outfit, are in `journeys_r2/`.

Defects found during review and fixed in `5e54a5d`:
- the asymmetric "Selected for you" grid collapsed to full-width stacked items, because the CSS refactor had
  dropped its `.edit` rules;
- Add to cart buttons were misaligned within a row when colour labels wrapped, fixed with equal-height cards;
- "Add to cart" wrapped onto two lines in the narrow mobile hero column, fixed with a full-width lead plus a
  two-up row and a no-wrap label;
- the cart drawer used the generic fade, now a slide-in from the right.

Wordmark check: readable at all three widths, with the italic initial and echo visible, no collision with the
menu, profile or cart controls at 390 px, and exact centring.

## 9. Automated and browser results

```text
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider
  baseline before edits (e19a079): 178 passed, 1 warning
  final:                            191 passed, 1 warning
    new: tests/test_merch.py (5) · tests/test_api.py (+4, families/convention/same-name/validation)
         tests/test_ui_catalog_js.py (1 → runs tests/js/catalog.test.mjs: 14 node assertions groups)
         tests/test_ui_static.py (+3: shopper modules never explain, white canvas + black cart, cart drawer)

PLAYWRIGHT=<local playwright> node scripts/ui/journeys.js <journeys>   (server on :8031, Chrome)
  center,copy,home,swatch,switch,product,collections,studio,label,keyboard,motion,offline   106/106
  assistant,visual (local LLM and vision model; 36 s answer)                               10/10
```

The browser journeys, in the order of plan §11:

1. Home has no Why this, category metadata, counts, disclaimer or duplicate family.
2. Every visible card has a USD price and a black Add to cart.
3. Swatches expose all real colourways and update image, colour, link, price and cart payload.
4. Adding twice increments the quantity; decrement, remove and subtotal work; the cart survives a reload.
5. Switching profile leaves no stale cards or duplicate families, and the cart is per profile.
6. The product page uses swatches and the cart, with no Other colours gallery.
7. Assistant, visual-search and outfit cards are consistent, with no technical copy.
8. The shopper header and footer are clean.
9. DS Studio still shows its technical content.
10. Label still works; the save is mocked, so no gold labels were written.
11. Tabs are grey when inactive and black when selected, including by keyboard.
12. The wordmark is exactly centred at all three widths.
13. Visual review (§8).
14. No overflow, console errors or failed requests.
15. Reduced motion and keyboard-only flows still work.

Mocks were used only for the Label save, a forced `/api/product` network failure and a forced 503 from the
live Complete the Look.

## 10. Known limitations

- **No checkout.** Checkout is disabled; payment is out of scope.
- **Prices are prototype values** (see §4) and would be replaced by a real price feed.
- **Same-colour colourways** (8,791 pairs) are labelled "option 1/2" because the serving store has no print
  or appearance column.
- **Swatch scope.** Colourways are all photographed catalogue articles of the family, as the old Other colours
  module did. The live-availability filter from the serving bundle is not applied to swatches.
- **Swatch size.** Hit areas are 24 px, which meets WCAG 2.2 AA but is below the 44 px comfort target used
  for other controls; this is the usual retail density.
- **Impressions** are logged per shown, grouped card, which is the correct semantics. Earlier counts
  included duplicates, so impression totals are not comparable across the change.
- **Event side effects.** The journeys wrote ordinary demo events, including `add_to_cart` and `not_for_me`,
  to the gitignored `serving.sqlite`, as normal use does.

## 11. Commits (round two)

```text
e19a079 docs(ui): define consumer polish round two                                     (plan, pre-existing)
153f1ff feat(api): product families, real colourways and prototype USD prices for the shop UI
1eaff0f feat(ui): consumer polish round two — cart, swatches, prices, page-wide grouping, shopper copy
94d8813 test(ui): grouping, swatch and cart unit tests (node) and round-two browser journeys
5e54a5d fix(ui): visual review fixes for round two
81b7495 test(ui): stabilise cart-button rule check and keyboard journey timing
<this report> docs(report): consumer UI polish round two report
```

## 12. `git status --short` audit

Recorded after the report commit. **No tracked file is modified or staged.** The only untracked entries are
user-owned and were preserved untouched:
- 33 `* 2.*` sync duplicates;
- `reports/track_a_retrain/logs/`.

Screenshots and journey results live in the gitignored `data/interim/ui_editorial_screenshots/`. The server
on port 8010 was not touched; the verification server on 8031 was stopped.

## 13. Follow-ups after review (2026-10-08)

| feedback | change | verified by |
| --- | --- | --- |
| Swatches only worked once; Add to cart did not add after a colour change | Every colourway of a loaded family is registered, so a card resolves its colours and price after any swatch change. A successful add opens the cart drawer with the new line marked. | `swatch` journey regression: cycle every swatch back to the first and add each (`5fbd72b`) |
| Navigation-level words too small | Main navigation, collection tabs and category filters use `--fs-nav` (0.98 rem, previously 0.8125). Eyebrow and small uppercase labels use 0.76 rem (previously 0.6875). Footer and section links are slightly larger. | screenshots at 1440 / 1024 / 390 px |
| Wordmark | Every letter is now a Didone italic like the initial *e*, each at its own scale (0.92–1.30 em) on a shared baseline. It keeps one accessible name, `ensemble`, with decorative spans, and stays exactly centred with no collisions. | `center` journey; `test_landmarks_and_centered_lowercase_wordmark` |
| Remove the delete (×) button; use "Not for me" | The × is gone. **Not for me** sits beside the price, keeps the `not_for_me` event and dims the card in place with "Got it. We'll remember that next time." The promise is real: the product family is stored per profile in `localStorage` (`ensemble.notforme.<customer>`) and excluded from every page rendered afterwards. | `home` journey (event, message, gone after reload); JS test "families marked Not for me never render on later pages" |

Results after these follow-ups: 191 tests passed; browser journeys 110/110 fast and 10/10 model-backed.
