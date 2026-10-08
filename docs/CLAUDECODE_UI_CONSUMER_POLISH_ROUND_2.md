# Consumer UI Polish — Round 2 Execution Plan

## Mission

Apply the user's post-review feedback to the completed editorial redesign. The customer experience must now read and behave like a real fashion store rather than a recommendation-system demonstration. Preserve DS Studio as the place for technical explanations, model details, degraded-service notices, and evaluation context.

Work autonomously through implementation, visual review, regression tests, and reporting. This plan supplements `docs/CLAUDECODE_UI_EDITORIAL_REDESIGN_PLAN.md`; where the two conflict, this round-two plan controls.

## Repository and safety rules

- Continue on `ui-editorial-redesign` from the current committed state.
- Read `CLAUDE.md` and inspect the current implementation before editing.
- Preserve all existing recommendation, analytics, assistant, visual-search, outfit, Label, and DS API contracts.
- Preserve every user-owned untracked `* 2.*` file and `reports/track_a_retrain/logs/`.
- Do not push, merge, deploy, publish, or open a pull request.
- Use coherent conventional commits.
- Do not add a frontend framework, package manager, build step, remote font, analytics service, or unnecessary dependency.
- Keep the existing local server on port 8010 untouched. Use another port for verification and stop it afterward.

## 1. Remove recommendation explanations from the shopper experience

Remove every shopper-facing `Why this` control and explanation dialog entry point from:

- home and collection product cards;
- product detail pages;
- Complete the Look and related product cards;
- assistant, visual-search, and outfit result cards.

Do not fetch `/api/explain/...` from a shopper route. Recommendation evidence, SHAP detail, and diagnostic explanation may remain available only in DS Studio if there is a natural existing location for it. Do not create a large new DS feature solely to relocate the dialog.

Also remove shopper-facing `Not for me` controls unless they are presented as a clearly useful retail action such as removing/hiding a card. If retained, relabel it consumer-first (`Hide this`) and keep the existing event contract. It must not sit where the requested Add to cart action belongs.

## 2. Add a functional Add to cart experience

Every shopper-facing product card must have an `Add to cart` button with:

- white text;
- solid black background;
- restrained hover/focus/pressed states;
- a useful accessible name including the product name;
- disabled/pending protection against duplicate clicks.

The product detail page must use the same wording and style. Remove the `Demo` badge and all copy saying that checkout or prices do not exist.

Implement a lightweight prototype cart rather than a dead button:

- persist cart contents in `localStorage`;
- show a cart control/count in the shopper masthead;
- open an accessible cart drawer/dialog listing item image, name, selected colour, price, quantity controls, line total, subtotal, and remove action;
- allow the same article/colour to increment quantity rather than create duplicate cart lines;
- changing a product-card colour swatch changes which article/colour is added;
- preserve the existing `add_to_cart` analytics event with the correct selected article ID and surface;
- clearly handle storage or rendering errors without breaking shopping;
- cart state must not appear in DS Studio navigation unless represented by the shared customer masthead.

A full payment or checkout implementation is out of scope. A disabled or clearly non-transactional `Checkout` action is acceptable, but do not add technical disclaimer paragraphs to normal shopping pages.

## 3. Page-wide product deduplication and colour grouping

The same underlying product must appear only once on a shopper page, even if recommendation modules return it repeatedly.

First inspect the catalogue schema and establish the correct product-family key. For the H&M article convention, prefer the real product/style identifier already present in data; if absent from API responses, derive the documented product code safely from `article_id` only after verifying the convention against the local data. Do not group merely because two products share a name.

Implement these rules:

- exact duplicate article IDs are removed page-wide;
- colour variants belonging to one verified product family are grouped into one card;
- the first/highest-ranked occurrence owns the card's position and module;
- subsequent occurrences and colourways enrich that card rather than rendering again;
- if a later module becomes empty, show the remaining legitimate unique items or omit that module cleanly—never clone products to fill space;
- deduplication applies across the whole current shopper view, not only inside each module;
- assistant messages may retain historically returned products in prior messages, but each newly rendered result group must be deduplicated and colour-grouped.

Add focused tests for exact duplicates, same-family colour variants, unrelated same-name products, ordering, empty modules, and selected-swatch add-to-cart behavior.

## 4. Colour swatches

Under every shopper product image, show the product name and its available colour swatches as one clean information group. The swatches should sit beside or immediately below the name depending on available width.

Requirements:

- show all locally available colourways for that grouped product that can be supported by real catalogue data;
- include the currently selected article's colour;
- show the selected swatch with a black outline/check or another non-colour-only indicator;
- each swatch is a real button with an accessible label such as `Black, selected`;
- selecting a swatch updates the card image, link target/article ID, colour label, price, and Add to cart target without moving the card;
- show the colour name in shopper-friendly text near the swatches;
- do not show product category/type/department on shopper cards;
- use an explicit mapping from known catalogue colour groups to visual swatch colours, with a labelled neutral fallback for unknown or multicolour values;
- do not pretend a colourway exists unless a real catalogue article supports it.

The product detail view should use the same component/behavior rather than keeping a separate redundant `Other colours` gallery when variants can be selected directly. Similar items and Complete the Look remain separate product relationships.

## 5. USD prototype price catalogue

The source catalogue has no trustworthy retail USD price field, but the user wants the prototype to behave like a real US store. Add a documented **prototype merchandising-price layer** rather than claiming the values are historical H&M prices.

Implementation requirements:

- generate stable USD prices from real product attributes and a checked-in, documented pricing policy;
- use sensible apparel/accessory price bands and standard retail endings such as `.00`, `.50`, `.90`, or `.99`;
- the same product family should normally keep one price across colourways;
- output must be deterministic across reloads, routes, customers, and processes;
- centralize the logic so home, collections, product detail, cart, assistant cards, visual search, outfit results, and related-product modules always agree;
- format through `Intl.NumberFormat("en-US", {style: "currency", currency: "USD"})`;
- do not infer random prices at render time;
- explain in code comments and the completion report that these are portfolio/demo merchandising prices, not source-dataset prices; do not clutter shopper pages with that technical note.

Prefer a small shared catalog/merchandising function with unit tests. A backward-compatible API price field is acceptable if it makes consistency and testing stronger; do not modify model features or evaluation data.

## 6. Exact customer-copy changes

Replace the current home introduction:

- from: `A personal edit, ranked from your own purchases and what is selling now.`
- to: `A personal edit, made with you in mind.`

Use that exact replacement unless a grammatical context requires punctuation only.

Change the purchased-items description:

- from: `Pieces you have bought, most recent first.`
- to: `Pieces you have bought.`

Remove all customer-facing item counts and profile suffixes, including patterns such as:

- `Selected for you · 12 pieces`;
- `12 pieces · for Shopper, 26`;
- `6 pieces`, `colourways`, or similar counts beside customer section titles;
- repeated `most recent first` copy.

The profile selector itself may still say `Shopper, 26`; the instruction is to remove that diagnostic/profile suffix from section headings and descriptive content, not to make profiles indistinguishable.

Remove this footer/customer disclaimer completely:

`there are no prices and no checkout. Every metric shown is offline unless labelled as live telemetry of this demo server.`

Technical metric provenance belongs in DS Studio only.

Remove this shopper utility status completely:

`Live outfit service unavailable · showing stored suggestions`

The app may still use the safe fallback, but the global consumer header must not announce backend architecture. If the user actively requests an outfit and live processing fails, show a concise task-local message such as `Here are outfit ideas available right now.` without mentioning services, storage, models, or telemetry.

## 7. Style Assistant copy changes

Replace the assistant introduction with consumer-facing copy. Use:

`Tell us what you're looking for, or bring a photo. We'll help you find pieces that feel right.`

Remove:

- `Every piece shown comes from the recommender's own tools`;
- `the assistant never invents products`;
- other implementation-focused claims from the shopper experience.

Time estimates:

- replace `Takes a few seconds.` with `It may take a few seconds.`;
- replace `Takes about a minute on this laptop.` with `It may take about a minute.`;
- remove references to `this laptop`, local models, endpoints, or technical execution from customer copy.

Keep honest progress, success, and failure states.

## 8. Typography and wordmark refinement

Use the attached editorial-fashion reference only for direction: an expressive, high-contrast Didone/italic display treatment with visibly different letter or word scales. Do not copy the reference text, braces, artwork, or exact trademark styling.

Refine the lowercase `ensemble` wordmark:

- preserve exact mathematical centering at 1440, 1024, and 390 px;
- use the existing local/system high-contrast serif stack—no remote font;
- create a tasteful variation in scale/baseline or italic/roman rhythm within the wordmark, while keeping `ensemble` immediately readable;
- keep one accessible textual name (`aria-label="ensemble"`) and hide purely decorative letter spans from assistive technology;
- avoid a novelty ransom-note effect;
- verify that the logo does not collide with menu, profile, or cart controls at mobile width.

Product names and major shopper headings should use the refined fashion serif with a deliberate responsive size hierarchy. Utility copy, price, swatches, buttons, and navigation remain clean sans-serif.

## 9. White visual system and navigation state

Change the shopper page background to pure white. Remove the automatic dark-mode color change for the customer experience. Product imagery may keep its existing studio backdrop inside the image frame.

Update category/collection/filter navigation to match the second reference's interaction language:

- unselected category names are grey text on white;
- selected category is black text, with weight/underline if needed;
- do not use a black pill or black filled rectangle for selected categories;
- hover/focus becomes black while preserving a clear focus indicator;
- the treatment must work in horizontally scrollable mobile tab rows;
- use semantic current/selected states and do not communicate selection by colour alone.

The Shop / DS Studio experience switch may keep a distinct compact treatment if necessary, but shopper collection/category tabs follow the grey-to-black rule.

## 10. Shopper/DS separation audit

Audit all shopper routes and move or remove system-oriented language, including:

- ranking terminology;
- offline/live metric disclaimers;
- retrieval/model/tool/API terminology;
- fallback-service status;
- dataset limitations;
- evidence and SHAP explanations.

Do not remove these facts from DS Studio or technical reports. The goal is proper information architecture, not concealment from developers.

## 11. Required regression and browser verification

Run the existing 178-test baseline first, then add tests for this round and run the entire suite.

At minimum verify in a real browser:

1. Home contains no `Why this`, category metadata, item counts, technical disclaimer, or duplicate product family.
2. Every visible shopper product card has a formatted USD price and black `Add to cart` button.
3. A grouped product exposes every real available colour swatch; changing swatches updates image, colour, article link, price, and cart payload.
4. Adding twice increments quantity; decrement/remove and subtotal work; reload restores cart.
5. Customer switching does not cause stale product cards or duplicate families.
6. Product detail uses swatches and cart correctly and does not duplicate the old Other colours gallery.
7. Assistant, visual-search, and outfit result cards have consistent price/cart/swatch behavior and no prohibited technical copy.
8. Shopper footer/header contains no metric, dataset, or service-availability disclaimer.
9. DS Studio retains technical metrics and remains fully functional.
10. Label workflow remains functional.
11. Category tabs are grey when inactive and black when selected, including keyboard navigation.
12. Wordmark remains exactly centered and readable at 1440 × 900, 1024 × 768, and 390 × 844.
13. White background, typography hierarchy, swatches, prices, and cart are visually reviewed at all three widths.
14. No horizontal page overflow, uncaught console errors, or unexpected failed requests.
15. Reduced-motion and keyboard-only flows still work.

Use real API/catalog data. Mock only destructive Label submission and controlled error-state checks. Do not write to gold labels.

## 12. Definition of done

This round is complete only when:

- all 16 user requests are implemented, not merely documented;
- shopper routes contain no `Why this` or technical/demo disclaimers named above;
- page-wide grouping prevents repeated products and exposes real colour variants through accessible swatches;
- all shopper products consistently show a stable USD prototype price;
- black/white Add to cart controls feed a working persistent cart;
- category states, white background, and refined wordmark match the requested visual direction;
- all revised copy is visible in the correct contexts;
- the complete automated test suite passes;
- the real browser journeys and three-size screenshots pass visual inspection;
- all fixes are committed locally on `ui-editorial-redesign`;
- no user-owned untracked file is touched;
- nothing is pushed, merged, deployed, or published;
- `reports/UI_CONSUMER_POLISH_ROUND_2_REPORT.md` records the implementation, tests, screenshots, limitations, commits, and final git audit.

## Completion report requirements

Create `reports/UI_CONSUMER_POLISH_ROUND_2_REPORT.md` with:

- a numbered 1–16 mapping from user request to implementation and verification evidence;
- product-family grouping rule and edge cases;
- swatch data/mapping behavior;
- prototype USD pricing policy and an explicit statement that it is not historical source pricing;
- cart behavior and persistence model;
- exact copy removals/replacements;
- screenshot and viewport review notes;
- exact automated and browser test results;
- known limitations;
- commit list;
- final `git status --short` audit separating preserved user files from task changes.
