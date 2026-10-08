# Editorial Retail UI Redesign — Claude Code Execution Plan

## 1. Mission

Transform the current functional demo into a polished, portfolio-ready fashion retail experience while preserving every existing recommendation, analytics, assistant, visual-search, outfit-analysis, and data-science workflow.

The customer experience should feel like a restrained French editorial boutique: fashion-forward, image-led, spacious, and credible as an online retailer. The data-science experience should remain operational and information-dense, but visually belong to the same product.

This is an implementation task, not only a mock-up. Work autonomously until the definition of done is satisfied. Diagnose and fix ordinary failures yourself.

## 2. Required repository discipline

- Work only on branch `ui-editorial-redesign`.
- Read `CLAUDE.md`, `docs/DESIGN.md`, `docs/DATA_CONTRACTS.md`, `docs/DECISIONS.md`, and `docs/GLOSSARY.md` before implementation.
- Inspect the current frontend and backend contracts before changing them.
- Preserve all user-owned untracked files matching `* 2.*` and everything under `reports/track_a_retrain/logs/`.
- Do not push, merge, publish, deploy, or open a pull request.
- Make small conventional commits at coherent milestones.
- Do not alter trained models, offline metrics, recommendation logic, ranking outputs, or research artifacts unless a UI bug proves that a narrowly scoped compatibility fix is required.
- Never fabricate products, images, recommendation reasons, customer attributes, business metrics, or model results.
- Finish with a clean audit of tracked changes and an implementation report.

## 3. Product positioning and design direction

Use the reference brands only as design research, never as assets or layouts to copy. Synthesize these qualities:

- Zara / H&M / Uniqlo: fast, obvious retail navigation and low-friction product discovery.
- & Other Stories: editorial collection storytelling and asymmetric image rhythm.
- Dior / Cartier / YSL: restrained typography, generous negative space, confident masthead, refined motion.
- Aesop: calm information hierarchy, warm neutral palette, thoughtful microcopy.

The result should be recognizably original and suited to the `ensemble` recommendation concept.

### Core visual character

- Lowercase `ensemble` wordmark, always visually centered at the top of the page.
- Elegant high-contrast serif masthead using a local/system font stack such as `Didot`, `Bodoni 72`, `Bodoni MT`, `Times New Roman`, serif. Do not add a remote font dependency.
- Clean neutral sans-serif for navigation, controls, metadata, and data tables.
- Warm ivory canvas, near-black ink, stone borders, and one restrained oxblood or muted-bronze accent.
- Large photography, quiet type, hairline rules, deliberately uneven editorial composition.
- Avoid generic dashboard cards, excessive rounded corners, loud gradients, glassmorphism, and decorative animation.

## 4. Information architecture and control model

Keep the concepts below separate and unmistakable:

1. **Experience switch:** `Shop` and `DS Studio` select the customer-facing or data-science-facing experience.
2. **Customer/demo profile selector:** chooses which demo customer powers personalization. It is not the same as the experience switch.
3. **Primary shop navigation:** Discover, Collections, Style Assistant, and any currently available utility destinations.

On desktop, use a three-column masthead grid (`1fr auto 1fr`) so the wordmark is mathematically centered regardless of left/right control widths. On mobile, preserve the centered wordmark while reducing surrounding labels to compact accessible controls.

The Shop surface should present the selected person as a profile, not as a raw database selector. A compact profile control may show a friendly label and supporting persona details in its menu/drawer. The raw customer ID can remain available in a secondary detail for debugging, but it must not dominate the retail experience.

Persist the chosen demo customer across navigation and reloads, using a safe local browser mechanism, while retaining the current default fallback.

## 5. Implementation strategy

The current application is deliberately simple: FastAPI serves a plain HTML/CSS/JavaScript frontend. Preserve the no-build deployment model unless there is a demonstrated, documented reason that it cannot meet the requirements.

Prefer one of these two maintainable approaches:

- split the current page into `index.html`, a focused stylesheet, and one or more small JavaScript modules served by the existing static route; or
- retain one HTML asset but reorganize it into clearly labeled, cohesive sections if changing static routing would create unnecessary risk.

Do not introduce React, Vue, a bundler, Tailwind, a package manager, or a third-party carousel for this redesign. Use semantic HTML, modern CSS Grid/Flexbox, native dialogs where suitable, CSS transitions, and small browser APIs such as `IntersectionObserver`.

Keep existing endpoint signatures stable. Backward-compatible response additions are allowed only if the UI genuinely requires them and corresponding tests are added.

## 6. Global design system

Create consistent tokens for:

- canvas, surface, text, muted text, border, accent, success, warning, and error colors;
- display serif and utility sans font stacks;
- a restrained type scale for masthead, editorial title, section title, product name, price, metadata, and technical labels;
- spacing scale, grid gaps, content widths, and page gutters;
- image aspect ratios and tile size variants;
- motion durations/easing;
- focus ring, border, and minimal radius values;
- elevation only where a floating layer genuinely requires it.

Use fluid typography and spacing with `clamp()` where it improves responsiveness. Maintain good contrast and readable line lengths.

## 7. Global masthead and navigation

Implement a refined sticky or gently static header that does not consume excessive vertical space.

Required behavior:

- `ensemble` remains at the exact horizontal center at desktop and mobile widths.
- Left side holds navigation/menu context; right side holds profile and utility controls.
- Shop/DS Studio switching is explicit, keyboard accessible, and visually independent from the customer selector.
- Current destination is communicated both visually and semantically.
- Header remains usable at 390 px width without clipping or horizontal scrolling.
- Mobile navigation opens a lightweight, keyboard-operable drawer or menu with correct focus behavior.

## 8. Customer-facing Shop homepage

Replace the current uniform horizontal rows with a coherent editorial retail homepage built from the real `/api/home/{customer_id}` response.

### Hero and introduction

- Use a quiet personalized greeting and one concise editorial line.
- Do not invent a campaign image if the API does not provide one. Build the hero from real recommended product photography or typography-led composition.
- Avoid exposing recommendation-system jargon in primary retail copy.

### Editorial recommendation modules

Present available modules using clear customer language, for example:

- Selected for you
- A study in layers / Complete the look
- Recently considered
- New this week
- Buy it again

Use the actual module inventory returned by the API; gracefully omit unavailable modules. Do not force labels that misrepresent the source data.

### Asymmetric image composition

- Use deterministic, responsive CSS grid patterns with a mixture of portrait, standard, and feature tiles.
- Larger tiles should correspond to recommendation prominence or module storytelling, not random fake importance.
- Keep product images uncropped where garment understanding matters; use controlled `object-fit` rules and neutral image backdrops.
- Ensure every item remains reachable even when the number of products differs from the ideal layout.
- A catalogue-like section may use a regular grid after the editorial opening; the full site must not become visually chaotic.

### Product-card interaction

- Show product name, product type/category where useful, and price only when the API has a real price.
- Preserve existing product navigation and impression/click tracking.
- Retain recommendation explanations, but present them through a refined `Why this` disclosure, drawer, or dialog rather than dense card text.
- Preserve `Not for me` feedback and make its state and result understandable.
- Do not create alternate-image crossfades when only one real image exists.

## 9. Product detail experience

Redesign the product page as a credible retailer product detail page:

- prominent product photography/gallery area using only available real images;
- clear title, category/type, price when available, and concise metadata;
- accessible primary actions, with unavailable commerce actions clearly marked as demo behavior rather than pretending checkout exists;
- elegant explanation access for personalized reasoning;
- `Complete the Look`, other colours, and similar items arranged as distinct editorial sections;
- preserve every existing API call and interaction associated with these modules;
- sensible back navigation and deep-link behavior.

On large screens, a two-column image/information layout is preferred. On narrow screens, stack content without sticky elements obscuring controls.

## 10. Style Assistant, visual search, and outfit-photo flow

Unify these capabilities as one premium styling service while keeping their distinct operations understandable.

- Give the assistant a calm conversational layout with clear empty, thinking, tool-running, success, and failure states.
- Keep tool calls grounded in actual backend responses; never display invented analysis.
- Make image upload feel intentional: supported input guidance, visible preview, replace/remove controls, progress state, and privacy-minded microcopy.
- Distinguish `find visually similar products` from `analyze my outfit / suggest complements` so users know what will happen.
- Display search and outfit results with the same product component system used by the shop.
- Preserve assistant session behavior and current visual/outfit endpoints.
- Ensure upload controls and results are fully keyboard accessible.

## 11. DS Studio

Keep DS Studio as a professional production/research observability surface, not a second fashion homepage.

- Use the shared masthead, tokens, and navigation so it belongs to the product.
- Prioritize legible metrics, model/system status, recommendation diagnostics, and existing tables.
- Preserve exact values and labels from `/api/metrics` and related responses; never round or reinterpret results misleadingly.
- Improve table hierarchy, alignment, responsive overflow, empty states, and explanatory labels.
- Make it obvious when a metric is offline evaluation, demo telemetry, unavailable, or stale, if that context already exists in the data.
- Keep customer profile selection available where it is required for inspection, while clearly separating it from global experience switching.

## 12. Label view and utility surfaces

Retain the Label workflow and make it visually consistent, efficient, and unambiguous. Do not turn a utility task into an editorial layout. Preserve all existing controls, validation, feedback, and data contracts.

## 13. Motion and interaction quality

Motion should communicate hierarchy and state, not distract.

Implement a small motion language:

- subtle first-view fade/up reveals using `IntersectionObserver`;
- stagger only a few immediately visible editorial tiles;
- 2–3% image scale or gentle crop shift on hover-capable devices;
- restrained underline, opacity, and drawer/dialog transitions;
- a short cross-view transition that never delays navigation;
- optional very subtle hero parallax only if it remains smooth and adds no accessibility/performance regression.

Requirements:

- honor `prefers-reduced-motion: reduce` and remove nonessential animation;
- do not auto-scroll carousels;
- do not hide necessary information behind hover;
- avoid scroll-jacking and heavy animation libraries;
- maintain stable layout and prevent visible content jumps.

## 14. Responsive behavior

Explicitly design and verify at minimum:

- 1440 × 900 desktop;
- 1024 × 768 tablet/small desktop;
- 390 × 844 mobile.

At all sizes:

- no unintended horizontal overflow;
- header, selector, experience switch, dialogs, tables, images, and assistant remain usable;
- editorial grids simplify predictably rather than merely shrinking;
- touch targets are comfortably sized;
- key content order remains logical in the DOM and visually.

## 15. Loading, empty, error, and degraded states

Every API-driven surface must have intentional states:

- lightweight image/content skeleton or stable loading placeholder;
- useful empty-state copy without blaming the user;
- visible retry action for recoverable failures;
- fallback image treatment for missing/broken images;
- disabled state during mutations to prevent duplicate actions;
- assistant/tool error states that preserve the session;
- offline/backend-unavailable message that keeps navigation intact.

Do not silently swallow errors. Do not expose stack traces or raw exceptions to customers.

## 16. Accessibility requirements

- Semantic landmarks, headings, lists, buttons, links, forms, labels, and tables.
- Complete keyboard access and visible focus treatment.
- Correct focus entry/return and Escape behavior for drawers/dialogs.
- Useful alternative text based only on available product data; decorative imagery has empty alt text.
- Sufficient foreground/background contrast.
- Status updates announced appropriately without noisy live regions.
- Form validation associated with the relevant control.
- Motion reduction as specified above.
- Do not use color alone to communicate state.

## 17. Performance and privacy requirements

- Lazy-load below-the-fold product images and set dimensions/aspect ratios to reduce layout shift.
- Keep above-the-fold assets deliberate; do not eagerly fetch an entire catalogue.
- Avoid third-party trackers, remote fonts, and unnecessary dependencies.
- Do not transmit uploaded images anywhere beyond the existing application endpoints.
- Reuse cached API results where safe, but never allow one selected customer's personalized content to appear under another customer's profile.
- Avoid expensive DOM recreation and repeated listeners during route changes.
- The browser console must remain free of uncaught errors during the verified journeys.

## 18. Functional compatibility checklist

The redesign must preserve and verify:

- customer list loading and customer switching;
- personalized home modules;
- product deep links and back navigation;
- recommendation impression and click events;
- `Why this` explanations;
- `Not for me` feedback;
- Complete the Look, similar items, and other-colour flows;
- assistant session creation and messaging;
- assistant tool result rendering;
- visual product search;
- outfit-photo analysis and complement suggestions;
- DS metrics/tables;
- Label workflow;
- health/degraded behavior already represented by the UI.

## 19. Test and visual-verification plan

### Automated checks

- Run the full existing test suite before major edits to establish a baseline.
- Add focused regression tests for any changed static-file routing or backend response contract.
- Add lightweight UI/DOM tests for critical semantics and selectors if the repository has an appropriate existing test pattern.
- If adding a browser-test tool would require a large new dependency stack, do not add it solely for this task; use the available browser automation and document the reproducible checks instead.
- Run the complete existing suite after implementation and report exact pass/fail counts.

### Browser journeys

Launch the application and exercise at least these real flows:

1. Open Shop, load its default customer, and inspect all available modules.
2. Change customer and verify content/profile updates without stale cross-customer results.
3. Open a product, request an explanation, use feedback, and navigate its related modules.
4. Open Style Assistant and complete a representative text interaction.
5. Exercise visual-search and outfit-photo input through the point supported by available local test fixtures.
6. Switch to DS Studio and inspect all metrics/tables.
7. Open Label and verify its core workflow.
8. Repeat key navigation by keyboard.
9. Verify reduced-motion behavior.
10. Inspect the browser console and failed network requests.

### Visual QA

Capture and inspect screenshots at the three required viewport sizes for:

- Shop homepage;
- product detail;
- Style Assistant/image flow;
- DS Studio.

Check centering of the `ensemble` wordmark, image crop quality, spacing rhythm, type hierarchy, focus state, loading/error state, and absence of overflow. Iterate on visible defects instead of treating first-pass screenshots as completion.

## 20. Definition of done

The task is complete only when all of the following are true:

- The `ensemble` wordmark is visibly and mathematically centered at the top on desktop and mobile.
- Shop resembles a credible artistic fashion retailer rather than a generic recommendation dashboard.
- The homepage uses real recommendation data in a varied but coherent editorial layout.
- Customer/demo profile selection and Shop/DS Studio switching are separate and understandable.
- Product, assistant, visual search, outfit, DS, and Label surfaces share one intentional visual system.
- All existing critical workflows in section 18 remain functional.
- Loading, empty, error, and degraded states are present for API-driven surfaces.
- Accessibility, reduced motion, responsive behavior, and performance requirements are verified.
- There are no uncaught browser-console errors in the checked journeys.
- The full test suite passes, or any pre-existing failure is reproduced and clearly separated from new failures.
- Changes are organized into coherent conventional commits on `ui-editorial-redesign`.
- User-owned untracked files remain untouched.
- A final report exists at `reports/UI_EDITORIAL_REDESIGN_REPORT.md`.

## 21. Required completion report

Create `reports/UI_EDITORIAL_REDESIGN_REPORT.md` containing:

- concise before/after summary;
- implemented design system and information architecture;
- file-by-file change summary;
- route/feature compatibility matrix;
- accessibility, responsiveness, motion, and performance decisions;
- exact automated test commands and results;
- browser journeys and viewport screenshots inspected;
- console/network findings;
- known limitations and justified deferred work;
- commit list;
- final `git status --short` audit that explicitly distinguishes preserved user files from task changes.

Do not declare completion based only on code edits. Completion requires running the application, visually inspecting it, testing the real journeys, fixing defects found, and producing the report.
