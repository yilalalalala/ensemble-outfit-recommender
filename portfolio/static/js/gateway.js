// Gateway: the first page. Two editorial choices — Shop your edit / Style with AI — and a quiet link to DS
// Studio. Imagery is the shopper's own real recommendations; nothing redirects automatically, so the
// gateway is always one click (the wordmark) away. (docs/AI_STYLIST_PRODUCT_FLOW.md §2–3)
import {$, esc, getHome, state} from "./core.js";

const LAST_KEY = "ensemble.last";
const PLACES = {"#/shop": "your edit", "#/collections": "collections", "#/stylist": "the AI Stylist",
  "#/stylist/ask": "your conversation with the stylist", "#/stylist/find": "Find this piece", "#/stylist/look": "Complete my look"};

// Remember the last shopper destination (called by the router for shopper routes).
export function remember(hash) {
  const base = hash.split("?")[0];
  if (!PLACES[base]) return;
  try { localStorage.setItem(LAST_KEY, base); } catch { /* storage blocked: nothing to remember */ }
}
function last() { try { return localStorage.getItem(LAST_KEY); } catch { return null; } }

const tile = (it, cls, eager) => it ? `<img class="${cls}" src="${esc(it.image)}" alt="" width="1166" height="1750"
  ${eager ? `fetchpriority="high"` : `loading="lazy"`} decoding="async">` : "";

export async function view(ctx) {
  ctx.setTitle(null);
  const customer = state.customer;
  ctx.render(`<section class="gateway" aria-busy="true"><div class="skeleton-line" style="width:30%"></div>
    <div class="skeleton-line" style="height:3rem;width:60%"></div></section>`);
  let fy = [], tr = [];
  try {
    const d = await getHome(customer);
    fy = d.modules.find(m => m.id === "for_you")?.items || [];
    tr = d.modules.find(m => m.id === "trending")?.items || [];
  } catch { /* the choices still work without imagery */ }
  if (!ctx.current() || customer !== state.customer) return;
  const prev = last();
  const isNew = state.profile?.segment === "new";
  const root = ctx.render(`<section class="gateway" aria-labelledby="gw-title">
    <header class="gateway__head">
      <p class="eyebrow">${isNew ? "Welcome" : "Welcome back"}</p>
      <h1 class="display gateway__title" id="gw-title" tabindex="-1">How would you like to shop today?</h1>
    </header>
    <ul class="gateway__choices" role="list">
      <li><a class="gw-choice gw-choice--shop" href="#/shop">
        <span class="gw-choice__media gw-choice__media--trio" aria-hidden="true">${fy.slice(0, 3).map((it, i) => tile(it, "gw-img gw-img--34", i < 2)).join("")}</span>
        <span class="gw-choice__text">
          <span class="gw-choice__title">Shop your edit</span>
          <span class="gw-choice__line">Pieces chosen for you, from what you love.</span>
          <span class="gw-choice__go" aria-hidden="true">See your edit →</span>
        </span></a></li>
      <li><a class="gw-choice gw-choice--ai" href="#/stylist">
        <span class="gw-choice__media gw-choice__media--pair" aria-hidden="true">${tile(tr[0], "gw-img gw-img--11", true)}${tile(tr[1] || fy[1], "gw-img gw-img--11", false)}</span>
        <span class="gw-choice__text">
          <span class="gw-choice__title">Style with AI</span>
          <span class="gw-choice__line">Ask a stylist, find a piece from a photo, or finish an outfit.</span>
          <span class="gw-choice__go" aria-hidden="true">Meet the stylist →</span>
        </span></a></li>
    </ul>
    <div class="gateway__foot">
      ${prev ? `<p class="gateway__resume">Continue where you left off: <a class="link-btn" href="${prev}">${esc(PLACES[prev])}</a></p>` : "<span></span>"}
      <a class="gateway__system" href="#/studio">Explore the system</a>
    </div>
  </section>`);
  if (root) requestAnimationFrame(() => $("#gw-title", root)?.focus({preventScroll: true}));
}
