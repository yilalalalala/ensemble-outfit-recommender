// Gateway: the first page. A welcome and two choices — Shop your edit / Style with AI — kept deliberately
// plain. A touch sends a soft ripple from the point of contact; choosing a path fades the page away before the
// next view arrives. Both effects are skipped under prefers-reduced-motion. (docs/AI_STYLIST_PRODUCT_FLOW.md §2–3)
import {$, state} from "./core.js";

const LEAVE_MS = 480;

const calm = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// A ripple: a ring of light that grows from the touch point and fades. On a solid button it is drawn inside the
// button (white on ink); anywhere else it spreads across the page (warm grey on white).
function ripple(root, e) {
  if (calm() || e.button > 0) return;
  const host = e.target.closest(".gw-choice__btn") || root;
  const box = host.getBoundingClientRect();
  const size = host === root ? 360 : Math.hypot(box.width, box.height) * 2;
  const r = document.createElement("span");
  r.className = "ripple";
  r.setAttribute("aria-hidden", "true");
  r.style.cssText = `left:${e.clientX - box.left - size / 2}px;top:${e.clientY - box.top - size / 2}px;width:${size}px;height:${size}px`;
  host.append(r);
  r.addEventListener("animationend", () => r.remove(), {once: true});
}

export async function view(ctx) {
  ctx.setTitle(null);
  const isNew = state.profile?.segment === "new";
  const root = ctx.render(`<section class="gateway" aria-labelledby="gw-title">
    <p class="eyebrow">${isNew ? "Welcome" : "Welcome back"}</p>
    <h1 class="sr-only" id="gw-title" tabindex="-1">How would you like to shop today?</h1>
    <ul class="gateway__choices" role="list">
      <li class="gw-choice"><a class="btn btn--solid gw-choice__btn gw-choice--shop" href="#/shop">Shop your edit</a>
        <p class="gw-choice__line">Pieces chosen for you, from what you love.</p></li>
      <li class="gw-choice"><a class="btn btn--solid gw-choice__btn gw-choice--ai" href="#/stylist">Style with AI</a>
        <p class="gw-choice__line">Ask a stylist, find a piece from a photo, or finish an outfit.</p></li>
    </ul>
  </section>`);
  if (!root) return;
  const gw = $(".gateway", root);
  gw.addEventListener("pointerdown", e => ripple(gw, e));
  gw.addEventListener("click", e => {
    const a = e.target.closest("a.gw-choice__btn");
    if (!a || calm() || e.button > 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    if (gw.classList.contains("is-leaving")) return;
    a.classList.add("is-chosen");
    gw.classList.add("is-leaving");
    setTimeout(() => { if (ctx.current()) location.hash = a.getAttribute("href"); }, LEAVE_MS);
  });
  requestAnimationFrame(() => $("#gw-title", root)?.focus({preventScroll: true}));
}
