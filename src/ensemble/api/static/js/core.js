// Shared state, API access, event logging, the product tile, dialogs and motion.
// Everything shown is rendered from API responses; nothing here invents product data.

export const $ = (s, root = document) => root.querySelector(s);
export const $$ = (s, root = document) => [...root.querySelectorAll(s)];
export const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

export const state = {
  customer: null,        // customer_idx powering personalisation (the demo profile)
  customers: [],
  profile: null,
  backend: null,         // assistant / vision backend reported by the server ("ollama" = local)
};

// ---- API ---------------------------------------------------------------------------------------------
export class ApiError extends Error {
  constructor(status, message, code = null) { super(message); this.status = status; this.code = code; }
}

function friendly(status, body) {
  // Customer-safe copy: never surface stack traces or raw exceptions.
  const detail = body?.error?.message || (typeof body?.detail === "string" ? body.detail : null);
  if (status === 400 || status === 413 || status === 415 || status === 422) return detail || "That request could not be processed.";
  if (status === 404) return detail ? `Not found: ${detail}.` : "We could not find that.";
  if (status === 503) return "This service is temporarily unavailable on the demo server.";
  return "Something went wrong on the server.";
}

export async function api(path, opts) {
  let r;
  try {
    r = await fetch(path, opts);
  } catch {
    throw new ApiError(0, "We couldn't reach the ensemble server. Check that it is running, then try again.", "offline");
  }
  const type = r.headers.get("content-type") || "";
  const body = type.includes("json") ? await r.json().catch(() => null) : null;
  if (!r.ok) throw new ApiError(r.status, friendly(r.status, body), body?.error?.code ?? null);
  return body;
}

const homeCache = new Map();   // keyed by customer, so one profile's edit never shows under another
export function getHome(customer) {
  if (!homeCache.has(customer)) {
    const p = api(`/api/home/${customer}`);
    p.catch(() => homeCache.delete(customer));
    homeCache.set(customer, p);
  }
  return homeCache.get(customer);
}
export async function cachedHome(customer) {
  return homeCache.has(customer) ? homeCache.get(customer).catch(() => null) : null;
}

// ---- events (DESIGN §7.6: impression, click, add_to_cart, not_for_me) --------------------------------
export function logEvent(event, surface, article_id, anchor = null) {
  fetch("/api/events", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({customer_idx: state.customer, event, surface, article_id: +article_id, anchor: anchor == null ? null : +anchor}),
  }).then(r => { if (!r.ok) console.warn(`event ${event} not recorded (${r.status})`); })
    .catch(() => console.warn(`event ${event} not recorded (server unreachable)`));
}
export const impressions = (items, surface, anchor = null) => items.forEach(it => logEvent("impression", surface, it.article_id, anchor));

// ---- formatting ----------------------------------------------------------------------------------------
const dateFmt = new Intl.DateTimeFormat("en-GB", {day: "numeric", month: "short", year: "numeric", timeZone: "UTC"});
export const fmtDate = iso => iso ? dateFmt.format(new Date(iso + "T00:00:00Z")) : "";
export const plural = (n, one, many = one + "s") => `${n.toLocaleString("en-GB")} ${n === 1 ? one : many}`;
export const words = s => String(s ?? "").replace(/_/g, " ");

export function profileName(c) {
  if (!c) return "Guest";
  return `${c.segment === "new" ? "New shopper" : "Shopper"}${c.age != null ? `, ${c.age}` : ""}`;
}
export function profileDetail(c) {
  if (!c) return "";
  if (c.segment === "new") return "No purchase history yet";
  return `${plural(c.n_purchases, "item")} bought · last order ${fmtDate(c.last_purchase)}`;
}

// ---- announcements -------------------------------------------------------------------------------------
let announceTimer = null;
export function announce(text) {
  const el = $("#announcer");
  el.textContent = "";
  clearTimeout(announceTimer);
  announceTimer = setTimeout(() => { el.textContent = text; }, 60);
}

// ---- product tile --------------------------------------------------------------------------------------
// The registry keeps the full item (reasons, provenance) behind each rendered tile for the Why dialog.
export const registry = new Map();
let keySeq = 0;

const PHOTO_W = 1166, PHOTO_H = 1750;   // catalogue photography size: lets the browser reserve space

export function tile(it, surface, o = {}) {
  const key = `k${++keySeq}`;
  registry.set(key, {item: it, surface, anchor: o.anchor ?? null, why: o.why});
  const id = it.article_id;
  const alt = [it.prod_name, [it.colour_group_name, it.product_type_name].filter(Boolean).join(" ").toLowerCase()].filter(Boolean).join(", ");
  const hasWhy = o.why === "shap" || (o.why !== false && (it.reasons || []).length > 0);
  // "Unknown" is the catalogue's placeholder value; it says nothing to a shopper, so it is left out.
  const meta = o.meta ?? [it.product_type_name, it.colour_group_name].filter(v => v && v !== "Unknown").join(" · ");
  const index = o.index != null ? `<span class="tile__index" aria-hidden="true">${String(o.index).padStart(2, "0")}</span>` : "";
  return `<article class="tile ${o.cls || ""}" data-key="${key}" style="${o.style || ""}">
    <a class="tile__media${o.media || ""}" href="#/product/${id}" data-open="${id}" data-s="${esc(surface)}" ${o.anchor != null ? `data-anchor="${o.anchor}"` : ""} tabindex="-1" aria-hidden="true">
      <img src="${esc(it.image)}" alt="${esc(alt)}" width="${PHOTO_W}" height="${PHOTO_H}" ${o.eager ? `fetchpriority="high"` : `loading="lazy"`} decoding="async">
      ${index}
    </a>
    <div class="tile__body">
      <h3 class="tile__name"><a href="#/product/${id}" data-open="${id}" data-s="${esc(surface)}" ${o.anchor != null ? `data-anchor="${o.anchor}"` : ""}>${esc(it.prod_name)}</a></h3>
      ${meta ? `<p class="tile__meta">${esc(meta)}</p>` : ""}
      ${o.note ? `<p class="tile__note">${o.note}</p>` : ""}
      ${o.actions === false ? "" : `<div class="tile__actions">
        ${hasWhy ? `<button class="link-btn" type="button" data-why="${key}" aria-haspopup="dialog">Why this<span class="sr-only">: ${esc(it.prod_name)}</span></button>` : ""}
        <button class="link-btn" type="button" data-nfm="${key}">Not for me<span class="sr-only">: ${esc(it.prod_name)}</span></button>
      </div>`}
      <p class="tile__status" role="status" hidden></p>
    </div>
  </article>`;
}

export const grid = (items, surface, o = {}) =>
  `<ul class="grid ${o.cls || ""}" role="list">${items.map((it, i) => `<li>${tile(it, surface, {...o, ...(o.each?.(it, i) || {})})}</li>`).join("")}</ul>`;

// ---- dialogs -------------------------------------------------------------------------------------------
export function openDialog(dlg, opener = document.activeElement) {
  dlg._opener = opener;
  if (!dlg.open) dlg.showModal();
}
export function initDialogs() {
  $$("dialog").forEach(dlg => {
    dlg.addEventListener("click", e => {
      if (e.target === dlg || e.target.closest("[data-close]")) dlg.close();
    });
    dlg.addEventListener("close", () => {
      const o = dlg._opener;
      if (o && document.contains(o)) o.focus({preventScroll: true});
      dlg._opener = null;
      dlg.dispatchEvent(new CustomEvent("closed"));
    });
  });
}

// ---- Why this ------------------------------------------------------------------------------------------
const EVIDENCE = {
  personal_history: "From your own purchases",
  similarity: "From shoppers with similar baskets",
  trending: "From what is selling now",
  visual_compatibility: "Visual compatibility",
  co_purchase: "Bought together in past baskets",
  style_co_purchase: "Bought with this style in past baskets",
  popular_in_slot: "Popular in this category",
};
const evidenceOf = r => EVIDENCE[r.evidence || r.evidence_type] || null;

export function reasonList(reasons) {
  if (!reasons?.length) return `<p class="meta">No specific reason was recorded for this piece.</p>`;
  return `<ul class="reason-list">${reasons.map(r => `<li><span>${esc(r.text)}${evidenceOf(r) ? `<span class="meta">${esc(evidenceOf(r))}</span>` : ""}</span></li>`).join("")}</ul>`;
}

export function shapBars(shap) {
  const max = Math.max(...shap.map(s => Math.abs(s.value)), 1e-9);
  return `<div class="shap" role="list">${shap.map(s => {
    const w = 50 * Math.abs(s.value) / max, neg = s.value < 0;
    return `<div class="shap__row" role="listitem"><span class="shap__feat">${esc(s.feature)}</span>
      <span class="shap__track" aria-hidden="true"><span class="shap__fill ${neg ? "is-neg" : ""}" style="left:${neg ? 50 - w : 50}%;width:${w}%"></span></span>
      <span class="shap__val">${s.value > 0 ? "+" : ""}${s.value}</span></div>`;
  }).join("")}</div>`;
}

export async function openWhy(key, opener) {
  const entry = registry.get(key);
  if (!entry) return;
  const {item, surface, why} = entry;
  const dlg = $("#why-dialog"), body = $("#why-body");
  $("#why-title").textContent = item.prod_name;
  $("#why-eyebrow").textContent = surface === "for_you" ? "Why this is in your edit" : "Why this suggestion";
  const extra = [];
  if (item.provenance) extra.push(`Main evidence: ${words(item.provenance)}.`);
  body.innerHTML = reasonList(item.reasons) + (extra.length ? `<p class="meta">${esc(extra.join(" "))}</p>` : "") +
    (why === "shap" ? `<div class="skeleton-line" aria-hidden="true"></div><p class="meta" role="status">Loading the model's reasoning…</p>` : "");
  openDialog(dlg, opener);
  if (why !== "shap") return;
  try {
    const d = await api(`/api/explain/${state.customer}/${item.article_id}`);
    if (!dlg.open || $("#why-title").textContent !== item.prod_name) return;
    body.innerHTML = reasonList(d.reasons) + `
      <details class="disclosure"><summary>Model detail: SHAP contributions</summary>
        <div style="display:grid;gap:.75rem;padding-top:.5rem">
          <p class="meta">How much each feature moved this piece's ranking score (${esc(d.score)}) for this profile. Positive values push it up.
            Reasons above are shown only when the customer's own data supports them.</p>
          ${shapBars(d.shap)}
        </div></details>`;
  } catch (e) {
    body.innerHTML = reasonList(item.reasons) + `<p class="notice notice--error" role="alert">${esc(e.message)}</p>`;
  }
}

// ---- Not for me ----------------------------------------------------------------------------------------
export function notForMe(key, btn) {
  const entry = registry.get(key);
  if (!entry || btn.disabled) return;
  const card = btn.closest(".tile");
  logEvent("not_for_me", entry.surface, entry.item.article_id, entry.anchor);
  btn.disabled = true;
  btn.textContent = "Noted";
  card.classList.add("is-dismissed");
  const status = card.querySelector(".tile__status");
  status.hidden = false;
  status.textContent = "Feedback noted. Thank you.";
  announce(`Feedback noted for ${entry.item.prod_name}.`);
}

// ---- states --------------------------------------------------------------------------------------------
export const loadingTiles = (n, cls = "grid--4") =>
  `<ul class="grid ${cls}" aria-hidden="true">${Array.from({length: n}, () =>
    `<li><div class="tile"><div class="tile__media skeleton"></div><div class="skeleton-line"></div><div class="skeleton-line short"></div></div></li>`).join("")}</ul>`;

export function errorState(err, {title = "This page did not load", retry = true} = {}) {
  return `<div class="state" role="alert"><p class="state__title">${esc(title)}</p>
    <p class="meta">${esc(err?.message || "Something went wrong.")}</p>
    ${retry ? `<button class="btn btn--quiet" type="button" data-retry>Try again</button>` : ""}</div>`;
}

// ---- motion --------------------------------------------------------------------------------------------
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
let observer = null;
export function initMotion() {
  if (!("IntersectionObserver" in window)) return;
  const apply = () => document.documentElement.classList.toggle("motion", !reduced.matches);
  apply();
  reduced.addEventListener?.("change", apply);
  observer = new IntersectionObserver(entries => entries.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add("is-in"); observer.unobserve(e.target); }
  }), {rootMargin: "0px 0px -8% 0px", threshold: 0.08});
}
export function reveal(root) {
  if (!observer || reduced.matches) return;
  $$(".reveal", root).forEach(el => observer.observe(el));
}

// Missing or broken images fall back to the neutral placeholder rather than a broken icon.
export function initImageFallback() {
  document.addEventListener("error", e => {
    const img = e.target;
    if (img.tagName !== "IMG" || img.dataset.fallback) return;
    img.dataset.fallback = "1";
    img.src = "/static/placeholder.svg";
    img.classList.add("is-missing");
  }, true);
}
