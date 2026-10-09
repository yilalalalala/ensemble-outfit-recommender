// Shared state, API access, event logging, the shopper product card, dialogs and motion.
// Everything shown is rendered from API responses; nothing here invents product data.
import {colourName, createPageGroups, familyKey, money, swatchStyle, variantLabels} from "./catalog.js";


// The local app is same-origin.  A hosted static build declares its Modal endpoint
// in a meta tag, keeping every UI module on the same small API wrapper.
const API_ROOT = (document.querySelector('meta[name="ensemble-api-base"]')?.content || "").replace(/\/$/, "");
const apiUrl = path => `${API_ROOT}${path}`;
// The hosted shop (GitHub Pages + Modal) hides local-only research tools such as Label.
export const HOSTED = API_ROOT !== "";

export const $ = (s, root = document) => root.querySelector(s);
export const $$ = (s, root = document) => [...root.querySelectorAll(s)];
export const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

export const state = {
  customer: null,        // customer_idx powering personalisation (the demo profile)
  customers: [],
  profile: null,
  backend: null,         // assistant / vision backend reported by the server (DS context only)
};

// ---- API ---------------------------------------------------------------------------------------------
export class ApiError extends Error {
  constructor(status, message, code = null) { super(message); this.status = status; this.code = code; }
}

function friendly(status, body) {
  // Shopper-safe copy: no stack traces, services, models or servers.
  const detail = body?.error?.message || (typeof body?.detail === "string" ? body.detail : null);
  if (status === 400 || status === 413 || status === 415 || status === 422) return detail || "That request could not be completed.";
  if (status === 404) return "We couldn't find that.";
  if (status === 503) return "This isn't available right now. Please try again shortly.";
  return "Something went wrong. Please try again.";
}

export async function api(path, opts) {
  let r;
  try {
    r = await fetch(apiUrl(path), opts);
  } catch {
    throw new ApiError(0, "We can't connect right now. Please check your connection and try again.", "offline");
  }
  const type = r.headers.get("content-type") || "";
  const body = type.includes("json") ? await r.json().catch(() => null) : null;
  if (!r.ok) throw new ApiError(r.status, friendly(r.status, body), body?.error?.code ?? null);
  return body;
}

// Status-aware probe for the endpoints whose HTTP status is part of the answer (/readyz).
export async function probe(path) {
  try {
    const r = await fetch(apiUrl(path));
    return {status: r.status, ok: r.ok, body: await r.json().catch(() => null)};
  } catch {
    return {status: 0, ok: false, body: null};
  }
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

// ---- product families, colourways and prices (/api/catalog/families) ------------------------------------
const familyOfArticle = new Map();   // article_id -> product_code (the catalogue's own family key)
const families = new Map();          // product_code -> {product_code, price_usd, variants[]}
export const familyOf = id => familyOfArticle.get(+id) ?? null;
export const familyData = code => families.get(code) ?? null;

export async function loadFamilies(ids) {
  const want = [...new Set(ids.map(Number))].filter(id => !familyOfArticle.has(id));
  for (let i = 0; i < want.length; i += 400) {
    const chunk = want.slice(i, i + 400);
    const d = await api(`/api/catalog/families?articles=${chunk.join(",")}`);
    for (const [code, f] of Object.entries(d.families)) {
      families.set(code, f);
      // Every colourway belongs to the family too, so a card stays resolvable after any swatch change.
      for (const v of f.variants) familyOfArticle.set(+v.article_id, code);
    }
    for (const [a, code] of Object.entries(d.articles)) familyOfArticle.set(+a, code);
  }
}
// "Not for me": remembered per profile on this device and applied to every page rendered afterwards.
const nfmKey = () => `ensemble.notforme.${state.customer}`;
export function notForMeSet() {
  try { return new Set(JSON.parse(localStorage.getItem(nfmKey()) || "[]")); } catch { return new Set(); }
}
function rememberNotForMe(key) {
  const s = notForMeSet(); s.add(key);
  try { localStorage.setItem(nfmKey(), JSON.stringify([...s])); return true; } catch { return false; }
}
export const newPageGroups = () => createPageGroups(familyOf, notForMeSet());

// ---- events (DESIGN §7.6: impression, click, add_to_cart, not_for_me) --------------------------------
export function logEvent(event, surface, article_id, anchor = null) {
  fetch(apiUrl("/api/events"), {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({customer_idx: state.customer, event, surface, article_id: +article_id, anchor: anchor == null ? null : +anchor}),
  }).then(r => { if (!r.ok) console.warn(`event ${event} not recorded (${r.status})`); })
    .catch(() => console.warn(`event ${event} not recorded (server unreachable)`));
}
export const impressions = (items, surface, anchor = null) => items.forEach(it => logEvent("impression", surface, it.article_id, anchor));

// ---- formatting ----------------------------------------------------------------------------------------
const dateFmt = new Intl.DateTimeFormat("en-US", {day: "numeric", month: "short", year: "numeric", timeZone: "UTC"});
export const fmtDate = iso => iso ? dateFmt.format(new Date(iso + "T00:00:00Z")) : "";
export const plural = (n, one, many = one + "s") => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;
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

// ---- shopper product card ------------------------------------------------------------------------------
const PHOTO_W = 1166, PHOTO_H = 1750;   // catalogue photography size: lets the browser reserve space
const MAX_SWATCHES = 6;                 // cards; the product page shows every colourway

const altOf = (name, colour) => `${name}, ${colour.toLowerCase()}`;

// The family's colourways with display labels; always includes the card's own article.
export function variantsFor(it, code = null) {
  const fam = familyData(code || familyOf(it.article_id));
  const vs = fam?.variants?.length ? fam.variants : [{article_id: it.article_id, prod_name: it.prod_name, colour_group_name: it.colour_group_name, image: it.image}];
  const list = vs.some(v => v.article_id === it.article_id) ? vs : [{article_id: it.article_id, prod_name: it.prod_name, colour_group_name: it.colour_group_name, image: it.image}, ...vs];
  const labels = variantLabels(list);
  return {family: fam, list: list.map((v, i) => ({...v, label: labels[i]}))};
}

export function swatchButtons(it, list, max = Infinity) {
  // Selected colour first, then the rest in catalogue order; the selected one is never cut off.
  const sel = list.find(v => v.article_id === it.article_id);
  const rest = list.filter(v => v !== sel);
  const shown = [sel, ...rest].slice(0, Math.max(1, max));
  const more = list.length - shown.length;
  const btns = shown.map(v => {
    const s = swatchStyle(v.colour_group_name), on = v === sel;
    return `<button class="swatch${s.pattern ? ` is-${s.pattern}` : ""}" type="button" data-swatch="${v.article_id}"
      aria-pressed="${on}" aria-label="${esc(v.label)}${on ? ", selected" : ""}" title="${esc(v.label)}"
      ${s.fill ? `style="--sw:${s.fill}"` : ""}></button>`;
  }).join("");
  return {btns, more};
}

export function card(it, surface, o = {}) {
  const id = it.article_id;
  const {family, list} = variantsFor(it);
  const v = list.find(x => x.article_id === id);
  const price = family?.price_usd;
  const {btns, more} = swatchButtons(it, list, o.maxSwatches ?? MAX_SWATCHES);
  const anchorAttr = o.anchor != null ? `data-anchor="${o.anchor}"` : "";
  return `<article class="card" data-card data-article="${id}" data-family="${esc(family?.product_code ?? "")}" data-s="${esc(surface)}" ${anchorAttr}>
    <div class="card__media">
      <a class="card__img" href="#/product/${id}" data-open="${id}" data-s="${esc(surface)}" ${anchorAttr} tabindex="-1" aria-hidden="true">
        <img src="${esc(v.image)}" alt="${esc(altOf(v.prod_name, v.label))}" width="${PHOTO_W}" height="${PHOTO_H}" ${o.eager ? `fetchpriority="high"` : `loading="lazy"`} decoding="async">
      </a>
    </div>
    <div class="card__info">
      <h3 class="card__name"><a href="#/product/${id}" data-open="${id}" data-s="${esc(surface)}" ${anchorAttr}>${esc(v.prod_name)}</a></h3>
      <div class="card__colours">
        <div class="swatches" role="group" aria-label="Colours">${btns}${more > 0
          ? `<a class="swatch-more" href="#/product/${id}" data-open="${id}" data-s="${esc(surface)}" ${anchorAttr} aria-label="${more} more colours">+${more}</a>` : ""}</div>
        <span class="card__colour">${esc(v.label)}</span>
      </div>
      <div class="card__row"><p class="card__price">${esc(money(price))}</p>
        ${o.nfm === false ? "" : `<button class="link-btn card__nfm" type="button" data-nfm aria-label="Not for me: ${esc(v.prod_name)}">Not for me</button>`}</div>
      <p class="card__nfm-note" role="status" hidden></p>
      ${o.note ? `<p class="card__note">${o.note}</p>` : ""}
      <button class="btn-cart" type="button" data-add aria-label="Add ${esc(v.prod_name)}, ${esc(v.label)}, to cart" ${price == null ? "disabled" : ""}>Add to cart</button>
    </div>
  </article>`;
}

export const cardGrid = (items, surface, o = {}) =>
  `<ul class="grid ${o.cls || ""}" role="list">${items.map((it, i) => `<li>${card(it, surface, {...o, ...(o.each?.(it, i) || {})})}</li>`).join("")}</ul>`;

// Swatch selection: update the card in place (image, name, link target, colour, price, cart target).
export function selectSwatch(btn) {
  const el = btn.closest("[data-card]");
  const id = +btn.dataset.swatch;
  if (!el || +el.dataset.article === id) return;
  const {family, list} = variantsFor({article_id: +el.dataset.article}, el.dataset.family || null);
  const v = list.find(x => x.article_id === id);
  if (!v) { console.warn("swatch: colourway not found", id); return; }
  el.dataset.article = id;
  const img = $("img", el);
  delete img.dataset.fallback; img.classList.remove("is-missing");
  img.src = v.image; img.alt = altOf(v.prod_name, v.label);
  $$("[data-open]", el).forEach(a => { a.dataset.open = id; a.href = `#/product/${id}`; });
  const nameLink = $(".card__name a", el); if (nameLink) nameLink.textContent = v.prod_name;
  $(".card__colour", el).textContent = v.label;
  $(".card__price", el).textContent = money(family?.price_usd);
  const nfm = $("[data-nfm]", el); if (nfm && !nfm.disabled) nfm.setAttribute("aria-label", `Not for me: ${v.prod_name}`);
  $("[data-add]", el).setAttribute("aria-label", `Add ${v.prod_name}, ${v.label}, to cart`);
  $$(".swatch", el).forEach(s => {
    const o = list.find(x => x.article_id === +s.dataset.swatch);
    const on = +s.dataset.swatch === id;
    s.setAttribute("aria-pressed", String(on));
    s.setAttribute("aria-label", `${o.label}${on ? ", selected" : ""}`);
  });
  el.dispatchEvent(new CustomEvent("variant-change", {bubbles: true, detail: {article_id: id, variant: v, family}}));
}

// What the card's Add to cart currently targets.
export function cardTarget(el) {
  const id = +el.dataset.article;
  const {family, list} = variantsFor({article_id: id}, el.dataset.family || null);
  const v = list.find(x => x.article_id === id);
  return {article_id: id, family: family?.product_code ?? null, name: v.prod_name, colour: v.label, image: v.image,
          price: family?.price_usd, surface: el.dataset.s, anchor: el.dataset.anchor ?? null};
}

// "Not for me": keeps the existing not_for_me event, dims the card in place (no layout jump), and is
// remembered for this profile so the product family is left out of every page rendered afterwards.
export function notForMe(btn) {
  const el = btn.closest("[data-card]");
  if (!el || btn.disabled) return;
  logEvent("not_for_me", el.dataset.s, el.dataset.article, el.dataset.anchor ?? null);
  const saved = rememberNotForMe(el.dataset.family ? `f:${el.dataset.family}` : familyKey(familyOf, el.dataset.article));
  const name = $(".card__name a", el)?.textContent || "Item";
  el.classList.add("is-dismissed");
  btn.disabled = true;
  btn.textContent = "Noted";
  const note = $(".card__nfm-note", el);
  note.hidden = false;
  note.textContent = saved ? "Got it. We'll remember that next time." : "Got it. We'll keep that in mind.";
  announce(`${name}: got it, we'll remember that next time.`);
}

// ---- reasons and SHAP: DS Studio only -------------------------------------------------------------------
const EVIDENCE = {
  personal_history: "From the customer's own purchases",
  similarity: "From shoppers with similar baskets",
  trending: "From what is selling now",
  visual_compatibility: "Visual compatibility",
};
export function reasonList(reasons) {
  if (!reasons?.length) return `<p class="meta">No specific reason was recorded for this item.</p>`;
  return `<ul class="reason-list">${reasons.map(r => {
    const ev = EVIDENCE[r.evidence || r.evidence_type];
    return `<li><span>${esc(r.text)}${ev ? `<span class="meta">${esc(ev)}</span>` : ""}</span></li>`;
  }).join("")}</ul>`;
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
    });
  });
}

// ---- states --------------------------------------------------------------------------------------------
export const loadingTiles = (n, cls = "grid--4") =>
  `<ul class="grid ${cls}" aria-hidden="true">${Array.from({length: n}, () =>
    `<li><div class="card"><div class="card__media skeleton"></div><div class="skeleton-line"></div><div class="skeleton-line short"></div></div></li>`).join("")}</ul>`;

export function errorState(err, {title = "This page didn't load", retry = true} = {}) {
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

// Resolved against this module's own URL, so it is correct at the server root and under a
// GitHub project subpath alike.
const PLACEHOLDER = new URL("../placeholder.svg", import.meta.url).pathname;

// Missing or broken images fall back to the neutral placeholder rather than a broken icon.
export function initImageFallback() {
  document.addEventListener("error", e => {
    const img = e.target;
    if (img.tagName !== "IMG" || img.dataset.fallback) return;
    img.dataset.fallback = "1";
    img.src = PLACEHOLDER;
    img.classList.add("is-missing");
  }, true);
}

export {colourName, money};
