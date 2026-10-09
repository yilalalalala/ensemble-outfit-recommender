// Router, masthead navigation, experience switch and demo-profile selector.
import {$, $$, HOSTED, api, announce, esc, errorState, initDialogs, initImageFallback, initMotion, logEvent,
        notForMe, openDialog, probe, profileDetail, profileName, selectSwatch, state} from "./core.js";
import {addToCart, initCart, loadCart} from "./cart.js";
import * as shop from "./shop.js";
import * as stylist from "./stylist.js";
import * as studio from "./studio.js";

const STORE_KEY = "ensemble.profile";
const NAV = {
  shop: [["Discover", "#/"], ["Collections", "#/collections"], ["Style Assistant", "#/assistant"]],
  // Label writes the human gold labels to disk and reads private outfit photos, so the hosted shop
  // does not offer it (the API refuses it too); its route still resolves and explains itself (studio.js).
  studio: [["Evaluation", "#/studio"], ["Serving", "#/studio/serving"], ["Inspect", "#/studio/inspect"],
           ...(HOSTED ? [] : [["Label", "#/label"]])],
};

// ---- routing -------------------------------------------------------------------------------------------
function parse(hash) {
  const raw = (hash || "#/").replace(/^#/, "") || "/";
  const [path, query = ""] = raw.split("?");
  const parts = path.split("/").filter(Boolean);
  return {path: "/" + parts.join("/"), parts, params: new URLSearchParams(query)};
}

function resolve(r) {
  const [a, b] = r.parts;
  if (!a) return {exp: "shop", nav: "#/", view: shop.home};
  if (a === "collections") return {exp: "shop", nav: "#/collections", view: shop.collections};
  if (a === "product" && /^\d+$/.test(b || "")) return {exp: "shop", nav: null, view: shop.product};
  if (a === "assistant") return {exp: "shop", nav: "#/assistant", view: stylist.view};
  if (a === "ds") return {exp: "studio", nav: "#/studio", view: studio.evaluation};          // legacy deep link
  if (a === "studio" && !b) return {exp: "studio", nav: "#/studio", view: studio.evaluation};
  if (a === "studio" && b === "serving") return {exp: "studio", nav: "#/studio/serving", view: studio.serving};
  if (a === "studio" && b === "inspect") return {exp: "studio", nav: "#/studio/inspect", view: studio.inspect};
  if (a === "label") return {exp: "studio", nav: "#/label", view: studio.label};
  return {exp: "shop", nav: null, view: notFound};
}

async function notFound(ctx) {
  ctx.setTitle("Not found");
  ctx.render(`<div class="state state--center"><h1 class="state__title" tabindex="-1">This page does not exist</h1>
    <p class="meta">The link may be out of date.</p><a class="btn" href="#/">Back to Discover</a></div>`);
}

let token = 0, current = null, navCount = 0;

async function route({focus = true} = {}) {
  const r = parse(location.hash);
  const res = resolve(r);
  const my = ++token;
  navCount++;
  state.navCount = navCount;
  current = res;
  setNav(res);
  const main = $("#main");
  const ctx = {
    route: r, exp: res.exp,
    current: () => my === token,
    setTitle: t => { document.title = t ? `${t} — ensemble` : "ensemble"; },
    // Replace the view only if this navigation is still the latest one (prevents stale cross-profile content).
    render(html) {
      if (my !== token) return null;
      main.innerHTML = `<div class="view view-enter">${html}</div>`;
      return main.firstElementChild;
    },
    rerun: () => route({focus: false}),
  };
  document.body.classList.toggle("studio", res.exp === "studio");
  try {
    await res.view(ctx);
  } catch (err) {
    console.warn("view failed:", err);
    if (my === token) ctx.render(errorState(err));
  }
  if (my === token && focus && navCount > 1) {
    window.scrollTo({top: 0});
    main.focus({preventScroll: true});
  }
}

function setNav(res) {
  const links = NAV[res.exp].map(([label, href]) =>
    `<li><a href="${href}" ${href === res.nav ? `aria-current="page"` : ""}>${esc(label)}</a></li>`).join("");
  $("#primary-nav ul").innerHTML = links;
  $("#drawer-links").innerHTML = links;
  $("#primary-nav").setAttribute("aria-label", res.exp === "studio" ? "DS Studio" : "Shop");
  $$("[data-exp]").forEach(a => {
    if (a.dataset.exp === res.exp) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  $("#profile-k").textContent = res.exp === "studio" ? "Inspecting profile" : "Shopping as";
  // The cart belongs to the shopper masthead; technical status belongs to DS Studio only.
  $("#cart-btn").hidden = res.exp === "studio";
  paintUtility(res.exp);
  $("#wordmark").setAttribute("href", res.exp === "studio" ? "#/studio" : "#/");
}

function paintUtility(exp) {
  const note = $("#utility-note");
  if (exp !== "studio") { note.innerHTML = ""; return; }
  const ready = state.ready;
  note.innerHTML = ready === false
    ? `<span class="status-dot is-bad" aria-hidden="true"></span>Serving bundle not ready · live Complete the Look falls back to stored suggestions`
    : `${ready ? `<span class="status-dot is-ok" aria-hidden="true"></span>Serving bundle ready · ` : ""}Offline evaluation unless labelled live telemetry`;
}

// ---- demo profile --------------------------------------------------------------------------------------
function readStored() {
  try { return localStorage.getItem(STORE_KEY); } catch { return null; }
}
function store(id) {
  try { localStorage.setItem(STORE_KEY, String(id)); } catch { /* private mode: keep the in-memory choice */ }
}

function paintProfile() {
  const c = state.profile;
  $("#profile-mono").textContent = c?.age ?? "·";
  $("#profile-v").textContent = c ? profileName(c) : "No profile";
  $("#profile-sr").textContent = c ? `Change profile. Current: ${profileName(c)}.` : "Choose a profile";
  $("#profile-btn").title = c ? `${profileName(c)} · ${profileDetail(c)} · customer ${c.customer_idx}` : "";
  $$(".profile-option").forEach(b => b.setAttribute("aria-current", String(+b.dataset.id === state.customer)));
}

function renderProfiles() {
  const groups = [["returning", "Returning customers"], ["new", "New customers · no purchase history"]];
  $("#profile-groups").innerHTML = groups.map(([seg, title]) => {
    const list = state.customers.filter(c => c.segment === seg);
    if (!list.length) return "";
    return `<section class="profile-group" aria-labelledby="pg-${seg}"><h3 id="pg-${seg}">${title}</h3>
      <ul class="profile-list">${list.map(c => `<li><button class="profile-option" type="button" data-id="${c.customer_idx}"
        data-q="${c.age ?? ""} ${c.customer_idx}">
        <span class="profile-option__name">${esc(profileName(c))}</span>
        <span class="profile-option__detail">${esc(profileDetail(c))}</span>
        <span class="profile-option__id">ID ${c.customer_idx}</span></button></li>`).join("")}</ul></section>`;
  }).join("");
  filterProfiles("");
}

function filterProfiles(q) {
  const t = q.trim();
  let shown = 0;
  $$(".profile-option").forEach(b => {
    const hit = !t || b.dataset.q.split(" ").some(v => v.startsWith(t));
    b.parentElement.hidden = !hit;
    shown += hit;
  });
  $$(".profile-group").forEach(g => { g.hidden = !g.querySelector("li:not([hidden])"); });
  $("#profile-count").textContent = `${shown} of ${state.customers.length} profiles`;
}

function selectProfile(id, {announceIt = true} = {}) {
  const c = state.customers.find(x => x.customer_idx === id);
  if (!c) return false;
  state.customer = id;
  state.profile = c;
  store(id);
  paintProfile();
  if (announceIt) announce(`Now ${current?.exp === "studio" ? "inspecting" : "shopping as"} ${profileName(c)}.`);
  return true;
}

async function loadProfiles() {
  try {
    state.customers = await api("/api/customers");
  } catch (err) {
    $("#profile-v").textContent = "Profiles unavailable";
    throw err;
  }
  renderProfiles();
  const stored = +readStored();
  const fallback = state.customers.find(c => c.segment === "returning") || state.customers[0];
  if (!(stored && selectProfile(stored, {announceIt: false}))) selectProfile(fallback?.customer_idx, {announceIt: false});
}

// ---- wiring --------------------------------------------------------------------------------------------
function wire() {
  initDialogs();
  initCart();
  initImageFallback();
  initMotion();

  $("#profile-btn").addEventListener("click", e => {
    $("#profile-filter").value = "";
    filterProfiles("");
    openDialog($("#profile-dialog"), e.currentTarget);
    const cur = $(`.profile-option[data-id="${state.customer}"]`);
    cur?.scrollIntoView({block: "center"});
    (cur || $("#profile-filter")).focus();
  });
  $("#profile-filter").addEventListener("input", e => filterProfiles(e.target.value));
  $("#profile-groups").addEventListener("click", e => {
    const b = e.target.closest(".profile-option");
    if (!b) return;
    const changed = +b.dataset.id !== state.customer;
    selectProfile(+b.dataset.id);
    $("#profile-dialog").close();
    if (changed) { loadCart(); route({focus: false}); }
  });

  const drawer = $("#drawer"), menuBtn = $("#menu-btn");
  menuBtn.addEventListener("click", () => { menuBtn.setAttribute("aria-expanded", "true"); openDialog(drawer, menuBtn); });
  drawer.addEventListener("close", () => menuBtn.setAttribute("aria-expanded", "false"));
  drawer.addEventListener("click", e => { if (e.target.closest("a")) { drawer._opener = null; drawer.close(); } });

  $("[data-skip]").addEventListener("click", e => { e.preventDefault(); $("#main").focus(); });

  // One delegated listener for every product tile on every view (no per-render listeners).
  document.addEventListener("click", e => {
    const open = e.target.closest("[data-open]");
    if (open) { logEvent("click", open.dataset.s, open.dataset.open, open.dataset.anchor ?? null); return; }
    const sw = e.target.closest("[data-swatch]");
    if (sw) { selectSwatch(sw); return; }
    const add = e.target.closest("[data-add]");
    if (add) { addToCart(add); return; }
    const nfm = e.target.closest("[data-nfm]");
    if (nfm) { notForMe(nfm); return; }
    if (e.target.closest("[data-retry]")) route({focus: false});
  });

  window.addEventListener("hashchange", () => route());
}

async function health() {
  // Degraded mode is shown, not hidden: if the live outfit service is not ready, say so in the utility bar.
  const r = await probe("/readyz");
  state.ready = r.ok && r.body?.ready !== false;
  state.readiness = r.body || {};
  if (document.body.classList.contains("studio")) paintUtility("studio");
}

(async () => {
  wire();
  health();
  try {
    await loadProfiles();
    loadCart();
  } catch (err) {
    $("#main").innerHTML = `<div class="view">${errorState(err, {title: "The shop could not open"})}</div>`;
    $("#main [data-retry]")?.addEventListener("click", () => location.reload(), {once: true});
    return;
  }
  route({focus: false});
})();
