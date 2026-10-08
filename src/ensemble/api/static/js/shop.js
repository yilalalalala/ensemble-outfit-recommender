// Shop: Discover (editorial home), Collections and the product page.
// Module inventory comes from /api/home/{customer}; unavailable modules are omitted, never faked.
import {$, $$, api, cachedHome, esc, fmtDate, getHome, grid, impressions, loadingTiles, logEvent, plural,
        profileDetail, profileName, reveal, state, tile, words} from "./core.js";

// Customer-language titles for the API's module ids. Unknown ids fall back to the API title.
const EDITS = {
  for_you: {title: "Selected for you", tab: "Selected for you"},
  buy_again: {title: "Buy it again", tab: "Buy it again"},
  trending: {title: "Trending this week", tab: "Trending this week"},
};
const editTitle = m => EDITS[m.id]?.title || m.title;

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? "Good evening." : h < 12 ? "Good morning." : h < 18 ? "Good afternoon." : "Good evening.";
}

// Deterministic asymmetric rhythm for the edit. Rows alternate a feature row (the row's highest-ranked
// piece is the large tile) and a staggered row of three; mobile simplifies to two columns.
const ROW_A = [{span: 6, sm: 2}, {span: 3, drop: "var(--s-8)"}, {span: 3, align: "end", dropSm: "var(--s-6)"}];
const ROW_B = [{span: 4, drop: "var(--s-7)"}, {span: 4, dropSm: "var(--s-6)"}, {span: 4, drop: "var(--s-8)"}];
function editLayout(n) {
  const out = [];
  for (let row = 0; out.length < n; row++) out.push(...(row % 2 ? ROW_B : ROW_A));
  return out.slice(0, n);
}

function editGrid(items, surface, startIndex, o = {}) {
  const layout = editLayout(items.length);
  return `<ul class="edit" role="list">${items.map((it, i) => {
    const l = layout[i];
    const style = `--span:${l.span};--span-sm:${l.sm || 1};${l.drop ? `--drop:${l.drop};` : ""}${l.dropSm ? `--drop-sm:${l.dropSm};` : ""}${l.align ? `--align:${l.align};` : ""}`;
    return `<li style="${style}">${tile(it, surface, {...o, index: startIndex + i, cls: "reveal", style: `--i:${i % 3}`})}</li>`;
  }).join("")}</ul>`;
}

function homeSkeleton() {
  return `<section class="hero" aria-busy="true" aria-label="Loading your edit">
    <div class="hero__copy"><div class="skeleton-line" style="width:40%"></div><div class="skeleton-line" style="height:3rem"></div><div class="skeleton-line short"></div></div>
    <div class="hero__lead"><div class="tile__media skeleton"></div></div>
    <div class="hero__second"><div class="tile__media skeleton"></div></div>
  </section>`;
}

function heroFigure(it, cls, idx, eager) {
  return `<figure class="hero__figure ${cls} reveal" style="--i:${idx - 1}">${tile(it, "for_you", {index: idx, why: "shap", eager, meta: it.product_type_name})}</figure>`;
}

export async function home(ctx) {
  ctx.setTitle(null);
  ctx.render(homeSkeleton());
  const customer = state.customer;
  const d = await getHome(customer);
  if (!ctx.current() || customer !== state.customer) return;
  const c = d.customer;
  const mods = d.modules.filter(m => m.items?.length);
  const by = Object.fromEntries(mods.map(m => [m.id, m]));
  const fy = by.for_you?.items || [];
  const isNew = c.segment === "new";

  const lede = isNew
    ? "No purchase history yet, so this first edit is drawn from what shoppers your age are buying this week."
    : "A personal edit, ranked from your own purchases and what is selling now.";
  const hero = `<section class="hero" aria-labelledby="hero-title">
      <div class="hero__copy">
        <p class="eyebrow">${fy.length ? `Selected for you · ${plural(fy.length, "piece")}` : "Your edit"}</p>
        <h1 class="display" id="hero-title" tabindex="-1">${greeting()}</h1>
        <p class="lede">${lede}</p>
      </div>
      ${fy[0] ? heroFigure(fy[0], "hero__lead", 1, true) : ""}
      ${fy[1] ? heroFigure(fy[1], "hero__second", 2, true) : ""}
      ${fy[2] ? heroFigure(fy[2], "hero__third", 3, false) : ""}
      <div class="hero__meta">
        <p class="meta"><span class="label">Shopping as</span><br>${esc(profileName(c))} · ${esc(profileDetail(c))}</p>
        <p><a class="link-btn" href="#/collections">See the whole edit</a> &nbsp; <a class="link-btn" href="#/assistant">Ask the stylist</a></p>
      </div>
    </section>`;

  const sections = [];
  if (fy.length > 3) sections.push(`<section class="section" aria-labelledby="s-for-you">
      <div class="section__head"><div><p class="eyebrow">Continued</p><h2 class="section-title" id="s-for-you">${esc(editTitle(by.for_you))}</h2></div>
        <p class="section__aside">Larger pieces rank higher in your edit</p></div>
      ${editGrid(fy.slice(3), "for_you", 4, {why: "shap"})}</section>`);
  if (fy.length && !isNew) sections.push(`<section class="section" id="home-look" aria-labelledby="s-look" aria-busy="true">
      <div class="section__head"><div><p class="eyebrow">Complete the look</p><h2 class="section-title" id="s-look">Built around No. 01</h2></div></div>
      ${loadingTiles(4)}</section>`);
  if (by.buy_again) sections.push(`<section class="section again" aria-labelledby="s-again">
      <div class="section__head"><div><p class="eyebrow">From your orders</p><h2 class="section-title" id="s-again">${esc(editTitle(by.buy_again))}</h2>
        <p class="lede">Pieces you have bought, most recent first.</p></div></div>
      ${grid(by.buy_again.items, "buy_again", {cls: "grid--6 grid--compact", each: it => ({note: it.last_bought ? `Last bought ${esc(fmtDate(it.last_bought))}${it.times > 1 ? ` · ${it.times} times` : ""}` : ""})})}</section>`);
  if (by.trending) sections.push(`<section class="section" aria-labelledby="s-trend">
      <div class="section__head"><div><p class="eyebrow">This week</p><h2 class="section-title" id="s-trend">${esc(editTitle(by.trending))}</h2>
        <p class="lede">Most bought this week by shoppers in your age group.</p></div>
        <a class="link-btn" href="#/collections/trending">Browse by category</a></div>
      ${grid(by.trending.items, "trending", {cls: "grid--6 grid--compact"})}</section>`);
  for (const m of mods.filter(m => !EDITS[m.id])) sections.push(`<section class="section" aria-labelledby="s-${esc(m.id)}">
      <div class="section__head"><h2 class="section-title" id="s-${esc(m.id)}">${esc(m.title)}</h2></div>${grid(m.items, m.id)}</section>`);
  if (!fy.length && !mods.length) sections.push(`<div class="state"><p class="state__title">Nothing to show yet</p>
      <p class="meta">There are no recommendations for this profile. Try another profile from the menu at the top right.</p></div>`);

  const root = ctx.render(hero + sections.join(""));
  if (!root) return;
  // Impressions are logged for every rendered recommendation, as before the redesign.
  for (const m of mods) impressions(m.items, m.id);
  reveal(root);
  if (fy.length && !isNew) homeLook(ctx, fy[0], customer);
}

async function homeLook(ctx, anchor, customer) {
  const sec = $("#home-look");
  try {
    const v = await api(`/api/v2/complete-the-look?anchor=${anchor.article_id}&k=4&customer=${customer}`);
    if (!ctx.current() || !document.contains(sec)) return;
    const mods = v.modules.filter(m => m.items.length).slice(0, 3);
    if (!mods.length) { sec.remove(); return; }
    sec.removeAttribute("aria-busy");
    sec.innerHTML = `<div class="section__head"><div><p class="eyebrow">Complete the look</p>
        <h2 class="section-title" id="s-look">Built around No. 01</h2>
        <p class="lede">What our outfit model pairs with the ${esc(anchor.prod_name)}${v.personalized ? ", ordered for you" : ""}.</p></div>
        <a class="link-btn" href="#/product/${anchor.article_id}">See the full look</a></div>
      <div class="look">
        <div class="look__anchor reveal">${tile(anchor, "complete_the_look", {actions: false, meta: anchor.product_type_name, cls: ""})}
          <p class="meta look__anchor-caption">The anchor piece · No. 01 in your edit</p></div>
        <div class="look__slots">${mods.map(m => `<div class="look__slot"><h3>${esc(m.title)}</h3>
          ${grid(m.items.slice(0, 4).map(ctlCard), "complete_the_look", {cls: "grid--4 grid--compact", anchor: anchor.article_id})}</div>`).join("")}</div>
      </div>`;
    mods.forEach(m => impressions(m.items.slice(0, 4), "complete_the_look", anchor.article_id));
    reveal(sec);
  } catch (err) {
    console.warn("home Complete the Look unavailable:", err.message);
    if (document.contains(sec)) sec.remove();   // optional module: omit rather than show a broken section
  }
}

// v2 Complete the Look item -> tile item (keeps reasons and provenance for the Why dialog)
const ctlCard = it => ({...it.card, reasons: it.reasons, provenance: it.provenance});

// ---- Collections ---------------------------------------------------------------------------------------
export async function collections(ctx) {
  ctx.setTitle("Collections");
  ctx.render(`<div class="crumbs"><span>Collections</span></div>${loadingTiles(8)}`);
  const customer = state.customer;
  const d = await getHome(customer);
  if (!ctx.current() || customer !== state.customer) return;
  const mods = d.modules.filter(m => m.items?.length);
  if (!mods.length) {
    ctx.render(`<div class="state state--center"><h1 class="state__title" tabindex="-1">No collections for this profile</h1>
      <p class="meta">Choose another profile from the top right.</p></div>`);
    return;
  }
  const want = ctx.route.parts[1];
  const m = mods.find(x => x.id === want) || mods[0];
  const types = Object.entries(m.items.reduce((a, it) => (a[it.product_type_name] = (a[it.product_type_name] || 0) + 1, a), {}))
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const root = ctx.render(`
    <div class="stylist-head" style="padding-bottom:var(--s-4)">
      <div><p class="eyebrow">Collections</p><h1 class="title" tabindex="-1">${esc(editTitle(m))}</h1></div>
      <p class="meta">${plural(m.items.length, "piece")} · for ${esc(profileName(d.customer))}</p>
    </div>
    <nav class="tabs" aria-label="Collections">${mods.map(x =>
      `<a href="#/collections/${esc(x.id)}" ${x.id === m.id ? `aria-current="page"` : ""}>${esc(EDITS[x.id]?.tab || x.title)} <span class="meta">${x.items.length}</span></a>`).join("")}</nav>
    <div class="filters" role="group" aria-label="Filter by category">
      <button class="filter" type="button" aria-pressed="true" data-type="">All ${m.items.length}</button>
      ${types.map(([t, n]) => `<button class="filter" type="button" aria-pressed="false" data-type="${esc(t)}">${esc(t)} ${n}</button>`).join("")}
    </div>
    <p class="sr-only" id="filter-status" role="status"></p>
    ${grid(m.items, m.id, {cls: "grid--4", why: m.id === "for_you" ? "shap" : undefined,
      each: (it, i) => ({index: m.id === "for_you" ? i + 1 : null,
        note: it.last_bought ? `Last bought ${esc(fmtDate(it.last_bought))}` : ""})})}`);
  if (!root) return;
  impressions(m.items, m.id);
  const items = $$(".grid > li", root);
  $(".filters", root).addEventListener("click", e => {
    const b = e.target.closest(".filter");
    if (!b) return;
    $$(".filter", root).forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    let n = 0;
    items.forEach((li, i) => { const hit = !b.dataset.type || m.items[i].product_type_name === b.dataset.type; li.hidden = !hit; n += hit; });
    $("#filter-status").textContent = `${plural(n, "piece")} shown`;
  });
}

// ---- Product -------------------------------------------------------------------------------------------
const FALLBACK = {
  anchor_pool: "Paired directly with this piece",
  style_sibling: "Paired with another colour of this style",
  visual_neighbor: "Paired with the closest similar piece",
  slot_popularity: "Popular in this category",
};
const fallbackWords = lvl => FALLBACK[lvl] || words(lvl);

export async function product(ctx) {
  const id = +ctx.route.parts[1];
  ctx.render(`<div class="crumbs"><span class="skeleton-line short" style="width:10rem"></span></div>
    <div class="pdp" aria-busy="true"><div class="pdp__media skeleton"></div><div class="pdp__info"><div class="skeleton-line"></div><div class="skeleton-line short"></div></div></div>`);
  const customer = state.customer;
  const ctlReq = api(`/api/v2/complete-the-look?anchor=${id}&k=8${customer != null ? `&customer=${customer}` : ""}`)
    .catch(err => ({error: err}));
  const d = await api(`/api/product/${id}`);
  if (!ctx.current()) return;
  const a = d.article;
  ctx.setTitle(a.prod_name);
  logEvent("click", "product_page", a.article_id);
  const home = await cachedHome(customer);
  const mine = home?.modules.find(m => m.id === "for_you")?.items.find(it => it.article_id === a.article_id);
  const whyKey = mine ? tileKeyFor(mine) : null;

  const facts = [["Colour", a.colour_group_name], ["Product type", a.product_type_name], ["Department", a.department_name],
                 ["Section", a.index_group_name], ["Garment group", a.garment_group_name], ["Article no.", String(a.article_id).padStart(10, "0")]]
    .filter(([, v]) => v);
  const root = ctx.render(`
    <nav class="crumbs" aria-label="Breadcrumb">
      <button class="link-btn" type="button" data-back>← Back</button><span aria-hidden="true">/</span>
      <a href="#/">Discover</a><span aria-hidden="true">/</span><span>${esc(a.index_group_name)}</span><span aria-hidden="true">/</span><span aria-current="page">${esc(a.product_type_name)}</span>
    </nav>
    <div class="pdp">
      <figure class="pdp__media"><img src="${esc(a.image)}" alt="${esc(`${a.prod_name}, ${a.colour_group_name} ${a.product_type_name}`.toLowerCase())}" width="1166" height="1750" fetchpriority="high"></figure>
      <div class="pdp__info">
        <p class="eyebrow">${esc(a.product_type_name)} · ${esc(a.index_group_name)}</p>
        <h1 class="pdp__title" tabindex="-1">${esc(a.prod_name)}</h1>
        <p class="meta">${esc(a.colour_group_name)}</p>
        ${a.detail_desc ? `<p class="pdp__desc">${esc(a.detail_desc)}</p>` : ""}
        <div class="pdp__actions">
          <button class="btn btn--solid" type="button" id="add-bag" aria-describedby="bag-status">Add to bag <span class="tag" style="color:inherit;border-color:currentColor">Demo</span></button>
          <p class="inline-status" id="bag-status" role="status">Demo shop: there is no checkout or price in this dataset.</p>
          <a class="btn btn--quiet" href="#/assistant?anchor=${a.article_id}">Ask the stylist about this piece</a>
          ${whyKey ? `<button class="link-btn" type="button" data-why="${whyKey}" aria-haspopup="dialog">Why this is in your edit</button>` : ""}
        </div>
        <dl class="facts">${facts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
      </div>
    </div>
    <section class="section" id="pdp-look" aria-labelledby="s-ctl" aria-busy="true">
      <div class="section__head"><div><p class="eyebrow">Outfit</p><h2 class="section-title" id="s-ctl">Complete the look</h2></div></div>
      ${loadingTiles(4)}
    </section>
    ${d.other_colours.length ? `<section class="section" aria-labelledby="s-col">
      <div class="section__head"><div><p class="eyebrow">Same style</p><h2 class="section-title" id="s-col">Other colours</h2></div>
        <p class="section__aside">${plural(d.other_colours.length, "colourway")}</p></div>
      ${grid(d.other_colours, "other_colours", {cls: "grid--6 grid--compact", meta: undefined, each: it => ({meta: it.colour_group_name})})}</section>` : ""}
    ${d.similar.length ? `<section class="section" aria-labelledby="s-sim">
      <div class="section__head"><div><p class="eyebrow">More ${esc(a.product_type_name.toLowerCase())}</p><h2 class="section-title" id="s-sim">Similar items</h2></div>
        <p class="section__aside">Same type in ${esc(a.index_group_name)}, most popular first</p></div>
      ${grid(d.similar, "similar", {cls: "grid--6 grid--compact"})}</section>` : ""}`);
  if (!root) return;
  impressions(d.other_colours, "other_colours");
  impressions(d.similar, "similar");

  $("[data-back]", root).addEventListener("click", () => {
    if ((state.navCount || 0) > 1) history.back(); else location.hash = "#/";
  });
  const bag = $("#add-bag", root);
  bag.addEventListener("click", () => {
    if (bag.disabled) return;
    logEvent("add_to_cart", "product_page", a.article_id);
    bag.disabled = true;
    bag.firstChild.textContent = "Added (demo) ";
    const s = $("#bag-status", root);
    s.textContent = "Demo only: nothing is purchased. Your interest was recorded as an add-to-bag event.";
    s.classList.add("is-ok");
  });

  const v = await ctlReq;
  if (!ctx.current()) return;
  const sec = $("#pdp-look", root);
  sec.removeAttribute("aria-busy");
  sec.innerHTML = renderLook(a, d, v);
  reveal(root);
}

function tileKeyFor(item) {
  // A hidden registry entry so the product page can open the same SHAP explanation as the home tile.
  const holder = document.createElement("div");
  holder.innerHTML = tile(item, "for_you", {why: "shap"});
  return holder.querySelector("[data-why]").dataset.why;
}

function renderLook(a, d, v) {
  const head = `<div class="section__head"><div><p class="eyebrow">Outfit</p><h2 class="section-title" id="s-ctl">Complete the look</h2>`;
  if (v && !v.error) {
    const mods = v.modules.filter(m => m.items.length);
    if (mods.length) {
      mods.forEach(m => impressions(m.items, "complete_the_look", a.article_id));
      return `${head}<p class="lede">${v.personalized ? "Ordered for your profile from pieces that pair with this one." : "Ordered by how well each piece pairs with this one."}</p></div></div>
        <div style="display:grid;gap:var(--s-7)">${mods.map(m => `<div class="look__slot reveal">
          <h3>${esc(m.title)} <span class="tag ${m.personalized ? "tag--accent" : ""}">${m.personalized ? "Personalised order" : "Compatibility order"}</span>
            <span class="tag">${esc(fallbackWords(m.fallback_level))}</span></h3>
          ${grid(m.items.map(ctlCard), "complete_the_look", {cls: "grid--4", anchor: a.article_id})}</div>`).join("")}</div>
        <details class="disclosure" style="margin-top:var(--s-6)"><summary>How this look was assembled</summary>
          <p class="meta" style="padding-top:.5rem">${v.personalized ? "Personalised for this profile" : "Not personalised (no purchase history for this profile)"} ·
            model ${esc(v.model_version)} · catalogue ${esc(v.catalog_version)} · request ${esc(v.request_id)}.
            Each piece's “Why this” lists the evidence behind it.</p></details>`;
    }
  }
  // Fallback to the stored (v1) table when the live service is unavailable or has no module for this anchor.
  const degraded = v?.error ? `<p class="notice" style="margin-bottom:var(--s-5)"><strong>Showing stored suggestions.</strong> The live outfit service did not answer (${esc(v.error.message)}).</p>` : "";
  if (!d.complete_the_look.length) return `${head}</div></div>${degraded}
    <div class="state"><p class="state__title">No outfit suggestions for this piece</p>
    <p class="meta">It is outside the wearable categories or not in the live catalogue.</p></div>`;
  d.complete_the_look.forEach(m => impressions(m.items, "complete_the_look", a.article_id));
  return `${head}</div></div>${degraded}<div style="display:grid;gap:var(--s-7)">${d.complete_the_look.map(m => `<div class="look__slot">
    <h3>${esc(m.title)}</h3>${grid(m.items, "complete_the_look", {cls: "grid--4", anchor: a.article_id})}</div>`).join("")}</div>`;
}
