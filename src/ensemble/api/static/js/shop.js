// Shop: Collections (one tab per home module: Selected for you · Buy it again · Trending) and the product page.
// Module inventory comes from /api/home/{customer}; unavailable modules are omitted, never faked.
// Every page groups products by family page-wide: the first occurrence keeps its position, colourways
// become swatches on that card, and later repeats are dropped (never cloned to fill space).
import {$, $$, api, card, cardGrid, esc, fmtDate, getHome, impressions, loadFamilies, loadingTiles, logEvent, money,
        newPageGroups, reveal, state, swatchButtons, variantsFor} from "./core.js";

// Shopper titles for the API's module ids. Unknown ids fall back to the API title.
const EDITS = {
  for_you: {title: "Selected for you", lede: null},
  buy_again: {title: "Buy it again", lede: "Pieces you have bought."},
  trending: {title: "Trending this week", lede: "Popular this week with shoppers your age."},
};
const editTitle = m => EDITS[m.id]?.title || m.title;

const ids = list => list.map(it => it.article_id);
const SIMILAR_SHOWN = 4;   // product page: "Similar items" shows one row of four

// ---- Collections ---------------------------------------------------------------------------------------
export async function collections(ctx) {
  ctx.setTitle("Collections");
  ctx.render(`<div class="crumbs"><span>Collections</span></div>${loadingTiles(8)}`);
  const customer = state.customer;
  const d = await getHome(customer);
  if (!ctx.current() || customer !== state.customer) return;
  const mods = d.modules.filter(m => m.items?.length);
  if (!mods.length) {
    ctx.render(`<div class="state state--center"><h1 class="state__title" tabindex="-1">Nothing here yet</h1>
      <p class="meta">Try another profile from the top right.</p></div>`);
    return;
  }
  const want = ctx.route.parts[1];
  const m = mods.find(x => x.id === want) || mods[0];
  await loadFamilies(ids(m.items));
  if (!ctx.current() || customer !== state.customer) return;
  const items = newPageGroups().take(m.items);
  const types = [...new Set(items.map(it => it.product_type_name).filter(t => t && t !== "Unknown"))].sort();
  const root = ctx.render(`
    <div class="collection-head"><p class="eyebrow">Collections</p><h1 class="title" tabindex="-1">${esc(editTitle(m))}</h1></div>
    <nav class="tabs" aria-label="Collections">${mods.map(x =>
      `<a href="#/collections/${esc(x.id)}" ${x.id === m.id ? `aria-current="page"` : ""}>${esc(editTitle(x))}</a>`).join("")}</nav>
    ${types.length > 1 ? `<div class="filters" role="group" aria-label="Filter by category">
      <button class="filter" type="button" aria-pressed="true" data-type="">All</button>
      ${types.map(t => `<button class="filter" type="button" aria-pressed="false" data-type="${esc(t)}">${esc(t)}</button>`).join("")}
    </div>` : ""}
    <p class="sr-only" id="filter-status" role="status"></p>
    ${cardGrid(items, m.id, {cls: "grid--4", each: it => ({note: it.last_bought ? `Last bought ${esc(fmtDate(it.last_bought))}` : ""})})}`);
  if (!root) return;
  impressions(items, m.id);
  const lis = $$(".grid > li", root);
  $(".filters", root)?.addEventListener("click", e => {
    const b = e.target.closest(".filter");
    if (!b) return;
    $$(".filter", root).forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    lis.forEach((li, i) => { li.hidden = !!b.dataset.type && items[i].product_type_name !== b.dataset.type; });
    $("#filter-status").textContent = `Showing ${b.dataset.type || "all categories"}`;
  });
}

// ---- Product -------------------------------------------------------------------------------------------
export async function product(ctx) {
  const id = +ctx.route.parts[1];
  ctx.render(`<div class="crumbs"><span class="skeleton-line short" style="width:10rem"></span></div>
    <div class="pdp" aria-busy="true"><div class="pdp__media skeleton"></div><div class="pdp__info"><div class="skeleton-line"></div><div class="skeleton-line short"></div></div></div>`);
  const d = await api(`/api/product/${id}`);
  if (!ctx.current()) return;
  const a = d.article;
  await loadFamilies([a.article_id, ...ids(d.similar)]);
  if (!ctx.current()) return;
  ctx.setTitle(a.prod_name);
  logEvent("click", "product_page", a.article_id);

  const page = newPageGroups();
  page.claim(a.article_id);   // the product itself owns its family: its colourways are swatches, never cards
  const similar = page.take(d.similar).slice(0, SIMILAR_SHOWN);   // one row of four, after family grouping

  const {family, list} = variantsFor(a);
  const cur = list.find(x => x.article_id === a.article_id);
  const {btns} = swatchButtons(a, list);
  const facts = [["Colour", cur.label, "fact-colour"], ["Category", a.product_type_name], ["Department", a.department_name],
                 ["Article no.", String(a.article_id).padStart(10, "0"), "fact-article"]]
    .filter(([, val]) => val && val !== "Unknown");
  const root = ctx.render(`
    <nav class="crumbs" aria-label="Breadcrumb">
      <button class="link-btn" type="button" data-back>← Back</button><span aria-hidden="true">/</span>
      <a href="#/collections">Collections</a><span aria-hidden="true">/</span><span>${esc(a.index_group_name)}</span><span aria-hidden="true">/</span><span aria-current="page">${esc(a.product_type_name)}</span>
    </nav>
    <div class="pdp" data-card data-article="${a.article_id}" data-family="${esc(family?.product_code ?? "")}" data-s="product_page">
      <figure class="pdp__media"><img src="${esc(cur.image)}" alt="${esc(`${cur.prod_name}, ${cur.label.toLowerCase()}`)}" width="1166" height="1750" fetchpriority="high"></figure>
      <div class="pdp__info">
        <p class="eyebrow">${esc(a.index_group_name)}</p>
        <h1 class="pdp__title" tabindex="-1">${esc(cur.prod_name)}</h1>
        <p class="pdp__price card__price">${esc(money(family?.price_usd))}</p>
        <div class="pdp__colours">
          <p class="pdp__colour-label">Colour: <span class="card__colour">${esc(cur.label)}</span></p>
          <div class="swatches swatches--lg" role="group" aria-label="Colours">${btns}</div>
        </div>
        <div class="pdp__actions">
          <button class="btn-cart btn-cart--lg" type="button" data-add aria-label="Add ${esc(cur.prod_name)}, ${esc(cur.label)}, to cart" ${family?.price_usd == null ? "disabled" : ""}>Add to cart</button>
          <a class="btn btn--quiet" href="#/stylist/ask?anchor=${a.article_id}">Ask the stylist about this piece</a>
        </div>
        ${a.detail_desc ? `<p class="pdp__desc">${esc(a.detail_desc)}</p>` : ""}
        <dl class="facts">${facts.map(([k, val, cls]) => `<dt>${esc(k)}</dt><dd ${cls ? `class="${cls}"` : ""}>${esc(val)}</dd>`).join("")}</dl>
      </div>
    </div>
    ${similar.length ? `<section class="section" aria-labelledby="s-sim">
      <div class="section__head"><div><p class="eyebrow">You may also like</p><h2 class="section-title" id="s-sim">Similar items</h2></div></div>
      ${cardGrid(similar, "similar", {cls: "grid--4"})}</section>` : ""}`);
  if (!root) return;
  impressions(similar, "similar");

  $("[data-back]", root).addEventListener("click", () => {
    if ((state.navCount || 0) > 1) history.back(); else location.hash = "#/collections";
  });
  // Selecting a colour on the product page updates it in place and keeps the address shareable.
  $(".pdp", root).addEventListener("variant-change", e => {
    const {variant: nv} = e.detail;
    $(".pdp__title", root).textContent = nv.prod_name;
    $(".fact-colour", root).textContent = nv.label;
    $(".fact-article", root).textContent = String(nv.article_id).padStart(10, "0");
    ctx.setTitle(nv.prod_name);
    history.replaceState(null, "", `#/product/${nv.article_id}`);
  });
  reveal(root);
}
