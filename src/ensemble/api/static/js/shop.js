// Shop: Discover (editorial home), Collections and the product page.
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

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? "Good evening." : h < 12 ? "Good morning." : h < 18 ? "Good afternoon." : "Good evening.";
}

// Deterministic asymmetric rhythm: feature rows give the row's first (highest-placed) piece the large tile,
// alternating with a staggered row of three; mobile simplifies to two columns.
const ROW_A = [{span: 6, sm: 2}, {span: 3, drop: "var(--s-8)"}, {span: 3, align: "end", dropSm: "var(--s-6)"}];
const ROW_B = [{span: 4, drop: "var(--s-7)"}, {span: 4, dropSm: "var(--s-6)"}, {span: 4, drop: "var(--s-8)"}];
function editLayout(n) {
  const out = [];
  for (let row = 0; out.length < n; row++) out.push(...(row % 2 ? ROW_B : ROW_A));
  return out.slice(0, n);
}
function editGrid(items, surface) {
  const layout = editLayout(items.length);
  return `<ul class="edit" role="list">${items.map((it, i) => {
    const l = layout[i];
    const style = `--span:${l.span};--span-sm:${l.sm || 1};${l.drop ? `--drop:${l.drop};` : ""}${l.dropSm ? `--drop-sm:${l.dropSm};` : ""}${l.align ? `--align:${l.align};` : ""}`;
    return `<li style="${style}"><div class="reveal" style="--i:${i % 3}">${card(it, surface)}</div></li>`;
  }).join("")}</ul>`;
}

function homeSkeleton() {
  return `<section class="hero" aria-busy="true" aria-label="Loading">
    <div class="hero__copy"><div class="skeleton-line" style="width:40%"></div><div class="skeleton-line" style="height:3rem"></div><div class="skeleton-line short"></div></div>
    <div class="hero__lead"><div class="card__media skeleton"></div></div>
    <div class="hero__second"><div class="card__media skeleton"></div></div>
  </section>`;
}

const ctlCard = it => ({...it.card});
const ids = list => list.map(it => it.article_id);

// ---- Discover ------------------------------------------------------------------------------------------
export async function home(ctx) {
  ctx.setTitle(null);
  ctx.render(homeSkeleton());
  const customer = state.customer;
  const d = await getHome(customer);
  if (!ctx.current() || customer !== state.customer) return;
  const c = d.customer;
  const isNew = c.segment === "new";
  const mods = d.modules.filter(m => m.items?.length);
  const by = Object.fromEntries(mods.map(m => [m.id, m]));
  const fyAll = by.for_you?.items || [];
  // Complete the Look around the first pick is part of the page, so it is grouped in display order too.
  const look = fyAll.length && !isNew
    ? await api(`/api/v2/complete-the-look?anchor=${fyAll[0].article_id}&k=4&customer=${customer}`).catch(err => { console.warn("Complete the Look unavailable:", err.message); return null; })
    : null;
  const lookMods = (look?.modules || []).filter(m => m.items.length).slice(0, 3);
  await loadFamilies([...mods.flatMap(m => ids(m.items)), ...lookMods.flatMap(m => m.items.map(i => i.article_id))]);
  if (!ctx.current() || customer !== state.customer) return;

  const page = newPageGroups();
  const fy = page.take(fyAll);
  const hero = fy.slice(0, 3), rest = fy.slice(3);
  const lookGroups = lookMods.map(m => ({...m, items: page.take(m.items.map(ctlCard)).slice(0, 4)})).filter(m => m.items.length);
  const again = by.buy_again ? page.take(by.buy_again.items) : [];
  const trending = by.trending ? page.take(by.trending.items) : [];
  const others = mods.filter(m => !EDITS[m.id]).map(m => ({...m, items: page.take(m.items)})).filter(m => m.items.length);

  const lede = isNew ? "A first edit, chosen from what shoppers your age love this week." : "A personal edit, made with you in mind.";
  const fig = (it, cls, i, eager) => `<figure class="hero__figure ${cls} reveal" style="--i:${i}">${card(it, "for_you", {eager})}</figure>`;
  const sections = [`<section class="hero" aria-labelledby="hero-title">
      <div class="hero__copy">
        <p class="eyebrow">Selected for you</p>
        <h1 class="display" id="hero-title" tabindex="-1">${greeting()}</h1>
        <p class="lede">${lede}</p>
        <p class="hero__links"><a class="link-btn" href="#/collections">Shop the edit</a><a class="link-btn" href="#/assistant">Ask the stylist</a></p>
      </div>
      ${hero[0] ? fig(hero[0], "hero__lead", 0, true) : ""}
      ${hero[1] ? fig(hero[1], "hero__second", 1, true) : ""}
      ${hero[2] ? fig(hero[2], "hero__third", 2, false) : ""}
    </section>`];
  if (rest.length) sections.push(`<section class="section" aria-labelledby="s-for-you">
      <div class="section__head"><div><p class="eyebrow">Continued</p><h2 class="section-title" id="s-for-you">${esc(editTitle(by.for_you))}</h2></div></div>
      ${editGrid(rest, "for_you")}</section>`);
  if (lookGroups.length) sections.push(`<section class="section" aria-labelledby="s-look">
      <div class="section__head"><div><p class="eyebrow">Complete the look</p>
        <h2 class="section-title" id="s-look">Styled with the ${esc(fyAll[0].prod_name)}</h2></div>
        <a class="link-btn" href="#/product/${fyAll[0].article_id}">See the piece</a></div>
      <div class="look-groups">${lookGroups.map(m => `<div class="look__slot reveal"><h3 class="look__title">${esc(m.title)}</h3>
        ${cardGrid(m.items, "complete_the_look", {cls: "grid--4", anchor: fyAll[0].article_id})}</div>`).join("")}</div></section>`);
  if (again.length) sections.push(`<section class="section" aria-labelledby="s-again">
      <div class="section__head"><div><p class="eyebrow">From your orders</p><h2 class="section-title" id="s-again">${esc(editTitle(by.buy_again))}</h2>
        <p class="lede">${EDITS.buy_again.lede}</p></div></div>
      ${cardGrid(again, "buy_again", {cls: "grid--4", each: it => ({note: it.last_bought ? `Last bought ${esc(fmtDate(it.last_bought))}` : ""})})}</section>`);
  if (trending.length) sections.push(`<section class="section" aria-labelledby="s-trend">
      <div class="section__head"><div><p class="eyebrow">This week</p><h2 class="section-title" id="s-trend">${esc(editTitle(by.trending))}</h2>
        <p class="lede">${EDITS.trending.lede}</p></div>
        <a class="link-btn" href="#/collections/trending">Shop by category</a></div>
      ${cardGrid(trending, "trending", {cls: "grid--4"})}</section>`);
  for (const m of others) sections.push(`<section class="section" aria-labelledby="s-${esc(m.id)}">
      <div class="section__head"><h2 class="section-title" id="s-${esc(m.id)}">${esc(m.title)}</h2></div>${cardGrid(m.items, m.id)}</section>`);
  if (!fy.length && !again.length && !trending.length) sections.push(`<div class="state"><p class="state__title">Nothing here yet</p>
      <p class="meta">Try another profile from the top right.</p></div>`);

  const root = ctx.render(sections.join(""));
  if (!root) return;
  // Impressions are logged for every product card actually shown.
  impressions(fy, "for_you");
  lookGroups.forEach(m => impressions(m.items, "complete_the_look", fyAll[0].article_id));
  impressions(again, "buy_again");
  impressions(trending, "trending");
  others.forEach(m => impressions(m.items, m.id));
  reveal(root);
}

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
  const customer = state.customer;
  const [d, v] = await Promise.all([
    api(`/api/product/${id}`),
    api(`/api/v2/complete-the-look?anchor=${id}&k=8${customer != null ? `&customer=${customer}` : ""}`)
      .catch(err => { console.warn("Complete the Look (live) unavailable, using stored suggestions:", err.message); return null; }),
  ]);
  if (!ctx.current()) return;
  const a = d.article;
  // Live suggestions when available; otherwise the stored table, without announcing the difference to shoppers.
  const lookMods = v?.modules?.some(m => m.items.length)
    ? v.modules.filter(m => m.items.length).map(m => ({title: m.title, items: m.items.map(ctlCard)}))
    : d.complete_the_look.map(m => ({title: m.title, items: m.items}));
  await loadFamilies([a.article_id, ...lookMods.flatMap(m => ids(m.items)), ...ids(d.similar)]);
  if (!ctx.current()) return;
  ctx.setTitle(a.prod_name);
  logEvent("click", "product_page", a.article_id);

  const page = newPageGroups();
  page.claim(a.article_id);   // the product itself owns its family: its colourways are swatches, never cards
  const look = lookMods.map(m => ({...m, items: page.take(m.items)})).filter(m => m.items.length);
  const similar = page.take(d.similar);

  const {family, list} = variantsFor(a);
  const cur = list.find(x => x.article_id === a.article_id);
  const {btns} = swatchButtons(a, list);
  const facts = [["Colour", cur.label, "fact-colour"], ["Category", a.product_type_name], ["Department", a.department_name],
                 ["Article no.", String(a.article_id).padStart(10, "0"), "fact-article"]]
    .filter(([, val]) => val && val !== "Unknown");
  const root = ctx.render(`
    <nav class="crumbs" aria-label="Breadcrumb">
      <button class="link-btn" type="button" data-back>← Back</button><span aria-hidden="true">/</span>
      <a href="#/">Discover</a><span aria-hidden="true">/</span><span>${esc(a.index_group_name)}</span><span aria-hidden="true">/</span><span aria-current="page">${esc(a.product_type_name)}</span>
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
          <a class="btn btn--quiet" href="#/assistant?anchor=${a.article_id}">Ask the stylist about this piece</a>
        </div>
        ${a.detail_desc ? `<p class="pdp__desc">${esc(a.detail_desc)}</p>` : ""}
        <dl class="facts">${facts.map(([k, val, cls]) => `<dt>${esc(k)}</dt><dd ${cls ? `class="${cls}"` : ""}>${esc(val)}</dd>`).join("")}</dl>
      </div>
    </div>
    ${look.length ? `<section class="section" aria-labelledby="s-ctl">
      <div class="section__head"><div><p class="eyebrow">Outfit</p><h2 class="section-title" id="s-ctl">Complete the look</h2>
        <p class="lede">Pieces that go with this one.</p></div></div>
      <div class="look-groups">${look.map(m => `<div class="look__slot reveal"><h3 class="look__title">${esc(m.title)}</h3>
        ${cardGrid(m.items, "complete_the_look", {cls: "grid--4", anchor: a.article_id})}</div>`).join("")}</div></section>` : ""}
    ${similar.length ? `<section class="section" aria-labelledby="s-sim">
      <div class="section__head"><div><p class="eyebrow">You may also like</p><h2 class="section-title" id="s-sim">Similar items</h2></div></div>
      ${cardGrid(similar, "similar", {cls: "grid--4"})}</section>` : ""}`);
  if (!root) return;
  look.forEach(m => impressions(m.items, "complete_the_look", a.article_id));
  impressions(similar, "similar");

  $("[data-back]", root).addEventListener("click", () => {
    if ((state.navCount || 0) > 1) history.back(); else location.hash = "#/";
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
