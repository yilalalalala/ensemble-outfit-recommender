const PRODUCTS = [
  {id: "0922037001", name: "Skye flared trousers", price: 34.99, image: "assets/skye-flared-trousers.jpg", kind: "Bottoms", colours: [{name:"Beige",value:"#c6ae8d"},{name:"Yellowish brown",value:"#96704e"},{name:"Black",value:"#171717"}]},
  {id: "0910448001", name: "Petite high-waist trousers", price: 39.99, image: "assets/petite-high-waist-trousers.jpg", kind: "Bottoms", colours: [{name:"Black",value:"#171717"}]},
  {id: "0909721003", name: "Tree pleated trousers", price: 34.99, image: "assets/tree-pleated-trousers.jpg", kind: "Bottoms", colours: [{name:"Dark grey",value:"#55524f"},{name:"Black",value:"#171717"},{name:"Beige",value:"#c6ae8d"}]},
  {id: "0902419001", name: "Amelie fluffy sweater", price: 29.99, image: "assets/amelie-fluffy-sweater.jpg", kind: "Tops", colours: [{name:"White",value:"#f1eee6"}]},
];

const state = {view: "shop", page: "discover", cart: JSON.parse(localStorage.getItem("ensemble.portfolio.cart") || "[]")};
const $ = (s, root=document) => root.querySelector(s);
const money = n => new Intl.NumberFormat("en-US", {style:"currency", currency:"USD"}).format(n);

function productCard(p, index=0) {
  return `<article class="product" data-id="${p.id}">
    <div class="product-media"><img src="${p.image}" alt="${p.name}" ${index ? 'loading="lazy"' : ''}></div>
    <div class="product-info">
      <div class="product-row"><h3 class="product-name">${p.name}</h3><span class="price">${money(p.price)}</span></div>
      <div class="swatches"><span>${p.colours.length > 1 ? "Available colours" : p.colours[0].name}</span>${p.colours.map((c,i)=>`<button class="swatch ${i===0?'is-active':''}" style="background:${c.value}" aria-label="${c.name}${i===0?', selected':''}" title="${c.name}" data-colour="${i}"></button>`).join("")}</div>
      <button class="add" data-add="${p.id}">Add to cart</button>
    </div>
  </article>`;
}

function discover() {
  return `<div class="page">
    <section class="hero-copy"><p class="eyebrow">Selected for you</p><h1 class="display">A personal edit,<br><em>made with you in mind.</em></h1><p class="lede">Considered silhouettes, quiet colour, and pieces chosen to work beautifully together.</p></section>
    <section class="edit-grid" aria-label="Selected products">${PRODUCTS.slice(0,2).map(productCard).join("")}</section>
    <div class="section-head"><div><p class="eyebrow">Styled together</p><h2>Complete the look</h2></div><p>From basket signals, visual compatibility and personal taste.</p></div>
    <section class="look-strip">${PRODUCTS.slice(2).map((p,i)=>productCard(p,i)).join("")}</section>
  </div>`;
}

function collections(active="All") {
  const kinds = ["All", ...new Set(PRODUCTS.map(p=>p.kind))];
  const shown = active === "All" ? PRODUCTS : PRODUCTS.filter(p=>p.kind === active);
  return `<div class="page"><p class="eyebrow">The collection</p><h1 class="display">Pieces to live in.</h1>
    <nav class="tabs" aria-label="Categories">${kinds.map(k=>`<button class="tab ${k===active?'is-active':''}" data-filter="${k}">${k}</button>`).join("")}</nav>
    <section class="collection-grid">${shown.map(productCard).join("")}</section></div>`;
}

function assistant() {
  return `<div class="page assistant-layout"><section><p class="eyebrow">Style Assistant</p><h1 class="assistant-title">Tell us what you’re looking for.</h1><p class="lede">Bring an occasion, a mood, or a piece you already own. The production application grounds every answer in retrieved products.</p></section>
    <section class="assistant-card"><label for="ask">Ask the stylist</label><input id="ask" placeholder="What goes with wide-leg trousers?">
      <div class="suggestions">${["“What should I wear to a summer wedding?”","“Find a polished layer for autumn.”","“What shoes work with wide-leg trousers?”"].map(x=>`<button data-suggest>${x}</button>`).join("")}</div>
      <button class="add" id="ask-btn">Find pieces</button><div class="answer" id="answer"><p>Try one of the questions above, or write your own.</p></div></section></div>`;
}

function studio() {
  return `<div class="page"><header class="studio-head"><div><p class="eyebrow">DS Studio · offline evaluation</p><h1>What the models proved.</h1></div><p>The public demo is static; these values come from the repository’s frozen temporal evaluations and Kaggle result. Full protocols, confidence intervals and model cards are linked in the source.</p></header>
    <section class="kpis"><div class="kpi"><strong>0.0333</strong><span>Kaggle private MAP@12</span></div><div class="kpi"><strong>+56%</strong><span>Complete-the-Look Recall@12</span></div><div class="kpi"><strong>0.59</strong><span>Visual-search Recall@1</span></div><div class="kpi"><strong>100%</strong><span>Assistant tool accuracy</span></div></section>
    <section class="method-grid"><article class="method"><p class="eyebrow">Track A</p><h3>Retrieve, then rank</h3><p>Nine retrieval channels feed a LightGBM LambdaRank model with point-in-time customer, article and interaction features.</p></article><article class="method"><p class="eyebrow">Track B</p><h3>Complete the outfit</h3><p>Basket associations, a logQ-corrected two-tower model and FashionCLIP features are fused by a learned ranker.</p></article><article class="method"><p class="eyebrow">Visual + Assistant</p><h3>Grounded discovery</h3><p>A DeepFashion2 adapter closes the street-to-shop gap; tool grounding prevents the assistant from citing products it did not retrieve.</p></article></section>
  </div>`;
}

function render() {
  const main = $("#content");
  document.body.classList.toggle("studio", state.view === "studio");
  $$(".view-switch").forEach(b=>b.classList.toggle("is-active", b.dataset.view===state.view));
  $(".nav-left").style.visibility = state.view === "shop" ? "visible" : "hidden";
  $(".profile").style.visibility = state.view === "shop" ? "visible" : "hidden";
  $(".bag").style.visibility = state.view === "shop" ? "visible" : "hidden";
  if (state.view === "studio") main.innerHTML = studio();
  else if (state.page === "collections") main.innerHTML = collections();
  else if (state.page === "assistant") main.innerHTML = assistant();
  else main.innerHTML = discover();
  $$(".nav-link").forEach(b=>b.classList.toggle("is-active", b.dataset.page===state.page));
  window.scrollTo({top: 0, behavior: "instant"});
}

function saveCart() { localStorage.setItem("ensemble.portfolio.cart", JSON.stringify(state.cart)); paintCart(); }
function paintCart() {
  $("#cart-count").textContent = state.cart.reduce((n,x)=>n+x.qty,0);
  $("#cart-lines").innerHTML = state.cart.length ? state.cart.map(x=>{const p=PRODUCTS.find(p=>p.id===x.id);return `<article class="cart-line"><img src="${p.image}" alt=""><div><h3>${p.name}</h3><span>${money(p.price)} · Qty ${x.qty}</span><br><button data-remove="${p.id}">Remove</button></div><strong>${money(p.price*x.qty)}</strong></article>`}).join("") : `<p class="empty">Your cart is empty.</p>`;
  $("#subtotal").textContent = money(state.cart.reduce((n,x)=>n+PRODUCTS.find(p=>p.id===x.id).price*x.qty,0));
}
function toast(message) { const el=$(".toast"); el.textContent=message; el.classList.add("show"); clearTimeout(toast.t); toast.t=setTimeout(()=>el.classList.remove("show"),1800); }
function $$(s, root=document) { return [...root.querySelectorAll(s)]; }

document.addEventListener("click", e => {
  const view=e.target.closest("[data-view]"); if(view){state.view=view.dataset.view;render();return;}
  const page=e.target.closest("[data-page]"); if(page){state.view="shop";state.page=page.dataset.page;render();return;}
  const filter=e.target.closest("[data-filter]"); if(filter){$("#content").innerHTML=collections(filter.dataset.filter);return;}
  const sw=e.target.closest(".swatch"); if(sw){sw.closest(".swatches").querySelectorAll(".swatch").forEach(x=>x.classList.remove("is-active"));sw.classList.add("is-active");toast("Colour selected");return;}
  const add=e.target.closest("[data-add]"); if(add){const line=state.cart.find(x=>x.id===add.dataset.add);line?line.qty++:state.cart.push({id:add.dataset.add,qty:1});saveCart();toast("Added to cart");return;}
  const remove=e.target.closest("[data-remove]"); if(remove){state.cart=state.cart.filter(x=>x.id!==remove.dataset.remove);saveCart();return;}
  const suggestion=e.target.closest("[data-suggest]"); if(suggestion){$("#ask").value=suggestion.textContent.replace(/[“”]/g,"");return;}
  if(e.target.closest("#ask-btn")){const q=$("#ask").value.trim();$("#answer").innerHTML=q?`<p><strong>A considered starting point:</strong> the ${PRODUCTS[2].name.toLowerCase()} over the ${PRODUCTS[0].name.toLowerCase()}, finished with ${PRODUCTS[3].name.toLowerCase()}.</p>`:`<p>Tell us what you’re dressing for first.</p>`;return;}
});

$(".bag").addEventListener("click",()=>$("#cart").showModal());
$(".close").addEventListener("click",()=>$("#cart").close());
$("#cart").addEventListener("click",e=>{if(e.target===$("#cart"))$("#cart").close();});
paintCart();
render();
