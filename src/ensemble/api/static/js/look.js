// Complete my look: 1 Photo → 2 Review → 3 Your edit (docs/AI_STYLIST_PRODUCT_FLOW.md §4).
// Detection always lands on Review; the edit is built only from garments the shopper confirmed. There is one
// direction (complete the look): dressing for an occasion is what Ask does, and similar pieces is Find.
import {$, $$, ApiError, announce, api, cardGrid, esc, impressions, state} from "./core.js";
import {photoPicker, stage, toBlob} from "./photo.js";
import {focusHeading, grouped, shell, stagedStatus} from "./stylist.js";

const STEPS = ["Photo", "Review", "Your edit"];
const KEYS = ["photo", "review", "edit"];
const store = stage("ensemble.look.v1");
const CATEGORY_LABELS = {top: "Top", outerwear: "Outerwear", knitwear: "Knitwear", bottom: "Bottoms", dress: "Dress",
  jumpsuit: "Jumpsuit", shoes: "Shoes", bag: "Bag", jewellery: "Jewellery", hat: "Hat", scarf: "Scarf", belt: "Belt",
  sunglasses: "Sunglasses", socks: "Socks"};
const load = () => store.get() || {};
const save = patch => { const s = {...load(), ...patch}; store.set(s); return s; };
const go = step => { location.hash = `#/stylist/look?step=${step}`; };
let seq = 0;   // garment ids for added rows

// The furthest step the saved state allows; a deep link to a later step falls back to it.
function reachable(s) {
  if (s.result) return 2;
  if (s.garments && s.photo) return 1;
  return 0;
}

export async function view(ctx) {
  ctx.setTitle("Complete my look");
  const s = load();
  const asked = ctx.route.params.get("step") || "photo";
  const want = Math.max(0, KEYS.indexOf(asked === "direction" ? "review" : asked));   // old links to the removed step
  const step = Math.min(want, reachable(s));
  if (step !== want) { history.replaceState(null, "", `#/stylist/look?step=${KEYS[step]}`); }
  const render = [photoStep, reviewStep, editStep][step];
  await render(ctx, s);
}

const frame = (ctx, step, opts) => ctx.render(shell({task: "look", steps: STEPS, step, ...opts}));
const startOver = `<button class="link-btn" type="button" data-start-over>Start over</button>`;

function wireStartOver(root) {
  $("[data-start-over]", root)?.addEventListener("click", () => {
    store.clear();
    announce("Started over.");
    if (location.hash === "#/stylist/look?step=photo") location.reload(); else go("photo");
  });
}

// ---- 1 · Photo ----------------------------------------------------------------------------------------------
async function photoStep(ctx, s) {
  const root = frame(ctx, 0, {
    title: "Show us your outfit", back: "#/stylist",
    lede: "One full-length photo of what you are wearing. We will show you what we spot before suggesting anything.",
    body: `<div class="look-photo" id="look-photo"></div>
      <p class="privacy">Your photo is only used to find pieces for you.</p>
      <div class="task__status" id="look-status" aria-live="polite"></div>`,
    primary: `<button class="btn btn--solid" type="button" id="look-detect" ${s.photo ? "" : "disabled"}>Look at my outfit</button>`,
    secondary: s.photo ? startOver : "",
  });
  if (!root) return;
  focusHeading(root);
  wireStartOver(root);
  const btn = $("#look-detect", root), status = $("#look-status", root);
  photoPicker($("#look-photo", root), {
    photo: s.photo, help: "JPEG, PNG or WebP, up to 12 MB. Full-length and well lit works best.",
    onChange: p => {
      store.set(p ? {photo: p} : {});   // a new photo invalidates everything after it
      btn.disabled = !p;
      status.innerHTML = "";
    },
  });
  btn.addEventListener("click", async () => {
    const cur = load();
    if (!cur.photo) return;
    btn.disabled = true;
    const stop = stagedStatus(status, ["Looking at your outfit…", "Picking out each piece…", "Almost there…"]);
    try {
      const fd = new FormData();
      fd.append("photo", toBlob(cur.photo.dataUrl), "outfit.jpg");
      const r = await api("/api/stylist/detect", {method: "POST", body: fd});
      if (!ctx.current()) return;
      save({garments: r.garments.map(g => ({...g, source: "detected"})), reviewed: false, result: null, detectedAt: Date.now()});
      go("review");
    } catch (e) {
      const resting = e instanceof ApiError && e.code === "ai_budget_reached";
      status.innerHTML = `<div class="state" role="alert"><p class="state__title">${resting ? "This feature is resting for now" : "We couldn't look at that photo just now"}</p>
        <p class="meta">${resting ? "Please try again later." : "Please try again, or add the pieces yourself."}</p>
        <p class="state__actions">${resting ? "" : `<button class="btn btn--quiet" type="button" id="look-retry">Try again</button>`}
          <button class="link-btn" type="button" id="look-manual">Add pieces yourself</button></p></div>`;
      $("#look-retry", status)?.addEventListener("click", () => btn.click());
      $("#look-manual", status).addEventListener("click", () => { save({garments: [], reviewed: false, result: null}); go("review"); });
      btn.disabled = false;
    } finally { stop(); }
  });
}

// ---- 2 · Review ---------------------------------------------------------------------------------------------
// "black strappy sandals", never "black black strappy sandals" when the description already names the colour.
const label = g => {
  const c = (g.colour || "").trim();
  const what = g.description || (c ? (CATEGORY_LABELS[g.category] || g.category).toLowerCase() : CATEGORY_LABELS[g.category] || g.category);
  return !c || what.toLowerCase().startsWith(c.toLowerCase()) ? what : `${c} ${what}`;
};

function garmentRow(g, i) {
  const id = esc(g.id);
  return `<li class="garment" data-gid="${id}">
    <span class="garment__n" aria-hidden="true">${i + 1}</span>
    <div class="garment__fields">
      <p class="garment__name" id="gname-${id}">${esc(label(g))}</p>
      <div class="garment__edit">
        <label><span class="field-label">Piece</span>
          <select class="input" data-field="category" aria-describedby="gname-${id}">${Object.entries(CATEGORY_LABELS).map(([k, v]) =>
            `<option value="${k}" ${k === g.category ? "selected" : ""}>${v}</option>`).join("")}</select></label>
        <label><span class="field-label">Colour</span>
          <input class="input" data-field="colour" value="${esc(g.colour)}" maxlength="40" autocomplete="off" aria-describedby="gname-${id}"></label>
      </div>
    </div>
    <button class="link-btn garment__remove" type="button" data-remove aria-label="Remove ${esc(label(g))}">Remove</button>
  </li>`;
}

async function reviewStep(ctx, s) {
  const n = s.garments.length;
  const root = frame(ctx, 1, {
    title: "Is this right?", back: "#/stylist/look?step=photo",
    lede: n ? `We spotted ${n} piece${n === 1 ? "" : "s"}. Fix anything we got wrong, remove what isn't there, or add what we missed.`
            : "We couldn't spot any pieces. Add what you are wearing and we will take it from there.",
    body: `<div class="review">
        <figure class="review__photo"><div class="review__frame">
          <img src="${s.photo.dataUrl}" alt="Your outfit photo with the spotted pieces numbered" width="${s.photo.w}" height="${s.photo.h}">
          <div id="review-boxes" aria-hidden="true"></div></div></figure>
        <div class="review__list">
          <ol class="garments" id="garments" role="list"></ol>
          <button class="btn btn--quiet review__add" type="button" id="garment-add">Add a piece we missed</button>
          <p class="inline-status is-error" id="review-err" role="alert"></p>
          <div class="task__status" id="review-status" aria-live="polite"></div>
        </div>
      </div>`,
    primary: `<button class="btn btn--solid" type="button" id="review-next">Complete my look</button>`,
    secondary: startOver,
  });
  if (!root) return;
  focusHeading(root);
  wireStartOver(root);
  let garments = s.garments.map(g => ({...g}));
  const list = $("#garments", root), boxes = $("#review-boxes", root), err = $("#review-err", root);
  const persist = () => save({garments, reviewed: false, result: null});
  const paint = () => {
    list.innerHTML = garments.map(garmentRow).join("");
    boxes.innerHTML = garments.map((g, i) => g.box ? `<span class="review__box" style="left:${g.box[0] * 100}%;top:${g.box[1] * 100}%;
      width:${(g.box[2] - g.box[0]) * 100}%;height:${(g.box[3] - g.box[1]) * 100}%"><span>${i + 1}</span></span>` : "").join("");
  };
  paint();
  list.addEventListener("change", e => {
    const row = e.target.closest("[data-gid]"), f = e.target.dataset.field;
    if (!row || !f) return;
    const g = garments.find(x => x.id === row.dataset.gid);
    g[f] = e.target.value.trim();
    if (f === "category") g.description = "";   // the old description named the old piece
    g.edited = true;
    $(".garment__name", row).textContent = label(g);
    persist();
  });
  list.addEventListener("click", e => {
    const b = e.target.closest("[data-remove]");
    if (!b) return;
    const row = b.closest("[data-gid]"), idx = garments.findIndex(x => x.id === row.dataset.gid);
    const gone = garments.splice(idx, 1)[0];
    persist();
    paint();
    announce(`${label(gone)} removed.`);
    ($$("[data-remove]", list)[Math.min(idx, garments.length - 1)] || $("#garment-add", root)).focus();
  });
  $("#garment-add", root).addEventListener("click", () => {
    garments.push({id: `a${Date.now().toString(36)}${seq++}`, category: "shoes", colour: "", description: "", box: null, source: "added"});
    persist();
    paint();
    const row = list.lastElementChild;
    $("select", row).focus();
    announce("Piece added. Choose what it is.");
  });
  const next = $("#review-next", root), status = $("#review-status", root);
  next.addEventListener("click", async () => {
    if (!garments.length) { err.textContent = "Add at least one piece you are wearing."; $("#garment-add", root).focus(); return; }
    err.textContent = "";
    save({garments, reviewed: true, result: null});
    next.disabled = true;
    const stop = stagedStatus(status, ["Finding pieces that go with it…", "Choosing the strongest pairings…", "Putting your edit together…"]);
    try {
      const cur = load();
      const fd = new FormData();
      fd.append("photo", toBlob(cur.photo.dataUrl), "outfit.jpg");
      fd.append("garments", JSON.stringify(cur.garments.map(({category, colour, description, box}) => ({category, colour, description, box}))));
      fd.append("goal", "complete");
      if (state.customer != null) fd.append("customer", state.customer);
      const r = await api("/api/stylist/edit", {method: "POST", body: fd});
      if (!ctx.current()) return;
      save({goal: "complete", result: r});
      go("edit");
    } catch (e) {
      const resting = e instanceof ApiError && e.code === "ai_budget_reached";
      status.innerHTML = `<div class="state" role="alert"><p class="state__title">We couldn't build your edit just now</p>
        <p class="meta">${resting ? "This feature is resting for now. Please try again later." : "Your pieces are saved. Please try again."}</p>
        ${resting ? "" : `<button class="btn btn--quiet" type="button" id="review-retry">Try again</button>`}</div>`;
      $("#review-retry", status)?.addEventListener("click", () => next.click());
      next.disabled = false;
    } finally { stop(); }
  });
}

// ---- 3 · Your edit ------------------------------------------------------------------------------------------
async function editStep(ctx, s) {
  const r = s.result;
  const picksIn = r.picks.map(p => ({...p.card, _reason: p.reason}));
  const [grouped4] = await grouped([picksIn]);
  if (!ctx.current()) return;
  const picks = grouped4.slice(0, 4);   // one row of four; no "more options"
  const surface = r.goal === "similar" ? "snap" : "complete_the_look";
  const root = frame(ctx, 2, {
    title: "Your edit", back: "#/stylist/look?step=review",
    lede: esc(r.summary),
    body: picks.length ? `<section class="look-edit" aria-label="Your edit" style="--pick-cols:4">
        ${cardGrid(picks, surface, {cls: "grid--4 look-edit__picks", anchor: r.anchor ?? undefined,
          each: it => ({note: esc(it._reason)})})}
      </section>`
      : `<div class="state"><p class="state__title">We couldn't find pieces for this one</p>
         <p class="meta">Check the pieces we spotted and try again.</p>
         <p class="state__actions"><a class="btn btn--quiet" href="#/stylist/look?step=review">Check the pieces</a></p></div>`,
    primary: picks.length ? `<a class="btn btn--solid" href="#/stylist/ask?context=look">Refine this edit</a>` : "",
    secondary: startOver,
  });
  if (!root) return;
  focusHeading(root);
  wireStartOver(root);
  impressions(picks, surface, r.anchor ?? null);
  announce("Your edit is ready.");
}
