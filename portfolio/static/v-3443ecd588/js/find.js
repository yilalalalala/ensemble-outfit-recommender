// Find this piece: 1 Photo → 2 Choose → 3 Results, with an optional hand-off to the stylist
// (docs/AI_STYLIST_PRODUCT_FLOW.md §4–5). The pieces in the photo are spotted for the shopper (/api/stylist/detect);
// they tap the ones they want (or drag around one), each chosen piece shows what it is and can be corrected, and
// every chosen piece gets its four closest matches from the visual search endpoint.
import {$, ApiError, announce, api, cardGrid, esc, impressions} from "./core.js";
import {photoPicker, stage, toBlob} from "./photo.js";
import {focusHeading, grouped, shell, stagedStatus} from "./stylist.js";

const STEPS = ["Photo", "Choose", "Results"];
const KEYS = ["photo", "frame", "results"];
const store = stage("ensemble.find.v2");
const PER_PIECE = 4;
const CATEGORIES = {top: "Top", outerwear: "Outerwear", knitwear: "Knitwear", bottom: "Bottoms", dress: "Dress",
  jumpsuit: "Jumpsuit", shoes: "Shoes", bag: "Bag", jewellery: "Jewellery", hat: "Hat", scarf: "Scarf", belt: "Belt",
  sunglasses: "Sunglasses"};

const load = () => store.get() || {};
const save = patch => { const s = {...load(), ...patch}; store.set(s); return s; };
const go = step => { location.hash = `#/stylist/find?step=${step}`; };
const reachable = s => (s.results ? 2 : s.photo ? 1 : 0);

// "beige sweatshirt", never "beige beige sweatshirt"; a piece the shopper framed reads as its category.
export function pieceLabel(p) {
  const what = (p.description || (CATEGORIES[p.category] || p.category)).trim();
  const c = (p.colour || "").trim();
  return !c || new RegExp(`\\b${c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(what) ? what : `${c} ${what}`;
}

// What a framed area most likely is: the spotted piece it overlaps most (share of the framed area), if any.
export function categoryFor(box, pieces, fallback = "top") {
  let best = null, score = 0;
  for (const p of pieces) {
    if (!p.box || p.added) continue;
    const w = Math.min(box[2], p.box[2]) - Math.max(box[0], p.box[0]);
    const h = Math.min(box[3], p.box[3]) - Math.max(box[1], p.box[1]);
    const share = w > 0 && h > 0 ? (w * h) / ((box[2] - box[0]) * (box[3] - box[1])) : 0;
    if (share > score) { score = share; best = p; }
  }
  return score >= 0.3 ? {category: best.category, colour: best.colour || "", description: best.description || "", match: best} : {category: fallback, colour: "", description: ""};
}

export async function view(ctx) {
  ctx.setTitle("Find this piece");
  const s = load();
  const want = Math.max(0, KEYS.indexOf(ctx.route.params.get("step") || "photo"));
  const step = Math.min(want, reachable(s));
  if (step !== want) history.replaceState(null, "", `#/stylist/find?step=${KEYS[step]}`);
  await [photoStep, chooseStep, resultsStep][step](ctx, s);
}

const frame = (ctx, step, opts) => ctx.render(shell({task: "find", steps: STEPS, step, ...opts}));
const startOver = `<button class="link-btn" type="button" data-start-over>Start over</button>`;
function wireStartOver(root) {
  $("[data-start-over]", root)?.addEventListener("click", () => {
    store.clear();
    announce("Started over.");
    if (location.hash === "#/stylist/find?step=photo") location.reload(); else go("photo");
  });
}

async function photoStep(ctx, s) {
  const root = frame(ctx, 0, {
    title: "Show us the piece", back: "#/stylist",
    lede: "A photo with the piece you like. We will pick out what is in it.",
    body: `<div class="look-photo" id="find-photo"></div><p class="privacy">Your photo is only used to find pieces for you.</p>`,
    primary: `<button class="btn btn--solid" type="button" id="find-next" ${s.photo ? "" : "disabled"}>Continue</button>`,
    secondary: s.photo ? startOver : "",
  });
  if (!root) return;
  focusHeading(root);
  wireStartOver(root);
  const btn = $("#find-next", root);
  photoPicker($("#find-photo", root), {
    photo: s.photo, help: "JPEG, PNG or WebP, up to 12 MB.",
    onChange: p => { store.set(p ? {photo: p} : {}); btn.disabled = !p; },   // a new photo starts the choice afresh
  });
  btn.addEventListener("click", () => load().photo && go("frame"));
}

async function chooseStep(ctx, s) {
  const root = frame(ctx, 1, {
    title: "Which piece?", back: "#/stylist/find?step=photo",
    lede: "Tap the pieces you want to find. Or drag around one yourself.",
    body: `<div class="picker">
        <figure class="picker__photo"><div class="picker__frame" id="picker">
          <img src="${s.photo.dataUrl}" draggable="false" alt="Your photo. Tap a piece to choose it, or drag to frame one." width="${s.photo.w}" height="${s.photo.h}">
          <div id="picker-boxes"></div><div class="picker__drag" id="picker-drag" hidden></div></div></figure>
        <div class="picker__side">
          <div class="task__status" id="find-status" aria-live="polite"></div>
          <ul class="pieces" id="pieces" role="list" aria-label="Pieces in your photo"></ul>
          <p class="picker__hint" id="picker-hint" aria-live="polite"></p>
          <p class="inline-status is-error" id="find-err" role="alert"></p>
        </div>
      </div>`,
    primary: `<button class="btn btn--solid" type="button" id="find-go">Find similar</button>`,
    secondary: startOver,
  });
  if (!root) return;
  focusHeading(root);
  wireStartOver(root);
  const status = $("#find-status", root), list = $("#pieces", root), boxes = $("#picker-boxes", root), err = $("#find-err", root);
  const el = $("#picker", root), img = $("img", el), drag = $("#picker-drag", root), btn = $("#find-go", root), hint = $("#picker-hint", root);
  let pieces = s.pieces || [], chosen = new Set(s.chosen || []);
  const persist = () => save({pieces, chosen: [...chosen], results: null});

  const paint = () => {
    // Every spotted piece is a tap target on the photo, but only chosen pieces show a frame. Larger areas first,
    // so where pieces overlap the smaller one is on top and gets the tap.
    const area = b => (b[2] - b[0]) * (b[3] - b[1]);
    boxes.innerHTML = pieces.filter(p => p.box).sort((a, b) => area(b.box) - area(a.box)).map(p => {
      const [x0, y0, x1, y1] = p.box, on = chosen.has(p.id);
      return `<button class="picker__box${on ? " is-on" : ""}" type="button" data-pid="${esc(p.id)}" aria-pressed="${on}"
        aria-label="${esc(pieceLabel(p))}" style="left:${x0 * 100}%;top:${y0 * 100}%;width:${(x1 - x0) * 100}%;height:${(y1 - y0) * 100}%"></button>`;
    }).join("");
    list.innerHTML = pieces.map(p => {
      const on = chosen.has(p.id), cats = {...CATEGORIES, ...(CATEGORIES[p.category] ? {} : {[p.category]: p.category})};
      return `<li class="piece${on ? " is-on" : ""}" data-pid="${esc(p.id)}">
        <button class="piece__pick" type="button" aria-pressed="${on}">
          <span class="piece__name">${esc(pieceLabel(p))}</span><span class="piece__tick" aria-hidden="true"></span></button>
        ${on ? `<label class="piece__kind"><span class="sr-only">What is the ${esc(pieceLabel(p))}?</span>
          <select data-kind>${Object.entries(cats).map(([k, v]) => `<option value="${esc(k)}" ${k === p.category ? "selected" : ""}>${esc(v)}</option>`).join("")}</select></label>` : ""}
        ${p.added ? `<button class="piece__remove" type="button" data-remove aria-label="Remove ${esc(pieceLabel(p))}">×</button>` : ""}
      </li>`;
    }).join("");
    btn.disabled = !chosen.size;
  };
  const toggle = (id, from = null) => {
    const p = pieces.find(x => x.id === id);
    if (!p) return;
    if (chosen.has(id)) chosen.delete(id); else chosen.add(id);
    err.textContent = "";
    persist(); paint();
    if (from) $(`[data-pid="${CSS.escape(id)}"]${from === "list" ? " .piece__pick" : ""}`, from === "list" ? list : boxes)?.focus();
    hint.textContent = chosen.has(id) && !p.box ? `Drag around the ${pieceLabel(p).toLowerCase()} on the photo to frame it.` : "";
    announce(`${pieceLabel(p)} ${chosen.has(id) ? "chosen" : "removed from your choice"}.`);
  };

  // Spot the pieces once per photo; if spotting is unavailable, the shopper frames pieces by hand.
  if (!s.spotted) {
    btn.disabled = true;
    const stop = stagedStatus(status, ["Looking at your photo…", "Picking out each piece…"], 3500);
    try {
      const fd = new FormData();
      fd.append("photo", toBlob(s.photo.dataUrl), "piece.jpg");
      const r = await api("/api/stylist/detect", {method: "POST", body: fd});
      if (!ctx.current()) return;
      pieces = r.garments.map(g => ({id: g.id, category: g.category, colour: g.colour || "", description: g.description || "", box: g.box}));
      pieces.forEach(p => { p.home = p.box; });   // where it was spotted: frames can be moved, matching uses this
      chosen = new Set();                         // nothing framed until the shopper picks
      status.innerHTML = pieces.length ? "" : `<p class="meta">We couldn't pick out the pieces. Drag around the one you want.</p>`;
    } catch (e) {
      if (!ctx.current()) return;
      pieces = []; chosen = new Set();
      status.innerHTML = `<p class="meta">Drag around the piece you want on the photo.</p>`;
    } finally { stop(); }
    save({spotted: true, pieces, chosen: [...chosen], results: null});
  } else if (!pieces.length) {
    status.innerHTML = `<p class="meta">Drag around the piece you want on the photo.</p>`;
  }
  paint();

  let dragged = false;   // a drag (draw or move) is not also a tap
  boxes.addEventListener("click", e => {
    const b = e.target.closest("[data-pid]");
    if (b && !dragged) toggle(b.dataset.pid, "photo");
  });
  list.addEventListener("click", e => {
    const row = e.target.closest("[data-pid]");
    if (!row) return;
    if (e.target.closest("[data-remove]")) {
      pieces = pieces.filter(p => p.id !== row.dataset.pid); chosen.delete(row.dataset.pid);
      persist(); paint(); announce("Piece removed.");
      return;
    }
    if (e.target.closest(".piece__pick")) toggle(row.dataset.pid, "list");
  });
  list.addEventListener("change", e => {
    if (!e.target.matches("[data-kind]")) return;
    const p = pieces.find(x => x.id === e.target.closest("[data-pid]").dataset.pid);
    p.category = e.target.value; p.description = ""; p.edited = true;   // the old description named the old piece
    persist(); paint();
    $(`[data-pid="${CSS.escape(p.id)}"] [data-kind]`, list)?.focus();
  });

  // The spotted pieces where they were found (frames may have been moved since): what a framed area is matched to.
  const spotted = except => pieces.filter(p => !p.added && p.home && p.id !== except).map(p => ({...p, box: p.home}));
  // A frame drawn or moved onto a spotted piece chooses that piece (it is never listed twice); elsewhere it is the
  // shopper's own piece, named after the spotted piece it covers or their last choice.
  function place(boxIn, moving = null) {
    const last = pieces.find(p => chosen.has(p.id) && p !== moving);
    const guess = categoryFor(boxIn, spotted(moving?.id), last?.category || "top");
    const match = guess.match && pieces.find(p => p.id === guess.match.id);
    if (match) {
      if (moving && moving !== match) {
        if (moving.added) pieces = pieces.filter(p => p !== moving); else { chosen.delete(moving.id); moving.box = moving.home; }
      }
      match.box = boxIn; chosen.add(match.id);
      return match;
    }
    if (moving) { moving.box = boxIn; return moving; }
    // A piece that was spotted without a box (it happens with small accessories) and that the shopper ticked:
    // the frame they draw is that piece, not a new one.
    const unframed = [...chosen].reverse().map(id => pieces.find(p => p.id === id)).find(p => p && !p.box && !p.added);
    if (unframed) { unframed.box = boxIn; unframed.home = boxIn; return unframed; }
    const p = {id: `f${Date.now().toString(36)}`, category: guess.category, colour: "", description: "", box: boxIn, added: true};
    pieces.push(p); chosen.add(p.id);
    return p;
  }

  // Drag on the photo to frame a piece; drag a frame to move it. A tap (no drag) chooses or un-chooses a piece.
  const at = e => { const r = img.getBoundingClientRect();
    return [Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1), Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1)]; };
  const show = (node, b) => Object.assign(node.style, {left: b[0] * 100 + "%", top: b[1] * 100 + "%", width: (b[2] - b[0]) * 100 + "%", height: (b[3] - b[1]) * 100 + "%"});
  let start = null, box = null, moving = null, moveNode = null;
  el.addEventListener("pointerdown", e => {
    if (e.button > 0) return;
    start = at(e); box = null; dragged = false;
    const hit = e.target.closest(".picker__box.is-on");
    moving = hit ? pieces.find(p => p.id === hit.dataset.pid) : null;
    moveNode = hit;
  });
  el.addEventListener("pointermove", e => {
    if (!start) return;
    const p = at(e);
    if (!box && Math.hypot(p[0] - start[0], p[1] - start[1]) < 0.02) return;   // still a tap
    if (!box) el.setPointerCapture(e.pointerId);
    if (moving) {
      const [x0, y0, x1, y1] = moving.box, w = x1 - x0, h = y1 - y0;
      const nx = Math.min(Math.max(x0 + p[0] - start[0], 0), 1 - w), ny = Math.min(Math.max(y0 + p[1] - start[1], 0), 1 - h);
      box = [nx, ny, nx + w, ny + h];
      show(moveNode, box);
    } else {
      box = [Math.min(start[0], p[0]), Math.min(start[1], p[1]), Math.max(start[0], p[0]), Math.max(start[1], p[1])];
      drag.hidden = false;
      show(drag, box);
    }
  });
  const end = () => {
    if (!start) return;
    start = null; drag.hidden = true;
    if (!box || box[2] - box[0] < 0.04 || box[3] - box[1] < 0.04) { moving = null; return; }   // a tap, not a drag
    dragged = true; setTimeout(() => { dragged = false; });
    const p = place(box, moving);
    moving = null;
    hint.textContent = "";
    persist(); paint();
    announce(`Framed: ${pieceLabel(p)}. Change what it is if that is not right.`);
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", () => { start = null; moving = null; drag.hidden = true; paint(); });

  btn.addEventListener("click", async () => {
    const picked = pieces.filter(p => chosen.has(p.id));
    if (!picked.length) { err.textContent = "Tap a piece first, or drag around one."; return; }
    btn.disabled = true;
    const stop = stagedStatus(status, ["Searching our collection…", "Ranking the closest matches…"], 3000);
    try {
      const results = [];
      for (const p of picked) {   // one at a time: gentle on the (CPU-only) hosted server
        const fd = new FormData();
        fd.append("photo", toBlob(s.photo.dataUrl), "piece.jpg");
        fd.append("category", p.category);
        fd.append("box", p.box ? p.box.join(",") : "");
        const r = await api("/api/visual-search", {method: "POST", body: fd});
        if (!ctx.current()) return;
        results.push({id: p.id, label: pieceLabel(p), matches: r.matches});
      }
      if (!ctx.current()) return;
      save({pieces, chosen: [...chosen], results});
      go("results");
    } catch (e) {
      status.innerHTML = `<div class="state" role="alert"><p class="state__title">We couldn't search with that photo just now</p>
        <p class="meta">${e instanceof ApiError && e.status === 400 ? esc(e.message) : "Please try again."}</p>
        <button class="btn btn--quiet" type="button" id="find-retry">Try again</button></div>`;
      $("#find-retry", status).addEventListener("click", () => btn.click());
      btn.disabled = false;
    } finally { stop(); }
  });
}

async function resultsStep(ctx, s) {
  // One family per card across the whole page, then the four closest for each chosen piece.
  const groups = await grouped(s.results.map(r => r.matches.map(m => ({...m, reasons: undefined}))));
  if (!ctx.current()) return;
  const sections = s.results.map((r, i) => ({...r, items: groups[i].slice(0, PER_PIECE)})).filter(r => r.items.length);
  const first = sections[0]?.items[0];
  const root = frame(ctx, 2, {
    title: sections.length ? "Pieces that look similar" : "No close matches",
    back: "#/stylist/find?step=frame",
    lede: sections.length ? "" : "Try choosing the piece again, or change what it is.",
    body: sections.map(r => `<section class="find-group" aria-label="Like your ${esc(r.label)}">
        <h2 class="find-group__title">Like your ${esc(r.label.toLowerCase())}</h2>
        ${cardGrid(r.items, "visual_search", {cls: "grid--4"})}</section>`).join(""),
    primary: first ? `<a class="btn btn--solid" href="#/stylist/ask?anchor=${first.article_id}">Ask the stylist what goes with it</a>`
                   : `<a class="btn btn--solid" href="#/stylist/find?step=frame">Choose again</a>`,
    secondary: `${first ? `<a class="link-btn" href="#/stylist/find?step=frame">Choose another piece</a>` : ""}${startOver}`,
  });
  if (!root) return;
  focusHeading(root);
  wireStartOver(root);
  sections.forEach(r => impressions(r.items, "visual_search"));
  announce(sections.length ? "Similar pieces found." : "No similar pieces found.");
}
