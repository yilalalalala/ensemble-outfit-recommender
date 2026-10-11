// Find this piece: 1 Photo → 2 Frame → 3 Results, with an optional hand-off to the stylist
// (docs/AI_STYLIST_PRODUCT_FLOW.md §4–5). Results come from the existing visual search endpoint.
import {$, ApiError, announce, api, cardGrid, esc, impressions} from "./core.js";
import {photoPicker, stage, toBlob} from "./photo.js";
import {focusHeading, grouped, shell, stagedStatus} from "./stylist.js";

const STEPS = ["Photo", "Frame", "Results"];
const KEYS = ["photo", "frame", "results"];
const store = stage("ensemble.find.v1");
const CATEGORIES = {top: "Top", outerwear: "Outerwear", knitwear: "Knitwear", bottom: "Bottoms", dress: "Dress",
  jumpsuit: "Jumpsuit", shoes: "Shoes", bag: "Bag", jewellery: "Jewellery", hat: "Hat", scarf: "Scarf", belt: "Belt",
  sunglasses: "Sunglasses"};

const load = () => store.get() || {};
const save = patch => { const s = {...load(), ...patch}; store.set(s); return s; };
const go = step => { location.hash = `#/stylist/find?step=${step}`; };
const reachable = s => (s.results ? 2 : s.photo ? 1 : 0);

export async function view(ctx) {
  ctx.setTitle("Find this piece");
  const s = load();
  const want = Math.max(0, KEYS.indexOf(ctx.route.params.get("step") || "photo"));
  const step = Math.min(want, reachable(s));
  if (step !== want) history.replaceState(null, "", `#/stylist/find?step=${KEYS[step]}`);
  await [photoStep, frameStep, resultsStep][step](ctx, s);
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
    lede: "A photo with the piece you like. Next, you can frame just that piece.",
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
    onChange: p => { store.set(p ? {photo: p} : {}); btn.disabled = !p; },
  });
  btn.addEventListener("click", () => load().photo && go("frame"));
}

async function frameStep(ctx, s) {
  const root = frame(ctx, 1, {
    title: "Frame the piece", back: "#/stylist/find?step=photo",
    lede: "Drag a box around the piece you want, or search the whole photo. Then tell us what it is.",
    body: `<div class="framer">
        <figure class="framer__photo"><div class="framer__frame" id="framer">
          <img src="${s.photo.dataUrl}" alt="Your photo: drag to frame one piece" width="${s.photo.w}" height="${s.photo.h}">
          <div class="framer__box" id="framer-box" hidden></div></div></figure>
        <div class="framer__controls">
          <fieldset class="framer__mode"><legend class="field-label">What should we search?</legend>
            <label class="radio"><input type="radio" name="area" value="whole" ${s.box ? "" : "checked"}> The whole photo</label>
            <label class="radio"><input type="radio" name="area" value="box" ${s.box ? "checked" : ""}> Just the framed area</label>
          </fieldset>
          <p class="meta" id="framer-hint">${s.box ? "Framed area selected. Drag again to change it." : "Drag on the photo to frame a piece."}</p>
          <label><span class="field-label">The piece is</span>
            <select class="input" id="find-cat">${Object.entries(CATEGORIES).map(([k, v]) =>
              `<option value="${k}" ${k === (s.category || "top") ? "selected" : ""}>${v}</option>`).join("")}</select></label>
          <div class="task__status" id="find-status" aria-live="polite"></div>
        </div>
      </div>`,
    primary: `<button class="btn btn--solid" type="button" id="find-go">Find similar</button>`,
    secondary: startOver,
  });
  if (!root) return;
  focusHeading(root);
  wireStartOver(root);
  let box = s.box || null;
  const el = $("#framer", root), img = $("img", el), boxEl = $("#framer-box", root), hint = $("#framer-hint", root);
  const radios = [...root.querySelectorAll("input[name=area]")];
  const paint = () => {
    boxEl.hidden = !box;
    if (box) Object.assign(boxEl.style, {left: box[0] * 100 + "%", top: box[1] * 100 + "%",
      width: (box[2] - box[0]) * 100 + "%", height: (box[3] - box[1]) * 100 + "%"});
    radios.find(r => r.value === (box ? "box" : "whole")).checked = true;
    radios.find(r => r.value === "box").disabled = !box;
    hint.textContent = box ? "Framed area selected. Drag again to change it." : "Drag on the photo to frame a piece.";
    save({box});
  };
  const at = e => { const r = img.getBoundingClientRect();
    return [Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1), Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1)]; };
  let start = null;
  el.addEventListener("pointerdown", e => { start = at(e); el.setPointerCapture(e.pointerId); e.preventDefault(); });
  el.addEventListener("pointermove", e => {
    if (!start) return;
    const p = at(e);
    box = [Math.min(start[0], p[0]), Math.min(start[1], p[1]), Math.max(start[0], p[0]), Math.max(start[1], p[1])];
    boxEl.hidden = false;
    Object.assign(boxEl.style, {left: box[0] * 100 + "%", top: box[1] * 100 + "%", width: (box[2] - box[0]) * 100 + "%", height: (box[3] - box[1]) * 100 + "%"});
  });
  const end = () => {
    if (!start) return;
    start = null;
    if (box && (box[2] - box[0] < 0.03 || box[3] - box[1] < 0.03)) box = null;   // a click, not a drag
    paint();
    announce(box ? "Area framed." : "Whole photo selected.");
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  radios.forEach(r => r.addEventListener("change", () => { if (r.value === "whole" && r.checked) { box = null; paint(); } }));
  paint();
  $("#find-cat", root).addEventListener("change", e => save({category: e.target.value}));

  const btn = $("#find-go", root), status = $("#find-status", root);
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    const category = $("#find-cat", root).value;
    const stop = stagedStatus(status, ["Reading the piece…", "Searching our collection…", "Ranking the closest matches…"], 3000);
    try {
      const fd = new FormData();
      fd.append("photo", toBlob(s.photo.dataUrl), "piece.jpg");
      fd.append("category", category);
      fd.append("box", box ? box.join(",") : "");
      const r = await api("/api/visual-search", {method: "POST", body: fd});
      if (!ctx.current()) return;
      save({category, box, results: r.matches});
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
  const [matches] = await grouped([s.results.map(m => ({...m, reasons: undefined}))]);
  if (!ctx.current()) return;
  const what = (CATEGORIES[s.category] || "piece").toLowerCase();
  const first = matches[0];
  const root = frame(ctx, 2, {
    title: matches.length ? "Pieces that look similar" : "No close matches",
    back: "#/stylist/find?step=frame",
    lede: matches.length ? `The closest ${esc(what)} pieces in our collection, most similar first.`
                         : "Try framing just the piece, or choose a different type.",
    body: matches.length ? cardGrid(matches, "visual_search", {cls: "grid--4"}) : "",
    primary: first ? `<a class="btn btn--solid" href="#/stylist/ask?anchor=${first.article_id}">Ask the stylist what goes with it</a>`
                   : `<a class="btn btn--solid" href="#/stylist/find?step=frame">Change the frame</a>`,
    secondary: `${first ? `<a class="link-btn" href="#/stylist/find?step=frame">Search another piece</a>` : ""}${startOver}`,
  });
  if (!root) return;
  focusHeading(root);
  wireStartOver(root);
  impressions(matches, "visual_search");
  announce(matches.length ? "Similar pieces found." : "No similar pieces found.");
}
