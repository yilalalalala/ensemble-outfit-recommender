// Style Assistant (M7b) with the two photo services (M7a visual search, "snap your outfit").
// Products shown here come only from tool / endpoint responses; the assistant's text never adds products.
// (That grounding is documented for DS readers; shoppers see consumer copy only.)
import {$, ApiError, DEMO, announce, api, cardGrid, esc, impressions, loadFamilies, newPageGroups, state} from "./core.js";
import {assistantExamples, assistantMatch, storedPhoto} from "./demo.js";

const CATEGORIES = ["top", "outerwear", "knitwear", "bottom", "dress", "jumpsuit", "shoes", "bag", "jewellery", "hat", "scarf", "belt", "sunglasses"];
const ACCEPT = "image/jpeg,image/png,image/webp";
const SUGGESTIONS = ["What shoes go with wide-leg trousers?", "Find me a warm knit for autumn", "What should I wear to a summer wedding?"];
// Published build: the chips are the saved questions, so every chip has a real saved answer behind it.
let suggestions = SUGGESTIONS, saved = [];
const savedLabel = "Saved answers from the local application. Pick a question below, or type one of them.";

// One conversation per profile while the page is open; a profile change starts a new one.
const sessions = new Map();   // customer -> {sid, backend, log: html, sentAnchors: Set}
let activeChat = null;        // the conversation on screen; the "Ask about this" hand-off talks to it
const askStylist = text => { if (activeChat?.current()) activeChat.send(text); };

const PRIVACY = "Your photo is only used to find pieces for you.";

// One result group (an answer, a search, an outfit analysis): load colourways and prices, then group
// products by family so the group never repeats a product.
async function grouped(lists) {
  await loadFamilies(lists.flat().map(it => it.article_id));
  const page = newPageGroups();
  return lists.map(l => page.take(l));
}

export async function view(ctx) {
  ctx.setTitle("Style Assistant");
  const customer = state.customer;
  const anchor = ctx.route.params.get("anchor");
  if (DEMO && !saved.length) {
    saved = await assistantExamples().catch(() => []);
    const picked = saved.filter(e => e.suggested).map(e => e.text);
    if (picked.length) suggestions = picked;
  }
  if (!ctx.current()) return;
  const root = ctx.render(`
    <div class="stylist-head">
      <div><p class="eyebrow">Style Assistant</p><h1 class="title" tabindex="-1">Ask the stylist.</h1></div>
      <p class="lede">Tell us what you’re looking for, or bring a photo. We’ll help you find pieces that <em>feel right</em>.</p>
    </div>
    <div class="stylist">
      <section class="panel" aria-labelledby="chat-h">
        <div class="panel__head"><h2 class="eyebrow" id="chat-h">Conversation</h2></div>
        <div class="chatlog" id="chatlog" role="log" aria-live="polite" aria-relevant="additions" tabindex="0" aria-label="Conversation"></div>
        <div class="attachment" id="chat-attach" hidden></div>
        <form class="composer" id="composer" novalidate>
          <button class="icon-btn" type="button" id="attach-btn" aria-describedby="attach-help">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8h3l2-2h6l2 2h3v11H4z"/><circle cx="12" cy="13" r="3.2"/></svg>
            <span class="sr-only">Attach a photo</span></button>
          <span class="sr-only" id="attach-help">Adds an outfit photo to your next message (JPEG, PNG or WebP).</span>
          <input type="file" id="chat-file" accept="${ACCEPT}" hidden>
          <label class="sr-only" for="chat-in">Message the stylist</label>
          <textarea id="chat-in" rows="1" placeholder="e.g. What goes with a beige trench?" aria-describedby="chat-err"></textarea>
          <button class="btn btn--solid" type="submit" id="chat-send">Send</button>
        </form>
        <p class="inline-status is-error" id="chat-err" role="alert"></p>
      </section>

      <section class="panel" aria-labelledby="photo-h">
        <div class="panel__head"><h2 class="eyebrow" id="photo-h">Bring a photo</h2></div>
        <div id="photo-slot"></div>
        <div class="services">
          <div class="service" aria-labelledby="svc1-h">
            <h3 id="svc1-h">Find visually similar products</h3>
            <p class="meta">Finds pieces in our collection that look like one item in your photo. Choose what it is; optionally drag on the photo to frame it, otherwise the whole photo is used. It may take a few seconds.</p>
            <div class="service__row">
              <label><span class="field-label">The piece is a</span>
                <select class="input" id="vs-cat">${CATEGORIES.map(c => `<option value="${c}">${c}</option>`).join("")}</select></label>
              <button class="btn btn--solid" type="button" id="vs-go" disabled>Find similar</button>
            </div>
          </div>
          <div class="service" aria-labelledby="svc2-h">
            <h3 id="svc2-h">Complete my outfit</h3>
            <p class="meta">We'll recognise each piece you're wearing, find it in our collection and suggest what would complete the outfit. It may take about a minute.</p>
            <div class="service__row"><button class="btn" type="button" id="snap-go" disabled>Analyse my outfit</button></div>
          </div>
          <p class="privacy" id="privacy">${PRIVACY}</p>
        </div>
        <div class="results" id="photo-results" aria-live="polite"></div>
      </section>
    </div>`);
  if (!root) return;
  if (DEMO) $("#attach-btn", root).hidden = true;   // uploads need the local application
  photoServices(root);

  // Conversation: reuse this profile's session if one exists.
  let s = sessions.get(customer);
  const log = $("#chatlog");
  if (s) {
    log.innerHTML = s.log;
    log.scrollTop = log.scrollHeight;
  } else {
    log.innerHTML = emptyChat();
    try {
      s = await newSession(customer);
    } catch (err) {
      if (!ctx.current()) return;
      log.innerHTML = `<div class="state" role="alert"><p class="state__title">The stylist isn't available right now</p>
        <p class="meta">${esc(err.message)}</p><button class="btn btn--quiet" type="button" data-retry>Try again</button></div>`;
      $("#composer").querySelectorAll("button, textarea").forEach(el => { el.disabled = true; });
      return;
    }
    if (!ctx.current()) return;
  }
  const chat = makeChat(ctx, customer);
  if (anchor && !s.sentAnchors.has(anchor)) { s.sentAnchors.add(anchor); chat.send(`What goes with article ${anchor}?`); }
}

async function newSession(customer) {
  const r = await api("/api/assistant/session", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({customer_idx: customer})});
  const s = {sid: r.session_id, backend: r.backend, log: "", sentAnchors: new Set()};
  state.backend = r.backend;
  sessions.set(customer, s);
  return s;
}

const chips = list => `<div class="suggestions">${list.map(t =>
  `<button class="suggestion" type="button" data-suggest="${esc(t)}">${esc(t)}</button>`).join("")}</div>`;

function emptyChat() {
  return `<div class="msg msg--bot" data-empty><p class="msg__who">Stylist</p>
    <p class="msg__text">I can find pieces, style something you own, or complete an outfit from a photo. What are you dressing for?</p>
    ${DEMO ? `<p class="saved-note">${esc(savedLabel)}</p>` : ""}
    ${chips(suggestions)}</div>`;
}

// Published build: a question with no saved answer is said to have none. Nothing is generated here.
function noSavedAnswer(all) {
  return `<div class="msg msg--bot" data-note><p class="msg__who">Stylist</p>
    <p class="msg__text">That one isn't among the saved questions. Here is what has been saved:</p>
    ${chips(all)}</div>`;
}

function renderAnswer(r, cards) {
  // Product references in the text ([[article_id]]) are shown as cards below the answer, so the markers are dropped.
  const text = esc(r.answer).replace(/\s*\[\[(\d+)\]\]/g, "");
  // The model writes light markdown; after escaping, only **bold** is turned into markup.
  const html = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  const next = DEMO ? saved.filter(e => e.followup_of && e.followup_of === r.example_id).map(e => e.text) : [];
  return `<div class="msg msg--bot"><p class="msg__who">Stylist</p><p class="msg__text">${html}</p>
    ${cards.length ? cardGrid(cards, "assistant", {cls: "grid--compact"}) : ""}
    ${next.length ? chips(next) : ""}</div>`;
}

function makeChat(ctx, customer) {
  const log = $("#chatlog"), form = $("#composer"), input = $("#chat-in"), sendBtn = $("#chat-send"), err = $("#chat-err");
  const attachBox = $("#chat-attach"), file = $("#chat-file");
  let photo = null, busy = false;
  const save = () => { const s = sessions.get(customer); if (s) s.log = log.innerHTML; };

  function setPhoto(f) {
    photo = f;
    if (!f) { attachBox.hidden = true; attachBox.innerHTML = ""; return; }
    const url = URL.createObjectURL(f);
    attachBox.hidden = false;
    attachBox.innerHTML = `<img src="${url}" alt="Attached photo preview"><span>${esc(f.name)} will be sent with your next message</span>
      <button class="link-btn" type="button" id="attach-remove">Remove</button>`;
    $("#attach-remove").onclick = () => { setPhoto(null); $("#attach-btn").focus(); };
  }

  async function send(text) {
    text = text.trim();
    if (busy) return;
    if (!text) { err.textContent = photo ? "Add a short message to go with the photo." : "Type a message first."; input.focus(); return; }
    err.textContent = "";
    busy = true;
    sendBtn.disabled = true;
    log.querySelector("[data-empty]")?.remove();
    const sent = photo;
    log.insertAdjacentHTML("beforeend", `<div class="msg msg--user"><p class="msg__who">You</p><p class="msg__text">${esc(text)}${sent ? " · photo attached" : ""}</p></div>
      <div class="msg msg--bot" id="pending"><p class="msg__who">Stylist</p><p class="thinking" role="status"><i></i><i></i><i></i> Finding pieces for you…</p></div>`);
    log.scrollTop = log.scrollHeight;
    setPhoto(null);
    const fd = new FormData();
    fd.append("text", text);
    if (sent) fd.append("photo", sent);
    if (DEMO && !(await assistantMatch(text))) {
      $("#pending")?.remove();
      log.insertAdjacentHTML("beforeend", noSavedAnswer(saved.map(e => e.text)));
      busy = false; sendBtn.disabled = false; log.scrollTop = log.scrollHeight; save();
      return;
    }
    try {
      let s = sessions.get(customer);
      let r;
      try {
        r = await api(`/api/assistant/${s.sid}/message`, {method: "POST", body: fd});
      } catch (e) {
        if (!(e instanceof ApiError && e.status === 404)) throw e;
        // The server forgot the session (e.g. it restarted): start a new one and retry once.
        s = await newSession(customer);
        r = await api(`/api/assistant/${s.sid}/message`, {method: "POST", body: fd});
        log.insertAdjacentHTML("beforeend", `<p class="meta">A new conversation was started because the previous one had expired.</p>`);
      }
      const [cards] = await grouped([r.cards]);
      $("#pending")?.remove();
      log.insertAdjacentHTML("beforeend", renderAnswer(r, cards));
      impressions(cards, "assistant");
      announce("The stylist replied.");
    } catch (e) {
      $("#pending")?.remove();
      log.insertAdjacentHTML("beforeend", `<div class="msg msg--bot" role="alert"><p class="msg__who">Stylist</p>
        <p class="msg__text">Sorry, I couldn't answer that just now.</p>
        <p><button class="link-btn" type="button" data-resend="${esc(text)}">Try again</button></p></div>`);
      if (sent) setPhoto(sent);
    } finally {
      busy = false;
      sendBtn.disabled = false;
      log.scrollTop = log.scrollHeight;
      save();
    }
  }

  form.addEventListener("submit", e => { e.preventDefault(); const v = input.value; input.value = ""; autosize(); send(v); });
  input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  const autosize = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 128) + "px"; };
  input.addEventListener("input", autosize);
  $("#attach-btn").addEventListener("click", () => file.click());
  file.addEventListener("change", () => { if (file.files[0]) setPhoto(file.files[0]); file.value = ""; input.focus(); });
  log.addEventListener("click", e => {
    const sug = e.target.closest("[data-suggest]");
    if (sug) { send(sug.dataset.suggest); return; }
    const re = e.target.closest("[data-resend]");
    if (re) { re.closest(".msg").remove(); send(re.dataset.resend); }
  });
  const chat = {send, current: ctx.current};
  activeChat = chat;
  return chat;
}

// ---- photo services -----------------------------------------------------------------------------------
function photoServices(root) {
  const slot = $("#photo-slot", root), out = $("#photo-results", root);
  const vsGo = $("#vs-go", root), snapGo = $("#snap-go", root);
  let file = null, box = null, url = null, stored = null;

  // Published build: one saved, openly licensed photo stands in for an upload, and both services
  // return exactly what the local application returned for it. Uploading needs the local application.
  async function savedPhoto() {
    stored = stored || await storedPhoto();
    slot.innerHTML = `<div class="preview">
      <div class="preview__frame" id="frame"><img src="${esc(stored.photo.src)}" alt="${esc(stored.photo.alt)}">
        <div class="preview__box" id="box" style="left:${stored.photo.box[0] * 100}%;top:${stored.photo.box[1] * 100}%;width:${(stored.photo.box[2] - stored.photo.box[0]) * 100}%;height:${(stored.photo.box[3] - stored.photo.box[1]) * 100}%"></div></div>
      <div class="preview__bar"><span class="meta" id="box-state">Framed area will be searched</span></div>
      <p class="saved-note">Saved example photo. Choosing your own photo needs the application running locally.</p>
      <p class="caption">${esc(stored.photo.credit)}</p></div>`;
    $("#vs-cat", root).value = stored.visual_search_category;
    vsGo.disabled = snapGo.disabled = false;
  }

  function empty() {
    slot.innerHTML = `<label class="dropzone" id="dropzone">
      <input type="file" id="photo-in" accept="${ACCEPT}" aria-describedby="photo-help">
      <span class="section-title" style="font-size:1.25rem">Choose a photo</span>
      <span class="meta" id="photo-help">JPEG, PNG or WebP. A clear, well-lit outfit photo works best.</span>
      <span class="btn btn--quiet" aria-hidden="true">Browse files</span></label>`;
    const input = $("#photo-in", slot), zone = $("#dropzone", slot);
    input.addEventListener("change", () => input.files[0] && load(input.files[0]));
    zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("is-over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("is-over"));
    zone.addEventListener("drop", e => { e.preventDefault(); zone.classList.remove("is-over"); const f = e.dataTransfer.files[0]; if (f) load(f); });
  }

  function load(f) {
    if (!ACCEPT.split(",").includes(f.type)) {
      empty();
      slot.insertAdjacentHTML("beforeend", `<p class="inline-status is-error" role="alert">That file is ${esc(f.type || "an unknown type")}. Please choose a JPEG, PNG or WebP image.</p>`);
      return;
    }
    if (url) URL.revokeObjectURL(url);
    file = f; box = null; url = URL.createObjectURL(f);
    slot.innerHTML = `<div class="preview">
      <div class="preview__frame" id="frame"><img src="${url}" alt="Your photo: drag to frame one piece"><div class="preview__box" id="box" hidden></div></div>
      <div class="preview__bar">
        <span class="meta" id="box-state">Whole photo will be searched</span>
        <button class="link-btn" type="button" id="box-clear" hidden>Clear frame</button>
        <label class="link-btn" style="cursor:pointer">Replace photo<input type="file" accept="${ACCEPT}" class="sr-only" id="photo-replace"></label>
        <button class="link-btn" type="button" id="photo-remove">Remove photo</button>
      </div></div>`;
    $("#photo-replace", slot).addEventListener("change", e => e.target.files[0] && load(e.target.files[0]));
    $("#photo-remove", slot).addEventListener("click", () => { if (url) URL.revokeObjectURL(url); file = null; url = null; box = null; empty(); sync(); $("#photo-in", slot).focus(); announce("Photo removed."); });
    $("#box-clear", slot).addEventListener("click", () => { box = null; paintBox(); });
    framing();
    sync();
    announce("Photo ready. Choose a service below.");
  }

  function paintBox() {
    const el = $("#box", slot);
    if (!el) return;
    el.hidden = !box;
    $("#box-clear", slot).hidden = !box;
    $("#box-state", slot).textContent = box ? "Framed area will be searched" : "Whole photo will be searched";
    if (box) Object.assign(el.style, {left: box[0] * 100 + "%", top: box[1] * 100 + "%", width: (box[2] - box[0]) * 100 + "%", height: (box[3] - box[1]) * 100 + "%"});
  }

  function framing() {
    const frame = $("#frame", slot), img = $("img", frame);
    let start = null;
    const at = e => { const r = img.getBoundingClientRect(); return [Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1), Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1)]; };
    frame.addEventListener("pointerdown", e => { start = at(e); frame.setPointerCapture(e.pointerId); e.preventDefault(); });
    frame.addEventListener("pointermove", e => {
      if (!start) return;
      const p = at(e);
      box = [Math.min(start[0], p[0]), Math.min(start[1], p[1]), Math.max(start[0], p[0]), Math.max(start[1], p[1])];
      paintBox();
    });
    const end = () => {
      if (!start) return;
      start = null;
      if (box && (box[2] - box[0] < 0.03 || box[3] - box[1] < 0.03)) box = null;   // a click, not a drag
      paintBox();
    };
    frame.addEventListener("pointerup", end);
    frame.addEventListener("pointercancel", end);
  }

  const sync = () => { vsGo.disabled = snapGo.disabled = DEMO ? false : !file; };

  function busy(on, label) {
    vsGo.disabled = snapGo.disabled = on || (DEMO ? false : !file);
    out.setAttribute("aria-busy", String(on));
    if (on) out.innerHTML = `<p class="thinking" role="status"><i></i><i></i><i></i> <span id="busy-label">${esc(label)}</span></p>`;
  }

  vsGo.addEventListener("click", async () => {
    if (!file && !DEMO) return;
    busy(true, "Searching our collection…");
    const fd = new FormData();
    fd.append("photo", file); fd.append("category", $("#vs-cat", root).value); fd.append("box", box ? box.join(",") : "");
    try {
      const r = DEMO ? (await storedPhoto()).visual_search : await api("/api/visual-search", {method: "POST", body: fd});
      const [matches] = await grouped([r.matches]);
      out.innerHTML = matches.length
        ? `<h3>Pieces that look similar</h3>${cardGrid(matches, "visual_search", {cls: "grid--compact"})}
           <p><button class="btn btn--quiet" type="button" id="ask-this">Ask the stylist what goes with it</button></p>`
        : `<div class="state"><p class="state__title">No close matches</p><p class="meta">Try framing a single piece, or choose a different category.</p></div>`;
      impressions(matches, "visual_search");
      if (matches.length) $("#ask-this", out).addEventListener("click", () => {
        askStylist(`What goes with the ${matches[0].prod_name} (article ${matches[0].article_id})?`);
        $("#chat-in").focus();
      });
      announce(matches.length ? "Similar pieces found." : "No similar pieces found.");
    } catch (e) {
      out.innerHTML = `<div class="state" role="alert"><p class="state__title">We couldn't search with that photo just now</p><p class="meta">Please try again.</p>
        <button class="btn btn--quiet" type="button" id="vs-retry">Try again</button></div>`;
      $("#vs-retry", out).addEventListener("click", () => vsGo.click());
    } finally { busy(false); out.scrollIntoView({block: "nearest"}); }
  });

  snapGo.addEventListener("click", async () => {
    if (!file && !DEMO) return;
    busy(true, "Looking at your outfit… This may take about a minute.");
    const t0 = Date.now();
    const timer = setInterval(() => { const l = $("#busy-label", out); if (l) l.textContent = `Looking at your outfit… ${Math.round((Date.now() - t0) / 1000)} s`; }, 1000);
    const fd = new FormData();
    fd.append("photo", file);
    try {
      const r = DEMO ? (await storedPhoto()).snap : await api("/api/snap", {method: "POST", body: fd});
      const garments = r.garments.filter(g => g.matches.length), fills = r.complete_the_look.filter(m => m.items.length);
      const groups = await grouped([...garments.map(g => g.matches.slice(0, 5)), ...fills.map(m => m.items)]);
      const gm = groups.slice(0, garments.length), fm = groups.slice(garments.length);
      const detected = r.garments.map(g => `${g.colour} ${g.category}`.trim());
      out.innerHTML = `${detected.length ? `<p class="notice"><strong>We spotted:</strong> ${esc(detected.join(", "))}</p>`
          : `<p class="meta">We couldn't make out the pieces in this photo. A full-length, well-lit photo works best.</p>`}
        ${garments.map((g, i) => gm[i].length ? `<div><h3>Like your ${esc(g.description)}</h3>${cardGrid(gm[i], "snap", {cls: "grid--compact"})}</div>` : "").join("")}
        ${fills.map((m, i) => fm[i].length ? `<div><h3>To complete the outfit: ${esc(m.title)}</h3>${cardGrid(fm[i], "complete_the_look", {cls: "grid--compact", anchor: r.anchor})}</div>` : "").join("")}`;
      gm.forEach(l => impressions(l, "snap"));
      fm.forEach(l => impressions(l, "complete_the_look", r.anchor));
      announce("Your outfit ideas are ready.");
    } catch (e) {
      out.innerHTML = `<div class="state" role="alert"><p class="state__title">We couldn't look at that photo just now</p><p class="meta">Please try again.</p>
        <button class="btn btn--quiet" type="button" id="snap-retry">Try again</button></div>`;
      $("#snap-retry", out).addEventListener("click", () => snapGo.click());
    } finally { clearInterval(timer); busy(false); out.scrollIntoView({block: "nearest"}); }
  });

  if (DEMO) savedPhoto(); else empty();
  sync();
}
