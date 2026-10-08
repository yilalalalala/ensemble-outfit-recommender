// Style Assistant (M7b) with the two photo services (M7a visual search, "snap your outfit").
// Products shown here come only from tool / endpoint responses; the assistant's text never adds products.
import {$, ApiError, announce, api, esc, grid, impressions, plural, profileName, state, words} from "./core.js";

const CATEGORIES = ["top", "outerwear", "knitwear", "bottom", "dress", "jumpsuit", "shoes", "bag", "jewellery", "hat", "scarf", "belt", "sunglasses"];
const ACCEPT = "image/jpeg,image/png,image/webp";
const SUGGESTIONS = ["What shoes go with wide-leg trousers?", "Find me a warm knit for autumn", "What should I wear to a summer wedding?"];

// One conversation per profile while the page is open; a profile change starts a new one.
const sessions = new Map();   // customer -> {sid, backend, log: html, sentAnchors: Set}
let activeChat = null;        // the conversation on screen; the "Ask about this" hand-off talks to it
const askStylist = text => { if (activeChat?.current()) activeChat.send(text); };

const isLocal = b => !b || b === "ollama";
function privacyCopy(backend) {
  return isLocal(backend)
    ? "Photos go only to this demo's own server. Visual search and outfit analysis run on this machine; nothing is stored after the request, except a chat photo kept in memory for the conversation."
    : `Photos go to this demo's server. Outfit analysis and chat photos are sent to the configured model provider (${esc(backend)}); visual search runs locally. Nothing is stored after the request, except a chat photo kept in memory for the conversation.`;
}

export async function view(ctx) {
  ctx.setTitle("Style Assistant");
  const customer = state.customer;
  const anchor = ctx.route.params.get("anchor");
  const root = ctx.render(`
    <div class="stylist-head">
      <div><p class="eyebrow">Style Assistant</p><h1 class="title" tabindex="-1">Ask the stylist.</h1></div>
      <p class="lede">Ask in plain language, or bring a photo. Every piece shown comes from the recommender's own tools; the assistant never invents products.</p>
    </div>
    <div class="stylist">
      <section class="panel" aria-labelledby="chat-h">
        <div class="panel__head"><h2 class="eyebrow" id="chat-h">Conversation · ${esc(profileName(state.profile))}</h2>
          <span class="meta" id="chat-backend">Connecting…</span></div>
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
            <p class="meta">Matches one piece in your photo to the catalogue. Choose what it is; optionally drag on the photo to frame it, otherwise the whole photo is used. Takes a few seconds.</p>
            <div class="service__row">
              <label><span class="field-label">The piece is a</span>
                <select class="input" id="vs-cat">${CATEGORIES.map(c => `<option value="${c}">${c}</option>`).join("")}</select></label>
              <button class="btn btn--solid" type="button" id="vs-go" disabled>Find similar</button>
            </div>
          </div>
          <div class="service" aria-labelledby="svc2-h">
            <h3 id="svc2-h">Complete my outfit</h3>
            <p class="meta">A vision model identifies each garment you are wearing, matches it to the catalogue and suggests what is missing. Takes about a minute on this laptop.</p>
            <div class="service__row"><button class="btn" type="button" id="snap-go" disabled>Analyse my outfit</button></div>
          </div>
          <p class="privacy" id="privacy">${privacyCopy(state.backend)}</p>
        </div>
        <div class="results" id="photo-results" aria-live="polite"></div>
      </section>
    </div>`);
  if (!root) return;
  photoServices(root);

  // Conversation: reuse this profile's session if one exists.
  let s = sessions.get(customer);
  const log = $("#chatlog");
  if (s) {
    log.innerHTML = s.log;
    log.scrollTop = log.scrollHeight;
    paintBackend(s.backend);
  } else {
    log.innerHTML = emptyChat();
    try {
      s = await newSession(customer);
    } catch (err) {
      if (!ctx.current()) return;
      $("#chat-backend").textContent = "Unavailable";
      log.innerHTML = `<div class="state" role="alert"><p class="state__title">The stylist is not available</p>
        <p class="meta">${esc(err.message)}</p><button class="btn btn--quiet" type="button" data-retry>Try again</button></div>`;
      $("#composer").querySelectorAll("button, textarea").forEach(el => { el.disabled = true; });
      return;
    }
    if (!ctx.current()) return;
    paintBackend(s.backend);
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

function paintBackend(b) {
  $("#chat-backend").innerHTML = `<span class="status-dot is-ok" aria-hidden="true"></span>${isLocal(b) ? "Local model" : `Model: ${esc(b)}`}`;
  $("#privacy").innerHTML = privacyCopy(b);
}

function emptyChat() {
  return `<div class="msg msg--bot" data-empty><p class="msg__who">Stylist</p>
    <p class="msg__text">I can find pieces, style something you own, or complete an outfit from a photo. What are you dressing for?</p>
    <div class="suggestions">${SUGGESTIONS.map(t => `<button class="filter" type="button" data-suggest="${esc(t)}">${esc(t)}</button>`).join("")}</div></div>`;
}

function renderAnswer(r) {
  let n = 0; const order = {};
  const text = esc(r.answer).replace(/\[\[(\d+)\]\]/g, (_, id) => `(${order[id] ??= ++n})`);
  // The model writes light markdown; after escaping, only **bold** is turned into markup.
  const html = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  const tools = r.trace.map(t => words(t.tool)).join(" → ");
  return `<div class="msg msg--bot"><p class="msg__who">Stylist</p><p class="msg__text">${html}</p>
    ${r.cards.length ? grid(r.cards, "assistant", {cls: "grid--compact", each: c => ({index: order[c.article_id] ?? null})}) : ""}
    <p class="msg__trace">${tools ? `<span class="tag">Used ${esc(tools)}</span>` : `<span class="tag">No tools used</span>`}
      <span>${esc(r.latency_s)} s${r.cost_usd ? ` · $${r.cost_usd.toFixed(4)}` : ""}</span>
      ${r.hallucinated.length ? `<span class="tag tag--warn">Removed ${plural(r.hallucinated.length, "unverified product")}</span>` : ""}</p></div>`;
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
      <div class="msg msg--bot" id="pending"><p class="msg__who">Stylist</p><p class="thinking" role="status"><i></i><i></i><i></i> Thinking · may call the recommender tools</p></div>`);
    log.scrollTop = log.scrollHeight;
    setPhoto(null);
    const fd = new FormData();
    fd.append("text", text);
    if (sent) fd.append("photo", sent);
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
      $("#pending")?.remove();
      log.insertAdjacentHTML("beforeend", renderAnswer(r));
      impressions(r.cards, "assistant");
      announce("The stylist replied.");
    } catch (e) {
      $("#pending")?.remove();
      log.insertAdjacentHTML("beforeend", `<div class="msg msg--bot" role="alert"><p class="msg__who">Stylist</p>
        <p class="msg__text">I couldn't answer that just now. ${esc(e.message)}</p>
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
  let file = null, box = null, url = null;

  function empty() {
    slot.innerHTML = `<label class="dropzone" id="dropzone">
      <input type="file" id="photo-in" accept="${ACCEPT}" aria-describedby="photo-help">
      <span class="section-title" style="font-size:1.25rem">Choose a photo</span>
      <span class="meta" id="photo-help">JPEG, PNG or WebP, up to the server's upload limit. A clear, well-lit outfit photo works best.</span>
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

  const sync = () => { vsGo.disabled = !file; snapGo.disabled = !file; };

  function busy(on, label) {
    vsGo.disabled = snapGo.disabled = on || !file;
    out.setAttribute("aria-busy", String(on));
    if (on) out.innerHTML = `<p class="thinking" role="status"><i></i><i></i><i></i> <span id="busy-label">${esc(label)}</span></p>`;
  }

  vsGo.addEventListener("click", async () => {
    if (!file) return;
    busy(true, "Searching the catalogue…");
    const fd = new FormData();
    fd.append("photo", file); fd.append("category", $("#vs-cat", root).value); fd.append("box", box ? box.join(",") : "");
    try {
      const r = await api("/api/visual-search", {method: "POST", body: fd});
      out.innerHTML = r.matches.length
        ? `<h3>Closest catalogue pieces</h3>${grid(r.matches, "visual_search", {cls: "grid--compact"})}
           <p><button class="btn btn--quiet" type="button" id="ask-this">Ask the stylist about the closest match</button></p>`
        : `<div class="state"><p class="state__title">No close matches</p><p class="meta">Try framing a single piece, or choose a different category.</p></div>`;
      impressions(r.matches, "visual_search");
      if (r.matches.length) $("#ask-this", out).addEventListener("click", () => {
        askStylist(`What goes with the ${r.matches[0].prod_name} (article ${r.matches[0].article_id})?`);
        $("#chat-in").focus();
      });
      announce(`${plural(r.matches.length, "match", "matches")} found.`);
    } catch (e) {
      out.innerHTML = `<div class="state" role="alert"><p class="state__title">Visual search did not complete</p><p class="meta">${esc(e.message)}</p>
        <button class="btn btn--quiet" type="button" id="vs-retry">Try again</button></div>`;
      $("#vs-retry", out).addEventListener("click", () => vsGo.click());
    } finally { busy(false); out.scrollIntoView({block: "nearest"}); }
  });

  snapGo.addEventListener("click", async () => {
    if (!file) return;
    busy(true, "Analysing your outfit… 0 s (usually about a minute)");
    const t0 = Date.now();
    const timer = setInterval(() => { const l = $("#busy-label", out); if (l) l.textContent = `Analysing your outfit… ${Math.round((Date.now() - t0) / 1000)} s (usually about a minute)`; }, 1000);
    const fd = new FormData();
    fd.append("photo", file);
    try {
      const r = await api("/api/snap", {method: "POST", body: fd});
      const detected = r.garments.map(g => `${g.colour} ${g.category}`.trim());
      out.innerHTML = `<div class="notice"><strong>Detected:</strong> ${esc(detected.join(", ") || "no garments")} ·
          <strong>Missing:</strong> ${esc(r.missing_slots.map(words).join(", ") || "nothing")}</div>
        ${r.garments.filter(g => g.matches.length).map(g => `<div><h3>${esc(g.description)}</h3>${grid(g.matches.slice(0, 5), "snap", {cls: "grid--compact"})}</div>`).join("")}
        ${r.complete_the_look.filter(m => m.items.length).map(m => `<div><h3>Fill the gap: ${esc(m.title)}</h3>${grid(m.items, "complete_the_look", {cls: "grid--compact", anchor: r.anchor})}</div>`).join("")}
        ${!r.garments.length ? `<p class="meta">No garments were recognised. A full-length, well-lit photo works best.</p>` : ""}`;
      r.garments.forEach(g => impressions(g.matches.slice(0, 5), "snap"));
      r.complete_the_look.forEach(m => impressions(m.items, "complete_the_look", r.anchor));
      announce("Outfit analysis finished.");
    } catch (e) {
      out.innerHTML = `<div class="state" role="alert"><p class="state__title">Outfit analysis did not complete</p><p class="meta">${esc(e.message)}</p>
        <button class="btn btn--quiet" type="button" id="snap-retry">Try again</button></div>`;
      $("#snap-retry", out).addEventListener("click", () => snapGo.click());
    } finally { clearInterval(timer); busy(false); out.scrollIntoView({block: "nearest"}); }
  });

  empty();
  sync();
}
