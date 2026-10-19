// AI Stylist: the hub, the shared task shell, and "Ask the stylist" (docs/AI_STYLIST_PRODUCT_FLOW.md).
// Products shown here come only from tool / endpoint responses; the assistant's text never adds products.
import {$, $$, ApiError, announce, api, cardGrid, esc, impressions, loadFamilies, newPageGroups, state, variantsFor} from "./core.js";
import {downscale, stage, toBlob, validate} from "./photo.js";

export const TASKS = [
  {id: "ask", href: "#/stylist/ask", title: "Ask the stylist",
   detail: "Advice for an occasion, a piece you own, or a style you want to try."},
  {id: "find", href: "#/stylist/find", title: "Find this piece",
   detail: "Tap the piece you like and we will find the four closest in our collection."},
  {id: "look", href: "#/stylist/look", title: "Complete my look",
   detail: "Check what we spotted, choose a direction, and get a short, considered edit."},
];

// One result group: load colourways and prices, then group products by family (no repeats within a group).
export async function grouped(lists) {
  await loadFamilies(lists.flat().map(it => it.article_id));
  const page = newPageGroups();
  return lists.map(l => page.take(l));
}

// ---- shared task shell -------------------------------------------------------------------------------------
// Back · progress · Change task, a short explanation, the step body, and exactly one dominant action.
export function shell({task, title, lede, steps = null, step = 0, back = null, body = "", primary = "", secondary = ""}) {
  const t = TASKS.find(x => x.id === task);
  const progress = steps ? `<ol class="steps" aria-label="Progress">${steps.map((s, i) =>
    `<li class="steps__step${i < step ? " is-done" : ""}" ${i === step ? `aria-current="step"` : ""}>
       <span class="steps__n" aria-hidden="true">${i + 1}</span><span class="steps__label">${esc(s)}</span>
       ${i < step ? `<span class="sr-only">, completed</span>` : ""}</li>`).join("")}</ol>
     <p class="steps__compact" aria-hidden="true">Step ${step + 1} of ${steps.length} · ${esc(steps[step])}</p>` : "";
  return `<div class="task" data-task="${task}">
    <nav class="task__bar" aria-label="AI Stylist">
      <a class="link-btn task__back" href="${back || "#/stylist"}">← Back</a>
      <span class="task__crumb"><a href="#/stylist">AI Stylist</a><span aria-hidden="true"> / </span><span>${esc(t.title)}</span></span>
      <a class="link-btn task__change" href="#/stylist">Change task</a>
    </nav>
    ${progress}
    <header class="task__head">
      <h1 class="task__title" tabindex="-1">${esc(title)}</h1>
      ${lede ? `<p class="task__lede">${lede}</p>` : ""}
    </header>
    <div class="task__body">${body}</div>
    ${primary || secondary ? `<div class="task__actions">${secondary ? `<div class="task__secondary">${secondary}</div>` : ""}${primary}</div>` : ""}
  </div>`;
}

// Move focus to the step heading after a step/route change (the router focuses <main> on navigation).
export function focusHeading(root) {
  requestAnimationFrame(() => $(".task__title", root)?.focus({preventScroll: false}));
}

// Staged loading copy: the label advances while a long request runs; returns a stop function.
export function stagedStatus(el, labels, every = 4500) {
  let i = 0;
  el.innerHTML = `<p class="thinking" role="status"><i></i><i></i><i></i> <span>${esc(labels[0])}</span></p>`;
  const span = $("span", el);
  const timer = setInterval(() => { if (i < labels.length - 1) span.textContent = labels[++i]; }, every);
  return () => clearInterval(timer);
}

// ---- hub ----------------------------------------------------------------------------------------------------
export async function hub(ctx) {
  ctx.setTitle("AI Stylist");
  const look = stage("ensemble.look.v1").get();
  const root = ctx.render(`<div class="hub">
    <header class="hub__head">
      <h1 class="display hub__title" tabindex="-1">What can we help with?</h1>
    </header>
    <ul class="hub__tasks" role="list">${TASKS.map((t, i) => `<li>
      <a class="task-card" href="${t.href}">
        <span class="task-card__n" aria-hidden="true">${String(i + 1).padStart(2, "0")}</span>
        <span class="task-card__title">${esc(t.title)}</span>
        <span class="task-card__detail">${esc(t.detail)}</span>
        <span class="task-card__go" aria-hidden="true">Start →</span>
      </a></li>`).join("")}</ul>
    ${look?.result ? `<p class="hub__resume"><a class="link-btn" href="#/stylist/look?step=edit">Return to your last edit</a></p>` : ""}
  </div>`);
  if (root) requestAnimationFrame(() => $(".hub__title", root)?.focus());
}

// ---- Ask the stylist -----------------------------------------------------------------------------------------
const SUGGESTIONS = ["What shoes go with wide-leg trousers?", "Find me a warm knit for autumn", "What should I wear to a summer wedding?"];
// One conversation per profile while the page is open; a profile change starts a new one.
const sessions = new Map();   // customer -> {sid, log, history, context, contextSent, sentAnchors}

async function createSession(customer, prev = null) {
  const r = await api("/api/assistant/session", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({customer_idx: customer})});
  const s = {sid: r.session_id, log: prev?.log || "", history: prev?.history || [], context: prev?.context || null,
             contextSent: false, sentAnchors: prev?.sentAnchors || new Set()};
  return s;
}

// The look being refined (from Complete my look), as text the stylist can ground on.
function lookContext() {
  const look = stage("ensemble.look.v1").get();
  if (!look?.result) return null;
  const worn = look.garments.map(g => `${g.description?.toLowerCase().startsWith((g.colour || "").toLowerCase()) ? "" : g.colour + " "}${g.description || g.category} (${g.category})`.trim()).join("; ");
  const goal = look.goal === "occasion" ? `dress it for ${look.occasion}` : look.goal === "similar" ? "find similar pieces" : "complete the look";
  const picks = look.result.picks.map(p => `${p.card.prod_name} [[${p.card.article_id}]]`).join(", ");
  return {look, text: `I'm refining an outfit from my photo. I'm wearing: ${worn}. My goal: ${goal}. ` +
    `You suggested: ${picks}. Use these article ids with your tools if helpful.`};
}

function renderAnswer(r, cards) {
  const text = esc(r.answer).replace(/\s*\[\[(\d+)\]\]/g, "");
  const html = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  return `<div class="msg msg--bot"><p class="msg__who">Stylist</p><p class="msg__text">${html}</p>
    ${cards.length ? cardGrid(cards, "assistant", {cls: "grid--compact"}) : ""}</div>`;
}

const chips = list => `<div class="suggestions">${list.map(t =>
  `<button class="suggestion" type="button" data-suggest="${esc(t)}">${esc(t)}</button>`).join("")}</div>`;

// The opening of a new conversation: the stylist greets the shopper in a speech bubble and the starter questions
// are reply bubbles, each with a small round photo of a real catalogue piece.
const SUGGESTION_THUMBS = [922037001, 917290001, 898287001];   // wide-leg trousers, a knit, a summer dress
const greeting = () => `<div class="greet" data-empty>
    <div class="greet__row"><span class="avatar" aria-hidden="true">e</span>
      <div class="bubble"><p class="bubble__title">Hi! What are you dressing for today?</p>
        <p class="bubble__text">Tell me in your own words, or pick one to start. I’ll find pieces that <em>feel right</em>.</p></div></div>
    <div class="replies" role="group" aria-label="Start with a question">${SUGGESTIONS.map((t, i) =>
      `<button class="reply" type="button" data-suggest="${esc(t)}"><img class="reply__img" data-thumb="${SUGGESTION_THUMBS[i]}" alt="" width="40" height="40">${esc(t)}</button>`).join("")}</div>
  </div>`;

// Fill the reply bubbles' photos from the catalogue (the image address differs locally and when hosted).
async function fillThumbs(log) {
  const imgs = $$("img[data-thumb]:not([src])", log);
  if (!imgs.length) return;
  try {
    await loadFamilies(imgs.map(i => +i.dataset.thumb));
    imgs.forEach(img => {
      const id = +img.dataset.thumb, v = variantsFor({article_id: id}).list.find(x => x.article_id === id);
      if (v?.image) img.src = v.image;
    });
  } catch { /* the bubbles read fine without their photos */ }
}

export async function ask(ctx) {
  ctx.setTitle("Ask the stylist");
  const customer = state.customer;
  const anchor = ctx.route.params.get("anchor");
  const refine = ctx.route.params.get("context") === "look" ? lookContext() : null;
  const contextCard = refine ? `<aside class="context-card" aria-label="What the stylist knows">
      <img src="${refine.look.photo.dataUrl}" alt="Your outfit photo" width="${refine.look.photo.w}" height="${refine.look.photo.h}">
      <div><p class="eyebrow">Refining your edit</p>
        <p class="context-card__text">${esc(refine.look.garments.map(g => `${g.description?.toLowerCase().startsWith((g.colour || "").toLowerCase()) ? "" : g.colour + " "}${g.description || g.category}`.trim()).join(" · "))}</p>
        <p class="meta">${refine.look.result.picks.length} pieces in your edit · <a href="#/stylist/look?step=edit">Back to your edit</a></p></div>
    </aside>` : "";
  const root = ctx.render(shell({
    task: "ask", title: refine ? "Refine your edit" : "Ask the stylist", back: refine ? "#/stylist/look?step=edit" : "#/stylist",
    lede: refine ? "Tell the stylist what to change: more formal, a different colour, a lower price…" : "",
    body: `${contextCard}<span class="orb orb--plum" aria-hidden="true"></span><span class="orb orb--sage" aria-hidden="true"></span>
      <section class="chat" aria-label="Conversation">
        <div class="chatlog" id="chatlog" role="log" aria-live="polite" aria-relevant="additions" tabindex="0" aria-label="Conversation"></div>
        <div class="attachment" id="chat-attach" hidden></div>
        <form class="composer" id="composer" novalidate>
          <button class="icon-btn" type="button" id="attach-btn" aria-describedby="attach-help">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8h3l2-2h6l2 2h3v11H4z"/><circle cx="12" cy="13" r="3.2"/></svg>
            <span class="sr-only">Attach a photo</span></button>
          <span class="sr-only" id="attach-help">Adds a photo to your next message (JPEG, PNG or WebP).</span>
          <input type="file" id="chat-file" accept="image/jpeg,image/png,image/webp" hidden>
          <label class="sr-only" for="chat-in">Message the stylist</label>
          <textarea id="chat-in" rows="1" placeholder="e.g. What goes with a beige trench?" aria-describedby="chat-err"></textarea>
          <button class="btn btn--solid composer__send" type="submit" id="chat-send">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5.5 11.5 12 5l6.5 6.5"/></svg><span class="sr-only">Send</span></button>
        </form>
        <p class="inline-status is-error" id="chat-err" role="alert"></p>
      </section>`,
  }));
  if (!root) return;
  focusHeading(root);
  const log = $("#chatlog", root);
  const key = refine ? `${customer}:look` : customer;   // a refinement is its own conversation
  let s = sessions.get(key);
  if (s?.log) {   // a conversation already under way: restore it
    log.innerHTML = s.log;
    log.scrollTop = log.scrollHeight;
  } else {        // nothing sent yet (first visit, or back again before asking): the opening greeting
    log.innerHTML = refine
      ? `<div class="msg msg--bot" data-empty><p class="msg__who">Stylist</p><p class="msg__text">I have your outfit and your edit. What would you like to change?</p>
         ${chips(["Make it more formal", "Show me other colours", "Something less expensive"])}</div>`
      : greeting();
    if (!s) try {
      s = await createSession(customer);
      sessions.set(key, s);
      s.context = refine?.text || null;
    } catch {
      if (!ctx.current()) return;
      log.innerHTML = `<div class="state" role="alert"><p class="state__title">The stylist isn't available right now</p>
        <p class="meta">Please try again in a moment.</p><button class="btn btn--quiet" type="button" data-retry>Try again</button></div>`;
      $$("#composer button, #composer textarea", root).forEach(el => { el.disabled = true; });
      return;
    }
    if (!ctx.current()) return;
  }
  fillThumbs(log);
  const chat = conversation(root, customer, key);
  if (anchor && !s.sentAnchors.has(anchor)) { s.sentAnchors.add(anchor); chat.send(`What goes with article ${anchor}?`); }
}

function conversation(root, customer, key) {
  const log = $("#chatlog", root), form = $("#composer", root), input = $("#chat-in", root);
  const sendBtn = $("#chat-send", root), err = $("#chat-err", root), attachBox = $("#chat-attach", root), file = $("#chat-file", root);
  let photo = null, busy = false;
  const save = () => { const s = sessions.get(key); if (s) s.log = log.innerHTML; };

  function setPhoto(p) {
    photo = p;
    if (!p) { attachBox.hidden = true; attachBox.innerHTML = ""; return; }
    attachBox.hidden = false;
    attachBox.innerHTML = `<img src="${p.dataUrl}" alt="Attached photo preview"><span>${esc(p.name)} will be sent with your next message</span>
      <button class="link-btn" type="button" id="attach-remove">Remove</button>`;
    $("#attach-remove", root).onclick = () => { setPhoto(null); $("#attach-btn", root).focus(); };
  }

  // The text actually sent: replays the context (the look being refined, or the last turns after a lost
  // session) once per server session, then the shopper's message.
  function outgoing(s, text) {
    const parts = [];
    if (!s.contextSent) {
      if (s.context) parts.push(s.context);
      if (s.replay) parts.push(`Earlier in our chat: ${s.replay}`);
    }
    parts.push(text);
    return parts.join("\n\n");
  }

  async function post(s, text, sent) {
    const fd = new FormData();
    fd.append("text", outgoing(s, text));
    if (sent) fd.append("photo", toBlob(sent.dataUrl), sent.name || "photo.jpg");
    const r = await api(`/api/assistant/${s.sid}/message`, {method: "POST", body: fd});
    s.contextSent = true;
    s.replay = null;
    return r;
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
      <div class="msg msg--bot" id="pending"><p class="msg__who">Stylist</p><p class="thinking" role="status"><i></i><i></i><i></i> <span>Thinking about your request…</span></p></div>`);
    const slow = setTimeout(() => { const l = $("#pending span", log); if (l) l.textContent = "Finding pieces for you…"; }, 3500);
    log.scrollTop = log.scrollHeight;
    setPhoto(null);
    let s = sessions.get(key);
    try {
      let r;
      try {
        r = await post(s, text, sent);
      } catch (e) {
        if (!(e instanceof ApiError && e.status === 404)) throw e;
        // The server forgot the session (e.g. it restarted): start a new one, replay what it needs, resend.
        const replay = s.history.slice(-4).map(h => `${h.role === "user" ? "I asked" : "you answered"}: ${h.text}`).join(" | ");
        s = await createSession(customer, s);
        sessions.set(key, s);
        s.replay = replay || null;
        r = await post(s, text, sent);
        log.insertAdjacentHTML("beforeend", `<p class="meta chat__note">We reconnected you to the stylist and kept your conversation.</p>`);
      }
      // Keep the product ids ([[id]]) in the replayable history so a recovered session can still refer to them.
      s.history.push({role: "user", text}, {role: "assistant", text: String(r.answer).slice(0, 500)});
      const [cards] = await grouped([r.cards]);
      $("#pending", log)?.remove();
      log.insertAdjacentHTML("beforeend", renderAnswer(r, cards));
      impressions(cards, "assistant");
      announce("The stylist replied.");
    } catch (e) {
      $("#pending", log)?.remove();
      const resting = e instanceof ApiError && e.code === "ai_budget_reached";
      log.insertAdjacentHTML("beforeend", `<div class="msg msg--bot" role="alert"><p class="msg__who">Stylist</p>
        <p class="msg__text">${resting ? "The stylist is resting for now. Please try again later." : "Sorry, I couldn't answer that just now."}</p>
        ${resting ? "" : `<p><button class="link-btn" type="button" data-resend="${esc(text)}">Try again</button></p>`}</div>`);
      if (sent) setPhoto(sent);
    } finally {
      clearTimeout(slow);
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
  $("#attach-btn", root).addEventListener("click", () => file.click());
  file.addEventListener("change", async () => {
    const f = file.files[0];
    file.value = "";
    if (!f) return;
    const bad = validate(f);
    if (bad) { err.textContent = bad; return; }
    setPhoto({...(await downscale(f)), name: f.name});
    input.focus();
  });
  log.addEventListener("click", e => {
    const sug = e.target.closest("[data-suggest]");
    if (sug) { send(sug.dataset.suggest); return; }
    const re = e.target.closest("[data-resend]");
    if (re) { re.closest(".msg").remove(); send(re.dataset.resend); }
  });
  return {send};
}

