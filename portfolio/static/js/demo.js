// Public demo data adapter (docs/CLAUDECODE_PUBLIC_DEMO_FULL_UI_PLAN.md, D-046).
//
// GitHub Pages serves static files only: no FastAPI, no SQLite, no model inference, no LLM. This
// module answers the exact same request paths the frontend already uses from frozen responses that
// were captured from the local application and checked into the published build. It is inert unless
// the page declares demo mode, so the FastAPI behaviour of the real app is unchanged.
//
// Demo mode is declared by the build, not guessed from the hostname: the generated index.html
// carries <meta name="ensemble-demo" content="<fixture dir>/">. The real index.html has no such tag.

const META = typeof document !== "undefined" ? document.querySelector('meta[name="ensemble-demo"]') : null;

// Directory the page is served from, so every fetch works under a GitHub project subpath
// (/<repo>/) exactly as it does at the server root (/).
export const ROOT = typeof location !== "undefined" ? location.pathname.replace(/[^/]*$/, "") : "/";

// null in the real application; the fixture directory (e.g. "demo/") in the published build.
export const DEMO = META ? (META.getAttribute("content") || "demo/") : null;

export class DemoError extends Error {
  constructor(status, message, code = null) { super(message); this.status = status; this.code = code; }
}

const cache = new Map();
async function load(name) {
  if (!cache.has(name)) {
    const p = fetch(`${ROOT}${DEMO}${name}`).then(r => {
      if (!r.ok) throw new DemoError(r.status === 404 ? 404 : 503, `fixture ${name} is not published`, "no_fixture");
      return r.json();
    });
    p.catch(() => cache.delete(name));
    cache.set(name, p);
  }
  return cache.get(name);
}

const split = path => {
  const [p, query = ""] = String(path).split("?");
  return {parts: p.split("/").filter(Boolean), params: new URLSearchParams(query)};
};

// ---- Complete the Look -------------------------------------------------------------------------------
// Captured per (anchor, customer) with the configured k. A smaller k truncates each module the way the
// live endpoint would; an unpublished anchor is a 404, which the product page already falls back from.
let ctlIndex = null;
async function ctlFile(anchor, customer) {
  // Only ask for a capture the build published: not every anchor has complementary items, and only a
  // profile's own picks were captured personalized. Anything else would be a 404 in the network log.
  ctlIndex = ctlIndex || new Set(await load("ctl/index.json"));
  if (customer != null && ctlIndex.has(`${anchor}-${customer}`)) return `ctl/${anchor}-${customer}.json`;
  if (ctlIndex.has(String(anchor))) return `ctl/${anchor}.json`;   // as served to a guest
  return null;
}

async function completeTheLook(params) {
  const anchor = params.get("anchor"), customer = params.get("customer");
  const k = Math.max(1, Math.min(24, +(params.get("k") || 8)));
  const file = await ctlFile(anchor, customer);
  if (!file) throw new DemoError(404, "no complementary items are published for this piece", "no_fixture");
  const d = await load(file);
  const slots = params.get("slots");
  const want = slots ? new Set(slots.split(",").map(s => s.trim())) : null;
  return {...d, request_id: "stored", modules: d.modules
    .filter(m => !want || want.has(m.slot))
    .map(m => ({...m, items: m.items.slice(0, k)}))};
}

// ---- product families --------------------------------------------------------------------------------
async function families(params) {
  const want = (params.get("articles") || "").split(",").map(s => s.trim()).filter(Boolean).map(Number);
  const all = await load("families.json");
  const articles = {}, out = {};
  for (const id of want) {
    const code = all.articles[id];
    if (code == null) continue;
    articles[id] = code;
    if (all.families[code]) out[code] = all.families[code];
  }
  return {articles, families: out};
}

// ---- Style Assistant ---------------------------------------------------------------------------------
const norm = s => String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

export async function assistantExamples() {
  return (await load("assistant.json")).examples;
}

// The stored turn for `text`, or null. Nothing is generated: an unmatched question is reported to the
// shopper as "not one of the saved questions" instead of being answered.
export async function assistantMatch(text) {
  const t = norm(text);
  if (!t) return null;
  const examples = await assistantExamples();
  return examples.find(e => norm(e.text) === t)
      || examples.find(e => norm(e.text).includes(t) && t.length >= 12)
      || null;
}

export const storedPhoto = () => load("photo.json");

// ---- the router --------------------------------------------------------------------------------------
// Every path the frontend requests. Anything not listed is an explicit 503 with the demo's own code,
// so a surface that needs a backend says so instead of hanging or showing an empty grid.
async function route(path, opts) {
  const {parts, params} = split(path);
  const method = (opts?.method || "GET").toUpperCase();

  if (path === "/readyz") return load("serving.json").then(d => d.readyz);
  if (parts[0] !== "api") throw new DemoError(404, "not published", "no_fixture");
  const [, a, b, c] = parts;

  if (a === "customers") return load("customers.json");
  if (a === "home") return load(`home/${+b}.json`);
  if (a === "product") return load(`product/${+b}.json`);
  if (a === "catalog" && b === "families") return families(params);
  if (a === "metrics") return load("metrics.json");
  if (a === "explain") return load(`explain/${+b}.json`).then(d => {
    const r = d[String(+c)];
    if (!r) throw new DemoError(404, "not in this customer's recommendations", "no_fixture");
    return r;
  });
  if (a === "events") return {ok: true, stored: false};   // documented no-op: the demo records nothing
  if (a === "v2" && b === "complete-the-look") return completeTheLook(params);
  if (a === "v2" && b === "meta") return load("serving.json").then(d => d.meta);
  if (a === "v2" && b === "metrics") return load("serving.json").then(d => d.metrics);

  if (a === "assistant" && b === "session") return {session_id: "stored", backend: "stored"};
  if (a === "assistant" && c === "message") {
    // stylist.js resolves the stored turn before calling this; an unmatched question never gets here.
    const text = opts?.body instanceof FormData ? opts.body.get("text") : "";
    const e = await assistantMatch(text);
    if (!e) throw new DemoError(409, "not a saved question", "no_fixture");
    return {answer: e.answer, trace: e.trace, latency_s: e.latency_s, cost_usd: e.cost_usd,
            hallucinated: e.hallucinated, cards: e.cards, example_id: e.id};
  }
  if (a === "visual-search" || a === "snap") throw new DemoError(503, "photo upload needs the local app", "no_backend");
  if (a === "label") throw new DemoError(503, "labelling needs the local app", "no_backend");
  throw new DemoError(404, "not published", "no_fixture");
}

// Mirrors the real api(): resolves to the parsed body, or throws with a status and a code.
export async function request(path, opts) {
  if (!DEMO) throw new DemoError(500, "demo mode is off");
  return route(path, opts);
}

// Raw-response shape for the two callers that read the status themselves (/readyz).
export async function requestRaw(path) {
  try {
    return {status: 200, ok: true, body: await route(path)};
  } catch (e) {
    return {status: e.status || 503, ok: false, body: null};
  }
}
