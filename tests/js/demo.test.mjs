// Unit tests for the public demo data adapter (src/ensemble/api/static/js/demo.js).
// Run by tests/test_ui_demo_js.py. The module is written for a browser, so the globals it reads
// (document, location, fetch) are stubbed here and the real fixtures are read from disk.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const BUILD = path.join(ROOT, "portfolio");
const FIXTURES = path.join(BUILD, "demo");
const SUBPATH = "/ensemble-outfit-recommender/";

// The published page: demo mode declared by the build, served from a project subpath.
const fetched = [];
globalThis.document = {querySelector: s => s === 'meta[name="ensemble-demo"]' ? {getAttribute: () => "demo/"} : null};
globalThis.location = {pathname: SUBPATH};
globalThis.fetch = async url => {
  fetched.push(url);
  assert.ok(url.startsWith(SUBPATH), `fixture request must stay under the project subpath: ${url}`);
  const file = path.join(FIXTURES, url.slice((SUBPATH + "demo/").length));
  if (!fs.existsSync(file)) return {ok: false, status: 404, json: async () => null};
  return {ok: true, status: 200, json: async () => JSON.parse(fs.readFileSync(file, "utf8"))};
};

const demo = await import("../../src/ensemble/api/static/js/demo.js");

const tests = [];
const test = (name, fn) => tests.push([name, fn]);
const read = n => JSON.parse(fs.readFileSync(path.join(FIXTURES, n), "utf8"));

test("demo mode and the document root are taken from the page, not the hostname", () => {
  assert.equal(demo.DEMO, "demo/");
  assert.equal(demo.ROOT, SUBPATH);
});

test("every fixture request is relative to the project subpath", async () => {
  fetched.length = 0;
  await demo.request("/api/customers");
  assert.deepEqual(fetched, [SUBPATH + "demo/customers.json"]);
});

test("the profile list, home modules and product pages are served from fixtures", async () => {
  const customers = await demo.request("/api/customers");
  assert.ok(customers.length >= 2);
  const c = customers[0].customer_idx;
  const home = await demo.request(`/api/home/${c}`);
  assert.equal(home.customer.customer_idx, c);
  assert.deepEqual(home.modules.map(m => m.id), ["for_you", "buy_again", "trending"]);
  const id = home.modules[0].items[0].article_id;
  const p = await demo.request(`/api/product/${id}`);
  assert.equal(p.article.article_id, id);
  for (const key of ["complete_the_look", "other_colours", "similar"]) assert.ok(Array.isArray(p[key]), key);
});

test("product photography is referenced by a relative path, never an absolute /images URL", async () => {
  const home = await demo.request(`/api/home/${(await demo.request("/api/customers"))[0].customer_idx}`);
  const images = home.modules.flatMap(m => m.items.map(i => i.image));
  assert.ok(images.length > 0);
  for (const src of images) assert.match(src, /^images\/\d+\.jpg$/, src);
});

test("families are sliced to the requested articles only", async () => {
  const all = read("families.json");
  const ids = Object.keys(all.articles).slice(0, 3).map(Number);
  const d = await demo.request(`/api/catalog/families?articles=${ids.join(",")}`);
  assert.deepEqual(Object.keys(d.articles).map(Number).sort((a, b) => a - b), [...ids].sort((a, b) => a - b));
  for (const code of Object.values(d.articles)) assert.ok(d.families[code], code);
  assert.ok(Object.keys(d.families).length <= ids.length);
});

// The build's own index is the list of captures, so a stray file in the directory (an editor backup,
// a file-sync agent's copy) is never mistaken for one.
const ctlKeys = () => read("ctl/index.json");

test("Complete the Look truncates to k and filters to the requested slots", async () => {
  const anchor = ctlKeys().find(n => !n.includes("-"));
  const full = await demo.request(`/api/v2/complete-the-look?anchor=${anchor}&k=8`);
  assert.ok(full.modules.length > 0);
  const two = await demo.request(`/api/v2/complete-the-look?anchor=${anchor}&k=2`);
  assert.ok(two.modules.every(m => m.items.length <= 2));
  const slot = full.modules[0].slot;
  const one = await demo.request(`/api/v2/complete-the-look?anchor=${anchor}&k=8&slots=${slot}`);
  assert.deepEqual(one.modules.map(m => m.slot), [slot]);
});

test("an unpublished customer's Complete the Look falls back to the guest capture", async () => {
  const anchor = ctlKeys().find(n => !n.includes("-"));
  const r = await demo.request(`/api/v2/complete-the-look?anchor=${anchor}&k=8&customer=999999999`);
  assert.ok(r.modules.length > 0);
});

test("an unpublished anchor is a 404, which the product page already falls back from", async () => {
  await assert.rejects(() => demo.request("/api/v2/complete-the-look?anchor=1&k=8"), e => e.status === 404);
});

test("event logging and the backends that need a server report themselves, never a fake result", async () => {
  assert.deepEqual(await demo.request("/api/events", {method: "POST"}), {ok: true, stored: false});
  for (const p of ["/api/visual-search", "/api/snap", "/api/label", "/api/label/tasks?round=2"]) {
    await assert.rejects(() => demo.request(p, {method: "POST"}), e => e.status === 503 && e.code === "no_backend", p);
  }
});

test("the saved assistant turns are matched exactly, and an unsaved question has no answer", async () => {
  const examples = await demo.assistantExamples();
  assert.ok(examples.length >= 4);
  for (const e of examples) {
    assert.ok(e.answer && e.answer.length > 10, e.id);
    assert.deepEqual(e.hallucinated, [], `${e.id} must cite only products a tool returned`);
    assert.ok(Array.isArray(e.cards));
  }
  const first = examples[0];
  assert.equal((await demo.assistantMatch(first.text)).id, first.id);
  assert.equal((await demo.assistantMatch("  " + first.text.toUpperCase() + " ")).id, first.id);
  assert.equal(await demo.assistantMatch("tell me a joke about kubernetes"), null);
  const session = await demo.request("/api/assistant/session", {method: "POST"});
  assert.equal(session.backend, "stored");
  const body = new FormData();
  body.append("text", first.text);
  const turn = await demo.request(`/api/assistant/${session.session_id}/message`, {method: "POST", body});
  assert.equal(turn.answer, first.answer);
  assert.equal(turn.example_id, first.id);
  const miss = new FormData();
  miss.append("text", "tell me a joke about kubernetes");
  await assert.rejects(() => demo.request(`/api/assistant/x/message`, {method: "POST", body: miss}),
                       e => e.status === 409);
});

test("the saved photo example carries its licence credit and both saved responses", async () => {
  const p = await demo.storedPhoto();
  assert.match(p.photo.src, /^photo\.jpg$/);
  assert.equal(p.photo.box.length, 4);
  assert.ok(p.photo.credit.includes("Photo:"));
  assert.ok(p.visual_search.matches.length > 0);
  assert.ok(p.snap.garments.length > 0);
  for (const m of p.visual_search.matches) assert.match(m.image, /^images\/\d+\.jpg$/);
});

test("DS Studio reads the stored reports and the captured bundle status", async () => {
  const m = await demo.request("/api/metrics");
  assert.ok(m.m3_ranker_test && m.m1_baselines_test);
  const ready = await demo.requestRaw("/readyz");
  assert.equal(ready.status, 200);
  assert.equal(ready.body.ready, true);
  const meta = await demo.request("/api/v2/meta");
  assert.ok(meta.bundle_version);
  assert.ok((await demo.request("/api/v2/metrics")).counters);
});

test("explanations are published per profile and 404 outside that profile's picks", async () => {
  const c = (await demo.request("/api/customers"))[0].customer_idx;
  const fy = (await demo.request(`/api/home/${c}`)).modules[0].items[0];
  const e = await demo.request(`/api/explain/${c}/${fy.article_id}`);
  assert.ok(Array.isArray(e.shap) && e.shap.length > 0);
  await assert.rejects(() => demo.request(`/api/explain/${c}/1`), x => x.status === 404);
});

let failed = 0;
for (const [name, fn] of tests) {
  try { await fn(); console.log("ok   " + name); }
  catch (e) { failed++; console.error("FAIL " + name + "\n     " + (e.message || e)); }
}
console.log(`\n${tests.length - failed}/${tests.length} passed`);
process.exit(failed ? 1 : 0);
