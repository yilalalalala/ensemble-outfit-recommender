// Unit tests for the pure shop logic (src/ensemble/api/static/js/catalog.js). Run by tests/test_ui_catalog_js.py.
import assert from "node:assert/strict";
import {Cart, colourName, createPageGroups, money, swatchStyle, variantLabels} from "../../src/ensemble/api/static/js/catalog.js";

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

// family map for the fixtures: 100/101 are colourways of family A; 200 has the same *name* but family B.
const FAM = {100: "A", 101: "A", 102: "A", 200: "B", 300: "C"};
const familyOf = id => FAM[id] ?? null;
const item = (id, name = "Skye trousers") => ({article_id: id, prod_name: name});

test("exact duplicate article ids are removed page-wide", () => {
  const page = createPageGroups(familyOf);
  assert.deepEqual(page.take([item(100), item(300)]).map(i => i.article_id), [100, 300]);
  assert.deepEqual(page.take([item(100), item(300)]).map(i => i.article_id), []);
});

test("colourways of one verified family collapse into the first card", () => {
  const page = createPageGroups(familyOf);
  assert.deepEqual(page.take([item(101), item(100), item(102)]).map(i => i.article_id), [101]);
});

test("unrelated products that share a name are not grouped", () => {
  const page = createPageGroups(familyOf);
  assert.deepEqual(page.take([item(100, "Skye"), item(200, "Skye")]).map(i => i.article_id), [100, 200]);
});

test("unknown families fall back to the exact article id only", () => {
  const page = createPageGroups(familyOf);
  assert.deepEqual(page.take([item(900), item(901), item(900)]).map(i => i.article_id), [900, 901]);
});

test("first occurrence keeps its position; a later module that becomes empty yields nothing", () => {
  const page = createPageGroups(familyOf);
  const first = page.take([item(300), item(100)]);
  const later = page.take([item(101), item(300)]);
  assert.deepEqual(first.map(i => i.article_id), [300, 100]);
  assert.deepEqual(later, []);
  assert.equal(page.size(), 2);
});

test("claim reserves a family (product page owns its own colourways)", () => {
  const page = createPageGroups(familyOf);
  assert.equal(page.claim(100), true);
  assert.deepEqual(page.take([item(102), item(200)]).map(i => i.article_id), [200]);
});

test("families marked Not for me never render on later pages", () => {
  const page = createPageGroups(familyOf, new Set(["f:A"]));
  assert.deepEqual(page.take([item(101), item(200), item(100)]).map(i => i.article_id), [200]);
});

test("duplicate colour groups inside one family get distinct labels", () => {
  const labels = variantLabels([{colour_group_name: "Black"}, {colour_group_name: "Beige"}, {colour_group_name: "Black"}, {colour_group_name: "Other Pink"}]);
  assert.deepEqual(labels, ["Black, option 1", "Beige", "Black, option 2", "Pink (other shade)"]);
});

test("swatch mapping is explicit with labelled neutral fallbacks", () => {
  assert.deepEqual(swatchStyle("Black"), {fill: "#111111", pattern: null});
  assert.equal(swatchStyle("Unknown").pattern, "neutral");
  assert.equal(swatchStyle("Other").pattern, "neutral");
  assert.equal(swatchStyle("Transparent").pattern, "clear");
  assert.equal(colourName("Unknown"), "Assorted");
});

test("USD formatting", () => {
  assert.equal(money(34.99), "$34.99");
  assert.equal(money(119), "$119.00");
  assert.equal(money(null), "");
});

const store = () => { const m = new Map(); return {getItem: k => m.get(k) ?? null, setItem: (k, v) => m.set(k, v), m}; };
const line = (id, price = 10, colour = "Black") => ({article_id: id, family: FAM[id], name: "Skye", colour, image: `/images/${id}.jpg`, price});

test("adding the same article twice increments one line", () => {
  const c = new Cart(store(), "k");
  c.add(line(100)); c.add(line(100));
  assert.equal(c.lines.length, 1);
  assert.equal(c.count, 2);
});

test("selecting another colourway adds a separate line for that article", () => {
  const c = new Cart(store(), "k");
  c.add(line(100, 34.99, "Beige")); c.add(line(101, 34.99, "Black"));
  assert.deepEqual(c.lines.map(l => [l.article_id, l.colour]), [[100, "Beige"], [101, "Black"]]);
  assert.equal(c.subtotal, 69.98);
});

test("quantity, remove and subtotal", () => {
  const c = new Cart(store(), "k");
  c.add(line(100, 19.99)); c.add(line(300, 5.5));
  c.setQty(100, 3);
  assert.equal(c.subtotal, 65.47);
  c.setQty(100, 0);
  assert.deepEqual(c.lines.map(l => l.article_id), [300]);
  c.remove(300);
  assert.equal(c.count, 0);
  assert.equal(c.subtotal, 0);
});

test("cart persists and reloads from storage", () => {
  const s = store();
  new Cart(s, "k").add(line(100));
  const again = new Cart(s, "k");
  assert.equal(again.count, 1);
  assert.equal(new Cart(s, "other").count, 0);
});

test("storage failures are reported, not thrown", () => {
  const broken = {getItem: () => "{not json", setItem: () => { throw new Error("quota"); }};
  const c = new Cart(broken, "k");
  assert.equal(c.error, "load");
  assert.equal(c.add(line(100)), false);
  assert.equal(c.error, "save");
  assert.equal(c.count, 1);
});

let failed = 0;
for (const [name, fn] of tests) {
  try { fn(); console.log(`ok - ${name}`); } catch (e) { failed++; console.log(`not ok - ${name}\n  ${e.message}`); }
}
console.log(`${tests.length - failed}/${tests.length} passed`);
process.exit(failed ? 1 : 0);
