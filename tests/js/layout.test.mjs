// The editorial grid presets (shop.js editPlan): every item count fills whole 12-column rows with no holes and
// nothing dropped, using only the allowed image ratios. Run by tests/test_ui_catalog_js.py.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const src = readFileSync(new URL("../../src/ensemble/api/static/js/shop.js", import.meta.url), "utf8");
const block = src.match(/const COMPACT[\s\S]*?\nfunction editGrid/)[0].replace(/\nfunction editGrid$/, "").replace("export function", "function");
const editPlan = new Function(`${block}; return editPlan;`)();

let failed = 0;
for (let n = 1; n <= 40; n++) {
  try {
    const plan = editPlan(n);
    assert.equal(plan.length, n, `n=${n}: every item is placed`);
    let col = 0;
    for (const t of plan) {
      assert.ok(["34", "11", "wide"].includes(t.ratio), `n=${n}: ratio ${t.ratio}`);
      col += t.span;
      assert.ok(col <= 12, `n=${n}: a row overflows`);
      if (col === 12) col = 0;
    }
    assert.ok(col === 0 || n === 1, `n=${n}: the last row leaves a hole`);
  } catch (e) { failed++; console.log(`not ok - ${e.message}`); }
}
console.log(`${40 - failed}/40 layouts ok`);
process.exit(failed ? 1 : 0);
