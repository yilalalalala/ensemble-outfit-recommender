// Unit test for the AI Stylist chat (src/ensemble/api/static/js/stylist.js): shoppers never see article numbers.
// stylist.js imports DOM modules, so the pure function is extracted from its source.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const src = readFileSync(new URL("../../src/ensemble/api/static/js/stylist.js", import.meta.url), "utf8");
const block = src.match(/export function humanize[\s\S]*?\n}\n/)[0].replace("export function", "function");
const humanize = new Function(`${block}\nreturn humanize;`)();

const cards = [{article_id: 883726003, prod_name: "Juan"}, {article_id: 751866001, prod_name: "MC Chestnut strap"}];
const cases = [
  ["Pair article 883726003 with wide trousers.", "Pair the Juan with wide trousers."],
  ["The **Juan** [[883726003]] works well.", "The **Juan** works well."],
  ["Try 0751866001 for evenings.", "Try MC Chestnut strap for evenings."],
  ["Article no. 123456789 is sold out.", "this piece is sold out."],
  ["Three tops under $20.", "Three tops under $20."],
];
let failed = 0;
for (const [input, want] of cases) {
  try { assert.equal(humanize(input, cards), want); console.log("ok", input); }
  catch (e) { failed++; console.log("FAIL", input, "→", humanize(input, cards)); }
}
process.exit(failed ? 1 : 0);
