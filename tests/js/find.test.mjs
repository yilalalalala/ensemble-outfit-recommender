// Unit tests for Find this piece (src/ensemble/api/static/js/find.js): the piece label and the category guess for
// a framed area. find.js imports DOM modules, so the two pure functions are extracted from its source.
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const src = readFileSync(new URL("../../src/ensemble/api/static/js/find.js", import.meta.url), "utf8");
const block = name => src.match(new RegExp(`export function ${name}[\\s\\S]*?\\n}\\n`))[0].replace("export function", "function");
const cats = src.match(/const CATEGORIES = \{[\s\S]*?\};/)[0];
const {pieceLabel, categoryFor} = new Function(`${cats}\n${block("pieceLabel")}\n${block("categoryFor")}\nreturn {pieceLabel, categoryFor};`)();

const spotted = [
  {id: "g1", category: "top", colour: "beige", description: "sweatshirt", box: [0.1, 0.1, 0.9, 0.5]},
  {id: "g2", category: "bottom", colour: "blue", description: "wide-leg jeans", box: [0.2, 0.45, 0.8, 0.95]},
  {id: "g3", category: "shoes", colour: "black", description: "flats", box: null},
];
const tests = [
  ["label joins colour and description once", () => {
    assert.equal(pieceLabel(spotted[0]), "beige sweatshirt");
    assert.equal(pieceLabel({category: "shoes", colour: "black", description: "black flats"}), "black flats");
    assert.equal(pieceLabel({category: "bag", colour: "", description: ""}), "Bag");
    assert.equal(pieceLabel({category: "top", colour: "yellow", description: "mustard yellow shirt"}), "mustard yellow shirt");
    assert.equal(pieceLabel({category: "dress", colour: "white", description: "a sleeveless white dress"}), "sleeveless white dress");
  }],
  ["a framed area takes the category of the spotted piece it mostly covers", () => {
    assert.equal(categoryFor([0.3, 0.6, 0.7, 0.9], spotted).category, "bottom");
    assert.equal(categoryFor([0.2, 0.15, 0.8, 0.4], spotted).category, "top");
    assert.equal(categoryFor([0.3, 0.6, 0.7, 0.9], spotted).colour, "blue");
  }],
  ["no real overlap falls back to the shopper's last choice", () => {
    assert.equal(categoryFor([0.0, 0.96, 0.1, 1.0], spotted, "bag").category, "bag");
    assert.equal(categoryFor([0.0, 0.0, 0.05, 0.05], [], "top").category, "top");
  }],
  ["pieces the shopper framed themselves never drive the guess", () => {
    const own = [{id: "f1", category: "hat", box: [0, 0, 1, 1], added: true}];
    assert.equal(categoryFor([0.2, 0.2, 0.4, 0.4], own, "top").category, "top");
  }],
];
let failed = 0;
for (const [name, fn] of tests) {
  try { fn(); console.log("ok", name); } catch (e) { failed++; console.log("FAIL", name, e.message); }
}
process.exit(failed ? 1 : 0);
