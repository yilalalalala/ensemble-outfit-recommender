// Pure shop logic (no DOM): product-family grouping, colour swatches, price formatting and the cart model.
// Kept dependency-free so it runs in the browser and under `node` for the unit tests.

// ---- money ---------------------------------------------------------------------------------------------
// Prices come from /api/catalog/families (prototype merchandising prices; see src/ensemble/api/merch.py).
const usd = new Intl.NumberFormat("en-US", {style: "currency", currency: "USD"});
export const money = v => (v == null || Number.isNaN(+v) ? "" : usd.format(+v));

// ---- colour swatches -----------------------------------------------------------------------------------
// Explicit mapping from the catalogue's colour groups to a swatch colour. Anything unmapped falls back to a
// labelled neutral pattern rather than a guessed colour.
export const SWATCH = {
  "Black": "#111111", "Dark Grey": "#4a4a4a", "Grey": "#8a8a8a", "Light Grey": "#cfcfcf", "White": "#ffffff",
  "Off White": "#f3efe6", "Light Beige": "#e8dcc4", "Beige": "#d6c3a1", "Dark Beige": "#b39b75", "Greyish Beige": "#bfb3a0",
  "Yellowish Brown": "#a97a3c", "Bronze/Copper": "#a8652f", "Gold": "#c9a64a", "Silver": "#c0c0c4",
  "Dark Blue": "#1f2a44", "Blue": "#3a64b0", "Light Blue": "#a9c6e8", "Other Blue": "#5f86c4",
  "Dark Turquoise": "#19757a", "Turquoise": "#2fb3a8", "Light Turquoise": "#9edfd8", "Other Turquoise": "#57c2b8",
  "Dark Green": "#234d32", "Green": "#3f8f4f", "Light Green": "#b5d9a3", "Greenish Khaki": "#7d7a4f", "Other Green": "#6e9c5a",
  "Dark Yellow": "#c9a227", "Yellow": "#f2d03b", "Light Yellow": "#f7eaa0", "Other Yellow": "#e9c85a",
  "Dark Orange": "#d0611f", "Orange": "#ec7f2c", "Light Orange": "#f6b77f", "Other Orange": "#e99a5c",
  "Dark Red": "#7a1c22", "Red": "#c0262d", "Light Red": "#e8777a", "Other Red": "#cf4b4f",
  "Dark Pink": "#c2507a", "Pink": "#e58fa8", "Light Pink": "#f4c9d3", "Other Pink": "#dc8fb2",
  "Dark Purple": "#4b2a5c", "Purple": "#7a4a9a", "Light Purple": "#c9b3dd", "Other Purple": "#9a78b8",
};
const FRIENDLY = {"Other": "Other colour", "Unknown": "Assorted", "Transparent": "Clear"};

export function colourName(group) {
  if (!group) return "Assorted";
  if (FRIENDLY[group]) return FRIENDLY[group];
  if (group.startsWith("Other ")) return `${group.slice(6)} (other shade)`;
  return group;
}

// {fill, pattern}: pattern is "clear" for transparent, "neutral" for unknown / multicolour values.
export function swatchStyle(group) {
  if (SWATCH[group]) return {fill: SWATCH[group], pattern: null};
  if (group === "Transparent") return {fill: null, pattern: "clear"};
  return {fill: null, pattern: "neutral"};
}

// Visible / accessible labels for a family's colourways. Two articles of one family can share a colour
// group (different print or shade); they are told apart as "Black, option 2" instead of a guessed name.
export function variantLabels(variants) {
  const total = {}, seen = {};
  for (const v of variants) total[v.colour_group_name] = (total[v.colour_group_name] || 0) + 1;
  return variants.map(v => {
    const name = colourName(v.colour_group_name);
    if (total[v.colour_group_name] < 2) return name;
    seen[v.colour_group_name] = (seen[v.colour_group_name] || 0) + 1;
    return `${name}, option ${seen[v.colour_group_name]}`;
  });
}

// ---- page-wide product grouping --------------------------------------------------------------------------
// One card per product family per page. `familyOf(article_id)` returns the catalogue's product_code, or
// null when unknown — then only the exact article id is used, so unrelated products are never merged
// (products are never grouped by name). The first occurrence keeps its position and module; later
// occurrences and colourways enrich that card instead of rendering again.
export function createPageGroups(familyOf) {
  const owner = new Map();   // family key -> article_id that owns the card
  const key = id => { const f = familyOf(id); return f ? `f:${f}` : `a:${id}`; };
  return {
    take(items) {
      const out = [];
      for (const it of items || []) {
        const k = key(it.article_id);
        if (owner.has(k)) continue;
        owner.set(k, it.article_id);
        out.push(it);
      }
      return out;
    },
    has: id => owner.has(key(id)),
    claim(id) { const k = key(id); if (owner.has(k)) return false; owner.set(k, id); return true; },
    size: () => owner.size,
  };
}

// ---- cart ----------------------------------------------------------------------------------------------
// One line per article (= family + colour); adding the same article again increments its quantity.
// `storage` is any {getItem, setItem}; failures are reported, never thrown into the page.
export const MAX_QTY = 10;

export class Cart {
  constructor(storage, key) {
    this.storage = storage; this.key = key; this.lines = []; this.error = null;
    this.load();
  }
  load() {
    try {
      const raw = this.storage?.getItem(this.key);
      const parsed = raw ? JSON.parse(raw) : [];
      this.lines = Array.isArray(parsed) ? parsed.filter(l => l && Number.isFinite(+l.article_id) && l.qty > 0) : [];
      this.error = null;
    } catch {
      this.lines = []; this.error = "load";
    }
    return this;
  }
  save() {
    try { this.storage?.setItem(this.key, JSON.stringify(this.lines)); this.error = null; return true; }
    catch { this.error = "save"; return false; }
  }
  find(id) { return this.lines.find(l => l.article_id === +id); }
  add(item) {
    const line = this.find(item.article_id);
    if (line) line.qty = Math.min(MAX_QTY, line.qty + 1);
    else this.lines.push({article_id: +item.article_id, family: item.family ?? null, name: item.name, colour: item.colour,
                          image: item.image, price: +item.price, qty: 1});
    return this.save();
  }
  setQty(id, qty) {
    const line = this.find(id);
    if (!line) return this.save();
    if (qty <= 0) this.lines = this.lines.filter(l => l !== line);
    else line.qty = Math.min(MAX_QTY, Math.floor(qty));
    return this.save();
  }
  remove(id) { this.lines = this.lines.filter(l => l.article_id !== +id); return this.save(); }
  get count() { return this.lines.reduce((a, l) => a + l.qty, 0); }
  get subtotal() { return Math.round(this.lines.reduce((a, l) => a + l.price * l.qty, 0) * 100) / 100; }
}
