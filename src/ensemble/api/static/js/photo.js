// Shared photo handling for the AI Stylist flows: validation, downscaling for per-tab storage, the picker UI,
// and sessionStorage-backed staged state (docs/AI_STYLIST_PRODUCT_FLOW.md §4).
import {$, announce, esc} from "./core.js";

export const ACCEPT = ["image/jpeg", "image/png", "image/webp"];
export const MAX_BYTES = 12 * 1024 * 1024;   // the API's upload limit
const MAX_SIDE = 1600;                       // what we keep per tab: enough for detection, small enough to store

export function validate(file) {
  if (!file) return "Choose a photo first.";
  if (!ACCEPT.includes(file.type)) return "That file is not a photo we can read. Please choose a JPEG, PNG or WebP.";
  if (file.size > MAX_BYTES) return "That photo is too large. Please choose one under 12 MB.";
  return null;
}

// Downscale (keeps orientation) to a JPEG data URL so a staged flow survives refresh within the tab.
export async function downscale(file) {
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(bmp, 0, 0, w, h);
  bmp.close?.();
  return {dataUrl: c.toDataURL("image/jpeg", 0.88), w, h};
}

export function toBlob(dataUrl) {
  const [head, b64] = dataUrl.split(",");
  const type = head.match(/data:([^;]+)/)?.[1] || "image/jpeg";
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Blob([buf], {type});
}

// ---- staged state (per tab) ------------------------------------------------------------------------------
const memory = new Map();   // fallback when sessionStorage is unavailable or full
export const stage = key => ({
  get() {
    try { const v = sessionStorage.getItem(key); if (v) return JSON.parse(v); } catch { /* fall through */ }
    return memory.get(key) || null;
  },
  set(value) {
    memory.set(key, value);
    try { sessionStorage.setItem(key, JSON.stringify(value)); } catch { /* quota or private mode: memory only */ }
  },
  clear() {
    memory.delete(key);
    try { sessionStorage.removeItem(key); } catch { /* ignore */ }
  },
});

// ---- picker -----------------------------------------------------------------------------------------------
// Renders an empty dropzone or the chosen photo with Replace / Remove. Calls onChange(photo|null) where
// photo = {dataUrl, w, h, name}.
export function photoPicker(slot, {photo = null, help, onChange}) {
  const errId = `${slot.id || "photo"}-err`;
  function empty(error = "") {
    slot.innerHTML = `<label class="dropzone">
        <input type="file" accept="${ACCEPT.join(",")}" aria-describedby="${errId}-help ${errId}">
        <span class="dropzone__icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M4 8h3l2-2h6l2 2h3v11H4z"/><circle cx="12" cy="13" r="3.2"/></svg></span>
        <span class="dropzone__title">Choose a photo</span>
        <span class="meta" id="${errId}-help">${esc(help)}</span>
        <span class="btn btn--quiet" aria-hidden="true">Browse files</span>
      </label>
      <p class="inline-status is-error" id="${errId}" role="alert">${esc(error)}</p>`;
    const input = $("input", slot), zone = $(".dropzone", slot);
    input.addEventListener("change", () => input.files[0] && load(input.files[0]));
    zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("is-over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("is-over"));
    zone.addEventListener("drop", e => { e.preventDefault(); zone.classList.remove("is-over"); if (e.dataTransfer.files[0]) load(e.dataTransfer.files[0]); });
  }
  function filled(p) {
    slot.innerHTML = `<figure class="photo-preview">
        <img src="${p.dataUrl}" alt="Your photo" width="${p.w}" height="${p.h}">
        <figcaption class="photo-preview__bar">
          <span class="meta">${esc(p.name || "Your photo")}</span>
          <label class="link-btn photo-preview__replace">Replace photo<input type="file" accept="${ACCEPT.join(",")}" class="sr-only"></label>
          <button class="link-btn" type="button" data-remove-photo>Remove</button>
        </figcaption>
      </figure>
      <p class="inline-status is-error" id="${errId}" role="alert"></p>`;
    $("input", slot).addEventListener("change", e => e.target.files[0] && load(e.target.files[0]));
    $("[data-remove-photo]", slot).addEventListener("click", () => {
      onChange(null); empty(); $("input", slot).focus(); announce("Photo removed.");
    });
  }
  async function load(file) {
    const err = validate(file);
    if (err) { photo ? filled(photo) : empty(); $(`#${errId}`, slot).textContent = err; return; }
    try {
      const p = {...(await downscale(file)), name: file.name};
      photo = p;
      filled(p);
      onChange(p);
      announce("Photo ready.");
    } catch {
      empty("We couldn't open that photo. Please try another one.");
    }
  }
  photo ? filled(photo) : empty();
}
