// Prototype cart: persisted in localStorage per demo profile, shown in a drawer from the masthead.
// Payment and checkout are out of scope; Checkout is shown disabled.
import {Cart, MAX_QTY, money} from "./catalog.js";
import {$, announce, cardTarget, esc, logEvent, openDialog, state} from "./core.js";

let cart = null;

function storage() {
  try { return window.localStorage; } catch { return null; }   // blocked storage: cart lives in memory
}
const memory = new Map();
const memoryStore = {getItem: k => memory.get(k) ?? null, setItem: (k, v) => memory.set(k, v)};

export function loadCart() {
  cart = new Cart(storage() || memoryStore, `ensemble.cart.${state.customer}`);
  paintBadge();
  if ($("#cart-dialog").open) renderDrawer();
}

function paintBadge() {
  const n = cart?.count || 0;
  $("#cart-count").textContent = n ? String(n) : "";
  $("#cart-count").hidden = !n;
  $("#cart-btn").setAttribute("aria-label", n ? `Cart, ${n} item${n === 1 ? "" : "s"}` : "Cart, empty");
}

function storageNote() {
  return cart.error ? `<p class="cart__note" role="alert">Your cart couldn't be saved on this device, so it will clear when you leave.</p>` : "";
}

function renderDrawer(justAdded = null) {
  const body = $("#cart-body");
  if (!cart.lines.length) {
    body.innerHTML = `${storageNote()}<div class="cart__empty"><p class="cart__empty-title">Your cart is empty.</p>
      <a class="btn-cart" href="#/collections" data-close>Continue shopping</a></div>`;
    $("#cart-foot").hidden = true;
    return;
  }
  body.innerHTML = storageNote() + (justAdded != null ? `<p class="cart__added" role="status">Added to your cart</p>` : "") + `<ul class="cart__lines" role="list">${cart.lines.map(l => `
    <li class="cart-line${l.article_id === justAdded ? " is-new" : ""}" data-line="${l.article_id}">
      <a class="cart-line__img" href="#/product/${l.article_id}" data-close tabindex="-1" aria-hidden="true"><img src="${esc(l.image)}" alt="" width="1166" height="1750"></a>
      <div class="cart-line__info">
        <p class="cart-line__name"><a href="#/product/${l.article_id}" data-close>${esc(l.name)}</a></p>
        <p class="cart-line__meta">${esc(l.colour)} · ${esc(money(l.price))}</p>
        <div class="qty" role="group" aria-label="Quantity for ${esc(l.name)}">
          <button type="button" data-qty="-1" aria-label="Decrease quantity of ${esc(l.name)}">−</button>
          <span class="qty__n" aria-live="polite">${l.qty}</span>
          <button type="button" data-qty="1" aria-label="Increase quantity of ${esc(l.name)}" ${l.qty >= MAX_QTY ? "disabled" : ""}>+</button>
        </div>
        <button class="link-btn" type="button" data-remove aria-label="Remove ${esc(l.name)} from cart">Remove</button>
      </div>
      <p class="cart-line__total">${esc(money(l.price * l.qty))}</p>
    </li>`).join("")}</ul>`;
  $("#cart-foot").hidden = false;
  $("#cart-subtotal").textContent = money(cart.subtotal);
}

export function initCart() {
  const dlg = $("#cart-dialog");
  $("#cart-btn").addEventListener("click", e => { renderDrawer(); openDialog(dlg, e.currentTarget); });
  dlg.addEventListener("click", e => {
    const line = e.target.closest("[data-line]");
    const qty = e.target.closest("[data-qty]");
    if (qty && line) {
      const id = +line.dataset.line, l = cart.find(id);
      cart.setQty(id, l.qty + +qty.dataset.qty);
      renderDrawer(); paintBadge();
      const again = $(`[data-line="${id}"] [data-qty="${qty.dataset.qty}"]`, dlg);
      (again && !again.disabled ? again : $("#cart-close")).focus();
      announce(cart.find(id) ? `Quantity ${cart.find(id).qty}.` : `${l.name} removed.`);
    } else if (e.target.closest("[data-remove]") && line) {
      const l = cart.find(+line.dataset.line);
      cart.remove(+line.dataset.line);
      renderDrawer(); paintBadge();
      announce(`${l?.name ?? "Item"} removed from your cart.`);
      $("#cart-close").focus();
    }
  });
}

// Add to cart from any shopper card or the product page: one pending click at a time.
export function addToCart(btn, target = null) {
  if (btn.disabled || btn.dataset.pending) return;
  const t = target || cardTarget(btn.closest("[data-card]"));
  if (t.price == null) {
    console.warn("add to cart: no price for article", t.article_id);
    announce("Sorry, this item can't be added right now.");
    return;
  }
  btn.dataset.pending = "1";
  btn.disabled = true;
  logEvent("add_to_cart", t.surface || "product_page", t.article_id, t.anchor);
  const saved = cart.add(t);
  paintBadge();
  const label = btn.textContent;
  btn.textContent = "Added";
  announce(saved ? `${t.name}, ${t.colour}, added to your cart.` : `${t.name} added. Your cart couldn't be saved on this device.`);
  // Show the cart with the new line, so the add is visible; closing it returns focus to this button.
  renderDrawer(t.article_id);
  openDialog($("#cart-dialog"), btn);
  setTimeout(() => { btn.textContent = label; btn.disabled = false; delete btn.dataset.pending; }, 1100);
}
