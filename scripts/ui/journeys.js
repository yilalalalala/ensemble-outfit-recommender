// Browser journeys for the shop UI (docs/CLAUDECODE_UI_EDITORIAL_REDESIGN_PLAN.md §19 and
// docs/CLAUDECODE_UI_CONSUMER_POLISH_ROUND_2.md §11). No dependency is added to the repo: point PLAYWRIGHT
// at any local Playwright install, and run against a server started on another port, e.g.
//   PYTHONPATH=src .venv/bin/uvicorn ensemble.api.app:app --port 8031
//   PLAYWRIGHT=/path/to/node_modules/playwright node scripts/ui/journeys.js [journey,journey] [base-url]
// Journeys: center copy home swatch cart switch product collections tabs studio label keyboard motion
//           offline assistant visual   (assistant/visual call the local models and take minutes).
// Mocked only: the Label save (never writes the user's gold labels) and controlled error states.
// Impressions/clicks/add_to_cart are logged to the demo events table, as in normal use.
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const fs = require('fs');
const BASE = process.argv[3] || 'http://localhost:8031/';
const ONLY = (process.argv[2] || '').split(',').filter(Boolean);
const SHOTS = process.env.SHOTS || 'data/interim/ui_editorial_screenshots/journeys_r2';
fs.mkdirSync(SHOTS, {recursive: true});
const PHOTO = process.env.PHOTO || 'data/raw/outfit_photos/IMG_1503.jpg';
const DEFAULT = 168058, NEW = 477553;

const results = [];
const ok = (name, cond, detail = '') => { results.push({name, ok: !!cond, detail}); console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${detail ? ' — ' + detail : ''}`); };

// Copy that must never appear on shopper routes (round 2 §1, §6, §7, §10).
const BANNED = [/why this/i, /recommender/i, /\branked?\b/i, /\branking\b/i, /\bmodel\b/i, /telemetry/i,
  /\boffline\b/i, /dataset/i, /no checkout/i, /no prices?/i, /\bdemo\b/i, /laptop/i, /stored suggestions/i,
  /\bservice\b/i, /\bAPI\b/, /SHAP/, /\bevidence\b/i, /\btools?\b/i, /never invents/i, /most recent first/i,
  /\b\d+ (pieces|colourways|items)\b/i, /for Shopper,/i, /Takes a few seconds/, /Takes about a minute/];

async function newPage(b, vp = [1440, 900], opts = {}) {
  const ctx = await b.newContext({viewport: {width: vp[0], height: vp[1]}, ...opts});
  const p = await ctx.newPage();
  p._errors = []; p._failed = []; p._events = [];
  p.on('pageerror', e => p._errors.push('pageerror: ' + e.message));
  p.on('console', m => { if (m.type() === 'error') p._errors.push('console: ' + m.text()); });
  p.on('requestfailed', r => p._failed.push(`${r.method()} ${r.url()} ${r.failure()?.errorText}`));
  p.on('response', r => { if (r.status() >= 400) p._failed.push(`${r.status()} ${r.request().method()} ${r.url()}`); });
  p.on('request', r => { if (r.url().endsWith('/api/events')) p._events.push(JSON.parse(r.postData())); });
  return {ctx, p};
}
// Add to cart opens the cart drawer; close it before interacting with the page again.
const closeCart = async p => { if (await p.$eval('#cart-dialog', d => d.open)) { await p.keyboard.press('Escape'); await p.waitForFunction(() => !document.querySelector('#cart-dialog').open); } };
const settle = async p => { await p.waitForLoadState('networkidle'); await p.waitForTimeout(400); };
const want = n => !ONLY.length || ONLY.includes(n);
const clean = (p, tag) => { ok(`no console errors (${tag})`, !p._errors.length, p._errors.join(' | ')); ok(`no failed requests (${tag})`, !p._failed.length, p._failed.join(' | ')); };

// Visible text of the shopper page (header, main, footer; closed dialogs are excluded by innerText).
const shopperText = p => p.evaluate(() => ['.utility', '.masthead', 'main', 'footer'].map(s => document.querySelector(s)?.innerText || '').join('\n'));
const bannedIn = text => BANNED.filter(r => r.test(text)).map(String);

// Every visible product card on the page: unique family and article, a USD price, a black Add to cart.
async function auditCards(p, scope = 'main') {
  await p.mouse.move(1, 1); await p.waitForTimeout(350);   // a hovered button is dark grey; audit the resting state after its transition
  return p.evaluate(scope => {
    const cards = [...document.querySelectorAll(`${scope} [data-card].card`)].filter(c => c.offsetParent !== null);
    const fams = cards.map(c => c.dataset.family || 'a:' + c.dataset.article), arts = cards.map(c => c.dataset.article);
    const bad = cards.filter(c => {
      const btn = c.querySelector('.btn-cart'), cs = btn && getComputedStyle(btn), price = c.querySelector('.card__price')?.textContent;
      return !btn || btn.textContent.trim() !== 'Add to cart' || cs.backgroundColor !== 'rgb(0, 0, 0)' || cs.color !== 'rgb(255, 255, 255)'
        || !/^\$\d+\.\d{2}$/.test(price || '') || !btn.getAttribute('aria-label').includes(c.querySelector('.card__name a').textContent)
        || !c.querySelector('.swatch[aria-pressed="true"]');
    }).map(c => c.dataset.article);
    return {n: cards.length, dupFamilies: fams.length - new Set(fams).size, dupArticles: arts.length - new Set(arts).size,
            emptyFamily: cards.filter(c => !c.dataset.family).length, bad, meta: document.querySelectorAll(`${scope} .tile__meta, ${scope} [data-why], ${scope} [data-hide]`).length};
  }, scope);
}

(async () => {
  const b = await chromium.launch({channel: 'chrome'});

  if (want('center')) for (const vp of [[1440, 900], [1024, 768], [390, 844]]) {
    const {ctx, p} = await newPage(b, vp, {colorScheme: 'dark'});
    for (const r of ['#/', '#/studio']) {
      await p.goto(BASE + r); await settle(p);
      const m = await p.evaluate(() => {
        const wm = document.querySelector('#wordmark'), w = wm.getBoundingClientRect();
        const boxes = ['#menu-btn', '#profile-btn', '#cart-btn', '#primary-nav'].map(s => document.querySelector(s))
          .filter(e => e && e.offsetParent !== null).map(e => e.getBoundingClientRect());
        const overlap = boxes.some(o => !(o.right <= w.left || o.left >= w.right || o.bottom <= w.top || o.top >= w.bottom));
        return {center: w.left + w.width / 2, vw: document.documentElement.clientWidth, label: wm.getAttribute('aria-label'), text: wm.textContent,
                hidden: [...wm.children].every(s => s.getAttribute('aria-hidden') === 'true'), overlap,
                bg: getComputedStyle(document.body).backgroundColor, overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth};
      });
      ok(`wordmark centered, readable, no collision ${vp[0]} ${r}`, Math.abs(m.center - m.vw / 2) < 0.75 && m.label === 'ensemble' && m.text === 'ensemble' && m.hidden && !m.overlap && m.overflow === 0,
         `center ${m.center.toFixed(2)} vs ${(m.vw / 2).toFixed(2)}, overlap ${m.overlap}, overflow ${m.overflow}`);
      if (r === '#/') ok(`white background even with OS dark mode ${vp[0]}`, m.bg === 'rgb(255, 255, 255)', m.bg);
    }
    await p.screenshot({path: `${SHOTS}/masthead_${vp[0]}.png`, clip: {x: 0, y: 0, width: vp[0], height: 120}});
    clean(p, `center ${vp[0]}`);
    await ctx.close();
  }

  if (want('copy') || want('home')) {
    const {ctx, p} = await newPage(b);
    for (const r of ['#/', '#/collections', '#/collections/trending', '#/collections/buy_again', '#/product/922037001', '#/assistant']) {
      await p.goto(BASE + r); await settle(p);
      const t = await shopperText(p);
      ok(`no prohibited technical/demo copy on ${r}`, !bannedIn(t).length, bannedIn(t).join(', '));
    }
    await p.goto(BASE); await settle(p);
    const t = await p.$eval('main', m => m.innerText);
    ok('home intro copy replaced exactly', t.includes('A personal edit, made with you in mind.') && !t.includes('ranked from your own purchases'), '');
    ok('purchased-items copy replaced', t.includes('Pieces you have bought.') && !/most recent first/i.test(t), '');
    ok('home has no item counts or profile suffix', !/Selected for you · \d+|\d+ pieces|for Shopper/.test(t), '');
    const foot = await p.$eval('footer', f => f.innerText);
    ok('shopper footer has no metric/dataset/checkout disclaimer', !/no prices|no checkout|offline|telemetry|dataset/i.test(foot), foot.replace(/\n/g, ' '));
    ok('shopper header has no service status', (await p.$eval('#utility-note', e => e.innerText)) === '', '');
    await p.goto(BASE + '#/assistant'); await settle(p);
    const a = await p.$eval('main', m => m.innerText);
    ok('assistant intro copy', a.includes("Tell us what you're looking for, or bring a photo. We'll help you find pieces that feel right."), '');
    ok('assistant time estimates', a.includes('It may take a few seconds.') && a.includes('It may take about a minute.') && !/laptop|Takes a few|Takes about/.test(a), '');
    clean(p, 'copy');
    await ctx.close();
  }

  if (want('home')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE); await settle(p);
    const titles = await p.$$eval('main h2.section-title', hs => hs.map(h => h.textContent.trim()));
    ok('home modules rendered', ['Selected for you', 'Buy it again', 'Trending this week'].every(t => titles.includes(t)) && titles.some(t => t.startsWith('Styled with')), titles.join(' / '));
    const a = await auditCards(p);
    ok('home: one card per product family page-wide', a.n > 20 && a.dupFamilies === 0 && a.dupArticles === 0 && a.emptyFamily === 0, JSON.stringify(a));
    ok('home: every card has USD price, swatch and black Add to cart', a.bad.length === 0, a.bad.join(','));
    ok('home: no Why this / category metadata / hide (×) on cards', a.meta === 0 && !(await p.$('main .tile__meta')), '');
    const imps = p._events.filter(e => e.event === 'impression');
    ok('impressions logged once per shown card, for this customer', imps.length === a.n && imps.every(e => e.customer_idx === DEFAULT), `${imps.length} impressions, ${a.n} cards`);
    // Not for me: logs not_for_me, confirms in place, and is remembered for this profile
    await p.evaluate(() => localStorage.removeItem('ensemble.notforme.168058'));
    ok('no hide (×) control on cards', !(await p.$('main [data-hide], main .card__hide')), '');
    const fam = await p.$eval('.hero__second [data-card]', c => c.dataset.family);
    await p.click('.hero__second [data-nfm]'); await p.waitForTimeout(200);
    const st = await p.$eval('.hero__second [data-card]', c => ({dim: c.classList.contains('is-dismissed'), note: c.querySelector('.card__nfm-note').textContent, dis: c.querySelector('[data-nfm]').disabled}));
    ok('Not for me logs the event and says it will be remembered', st.dim && st.dis && /remember that next time/.test(st.note) && p._events.some(e => e.event === 'not_for_me'), JSON.stringify(st));
    await p.reload(); await settle(p);
    ok('Not for me is remembered: that product is gone after reload', !(await p.$$eval('main [data-card]', (cs, f) => cs.some(c => c.dataset.family === f), fam)), fam);
    await p.evaluate(() => localStorage.removeItem('ensemble.notforme.168058'));
    await p.screenshot({path: `${SHOTS}/home.png`});
    clean(p, 'home');
    await ctx.close();
  }

  if (want('swatch') || want('cart')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE); await settle(p);
    await p.evaluate(() => localStorage.removeItem('ensemble.cart.168058'));
    // a card whose family has more than one real colourway
    const cardSel = await p.evaluate(() => {
      const c = [...document.querySelectorAll('main [data-card].card')].find(c => c.querySelectorAll('.swatch').length > 1);
      c.id = 'probe'; return c.dataset.article;
    });
    const fam = await p.evaluate(async id => (await (await fetch(`/api/catalog/families?articles=${id}`)).json()), cardSel);
    const code = fam.articles[cardSel], variants = fam.families[code].variants;
    const shown = await p.$$eval('#probe .swatch', s => s.length), more = await p.$eval('#probe', c => c.querySelector('.swatch-more')?.textContent || '');
    ok('card exposes every real colourway (≤6 swatches + "+N" link)', shown + (+more.replace('+', '') || 0) === variants.length, `${shown} swatches ${more}; ${variants.length} variants`);
    const labels = await p.$$eval('#probe .swatch', s => s.map(x => x.getAttribute('aria-label')));
    ok('swatches are labelled buttons, one marked selected', labels.filter(l => /, selected$/.test(l)).length === 1, labels.join(' | '));
    const priceBefore = await p.$eval('#probe .card__price', e => e.textContent);
    const target = await p.$eval('#probe .swatch[aria-pressed="false"]', s => s.dataset.swatch);
    await p.click(`#probe .swatch[data-swatch="${target}"]`);
    const after = await p.$eval('#probe', c => ({article: c.dataset.article, img: c.querySelector('img').getAttribute('src'), href: c.querySelector('.card__name a').getAttribute('href'),
      colour: c.querySelector('.card__colour').textContent, price: c.querySelector('.card__price').textContent, add: c.querySelector('[data-add]').getAttribute('aria-label'),
      pressed: c.querySelector(`.swatch[aria-pressed="true"]`).dataset.swatch}));
    const tv = variants.find(v => String(v.article_id) === target);
    ok('swatch updates image, link, colour, price and cart target in place', after.article === target && after.img === `/images/${target}.jpg` && after.href === `#/product/${target}`
       && after.pressed === target && after.price === priceBefore && after.add.includes(after.colour) && after.colour.startsWith(tv.colour_group_name.replace(/^Other /, '')), JSON.stringify(after));
    await p.click('#probe [data-add]');
    ok('Add to cart is protected while pending', await p.$eval('#probe [data-add]', bt => bt.disabled), '');
    ok('Add to cart opens the cart showing the new line', await p.$eval('#cart-dialog', d => d.open && !!d.querySelector(`.cart-line.is-new`)), '');
    await closeCart(p);
    const ev = p._events.filter(e => e.event === 'add_to_cart').at(-1);
    ok('add_to_cart event carries the selected colourway and surface', ev && String(ev.article_id) === target && ev.surface, JSON.stringify(ev));
    await p.waitForTimeout(1300);
    await p.click('#probe [data-add]'); await closeCart(p); await p.waitForTimeout(1300);
    ok('adding the same colourway twice increments quantity', (await p.$eval('#cart-count', e => e.textContent)) === '2', '');
    // second product, then the drawer
    const other = await p.$eval('main [data-card].card:not(#probe)', c => c.dataset.article);
    await p.click(`main [data-card][data-article="${other}"] [data-add]`); await closeCart(p); await p.waitForTimeout(1300);
    await p.click('#cart-btn'); await p.waitForSelector('#cart-dialog[open]');
    const cart = await p.$eval('#cart-dialog', d => ({lines: [...d.querySelectorAll('.cart-line')].map(l => ({id: l.dataset.line, qty: +l.querySelector('.qty__n').textContent,
      price: l.querySelector('.cart-line__meta').textContent, total: l.querySelector('.cart-line__total').textContent, img: !!l.querySelector('img'), name: l.querySelector('.cart-line__name').textContent})),
      subtotal: d.querySelector('#cart-subtotal').textContent, checkout: d.querySelector('.cart__foot .btn-cart')?.disabled}));
    const num = s => +s.replace(/[^0-9.]/g, '');
    const sum = cart.lines.reduce((a, l) => a + num(l.total), 0);
    ok('cart lists image, name, colour, price, quantity, line total and subtotal', cart.lines.length === 2 && cart.lines[0].qty === 2 && cart.lines.every(l => l.img && l.name)
       && Math.abs(num(cart.lines[0].total) - 2 * num(cart.lines[0].price.split('·').pop())) < 0.005 && Math.abs(num(cart.subtotal) - sum) < 0.005 && cart.checkout === true, JSON.stringify(cart));
    await p.waitForTimeout(500);
    await p.screenshot({path: `${SHOTS}/cart_drawer.png`});
    await p.click(`[data-line="${target}"] [data-qty="1"]`); await p.waitForTimeout(100);
    await p.click(`[data-line="${target}"] [data-qty="-1"]`); await p.waitForTimeout(100);
    ok('increment/decrement update quantity', (await p.$eval(`[data-line="${target}"] .qty__n`, e => e.textContent)) === '2', '');
    await p.click(`[data-line="${other}"] [data-remove]`); await p.waitForTimeout(100);
    ok('remove deletes the line and updates subtotal', (await p.$$('.cart-line')).length === 1 && (await p.$eval('#cart-count', e => e.textContent)) === '2', '');
    await p.keyboard.press('Escape');
    ok('cart closes on Escape, focus returns to the cart control', (await p.evaluate(() => document.activeElement.id)) === 'cart-btn', '');
    await p.reload(); await settle(p);
    ok('cart persists across reload', (await p.$eval('#cart-count', e => e.textContent)) === '2', '');
    // regression: a card's swatches keep working after any number of changes, and every colour can be added
    await p.evaluate(() => localStorage.removeItem('ensemble.cart.168058')); await p.reload(); await settle(p);
    const all = await p.$$eval('.hero__lead .swatch', s => s.map(x => x.dataset.swatch));
    const seq = [...all.slice(1), all[0]];
    let cycled = true;
    for (const id of seq) {
      await p.click(`.hero__lead .swatch[data-swatch="${id}"]`);
      cycled = cycled && (await p.$eval('.hero__lead [data-card]', c => c.dataset.article)) === id;
      await p.click('.hero__lead [data-add]'); await closeCart(p); await p.waitForTimeout(1200);
    }
    const lines = await p.evaluate(() => JSON.parse(localStorage.getItem('ensemble.cart.168058')).map(l => l.article_id));
    ok('swatches switch repeatedly (incl. back to the first) and each colour is added', cycled && lines.length === seq.length && seq.every(id => lines.includes(+id)), `${seq.join('→')} | cart ${lines.join(',')}`);
    // storage failure must not break shopping
    await p.evaluate(() => { Storage.prototype._set = Storage.prototype.setItem; Storage.prototype.setItem = () => { throw new Error('quota'); }; });
    await p.click('main [data-card].card [data-add]'); await p.waitForSelector('#cart-dialog[open]');
    ok('storage failure is reported in the cart, shopping continues', /couldn't be saved/.test(await p.$eval('#cart-body', e => e.innerText)), '');
    await p.keyboard.press('Escape');
    await p.evaluate(() => { Storage.prototype.setItem = Storage.prototype._set; localStorage.removeItem('ensemble.cart.168058'); });
    clean(p, 'swatch/cart');
    await ctx.close();
  }

  if (want('switch')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE); await settle(p);
    const first = await p.$$eval('main [data-card]', c => c.map(x => x.dataset.article));
    await p.click('#profile-btn'); await p.fill('#profile-filter', String(NEW));
    await p.click(`.profile-option[data-id="${NEW}"]`); await settle(p);
    const a = await auditCards(p);
    const now = await p.$$eval('main [data-card]', c => c.map(x => x.dataset.article));
    const t = await p.$eval('main', m => m.innerText);
    ok('profile switch replaces cards with no duplicate families', a.dupFamilies === 0 && now.join() !== first.join() && !(await p.$('#s-again')), JSON.stringify(a));
    ok('new-profile copy, no item counts', /A first edit/.test(t) && !/\d+ pieces/.test(t), '');
    const imps = p._events.filter(e => e.event === 'impression').slice(-a.n);
    ok('no stale cross-customer impressions', imps.every(e => e.customer_idx === NEW), '');
    ok('cart is per profile', (await p.$eval('#cart-count', e => e.hidden)) === true, '');
    await p.reload(); await settle(p);
    ok('profile persists across reload', (await p.$eval('#profile-v', e => e.textContent)) === 'New shopper, 56', '');
    await p.evaluate(() => localStorage.setItem('ensemble.profile', '168058'));
    clean(p, 'switch');
    await ctx.close();
  }

  if (want('product')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE); await settle(p);
    await p.click('.hero__lead .card__name a'); await settle(p);
    ok('product deep link from card', /#\/product\/922037001/.test(p.url()), p.url());
    ok('click events logged (card + product page)', p._events.some(e => e.event === 'click' && e.surface === 'for_you') && p._events.some(e => e.event === 'click' && e.surface === 'product_page'), '');
    const secs = await p.$$eval('main h2.section-title', hs => hs.map(h => h.textContent.trim()));
    ok('product page: Complete the look and Similar items; no Other colours gallery', secs.includes('Complete the look') && secs.includes('Similar items') && !secs.includes('Other colours'), secs.join(' / '));
    const fam = await p.evaluate(async () => (await (await fetch('/api/catalog/families?articles=922037001')).json()).families['0922037']);
    const sw = await p.$$eval('.pdp .swatch', s => s.map(x => x.dataset.swatch));
    ok('product page shows every real colourway as a swatch', sw.length === fam.variants.length && fam.variants.every(v => sw.includes(String(v.article_id))), `${sw.length} / ${fam.variants.length}`);
    const a = await auditCards(p);
    ok('product page: no repeated family (incl. the product itself)', a.dupFamilies === 0 && !(await p.$$eval('main [data-card].card', c => c.some(x => x.dataset.family === '0922037'))), JSON.stringify(a));
    ok('product page cards: price, swatch, black Add to cart', a.bad.length === 0, a.bad.join(','));
    const other = sw.find(x => x !== '922037001');
    await p.click(`.pdp .swatch[data-swatch="${other}"]`);
    const st = await p.evaluate(() => ({hash: location.hash, img: document.querySelector('.pdp__media img').getAttribute('src'), colour: document.querySelector('.fact-colour').textContent,
      label: document.querySelector('.pdp [data-add]').getAttribute('aria-label')}));
    ok('product swatch updates image, colour, address and cart target in place', st.hash === `#/product/${other}` && st.img === `/images/${other}.jpg` && st.label.includes(st.colour), JSON.stringify(st));
    await p.click('.pdp [data-add]'); await p.waitForTimeout(200);
    await closeCart(p);
    await p.mouse.move(5, 5); await p.waitForTimeout(1300);
    ok('product Add to cart logs add_to_cart for the selected colourway', p._events.some(e => e.event === 'add_to_cart' && String(e.article_id) === other && e.surface === 'product_page'), '');
    const btn = await p.$eval('.pdp [data-add]', b => { const c = getComputedStyle(b); return [c.backgroundColor, c.color, b.innerText.trim()]; });
    ok('product Add to cart is black with white text, same wording', btn[0] === 'rgb(0, 0, 0)' && btn[1] === 'rgb(255, 255, 255)' && /^add to cart$/i.test(btn[2]), btn.join(' '));
    await p.screenshot({path: `${SHOTS}/product.png`});
    const ctl = await p.$('#s-ctl ~ * .card__name a, section[aria-labelledby="s-ctl"] .card__name a');
    const ctlId = await ctl.getAttribute('data-open');
    await ctl.click(); await settle(p);
    ok('Complete the look card opens its product, click carries the anchor', p.url().endsWith('/product/' + ctlId) && p._events.some(e => e.event === 'click' && e.surface === 'complete_the_look' && e.anchor != null), p.url());
    await p.click('[data-back]'); await settle(p);
    ok('Back returns to the product', /\/product\/\d+$/.test(p.url()), p.url());
    await p.goto(BASE + '#/ds'); await settle(p);
    ok('legacy #/ds deep link opens DS Studio', (await p.$eval('h1', h => h.textContent)) === 'Offline evaluation', '');
    p._events.length = 0;
    await p.goto(BASE + '#/product/922037001'); await settle(p);
    ok('shopper routes never request /api/explain', !p._failed.some(f => f.includes('/api/explain')), '');
    clean(p, 'product');
    await ctx.close();
  }

  if (want('collections') || want('tabs')) {
    const {ctx, p} = await newPage(b);
    const explain = []; p.on('request', r => { if (r.url().includes('/api/explain')) explain.push(r.url()); });
    await p.goto(BASE + '#/collections/trending'); await settle(p);
    const a = await auditCards(p);
    ok('collection: unique families, priced cards', a.dupFamilies === 0 && a.bad.length === 0 && a.n > 0, JSON.stringify(a));
    const tabs = await p.$$eval('.tabs a', ts => ts.map(t => ({cur: t.getAttribute('aria-current'), color: getComputedStyle(t).color, bg: getComputedStyle(t).backgroundColor, border: getComputedStyle(t).borderBottomColor, text: t.textContent})));
    const sel = tabs.find(t => t.cur === 'page'), un = tabs.filter(t => t.cur !== 'page');
    ok('collection tabs: grey inactive, black selected with underline, no pill, no counts', sel.color === 'rgb(17, 17, 17)' && sel.border === 'rgb(17, 17, 17)' && un.every(t => t.color === 'rgb(118, 118, 118)')
       && tabs.every(t => t.bg === 'rgba(0, 0, 0, 0)' && !/\d/.test(t.text)), JSON.stringify(tabs));
    await p.focus('.filter[data-type]:not([data-type=""])'); await p.keyboard.press('Enter'); await p.waitForTimeout(200);
    const f = await p.$$eval('.filter', fs => fs.map(x => ({p: x.getAttribute('aria-pressed'), color: getComputedStyle(x).color, bg: getComputedStyle(x).backgroundColor, t: x.textContent})));
    const on = f.find(x => x.p === 'true');
    ok('category filter by keyboard: selected black, others grey, never a black pill, no counts', on && on.t !== 'All' && on.color === 'rgb(17, 17, 17)' && f.every(x => x.bg === 'rgba(0, 0, 0, 0)' && !/\d/.test(x.t))
       && f.filter(x => x.p !== 'true').every(x => x.color === 'rgb(118, 118, 118)'), JSON.stringify(on));
    const shown = await p.$$eval('main .grid > li:not([hidden])', l => l.length);
    ok('filter narrows the grid', shown > 0 && shown < a.n, `${a.n} -> ${shown}`);
    await p.screenshot({path: `${SHOTS}/collections_tabs.png`});
    ok('no /api/explain from shopper routes', !explain.length, explain.join(','));
    clean(p, 'collections');
    await ctx.close();
  }

  if (want('studio')) {
    const {ctx, p} = await newPage(b);
    for (const [r, h] of [['#/studio', 'Offline evaluation'], ['#/studio/serving', 'Serving status'], ['#/studio/inspect', 'Inspect a recommendation']]) {
      await p.goto(BASE + r); await settle(p);
      const d = await p.evaluate(() => ({h: document.querySelector('main h1')?.textContent, rows: document.querySelectorAll('table.data tbody tr').length,
        cur: document.querySelector('[data-exp="studio"]').getAttribute('aria-current'), cart: document.querySelector('#cart-btn').hidden,
        note: document.querySelector('#utility-note').innerText, foot: getComputedStyle(document.querySelector('.footer__studio')).display}));
      ok(`DS Studio ${r} (technical context kept, no cart)`, d.h === h && d.cur === 'page' && d.cart && /Offline evaluation|not ready/.test(d.note) && d.foot === 'block' && (r.endsWith('inspect') || d.rows > 0), JSON.stringify(d));
      await p.screenshot({path: `${SHOTS}/studio_${r.split('/').pop() || 'eval'}.png`});
    }
    await p.click('.inspect-item[data-i="2"]'); await p.waitForSelector('#inspect-detail .shap');
    ok('Inspect keeps reasons + SHAP in DS Studio', /evidence-gated reasons/i.test(await p.$eval('#inspect-detail', e => e.innerText)), '');
    clean(p, 'studio');
    await ctx.close();
  }

  if (want('label')) {
    const {ctx, p} = await newPage(b);
    let posted = null;
    await p.route('**/api/label', route => { posted = JSON.parse(route.request().postData()); route.fulfill({status: 200, contentType: 'application/json', body: '{"ok":true}'}); });
    await p.route('**/api/label/tasks**', async route => { const r = await route.fetch(); const j = await r.json(); if (j.length) j[0].done = false; route.fulfill({response: r, json: j}); });
    await p.goto(BASE + '#/label'); await settle(p);
    await p.click('.label-card[data-i="0"]'); await p.click('.label-card[data-i="2"]');
    const st = await p.$eval('.label-card[data-i="0"]', e => [e.getAttribute('aria-pressed'), e.innerText]);
    await p.click('#labsave'); await settle(p);
    ok('Label toggles (state not by colour alone) and saves', st[0] === 'true' && /relevant/i.test(st[1]) && posted && posted.relevant[0] === 1 && posted.relevant[2] === 1 && posted.relevant[1] === 0, JSON.stringify(posted));
    await p.screenshot({path: `${SHOTS}/label.png`});
    clean(p, 'label');
    await ctx.close();
  }

  if (want('keyboard')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE); await settle(p);
    await p.keyboard.press('Tab');
    ok('first Tab reaches the skip link', (await p.evaluate(() => document.activeElement.textContent)) === 'Skip to content', '');
    await p.keyboard.press('Enter');
    ok('skip link moves focus to main', (await p.evaluate(() => document.activeElement.id)) === 'main', '');
    // swatch + add to cart by keyboard
    await p.evaluate(() => document.querySelector('main .swatch[aria-pressed="false"]').focus());
    const id = await p.evaluate(() => document.activeElement.dataset.swatch);
    await p.keyboard.press('Enter');
    ok('swatch selectable by keyboard', (await p.evaluate(i => document.querySelector(`.swatch[data-swatch="${i}"]`).getAttribute('aria-pressed'), id)) === 'true', '');
    await p.evaluate(() => document.querySelector('#cart-btn').focus());
    await p.keyboard.press('Enter'); await p.waitForTimeout(200);
    ok('cart drawer opens from keyboard with focus inside', await p.evaluate(() => document.querySelector('#cart-dialog').open && document.querySelector('#cart-dialog').contains(document.activeElement)), '');
    await p.keyboard.press('Escape');
    await p.waitForFunction(() => !document.querySelector('#cart-dialog').open); await p.waitForTimeout(200);
    await p.evaluate(() => document.querySelector('[data-exp="shop"]').focus());
    await p.keyboard.press('Tab'); await p.keyboard.press('Enter'); await settle(p);
    ok('experience switch by keyboard', /#\/studio$/.test(p.url()), p.url());
    await p.evaluate(() => document.querySelector('#profile-btn').focus());
    await p.keyboard.press('Enter'); await p.waitForTimeout(200);
    ok('profile dialog opens from keyboard', await p.evaluate(() => document.querySelector('#profile-dialog').open && document.querySelector('#profile-dialog').contains(document.activeElement)), '');
    await p.keyboard.press('Escape'); await p.waitForTimeout(200);
    ok('Escape closes profile dialog and returns focus', (await p.evaluate(() => document.activeElement.id)) === 'profile-btn', '');
    await ctx.close();
    const m = await newPage(b, [390, 844]);
    await m.p.goto(BASE); await settle(m.p);
    await m.p.evaluate(() => document.querySelector('#menu-btn').focus());
    await m.p.keyboard.press('Enter'); await m.p.waitForTimeout(250);
    ok('mobile drawer opens from keyboard', await m.p.evaluate(() => document.querySelector('#drawer').open), '');
    await m.p.keyboard.press('Escape'); await m.p.waitForTimeout(250);
    ok('drawer closes on Escape, focus back on menu button', (await m.p.evaluate(() => document.activeElement.id)) === 'menu-btn', '');
    clean(m.p, 'keyboard');
    await m.ctx.close();
  }

  if (want('motion')) {
    for (const rm of ['reduce', 'no-preference']) {
      const {ctx, p} = await newPage(b, [1440, 900], {reducedMotion: rm});
      await p.goto(BASE); await settle(p);
      const d = await p.evaluate(() => ({motion: document.documentElement.classList.contains('motion'),
        hidden: [...document.querySelectorAll('.reveal')].filter(e => getComputedStyle(e).opacity === '0').length,
        anim: getComputedStyle(document.querySelector('.view')).animationName}));
      if (rm === 'reduce') ok('reduced motion: nothing hidden, no view animation', !d.motion && d.hidden === 0 && d.anim === 'none', JSON.stringify(d));
      else ok('motion: reveal + view transition active', d.motion && d.anim !== 'none', JSON.stringify(d));
      await ctx.close();
    }
  }

  if (want('offline')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE); await settle(p);
    await p.route('**/api/product/**', r => r.abort('failed'));
    await p.click('.hero__lead .card__name a'); await p.waitForTimeout(800);
    const txt = await p.$eval('main', m => m.innerText);
    ok('connection failure: friendly message with retry, nav intact', /can't connect/.test(txt) && !!(await p.$('[data-retry]')) && !!(await p.$('#primary-nav a')) && !bannedIn(txt).length, txt.slice(0, 120).replace(/\n/g, ' '));
    await p.unroute('**/api/product/**');
    await p.click('[data-retry]'); await settle(p);
    ok('retry recovers', !!(await p.$('.pdp')), '');
    await p.route('**/api/v2/complete-the-look**', r => r.fulfill({status: 503, contentType: 'application/json', body: '{"error":{"code":"not_ready","message":"bundle not loaded"}}'}));
    await p.reload(); await settle(p);
    const t2 = await shopperText(p);
    ok('live outfit failure falls back quietly (no service/storage wording)', /Complete the look/.test(t2) && !bannedIn(t2).length, bannedIn(t2).join(','));
    await ctx.close();
  }

  if (want('assistant')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE + '#/assistant'); await settle(p);
    await p.fill('#chat-in', 'What shoes go with wide-leg trousers?');
    await p.keyboard.press('Enter');
    ok('thinking state shown', !!(await p.waitForSelector('#pending .thinking', {timeout: 5000})), '');
    const t0 = Date.now();
    await p.waitForSelector('#pending', {state: 'detached', timeout: 600000});
    const ans = await p.$$eval('.msg--bot', m => m.at(-1).innerText);
    ok('assistant replies', !/couldn't answer/.test(ans) && ans.length > 20, `${Math.round((Date.now() - t0) / 1000)} s; ${ans.slice(0, 140).replace(/\n/g, ' ')}`);
    const a = await auditCards(p, '.msg--bot:last-child');
    ok('assistant cards: price, swatches, black Add to cart, no repeated family', a.n > 0 && a.bad.length === 0 && a.dupFamilies === 0, JSON.stringify(a));
    const t = await shopperText(p);
    ok('assistant shows no tool trace / technical copy', !(await p.$('.msg__trace')) && !bannedIn(t).length, bannedIn(t).join(','));
    await p.screenshot({path: `${SHOTS}/assistant_answer.png`});
    clean(p, 'assistant');
    await ctx.close();
  }

  if (want('visual')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE + '#/assistant'); await settle(p);
    await p.setInputFiles('#photo-in', PHOTO); await p.waitForTimeout(400);
    await p.selectOption('#vs-cat', 'top');
    await p.click('#vs-go');
    await p.waitForSelector('#photo-results .grid, #photo-results .state', {timeout: 300000});
    let a = await auditCards(p, '#photo-results');
    ok('visual search cards: price, swatches, black Add to cart, unique families', a.n > 0 && a.bad.length === 0 && a.dupFamilies === 0, JSON.stringify(a));
    await p.screenshot({path: `${SHOTS}/visual_search.png`});
    await p.click('#snap-go');
    await p.waitForSelector('#photo-results .notice, #photo-results .state, #photo-results .meta', {timeout: 600000});
    a = await auditCards(p, '#photo-results');
    const t = await p.$eval('#photo-results', e => e.innerText);
    ok('outfit results: consistent cards, unique families, consumer copy', a.n > 0 && a.bad.length === 0 && a.dupFamilies === 0 && !bannedIn(t).length, JSON.stringify(a) + ' ' + t.slice(0, 100).replace(/\n/g, ' '));
    await p.screenshot({path: `${SHOTS}/outfit.png`, fullPage: true});
    clean(p, 'visual');
    await ctx.close();
  }

  await b.close();
  const failed = results.filter(r => !r.ok);
  fs.writeFileSync(SHOTS + '/../journeys_r2_result.json', JSON.stringify(results, null, 1));
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
