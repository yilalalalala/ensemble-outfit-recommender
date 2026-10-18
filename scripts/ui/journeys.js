// Browser journeys for the shop UI (docs/CLAUDECODE_UI_EDITORIAL_REDESIGN_PLAN.md §19 and
// docs/CLAUDECODE_UI_CONSUMER_POLISH_ROUND_2.md §11). No dependency is added to the repo: point PLAYWRIGHT
// at any local Playwright install, and run against a server started on another port, e.g.
//   PYTHONPATH=src .venv/bin/uvicorn ensemble.api.app:app --port 8031
//   PLAYWRIGHT=/path/to/node_modules/playwright node scripts/ui/journeys.js [journey,journey] [base-url]
// Journeys: center copy home swatch cart switch product collections tabs studio label keyboard motion offline
//           gateway widths · look find ask recovery (these call the models and can take minutes).
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
  /\b\d+ (pieces|colourways|items)\b/i, /for Shopper,/i, /Takes a few seconds/, /Takes about a minute/,
  /FashionCLIP/i, /DeepFashion/i, /\branker\b/i, /tool calling/i];

// JOURNEY_WIDTH=390 runs the journeys at a phone width (default 1440 × 900).
const DEFAULT_VP = process.env.JOURNEY_WIDTH ? [+process.env.JOURNEY_WIDTH, 844] : [1440, 900];
async function newPage(b, vp = DEFAULT_VP, opts = {}) {
  const ctx = await b.newContext({viewport: {width: vp[0], height: vp[1]}, ...opts});
  const p = await ctx.newPage();
  p._errors = []; p._failed = []; p._events = [];
  p.on('pageerror', e => p._errors.push('pageerror: ' + e.message));
  p.on('console', m => { if (m.type() === 'error') p._errors.push('console: ' + m.text()); });
  // ERR_ABORTED = the page navigated away before a request finished (seen with a remote API); not a failure.
  p.on('requestfailed', r => { if (r.failure()?.errorText !== 'net::ERR_ABORTED') p._failed.push(`${r.method()} ${r.url()} ${r.failure()?.errorText}`); });
  p.on('response', r => { if (r.status() >= 400) p._failed.push(`${r.status()} ${r.request().method()} ${r.url()}`); });
  p.on('request', r => { if (r.url().endsWith('/api/events')) p._events.push(JSON.parse(r.postData())); });
  return {ctx, p};
}
// Add to cart opens the cart drawer; close it before interacting with the page again.
const closeCart = async p => { if (await p.$eval('#cart-dialog', d => d.open)) { await p.keyboard.press('Escape'); await p.waitForFunction(() => !document.querySelector('#cart-dialog').open); } };
// With a remote API, event POSTs can keep the network busy; idle is a best-effort wait, checks wait for elements.
// Screenshots wait for every image in the viewport to finish loading (R2 images can lag the layout).
const imagesReady = p => p.waitForFunction(() => [...document.images].filter(i => { const r = i.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.width > 0; }).every(i => i.complete && i.naturalWidth > 0), null, {timeout: 30000}).catch(() => {});
const settle = async p => { await p.waitForLoadState('networkidle', {timeout: 15000}).catch(() => {}); await p.waitForTimeout(400); };
const want = n => !ONLY.length || ONLY.includes(n);
const clean = (p, tag) => { ok(`no console errors (${tag})`, !p._errors.length, p._errors.join(' | ')); ok(`no failed requests (${tag})`, !p._failed.length, p._failed.join(' | ')); };

// Visible text of the shopper page (header, main and, where shown, the footer; closed dialogs are excluded by innerText).
const shopperText = p => p.evaluate(() => ['.utility', '.masthead', 'main', 'footer'].map(s => document.querySelector(s))
  .filter(e => e && getComputedStyle(e).display !== 'none').map(e => e.innerText).join('\n'));
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
    for (const r of ['#/', '#/collections', '#/collections/trending', '#/collections/buy_again', '#/product/922037001',
                     '#/stylist', '#/stylist/ask', '#/stylist/find', '#/stylist/look']) {
      await p.goto(BASE + r); await settle(p);
      const t = await shopperText(p);
      ok(`no prohibited technical/demo copy on ${r}`, !bannedIn(t).length, bannedIn(t).join(', '));
    }
    await p.goto(BASE + '#/collections'); await settle(p);
    const t = await p.$eval('main', m => m.innerText);
    ok('Collections: the three edits as tabs, no Your edit page', /^collections/i.test(t) && !/Your edit/.test(t) && ['Selected for you', 'Buy it again', 'Trending this week'].every(x => t.includes(x)), t.slice(0, 120));
    ok('Collections has no item counts or profile suffix', !/Selected for you · \d+|\d+ pieces|for Shopper/.test(t), '');
    ok('shoppers see no footer (no bottom links or words)', await p.$eval('footer', f => getComputedStyle(f).display === 'none'), '');
    ok('shop navigation is Collections · AI Stylist only', JSON.stringify(await p.$$eval('#primary-nav a', as => as.map(a => a.textContent))) === '["Collections","AI Stylist"]', '');
    ok('shopper header has no service status', (await p.$eval('#utility-note', e => e.innerText)) === '', '');
    await p.goto(BASE + '#/stylist/ask'); await settle(p);
    const a = await p.$eval('main', m => m.innerText);
    ok('ask intro copy', a.replace(/[’]/g, "'").includes("Tell us what you're looking for. We'll help you find pieces that feel right."), '');
    await p.goto(BASE + '#/stylist'); await settle(p);
    ok('hub copy and three tasks, no eyebrow above the question', (await p.$eval('main', m => m.innerText)).includes('What can we help with?') && (await p.$$('.task-card')).length === 3 && !(await p.$('.hub__head .eyebrow')), '');
    const fonts = await p.evaluate(() => ({h: getComputedStyle(document.querySelector('.hub__title')).fontFamily, body: getComputedStyle(document.querySelector('.lede')).fontFamily,
      nav: getComputedStyle(document.querySelector('#primary-nav a')).fontFamily, wm: getComputedStyle(document.querySelector('#wordmark')).fontFamily,
      loaded: [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family)}));
    ok('canvas typefaces: Tiro Gurmukhi headings, Inter text, Poppins navigation, Segoe UI wordmark (self-hosted)',
       /^"Tiro Gurmukhi"/.test(fonts.h) && /^Inter/.test(fonts.body) && /^Poppins/.test(fonts.nav) && /^"Segoe UI"/.test(fonts.wm)
       && ['Tiro Gurmukhi', 'Inter', 'Poppins'].every(f => fonts.loaded.includes(f)), JSON.stringify(fonts));
    clean(p, 'copy');
    await ctx.close();
  }

  if (want('home')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE + '#/collections'); await settle(p);
    const titles = await p.$$eval('main .tabs a', hs => hs.map(h => h.textContent.trim()));
    ok('Collections tabs rendered', ['Selected for you', 'Buy it again', 'Trending this week'].every(t => titles.includes(t)), titles.join(' / '));
    const a = await auditCards(p);
    ok('home: one card per product family page-wide', a.n >= 8 && a.dupFamilies === 0 && a.dupArticles === 0 && a.emptyFamily === 0, JSON.stringify(a));
    ok('home: every card has USD price, swatch and black Add to cart', a.bad.length === 0, a.bad.join(','));
    ok('home: no Why this / category metadata / hide (×) on cards', a.meta === 0 && !(await p.$('main .tile__meta')), '');
    const imps = p._events.filter(e => e.event === 'impression');
    ok('impressions logged once per shown card, for this customer', imps.length === a.n && imps.every(e => e.customer_idx === DEFAULT), `${imps.length} impressions, ${a.n} cards`);
    ok('no hide (×) and no Not for me on cards', !(await p.$('main [data-hide], main .card__hide, main [data-nfm]')) && !/Not for me/.test(await p.$eval('main', m => m.innerText)), '');
    const chips = await p.$$eval('main .swatch', ss => ss.map(x => ({fabric: x.classList.contains('is-fabric'), sw: getComputedStyle(x).getPropertyValue('--sw').trim()})));
    ok('swatches come from the photos: a colour-card chip or a fabric crop on every swatch', chips.length > 0 && chips.every(c => c.fabric || /^#[0-9a-f]{6}$/i.test(c.sw)), JSON.stringify(chips.slice(0, 3)));
    await p.screenshot({path: `${SHOTS}/home.png`});
    clean(p, 'home');
    await ctx.close();
  }

  if (want('swatch') || want('cart')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE + '#/collections'); await settle(p);
    await p.evaluate(() => localStorage.removeItem('ensemble.cart.168058'));
    // a card whose family has more than one real colourway
    const cardSel = await p.evaluate(() => {
      const c = [...document.querySelectorAll('main [data-card].card')].find(c => c.querySelectorAll('.swatch').length > 1);
      c.id = 'probe'; return c.dataset.article;
    });
    const fam = await p.evaluate(async id => (await (await fetch(`${document.querySelector('meta[name="ensemble-api-base"]')?.content || ''}/api/catalog/families?articles=${id}`)).json()), cardSel);
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
    // Locally images are /images/<id>.jpg; hosted they are R2 URLs ending in /<prefix>/<zero-padded id>.jpg.
    const imgOk = after.img === `/images/${target}.jpg` || after.img.endsWith(`/${target.padStart(10, '0')}.jpg`);
    ok('swatch updates image, link, colour, price and cart target in place', after.article === target && imgOk && after.href === `#/product/${target}`
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
    const all = await p.$$eval('main .grid > li:nth-child(1) .swatch', s => s.map(x => x.dataset.swatch));
    const seq = [...all.slice(1), all[0]];
    let cycled = true;
    for (const id of seq) {
      await p.click(`main .grid > li:nth-child(1) .swatch[data-swatch="${id}"]`);
      cycled = cycled && (await p.$eval('main .grid > li:nth-child(1) [data-card]', c => c.dataset.article)) === id;
      await p.click('main .grid > li:nth-child(1) [data-add]'); await closeCart(p); await p.waitForTimeout(1200);
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
    await p.goto(BASE + '#/collections'); await settle(p);
    const first = await p.$$eval('main [data-card]', c => c.map(x => x.dataset.article));
    await p.click('#profile-btn'); await p.fill('#profile-filter', String(NEW));
    await p.click(`.profile-option[data-id="${NEW}"]`); await settle(p);
    const a = await auditCards(p);
    const now = await p.$$eval('main [data-card]', c => c.map(x => x.dataset.article));
    const t = await p.$eval('main', m => m.innerText);
    ok('profile switch replaces cards with no duplicate families', a.dupFamilies === 0 && now.join() !== first.join() && !(await p.$('#s-again')), JSON.stringify(a));
    ok('new profile: Collections, no item counts', /^collections/i.test(t) && !/\d+ pieces/.test(t), '');
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
    await p.goto(BASE + '#/collections'); await settle(p);
    await p.click('main .grid > li:nth-child(1) .card__name a'); await settle(p);
    await p.waitForSelector('.pdp [data-add]'); await p.waitForSelector('#s-sim', {timeout: 60000}); await p.waitForTimeout(800);
    ok('product deep link from card', /#\/product\/922037001/.test(p.url()), p.url());
    ok('click events logged (card + product page)', p._events.some(e => e.event === 'click' && e.surface === 'for_you') && p._events.some(e => e.event === 'click' && e.surface === 'product_page'), '');
    const secs = await p.$$eval('main h2.section-title', hs => hs.map(h => h.textContent.trim()));
    ok('product page: Similar items only; no Complete the look, no Other colours gallery', secs.includes('Similar items') && !secs.includes('Complete the look') && !secs.includes('Other colours')
       && !p._failed.concat(p._events.map(e => e.surface)).some(x => /complete.the.look/.test(String(x))), secs.join(' / '));
    const fam = await p.evaluate(async () => (await (await fetch(`${document.querySelector('meta[name="ensemble-api-base"]')?.content || ''}/api/catalog/families?articles=922037001`)).json()).families['0922037']);
    const sw = await p.$$eval('.pdp .swatch', s => s.map(x => x.dataset.swatch));
    ok('product page shows every real colourway as a swatch', sw.length === fam.variants.length && fam.variants.every(v => sw.includes(String(v.article_id))), `${sw.length} / ${fam.variants.length}`);
    const a = await auditCards(p);
    ok('product page: no repeated family (incl. the product itself)', a.dupFamilies === 0 && !(await p.$$eval('main [data-card].card', c => c.some(x => x.dataset.family === '0922037'))), JSON.stringify(a));
    ok('product page cards: price, swatch, black Add to cart', a.bad.length === 0, a.bad.join(','));
    const other = sw.find(x => x !== '922037001');
    await p.click(`.pdp .swatch[data-swatch="${other}"]`);
    const st = await p.evaluate(() => ({hash: location.hash, img: document.querySelector('.pdp__media img').getAttribute('src'), colour: document.querySelector('.fact-colour').textContent,
      label: document.querySelector('.pdp [data-add]').getAttribute('aria-label')}));
    ok('product swatch updates image, colour, address and cart target in place', st.hash === `#/product/${other}` && (st.img === `/images/${other}.jpg` || st.img.endsWith(`/${other.padStart(10, '0')}.jpg`)) && st.label.includes(st.colour), JSON.stringify(st));
    await p.click('.pdp [data-add]'); await p.waitForTimeout(200);
    await closeCart(p);
    await p.mouse.move(5, 5); await p.waitForTimeout(1300);
    ok('product Add to cart logs add_to_cart for the selected colourway', p._events.some(e => e.event === 'add_to_cart' && String(e.article_id) === other && e.surface === 'product_page'), '');
    const btn = await p.$eval('.pdp [data-add]', b => { const c = getComputedStyle(b); return [c.backgroundColor, c.color, b.innerText.trim()]; });
    ok('product Add to cart is black with white text, same wording', btn[0] === 'rgb(0, 0, 0)' && btn[1] === 'rgb(255, 255, 255)' && /^add to cart$/i.test(btn[2]), btn.join(' '));
    await p.screenshot({path: `${SHOTS}/product.png`});
    const sim = await p.$('section[aria-labelledby="s-sim"] .card__name a');
    const simId = await sim.getAttribute('data-open');
    await sim.click(); await settle(p);
    ok('Similar item card opens its product', p.url().endsWith('/product/' + simId) && p._events.some(e => e.event === 'click' && e.surface === 'similar'), p.url());
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
    await p.goto(BASE + '#/shop/trending'); await settle(p);
    ok('old Your edit links land on Collections', /#\/collections\/trending$/.test(p.url()), p.url());
    const a = await auditCards(p);
    ok('collection: unique families, priced cards', a.dupFamilies === 0 && a.bad.length === 0 && a.n > 0, JSON.stringify(a));
    const tabs = await p.$$eval('.tabs a', ts => ts.map(t => ({cur: t.getAttribute('aria-current'), color: getComputedStyle(t).color, bg: getComputedStyle(t).backgroundColor, border: getComputedStyle(t).borderBottomColor, text: t.textContent})));
    const sel = tabs.find(t => t.cur === 'page'), un = tabs.filter(t => t.cur !== 'page');
    ok('collection tabs: grey inactive, black selected with underline, no pill, no counts', sel.color === 'rgb(17, 17, 17)' && sel.border === 'rgb(17, 17, 17)' && un.every(t => t.color === 'rgb(117, 111, 105)')
       && tabs.every(t => t.bg === 'rgba(0, 0, 0, 0)' && !/\d/.test(t.text)), JSON.stringify(tabs));
    await p.focus('.filter[data-type]:not([data-type=""])'); await p.keyboard.press('Enter'); await p.mouse.move(1, 1); await p.waitForTimeout(450);
    const f = await p.$$eval('.filter', fs => fs.map(x => ({p: x.getAttribute('aria-pressed'), color: getComputedStyle(x).color, bg: getComputedStyle(x).backgroundColor, t: x.textContent})));
    const on = f.find(x => x.p === 'true');
    ok('category filter by keyboard: selected black, others grey, never a black pill, no counts', on && on.t !== 'All' && on.color === 'rgb(17, 17, 17)' && f.every(x => x.bg === 'rgba(0, 0, 0, 0)' && !/\d/.test(x.t))
       && f.filter(x => x.p !== 'true').every(x => x.color === 'rgb(117, 111, 105)'), JSON.stringify(on));
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
      if (!r.endsWith('inspect')) await p.waitForSelector('table.data tbody tr', {timeout: 60000});
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
    await p.goto(BASE + '#/collections'); await settle(p);
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
    await m.p.goto(BASE + '#/collections'); await settle(m.p);
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
      await p.goto(BASE + '#/collections'); await settle(p);
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
    await p.goto(BASE + '#/collections'); await settle(p);
    await p.route('**/api/product/**', r => r.abort('failed'));
    await p.click('main .grid > li:nth-child(1) .card__name a'); await p.waitForTimeout(800);
    const txt = await p.$eval('main', m => m.innerText);
    ok('connection failure: friendly message with retry, nav intact', /can't connect/.test(txt) && !!(await p.$('[data-retry]')) && !!(await p.$('#primary-nav a')) && !bannedIn(txt).length, txt.slice(0, 120).replace(/\n/g, ' '));
    await p.unroute('**/api/product/**');
    await p.click('[data-retry]'); await settle(p);
    ok('retry recovers', !!(await p.$('.pdp')), '');
    await ctx.close();
  }

  // ---- AI Stylist product flow (docs/AI_STYLIST_PRODUCT_FLOW.md) --------------------------------------------
  const oneCta = p => p.$$eval('.task__actions .btn--solid', b => b.length);
  const heading = p => p.$eval('.task__title, h1', h => h.textContent.trim());
  const focusedHeading = p => p.evaluate(() => document.activeElement?.matches('h1'));

  if (want('gateway')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE + '#/'); await settle(p);
    ok('gateway heading (screen readers) and two choices', (await heading(p)) === 'How would you like to shop today?' && (await p.$eval('#gw-title', h => h.classList.contains('sr-only'))) && (await p.$$('.gw-choice')).length === 2, '');
    const gw = await p.evaluate(() => {
      const s = document.querySelector('.gateway'), t = document.querySelector('.gateway__choices').getBoundingClientRect();
      const r = s.getBoundingClientRect(), mid = r.left + r.width / 2;
      const bs = [...s.querySelectorAll('.gw-choice__btn')].map(b => b.getBoundingClientRect());
      return {imgs: s.querySelectorAll('img').length, solid: s.querySelectorAll('.gw-choice__btn.btn--solid').length,
        sameSize: bs.length === 2 && Math.abs(bs[0].width - bs[1].width) < 1 && Math.abs(bs[0].height - bs[1].height) < 1,
        fits: [...s.querySelectorAll('.gw-choice__btn')].every(b => b.scrollWidth <= b.clientWidth && b.getBoundingClientRect().width <= b.parentElement.getBoundingClientRect().width + 0.5),
        centred: Math.abs(t.left + t.width / 2 - mid) < 4, overflow: document.documentElement.scrollWidth > innerWidth};
    });
    ok('gateway is plain: no imagery, two solid buttons, centred, no horizontal scroll', !gw.imgs && gw.solid === 2 && gw.centred && !gw.overflow, JSON.stringify(gw));
    ok('both gateway buttons are the same size and their labels fit', gw.sameSize && gw.fits, JSON.stringify(gw));
    const box = await p.$eval('.gateway', s => { const r = s.getBoundingClientRect(); return {x: r.left + 40, y: r.top + 40}; });
    await p.mouse.move(box.x, box.y); await p.mouse.down();
    const rip = await p.$$eval('.gateway > .ripple', rs => rs.length); await p.mouse.up();
    const btn = await p.$eval('.gw-choice--ai', b => { const r = b.getBoundingClientRect(); return {x: r.left + 10, y: r.top + 10}; });
    await p.mouse.move(btn.x, btn.y); await p.mouse.down();
    const ripBtn = await p.$$eval('.gw-choice--ai > .ripple', rs => rs.length); await p.mouse.up();
    ok('a touch ripples from the page and from a button', rip === 1 && ripBtn === 1, JSON.stringify({rip, ripBtn}));
    await p.waitForTimeout(1000);
    ok('ripples clean themselves up', !(await p.$$('.ripple')).length, '');
    await p.goto(BASE + '#/'); await settle(p);
    await p.click('.gw-choice--shop');
    const leaving = await p.evaluate(() => ({cls: document.querySelector('.gateway')?.classList.contains('is-leaving'), url: location.hash}));
    ok('choosing a path fades the gateway before navigating', leaving.cls && leaving.url === '#/', JSON.stringify(leaving));
    await p.waitForURL(/#\/collections$/, {timeout: 3000}); await settle(p);
    const hero = await p.waitForSelector('main .grid .card', {timeout: 60000}).catch(() => null);
    ok('View your collections reaches Collections in one click', /#\/collections$/.test(p.url()) && !!hero, p.url());
    await p.goto(BASE + '#/'); await settle(p);
    await p.click('.gw-choice--ai'); await p.waitForURL(/#\/stylist$/, {timeout: 3000}); await settle(p);
    ok('Style with AI opens the hub with three tasks', /#\/stylist$/.test(p.url()) && (await p.$$('.task-card')).length === 3, p.url());
    await p.goto(BASE + '#/'); await settle(p);
    await p.click('a[data-exp="studio"]'); await settle(p);
    await p.waitForSelector('main h1');
    ok('the masthead switch reaches DS Studio', /#\/studio$/.test(p.url()) && (await heading(p)) === 'Offline evaluation', p.url());
    await p.goto(BASE + '#/assistant?anchor=922037001'); await settle(p);
    ok('legacy #/assistant deep link redirects to Ask (anchor kept)', /#\/stylist\/ask\?anchor=922037001$/.test(p.url()), p.url());
    ok('wordmark returns to the gateway', (await p.$eval('#wordmark', a => a.getAttribute('href'))) === '#/', '');
    const {ctx: calm, p: q} = await newPage(b, DEFAULT_VP, {reducedMotion: 'reduce'});
    await q.goto(BASE + '#/'); await settle(q);
    await q.click('.gw-choice--ai');
    const calmState = await q.evaluate(() => ({rip: document.querySelectorAll('.ripple').length, hash: location.hash}));
    ok('reduced motion: no ripple, navigation is immediate', !calmState.rip && calmState.hash === '#/stylist', JSON.stringify(calmState));
    clean(q, 'gateway reduced motion');
    await calm.close();
    clean(p, 'gateway');
    await ctx.close();
  }

  if (want('look')) {
    const {ctx, p} = await newPage(b);
    let editPayload = null;
    // Read the multipart body through interception (request events expose no body for Blob uploads), then continue.
    await p.route('**/api/stylist/edit', route => { editPayload = route.request().postDataBuffer()?.toString('latin1') || ''; return route.continue(); });
    await p.goto(BASE + '#/stylist/look'); await settle(p);
    await p.evaluate(() => sessionStorage.clear()); await p.reload(); await settle(p);
    ok('step 1: progress header, primary disabled until a photo', (await p.$eval('.steps__step[aria-current="step"]', e => e.innerText)).includes('Photo')
       && await p.$eval('#look-detect', bt => bt.disabled) && (await oneCta(p)) === 1, '');
    await p.setInputFiles('.dropzone input', {name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('not a photo')});
    ok('a non-photo file is rejected with clear copy', /not a photo we can read/.test(await p.$eval('#look-photo [role=alert]', e => e.textContent)), '');
    await p.setInputFiles('.dropzone input', PHOTO); await p.waitForSelector('.photo-preview img');
    ok('photo preview with replace and remove', !!(await p.$('.photo-preview__replace')) && !!(await p.$('[data-remove-photo]')), '');
    await p.click('#look-detect');
    ok('staged detection copy', /Looking at your outfit/.test(await p.$eval('#look-status', e => e.innerText)), '');
    await p.waitForFunction(() => location.hash.includes('step=review'), null, {timeout: 600000}); await settle(p);
    const n0 = await p.$$eval('.garment', g => g.length);
    ok('detection lands on Review (never straight to results)', (await heading(p)) === 'Is this right?' && n0 > 0 && await focusedHeading(p), `${n0} pieces`);
    ok('review shows numbered overlays on the photo', (await p.$$('.review__box')).length > 0, '');
    await imagesReady(p); await p.screenshot({path: `${SHOTS}/look_review.png`});
    await p.goto(BASE + '#/stylist/look?step=edit'); await settle(p);
    ok('deep link past an unfinished step falls back to the furthest valid step', /step=review/.test(p.url()), p.url());
    // correct the detection: recolour the first piece, remove the last, add a missed belt
    await p.fill('.garment:first-child [data-field="colour"]', 'navy'); await p.press('.garment:first-child [data-field="colour"]', 'Tab');
    await p.click('.garment:last-child [data-remove]');
    await p.click('#garment-add');
    await p.selectOption('.garment:last-child [data-field="category"]', 'belt');
    const edited = await p.$$eval('.garment', g => g.map(x => [x.querySelector('[data-field=category]').value, x.querySelector('[data-field=colour]').value]));
    await p.reload(); await settle(p);
    const restored = await p.$$eval('.garment', g => g.map(x => [x.querySelector('[data-field=category]').value, x.querySelector('[data-field=colour]').value]));
    ok('refresh restores the step and the corrections', JSON.stringify(edited) === JSON.stringify(restored) && edited[0][1] === 'navy' && edited.at(-1)[0] === 'belt', JSON.stringify(restored));
    await p.click('#review-next'); await settle(p);
    ok('step 3: direction choices, one dominant action', (await heading(p)) === 'What would you like?' && (await p.$$('.choice')).length === 3 && (await oneCta(p)) === 1, '');
    await p.check('input[name=goal][value=occasion]');
    await p.click('#dir-next');
    ok('occasion requires minimal context', /occasion/i.test(await p.$eval('#dir-err', e => e.textContent)), '');
    await p.click('[data-occ="a wedding"]');
    await p.click('#dir-next');
    await p.waitForFunction(() => location.hash.includes('step=edit'), null, {timeout: 600000}); await settle(p);
    await p.waitForSelector('.look-edit__picks .card', {timeout: 60000}).catch(() => {});
    const sent = JSON.parse((editPayload || '').match(/name="garments"\r\n\r\n([^\r]*)/)?.[1] || '[]');
    ok('edit request carries exactly the confirmed pieces', sent.length === edited.length && sent[0].colour === 'navy' && sent.at(-1).category === 'belt' && /name="goal"\r\n\r\noccasion/.test(editPayload), JSON.stringify(sent));
    const a = await auditCards(p, '.look-edit__picks');
    ok('step 4: 4–6 priority picks with price, swatches and black Add to cart, one family each', a.n >= 4 && a.n <= 6 && a.bad.length === 0 && a.dupFamilies === 0, JSON.stringify(a));
    ok('alternatives sit behind a quiet control', !(await p.$eval('.look-edit__more', d => d.open).catch(() => false)), '');
    ok('pick reasons are consumer phrases', await p.$$eval('.look-edit__picks .card__note', n => n.length > 0 && n.every(x => x.textContent.length > 5)), '');
    ok('step 4 has exactly one dominant action (Refine this edit)', (await oneCta(p)) === 1 && /Refine this edit/.test(await p.$eval('.task__actions .btn--solid', e => e.textContent)), '');
    await imagesReady(p); await p.screenshot({path: `${SHOTS}/look_edit.png`});
    const t = await shopperText(p);
    ok('no technical copy anywhere in the flow', !bannedIn(t).length, bannedIn(t).join(','));
    await p.click('.task__actions .btn--solid'); await settle(p);
    ok('Refine opens a dedicated conversation preloaded with the look', /context=look/.test(p.url()) && !!(await p.$('.context-card img')) && (await heading(p)) === 'Refine your edit', p.url());
    let askText = null;
    // Record the outgoing message text in the page (network capture of FormData bodies is unreliable).
    await p.evaluate(() => {
      const orig = window.fetch; window.__sent = [];
      window.fetch = (u, o) => { if (/\/api\/assistant\/[^/]+\/message$/.test(String(u)) && o?.body instanceof FormData) window.__sent.push(o.body.get('text')); return orig(u, o); };
    });
    await p.fill('#chat-in', 'Make it more formal'); await p.keyboard.press('Enter');
    await p.waitForSelector('#pending', {state: 'detached', timeout: 600000});
    askText = (await p.evaluate(() => window.__sent))[0] || null;
    ok('the refinement sends the look context with the first message', /I'm refining an outfit/.test(askText || '') && /Make it more formal/.test(askText || ''), (askText || 'no body captured').slice(0, 160).replace(/\s+/g, ' '));
    ok('refinement answer arrives', !/couldn't answer/.test(await p.$$eval('.msg--bot', m => m.at(-1).innerText)), '');
    await imagesReady(p); await p.screenshot({path: `${SHOTS}/look_refine.png`});
    await p.goto(BASE + '#/stylist/look?step=edit'); await settle(p);
    await p.click('[data-start-over]'); await settle(p);
    ok('Start over clears the staged state', /step=photo/.test(p.url()) && !!(await p.$('.dropzone')) && !(await p.evaluate(() => sessionStorage.getItem('ensemble.look.v1'))), p.url());
    clean(p, 'look');
    await ctx.close();
  }

  if (want('find')) {
    const {ctx, p} = await newPage(b);
    // Detection is stubbed so the check is deterministic; the matching below is the real visual search.
    await p.route('**/api/stylist/detect', r => r.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify({categories: [],
      garments: [{id: 'g1', category: 'top', colour: 'beige', description: 'oversized sweatshirt', box: [0.1, 0.2, 0.9, 0.55]},
                 {id: 'g2', category: 'bottom', colour: 'blue', description: 'wide-leg jeans', box: [0.25, 0.5, 0.75, 0.98]},
                 {id: 'g3', category: 'shoes', colour: 'black', description: 'flats', box: null}]})}));
    const searches = []; p.on('request', r => { if (r.url().endsWith('/api/visual-search')) searches.push(r.postData() || ''); });
    await p.goto(BASE + '#/stylist/find'); await settle(p);
    await p.evaluate(() => sessionStorage.clear()); await p.reload(); await settle(p);
    await p.setInputFiles('.dropzone input', PHOTO); await p.waitForSelector('.photo-preview img');
    await p.click('#find-next'); await p.waitForSelector('.piece'); await settle(p);
    ok('choose step lists the spotted pieces, none chosen yet, no radio buttons or category dropdown up front',
       (await heading(p)) === 'Which piece?' && (await p.$$('.piece')).length === 3 && !(await p.$('.piece.is-on')) && !(await p.$('input[type=radio]')) && !(await p.$('[data-kind]'))
       && await p.$eval('#find-go', b => b.disabled), '');
    await p.click('.picker__box[data-pid=g1]');
    await p.locator('.piece[data-pid=g2] .piece__pick').click();
    const on = await p.$$eval('.piece.is-on', rs => rs.map(r => ({id: r.dataset.pid, kind: r.querySelector('[data-kind]')?.value})));
    ok('tapping a piece on the photo or in the list chooses it and shows what it is', JSON.stringify(on) === '[{"id":"g1","kind":"top"},{"id":"g2","kind":"bottom"}]', JSON.stringify(on));
    ok('focus stays on the tapped row', await p.evaluate(() => document.activeElement?.closest('.piece')?.dataset.pid === 'g2'), '');
    await p.selectOption('.piece[data-pid=g1] [data-kind]', 'knitwear');
    ok('the shopper can correct what a piece is', (await p.$eval('.piece[data-pid=g1] .piece__name', e => e.textContent)) === 'beige Knitwear' || /knitwear/i.test(await p.$eval('.piece[data-pid=g1] .piece__name', e => e.textContent)), '');
    await p.locator('.piece[data-pid=g2] .piece__pick').click();
    ok('tapping again un-chooses it', !(await p.$('.piece[data-pid=g2].is-on')), '');
    const fr = await (await p.$('#picker img')).boundingBox();
    await p.mouse.move(fr.x + fr.width * .3, fr.y + fr.height * .6); await p.mouse.down();
    await p.mouse.move(fr.x + fr.width * .7, fr.y + fr.height * .9, {steps: 6}); await p.mouse.up();
    const added = await p.$$eval('.piece.is-on', rs => rs.map(r => r.querySelector('[data-kind]')?.value));
    ok('dragging frames a new piece and its category follows the spotted piece it covers', added.length === 2 && added[1] === 'bottom', JSON.stringify(added));
    await p.click('.piece:last-child [data-remove]');
    ok('a framed piece can be removed', (await p.$$('.piece')).length === 3, '');
    await p.click('#find-go');
    await p.waitForFunction(() => location.hash.includes('step=results'), null, {timeout: 300000}); await settle(p);
    const counts = await p.$$eval('.find-group', g => g.map(x => x.querySelectorAll('.card').length));
    ok('one search per chosen piece with the corrected category', searches.length === 1 && /knitwear/.test(searches[0]), String(searches.length));
    ok('the four closest per chosen piece', counts.length === 1 && counts[0] === 4, JSON.stringify(counts));
    const a = await auditCards(p, '.task__body');
    ok('real visually similar results with prices and cart', a.n === 4 && a.bad.length === 0 && a.dupFamilies === 0, JSON.stringify(a));
    ok('stylist refinement offered as the one next action', (await oneCta(p)) === 1 && /\/stylist\/ask\?anchor=\d+/.test(await p.$eval('.task__actions .btn--solid', e => e.getAttribute('href'))), '');
    await imagesReady(p); await p.screenshot({path: `${SHOTS}/find_results.png`});
    clean(p, 'find');
    await ctx.close();
  }

  if (want('ask') || want('recovery')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE + '#/stylist/ask'); await settle(p);
    ok('ask: starter prompts and one dominant action', (await p.$$('.suggestion')).length === 3 && (await p.$$('#composer .btn--solid')).length === 1, '');
    await p.click('.suggestion');
    await p.waitForSelector('#pending', {state: 'detached', timeout: 600000});
    const a1 = await auditCards(p, '.msg--bot:last-child');
    ok('grounded answer with consistent product cards', a1.bad.length === 0 && a1.dupFamilies === 0, JSON.stringify(a1));
    // Recovery: the server forgets the session once (as after a Modal restart).
    let forced = false, resent = null;
    await p.route(/\/api\/assistant\/[^/]+\/message$/, async route => {
      if (!forced) { forced = true; return route.fulfill({status: 404, contentType: 'application/json', body: '{"detail":"unknown session"}'}); }
      resent = route.request().postData(); return route.continue();
    });
    await p.fill('#chat-in', 'And a bag to go with it?'); await p.keyboard.press('Enter');
    await p.waitForSelector('#pending', {state: 'detached', timeout: 600000});
    const last = await p.$$eval('.msg--bot', m => m.at(-1).innerText);
    ok('lost session recovers: new session, earlier turns replayed, no raw error', forced && /Earlier in our chat/.test(resent || '') && /And a bag/.test(resent || '')
       && !/couldn't answer|404/.test(last) && /reconnected/.test(await p.$eval('#chatlog', e => e.innerText)), (resent || '').slice(0, 120));
    await imagesReady(p); await p.screenshot({path: `${SHOTS}/ask.png`});
    // The one injected 404 is the controlled lost-session check, not a failure.
    p._failed = p._failed.filter(f => !(f.startsWith('404 POST') && /\/api\/assistant\/[^/]+\/message$/.test(f)));
    p._errors = p._errors.filter(e => !/status of 404/.test(e) || p._failed.length);
    clean(p, 'ask');
    await ctx.close();
  }

  if (want('widths')) {
    const routes = ['#/', '#/collections', '#/stylist', '#/stylist/look', '#/stylist/find', '#/stylist/ask', '#/collections/trending', '#/product/922037001', '#/studio'];
    for (const w of [320, 375, 768, 1024, 1280, 1440]) {
      const {ctx, p} = await newPage(b, [w, 900]);
      const bad = [];
      for (const r of routes) {
        await p.goto(BASE + r); await settle(p);
        const o = await p.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
        if (o > 0) bad.push(`${r}:${o}`);
      }
      ok(`no horizontal overflow at ${w}px on ${routes.length} routes`, !bad.length, bad.join(' '));
      clean(p, `widths ${w}`);
      await ctx.close();
    }
  }

  await b.close();
  const failed = results.filter(r => !r.ok);
  fs.writeFileSync(SHOTS + '/../journeys_r2_result.json', JSON.stringify(results, null, 1));
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
