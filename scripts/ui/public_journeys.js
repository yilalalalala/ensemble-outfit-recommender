// Browser journeys for the PUBLISHED build (docs/CLAUDECODE_PUBLIC_DEMO_FULL_UI_PLAN.md §5).
// The site is served under a GitHub project subpath, so these runs prove the build works at
// https://<user>.github.io/<repo>/ and not only at a server root. No dependency is added to the repo.
//
//   python -m ensemble.api.public_demo build
//   mkdir -p /tmp/site && ln -sfn "$PWD/portfolio" /tmp/site/ensemble-outfit-recommender
//   .venv/bin/python -m http.server 8041 --directory /tmp/site
//   PLAYWRIGHT=/path/to/node_modules/playwright node scripts/ui/public_journeys.js \
//     [journey,...] [http://localhost:8041/ensemble-outfit-recommender/]
//
// Journeys: routes shop product assistant studio cart keyboard motion offline
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const fs = require('fs');
const BASE = (process.argv[3] || 'http://localhost:8041/ensemble-outfit-recommender/').replace(/\/?$/, '/');
const ONLY = (process.argv[2] || '').split(',').filter(Boolean);
const SHOTS = process.env.SHOTS || 'data/interim/public_demo_screenshots';
fs.mkdirSync(SHOTS, { recursive: true });
const ORIGIN = new URL(BASE).origin;
const SITE = new URL(BASE).pathname;          // e.g. /ensemble-outfit-recommender/

// The canonical DS Studio overview values. These must appear exactly, unrounded and unrecomputed.
const HEADLINE = { map_test: '0.03918', lift_test: '+45.7%', map_val: '0.03772', track_b_models: '10' };

const results = [];
const ok = (name, cond, detail = '') => { results.push({ name, ok: !!cond, detail }); console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${detail ? ' — ' + detail : ''}`); };
const want = n => !ONLY.length || ONLY.includes(n);

async function newPage(b, vp = [1440, 900], opts = {}) {
  const ctx = await b.newContext({ viewport: { width: vp[0], height: vp[1] }, ...opts });
  const p = await ctx.newPage();
  p._errors = []; p._failed = []; p._offsite = []; p._badImages = [];
  p.on('pageerror', e => p._errors.push('pageerror: ' + e.message));
  p.on('console', m => { if (m.type() === 'error') p._errors.push('console: ' + m.text()); });
  p.on('requestfailed', r => p._failed.push(`${r.method()} ${r.url()} ${r.failure()?.errorText}`));
  p.on('response', r => { if (r.status() >= 400) p._failed.push(`${r.status()} ${r.request().method()} ${r.url()}`); });
  // Nothing may leave the published site: no /api, no backend, no third party.
  p.on('request', r => {
    const u = new URL(r.url());
    if (u.origin !== ORIGIN || !u.pathname.startsWith(SITE)) p._offsite.push(r.url());
    if (/\/api\/|\/readyz/.test(u.pathname.slice(SITE.length - 1))) p._offsite.push(r.url());
  });
  return { ctx, p };
}
const settle = async p => { await p.waitForLoadState('networkidle'); await p.waitForTimeout(350); };
const clean = (p, tag) => {
  ok(`no console errors (${tag})`, !p._errors.length, p._errors.join(' | '));
  ok(`no failed requests (${tag})`, !p._failed.length, p._failed.join(' | '));
  ok(`no request leaves the published site (${tag})`, !p._offsite.length, p._offsite.join(' | '));
};
const overflow = p => p.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
// Every <img> on the page actually decodes. Cards below the fold are loading="lazy", so the page is
// scrolled first and the images are then given a chance to settle: a published build must show no
// broken photography and must never fall back to the placeholder.
async function brokenImages(p) {
  await p.evaluate(async () => {
    for (let y = 0; y < document.body.scrollHeight; y += window.innerHeight) { window.scrollTo(0, y); await new Promise(r => setTimeout(r, 120)); }
    window.scrollTo(0, 0);
  });
  await p.waitForLoadState('networkidle');
  await p.waitForTimeout(400);
  return p.evaluate(() => [...document.images]
    .filter(i => i.offsetParent !== null && i.getAttribute('src'))
    .filter(i => !i.complete || i.naturalWidth === 0 || i.src.endsWith('placeholder.svg'))
    .map(i => i.getAttribute('src')));
}

async function auditCards(p, scope = 'main') {
  await p.mouse.move(1, 1); await p.waitForTimeout(320);
  return p.evaluate(scope => {
    const cards = [...document.querySelectorAll(`${scope} [data-card].card`)].filter(c => c.offsetParent !== null);
    const fams = cards.map(c => c.dataset.family || 'a:' + c.dataset.article);
    const bad = cards.filter(c => {
      const btn = c.querySelector('.btn-cart'), cs = btn && getComputedStyle(btn);
      const price = c.querySelector('.card__price')?.textContent;
      return !btn || btn.textContent.trim() !== 'Add to cart' || cs.backgroundColor !== 'rgb(0, 0, 0)'
        || !/^\$\d+\.\d{2}$/.test(price || '') || !c.querySelector('.swatch[aria-pressed="true"]');
    }).map(c => c.dataset.article);
    return { n: cards.length, dupFamilies: fams.length - new Set(fams).size, bad,
             emptyFamily: cards.filter(c => !c.dataset.family).length };
  }, scope);
}

const ROUTES = ['#/', '#/collections', '#/collections/trending', '#/collections/buy_again', '#/assistant',
                '#/studio', '#/studio/serving', '#/studio/inspect', '#/label', '#/ds', '#/nope'];

(async () => {
  const b = await chromium.launch({ channel: 'chrome' });

  if (want('routes')) for (const vp of [[1440, 900], [1024, 768], [390, 844]]) {
    const { ctx, p } = await newPage(b, vp);
    for (const r of ROUTES) {
      await p.goto(BASE + r); await settle(p);
      const d = await p.evaluate(() => ({
        h1: document.querySelector('main h1, main .state__title')?.textContent?.trim() || '',
        loading: /Loading…|Opening the shop…/.test(document.querySelector('main')?.innerText || ''),
        text: (document.querySelector('main')?.innerText || '').length,
      }));
      const broken = await brokenImages(p), over = await overflow(p);
      const expectEmpty = r === '#/nope';
      ok(`${r} renders at ${vp[0]}`,
         d.h1.length > 0 && !d.loading && d.text > (expectEmpty ? 20 : 120) && over === 0 && !broken.length,
         `h1="${d.h1}" chars=${d.text} overflow=${over} broken=${broken.slice(0, 2).join(',')}`);
    }
    await p.goto(BASE); await settle(p);
    await p.screenshot({ path: `${SHOTS}/discover_${vp[0]}.png`, fullPage: vp[0] === 390 });
    await p.goto(BASE + '#/assistant'); await settle(p);
    await p.screenshot({ path: `${SHOTS}/assistant_${vp[0]}.png`, fullPage: vp[0] === 390 });
    await p.goto(BASE + '#/studio'); await settle(p);
    await p.screenshot({ path: `${SHOTS}/studio_${vp[0]}.png`, fullPage: vp[0] === 390 });
    clean(p, `routes ${vp[0]}`);
    await ctx.close();
  }

  if (want('shop')) {
    const { ctx, p } = await newPage(b);
    await p.goto(BASE); await settle(p);
    const titles = await p.$$eval('main h2.section-title', hs => hs.map(h => h.textContent.trim()));
    ok('Discover shows the real modules', ['Selected for you', 'Buy it again', 'Trending this week'].every(t => titles.includes(t))
       && titles.some(t => t.startsWith('Styled with')), titles.join(' / '));
    const a = await auditCards(p);
    ok('Discover: one card per family, priced, swatched, black Add to cart',
       a.n > 20 && a.dupFamilies === 0 && a.emptyFamily === 0 && a.bad.length === 0, JSON.stringify(a));
    const broken = await brokenImages(p);
    ok('catalogue photography loads', !broken.length, broken.slice(0, 3).join(','));
    // swatch changes the card in place
    const probe = await p.evaluate(() => {
      const c = [...document.querySelectorAll('main [data-card].card')].find(c => c.querySelectorAll('.swatch').length > 1);
      c.id = 'probe'; return c.dataset.article;
    });
    const target = await p.$eval('#probe .swatch[aria-pressed="false"]', s => s.dataset.swatch);
    await p.click(`#probe .swatch[data-swatch="${target}"]`); await p.waitForTimeout(200);
    const after = await p.$eval('#probe', c => ({ a: c.dataset.article, img: c.querySelector('img').getAttribute('src'),
      href: c.querySelector('.card__name a').getAttribute('href'), broken: !c.querySelector('img').naturalWidth }));
    ok('colour swatch updates the card and its photo', after.a === target && after.img === `images/${target}.jpg`
       && after.href === `#/product/${target}` && !after.broken, JSON.stringify(after));
    // "Not for me" is remembered for this profile
    await p.evaluate(() => localStorage.clear());
    await p.reload(); await settle(p);
    const fam = await p.$eval('.hero__second [data-card]', c => c.dataset.family);
    await p.click('.hero__second [data-nfm]'); await p.waitForTimeout(200);
    await p.reload(); await settle(p);
    ok('Not for me is remembered after reload',
       !(await p.$$eval('main [data-card]', (cs, f) => cs.some(c => c.dataset.family === f), fam)), fam);
    await p.evaluate(() => localStorage.clear());
    // profile switch repaints the page from that profile's own fixtures
    await p.reload(); await settle(p);
    const before = await p.$$eval('main [data-card]', c => c.map(x => x.dataset.article));
    await p.click('#profile-btn'); await p.waitForTimeout(150);
    const other = await p.$eval('.profile-option:not([aria-current="true"])', b => b.dataset.id);
    await p.click(`.profile-option[data-id="${other}"]`); await settle(p);
    const now = await p.$$eval('main [data-card]', c => c.map(x => x.dataset.article));
    ok('profile switch loads that profile\'s own recommendations', now.join() !== before.join() && now.length > 10,
       `${before.length} -> ${now.length}`);
    await p.evaluate(() => localStorage.clear());
    // collections tabs and category filter
    await p.goto(BASE + '#/collections/trending'); await settle(p);
    const ca = await auditCards(p);
    ok('Collections renders a filterable grid', ca.n > 0 && ca.dupFamilies === 0 && ca.bad.length === 0, JSON.stringify(ca));
    const filters = await p.$$('.filter[data-type]:not([data-type=""])');
    if (filters.length) {
      await filters[0].click(); await p.waitForTimeout(200);
      const shown = await p.$$eval('main .grid > li:not([hidden])', l => l.length);
      ok('category filter narrows the grid', shown > 0 && shown < ca.n, `${ca.n} -> ${shown}`);
    } else ok('category filter narrows the grid', false, 'no category filters rendered');
    clean(p, 'shop');
    await ctx.close();
  }

  if (want('product')) {
    const { ctx, p } = await newPage(b);
    await p.goto(BASE); await settle(p);
    await p.click('.hero__lead .card__name a'); await settle(p);
    ok('a card opens its product page', /#\/product\/\d+$/.test(p.url()), p.url());
    const secs = await p.$$eval('main h2.section-title', hs => hs.map(h => h.textContent.trim()));
    ok('product page shows Complete the look and Similar items',
       secs.includes('Complete the look') && secs.includes('Similar items'), secs.join(' / '));
    const pa = await auditCards(p);
    ok('product page cards are priced and unique', pa.dupFamilies === 0 && pa.bad.length === 0 && pa.n > 0, JSON.stringify(pa));
    const pbroken = await brokenImages(p);
    ok('no broken photography on the product page', !pbroken.length, pbroken.slice(0, 3).join(','));
    const sw = await p.$$eval('.pdp .swatch', s => s.map(x => x.dataset.swatch));
    if (sw.length > 1) {
      await p.click(`.pdp .swatch[data-swatch="${sw[1]}"]`); await p.waitForTimeout(250);
      const st = await p.evaluate(() => ({ hash: location.hash, img: document.querySelector('.pdp__media img').getAttribute('src'),
        broken: !document.querySelector('.pdp__media img').naturalWidth }));
      ok('product colour change updates photo and address', st.hash === `#/product/${sw[1]}` && st.img === `images/${sw[1]}.jpg` && !st.broken, JSON.stringify(st));
    } else ok('product colour change updates photo and address', true, 'single-colour product');
    // every Complete-the-Look and Similar card leads to a page that exists
    const links = await p.$$eval('main .grid .card__name a', as => as.slice(0, 10).map(a => a.getAttribute('href')));
    let bad = [];
    for (const href of links) {
      await p.goto(BASE + href); await settle(p);
      if (!(await p.$('.pdp__title'))) bad.push(href);
    }
    ok('every reachable product link resolves to a product page', !bad.length, bad.join(','));
    await p.screenshot({ path: `${SHOTS}/product_1440.png` });
    clean(p, 'product');
    await ctx.close();
  }

  if (want('assistant')) {
    const { ctx, p } = await newPage(b);
    await p.goto(BASE + '#/assistant'); await settle(p);
    const intro = await p.$eval('#chatlog', e => e.innerText);
    ok('the saved conversation is labelled as saved', /Saved answers from the local application/.test(intro), intro.slice(0, 80));
    const chips = await p.$$eval('.suggestion', bs => bs.map(b => b.textContent.trim()));
    ok('suggested questions are the saved ones', chips.length >= 2, chips.join(' | '));
    await p.click(`.suggestion`); await p.waitForTimeout(600);
    const msgs = await p.$$eval('.msg', ms => ms.map(m => ({who: m.querySelector('.msg__who')?.textContent, t: m.innerText})));
    ok('a saved question replays its saved answer',
       msgs.length >= 2 && msgs.at(-1).who === 'Stylist' && msgs.at(-1).t.length > 60 && !/pending|Finding pieces/.test(msgs.at(-1).t),
       msgs.at(-1)?.t.slice(0, 90).replace(/\n/g, ' '));
    const ac = await auditCards(p, '.msg--bot:last-child');
    ok('the saved answer shows its products as real cards', ac.n > 0 && ac.bad.length === 0 && ac.dupFamilies === 0, JSON.stringify(ac));
    // an unsaved question is reported, never answered
    await p.fill('#chat-in', 'tell me about kubernetes operators'); await p.keyboard.press('Enter');
    await p.waitForTimeout(600);
    const last = await p.$eval('.msg:last-child', m => m.innerText);
    ok('an unsaved question is reported, not answered', /isn.t among the saved questions/.test(last), last.slice(0, 90).replace(/\n/g, ' '));
    // the follow-up exchange
    await p.goto(BASE + '#/assistant'); await settle(p);
    const texts = await p.$$eval('.suggestion', bs => bs.map(b => b.textContent.trim()));
    await p.fill('#chat-in', 'Recommend me a top.'); await p.keyboard.press('Enter'); await p.waitForTimeout(700);
    const follow = await p.$$eval('.msg--bot:last-child .suggestion', bs => bs.map(b => b.textContent.trim()));
    ok('a saved follow-up is offered after its first turn', follow.length === 1 && /bottoms/.test(follow[0]), follow.join('|'));
    await p.click('.msg--bot:last-child .suggestion'); await p.waitForTimeout(700);
    ok('the follow-up replays its own saved answer', (await p.$$('.msg')).length >= 5, '');
    // the saved photo example
    const photo = await p.evaluate(() => ({ src: document.querySelector('#photo-slot img')?.getAttribute('src'),
      broken: !document.querySelector('#photo-slot img')?.naturalWidth,
      box: !!document.querySelector('#photo-slot .preview__box'),
      note: document.querySelector('#photo-slot .saved-note')?.innerText || '',
      credit: document.querySelector('#photo-slot .caption')?.innerText || '',
      upload: !!document.querySelector('#photo-in'), attach: document.querySelector('#attach-btn')?.hidden }));
    ok('the saved photo is shown, framed, credited, with upload withdrawn',
       photo.src === 'photo.jpg' && !photo.broken && photo.box && /Saved example photo/.test(photo.note)
       && /CC|public domain/i.test(photo.credit) && !photo.upload && photo.attach === true, JSON.stringify(photo));
    await p.click('#vs-go'); await p.waitForSelector('#photo-results .grid, #photo-results .state', { timeout: 20000 });
    const vs = await auditCards(p, '#photo-results');
    ok('visual search shows its saved matches', vs.n > 0 && vs.bad.length === 0, JSON.stringify(vs));
    await p.click('#snap-go'); await p.waitForSelector('#photo-results .notice', { timeout: 20000 });
    const snapText = await p.$eval('#photo-results', e => e.innerText);
    const sn = await auditCards(p, '#photo-results');
    ok('outfit analysis shows its saved garments and gap', /We spotted:/.test(snapText) && sn.n > 0 && sn.bad.length === 0,
       snapText.slice(0, 80).replace(/\n/g, ' '));
    await p.screenshot({ path: `${SHOTS}/assistant_saved_1440.png`, fullPage: true });
    clean(p, 'assistant');
    await ctx.close();
  }

  if (want('studio')) {
    const { ctx, p } = await newPage(b);
    await p.goto(BASE + '#/studio'); await settle(p);
    const kpis = await p.$$eval('.kpi', ks => ks.map(k => [k.querySelector('.kpi__v').textContent.trim(),
                                                           k.querySelector('.kpi__k').textContent.trim()]));
    const by = Object.fromEntries(kpis.map(([v, k]) => [k, v]));
    ok(`DS Studio shows the ranker test MAP@12 exactly ${HEADLINE.map_test}`,
       by['Ranker MAP@12 · test week'] === HEADLINE.map_test, JSON.stringify(by));
    ok(`DS Studio shows the test lift exactly ${HEADLINE.lift_test}`,
       by['Ranker lift vs best baseline · test'] === HEADLINE.lift_test, by['Ranker lift vs best baseline · test']);
    ok(`DS Studio shows the validation MAP@12 exactly ${HEADLINE.map_val}`,
       by['Ranker MAP@12 · validation week'] === HEADLINE.map_val, by['Ranker MAP@12 · validation week']);
    const tbKey = Object.keys(by).find(k => k.startsWith('Track B models compared'));
    ok(`DS Studio compares ${HEADLINE.track_b_models} Track B models on the test week`,
       by[tbKey] === HEADLINE.track_b_models && /test week$/.test(tbKey), `${tbKey} = ${by[tbKey]}`);
    const txt = await p.$eval('main', m => m.innerText);
    ok('results are labelled offline, with no claim of live traffic or A/B evidence',
       /All numbers are offline \(historical data\)/.test(txt) && /online A\/B test/.test(txt), '');
    const rows = await p.$$eval('table.data tbody tr', r => r.length);
    ok('the evaluation tables are populated', rows > 20, String(rows));
    const exact = await p.$eval('.kpi__v[title]', e => e.getAttribute('title'));
    ok('the unrounded value is kept for assistive technology', exact === '0.03917510232956311', exact);

    await p.goto(BASE + '#/studio/serving'); await settle(p);
    const serving = await p.$eval('main', m => m.innerText);
    ok('serving status is honest about being a captured snapshot',
       /captured when the published build was made/.test(serving) && /not public traffic/.test(serving)
       && /captured snapshot/i.test(serving) && /\bReady\b/i.test(serving), '');
    await p.goto(BASE + '#/studio/inspect'); await settle(p);
    await p.click('.inspect-item[data-i="2"]'); await p.waitForSelector('#inspect-detail .shap', { timeout: 10000 });
    const insp = await p.$eval('#inspect-detail', e => e.innerText);
    ok('Inspect shows the stored reasons and SHAP contributions for a pick',
       /evidence-gated reasons/i.test(insp) && /SHAP contributions/i.test(insp), insp.slice(0, 70).replace(/\n/g, ' '));
    const nav = await p.$$eval('#primary-nav a', as => as.map(a => a.textContent.trim()));
    ok('Label is not offered in the published build', !nav.includes('Label'), nav.join('/'));
    await p.goto(BASE + '#/label'); await settle(p);
    const lab = await p.$eval('main', m => m.innerText);
    ok('the Label route explains itself instead of failing', /runs locally only/.test(lab), lab.slice(0, 80).replace(/\n/g, ' '));
    const util = await p.$eval('#utility-note', e => e.innerText);
    ok('the studio utility bar states the provenance', /Stored responses from the local application/.test(util), util);
    await p.screenshot({ path: `${SHOTS}/studio_serving_1440.png` });
    clean(p, 'studio');
    await ctx.close();
  }

  if (want('cart')) {
    const { ctx, p } = await newPage(b);
    await p.goto(BASE); await settle(p);
    await p.evaluate(() => localStorage.clear());
    await p.reload(); await settle(p);
    await p.click('main [data-card].card [data-add]'); await p.waitForSelector('#cart-dialog[open]');
    const line = await p.$eval('#cart-dialog', d => ({ lines: d.querySelectorAll('.cart-line').length,
      subtotal: d.querySelector('#cart-subtotal').textContent, checkout: d.querySelector('.cart__foot .btn-cart').disabled }));
    ok('Add to cart opens the cart with the new line and a subtotal',
       line.lines === 1 && /^\$\d/.test(line.subtotal) && line.checkout === true, JSON.stringify(line));
    await p.keyboard.press('Escape'); await p.waitForTimeout(200);
    await p.reload(); await settle(p);
    ok('the cart persists across a reload', (await p.$eval('#cart-count', e => e.textContent)) === '1', '');
    await p.click('#cart-btn'); await p.waitForSelector('#cart-dialog[open]');
    await p.click('.cart-line [data-remove]'); await p.waitForTimeout(200);
    ok('removing the last line empties the cart', /Your cart is empty/.test(await p.$eval('#cart-body', e => e.innerText)), '');
    await p.evaluate(() => localStorage.clear());
    clean(p, 'cart');
    await ctx.close();
  }

  if (want('keyboard')) {
    const { ctx, p } = await newPage(b);
    await p.goto(BASE); await settle(p);
    await p.keyboard.press('Tab');
    ok('the first Tab reaches the skip link', (await p.evaluate(() => document.activeElement.textContent)) === 'Skip to content', '');
    await p.keyboard.press('Enter');
    ok('the skip link moves focus to main', (await p.evaluate(() => document.activeElement.id)) === 'main', '');
    await p.evaluate(() => document.querySelector('main .swatch[aria-pressed="false"]').focus());
    const id = await p.evaluate(() => document.activeElement.dataset.swatch);
    await p.keyboard.press('Enter'); await p.waitForTimeout(200);
    ok('a colour swatch is operable by keyboard',
       (await p.evaluate(i => document.querySelector(`.swatch[data-swatch="${i}"]`).getAttribute('aria-pressed'), id)) === 'true', '');
    await p.evaluate(() => document.querySelector('#cart-btn').focus());
    await p.keyboard.press('Enter'); await p.waitForTimeout(250);
    ok('the cart opens from the keyboard with focus inside',
       await p.evaluate(() => document.querySelector('#cart-dialog').open && document.querySelector('#cart-dialog').contains(document.activeElement)), '');
    await p.keyboard.press('Escape'); await p.waitForTimeout(250);
    ok('Escape closes the cart and returns focus, with no keyboard trap',
       (await p.evaluate(() => document.activeElement.id)) === 'cart-btn', '');
    // labels that assistive technology reads
    const labels = await p.evaluate(() => ({
      swatches: [...document.querySelectorAll('main .swatch')].every(s => s.getAttribute('aria-label')),
      adds: [...document.querySelectorAll('main [data-add]')].every(b => b.getAttribute('aria-label')),
      imgs: [...document.querySelectorAll('main .card img')].every(i => i.getAttribute('alt') !== null),
      live: !!document.querySelector('#announcer[aria-live="polite"]'),
      main: document.querySelectorAll('main').length === 1 }));
    ok('accessible names are present on cards, swatches and the live region', Object.values(labels).every(Boolean), JSON.stringify(labels));
    await p.evaluate(() => localStorage.clear());
    clean(p, 'keyboard');
    await ctx.close();
  }

  if (want('motion')) for (const rm of ['reduce', 'no-preference']) {
    const { ctx, p } = await newPage(b, [1440, 900], { reducedMotion: rm });
    await p.goto(BASE); await settle(p);
    const d = await p.evaluate(() => ({ motion: document.documentElement.classList.contains('motion'),
      hidden: [...document.querySelectorAll('.reveal')].filter(e => getComputedStyle(e).opacity === '0').length,
      anim: getComputedStyle(document.querySelector('.view')).animationName }));
    if (rm === 'reduce') ok('reduced motion: nothing hidden and no view animation', !d.motion && d.hidden === 0 && d.anim === 'none', JSON.stringify(d));
    else ok('motion: the reveal and view transitions are active', d.motion && d.anim !== 'none', JSON.stringify(d));
    await ctx.close();
  }

  if (want('offline')) {
    // A missing fixture must say so, not hang on a skeleton or show an empty grid.
    const { ctx, p } = await newPage(b);
    await p.route('**/demo/product/*.json', r => r.fulfill({ status: 404, body: 'missing' }));
    await p.goto(BASE + '#/product/' + 922037001); await p.waitForTimeout(900);
    const t = await p.$eval('main', m => m.innerText);
    ok('a missing fixture shows a recoverable message, not a dead page',
       /couldn't find that|didn't load/i.test(t) && !!(await p.$('[data-retry]')) && !!(await p.$('#primary-nav a')),
       t.slice(0, 90).replace(/\n/g, ' '));
    await p.unroute('**/demo/product/*.json');
    await p.click('[data-retry]'); await settle(p);
    ok('retry recovers once the fixture is reachable again', !!(await p.$('.pdp')), '');
    await ctx.close();
  }

  await b.close();
  const failed = results.filter(r => !r.ok);
  fs.writeFileSync(SHOTS + '/public_journeys_result.json', JSON.stringify(results, null, 1));
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
