// Browser journeys for the editorial UI (docs/CLAUDECODE_UI_EDITORIAL_REDESIGN_PLAN.md §19).
// No dependency is added to the repo: point PLAYWRIGHT at any local Playwright install, and run against a
// server started with `make serve PORT=8031`. Uses the installed Google Chrome.
//   PLAYWRIGHT=/path/to/node_modules/playwright node scripts/ui/journeys.js [journey,journey] [base-url]
// Journeys: center home product collections studio label keyboard motion offline assistant visual
// (assistant/visual call the local models and take minutes). The Label save is intercepted, so the
// user's gold-label file is never written; impressions/clicks are logged to the demo events table as in normal use.
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const fs = require('fs');
const BASE = process.argv[3] || 'http://localhost:8031/';
const ONLY = (process.argv[2] || '').split(',').filter(Boolean);
const SHOTS = process.env.SHOTS || 'data/interim/ui_editorial_screenshots/journeys';
fs.mkdirSync(SHOTS, {recursive: true});
const PHOTO = process.env.PHOTO || 'data/raw/outfit_photos/IMG_1503.jpg';

const results = [];
const ok = (name, cond, detail = '') => { results.push({name, ok: !!cond, detail}); console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${detail ? ' — ' + detail : ''}`); };

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
const settle = async p => { await p.waitForLoadState('networkidle'); await p.waitForTimeout(400); };
const want = n => !ONLY.length || ONLY.includes(n);

(async () => {
  const b = await chromium.launch({channel: 'chrome'});

  if (want('center')) for (const vp of [[1440, 900], [1024, 768], [390, 844]]) {
    const {ctx, p} = await newPage(b, vp);
    for (const r of ['#/', '#/studio']) {
      await p.goto(BASE + r); await settle(p);
      const m = await p.evaluate(() => {
        const w = document.querySelector('#wordmark').getBoundingClientRect();
        const inner = document.querySelector('.masthead__inner').getBoundingClientRect();
        return {center: w.left + w.width / 2, vw: document.documentElement.clientWidth, innerCenter: inner.left + inner.width / 2,
                text: document.querySelector('#wordmark').textContent, ff: getComputedStyle(document.querySelector('#wordmark')).fontFamily,
                overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth};
      });
      ok(`wordmark centered ${vp[0]} ${r}`, Math.abs(m.center - m.vw / 2) < 0.75 && m.text === 'ensemble' && m.overflow === 0,
         `center ${m.center.toFixed(2)} vs ${(m.vw / 2).toFixed(2)}, overflow ${m.overflow}`);
    }
    ok(`no console errors (center ${vp[0]})`, !p._errors.length, p._errors.join(' | '));
    await ctx.close();
  }

  if (want('home')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE); await settle(p);
    const titles = await p.$$eval('main h2.section-title', hs => hs.map(h => h.textContent.trim()));
    ok('home modules rendered', titles.includes('Selected for you') && titles.includes('Buy it again') && titles.includes('Trending this week'), titles.join(' / '));
    ok('home Complete the look module (live CTL)', titles.some(t => t.startsWith('Built around')), '');
    const tiles = await p.$$eval('main .tile', t => t.length);
    const imps = p._events.filter(e => e.event === 'impression');
    ok('impressions logged for rendered recommendations', imps.length >= 12 + 12 + 24, `${imps.length} impressions, ${tiles} tiles`);
    ok('impressions carry the selected customer', imps.every(e => e.customer_idx === 168058), '');
    const exp = await p.$eval('[data-exp="shop"]', a => a.getAttribute('aria-current'));
    const prof = await p.$eval('#profile-btn', bt => bt.textContent.replace(/\s+/g, ' ').trim());
    ok('experience switch and profile are separate controls', exp === 'page' && /Shopping as/.test(prof), prof);
    await p.screenshot({path: `${SHOTS}/home.png`});

    // Why this (SHAP) from a home tile, with Escape + focus return
    const why = await p.$('.hero__lead [data-why]');
    await why.focus(); await p.keyboard.press('Enter');
    await p.waitForSelector('#why-dialog[open] .shap', {timeout: 10000, state: 'attached'});
    ok('Why this dialog shows reasons and SHAP', await p.$eval('#why-body', e => e.querySelectorAll('.reason-list li').length > 0 && !!e.querySelector('details')), '');
    await p.screenshot({path: `${SHOTS}/why.png`});
    await p.keyboard.press('Escape'); await p.waitForTimeout(300);
    ok('dialog closes on Escape and focus returns', await p.evaluate(() => !document.querySelector('#why-dialog').open && document.activeElement?.dataset?.why != null), '');

    // Not for me
    const nfm = await p.$('.hero__second [data-nfm]');
    await nfm.click(); await p.waitForTimeout(300);
    const st = await p.$eval('.hero__second .tile', t => ({dis: t.classList.contains('is-dismissed'), status: t.querySelector('.tile__status').textContent, btn: t.querySelector('[data-nfm]').disabled}));
    ok('Not for me logs, dims and explains', st.dis && st.btn && /noted/i.test(st.status) && p._events.some(e => e.event === 'not_for_me'), JSON.stringify(st));

    // Customer switch via profile dialog
    await p.click('#profile-btn');
    await p.fill('#profile-filter', '477553');
    await p.click('.profile-option[data-id="477553"]'); await settle(p);
    const after = await p.evaluate(() => ({hero: document.querySelector('.hero .lede')?.textContent, again: !!document.querySelector('#s-again'),
      prof: document.querySelector('#profile-v').textContent, stored: localStorage.getItem('ensemble.profile')}));
    ok('switching profile updates content (new customer: no Buy it again)', /No purchase history/.test(after.hero) && !after.again && after.prof === 'New shopper, 56', JSON.stringify(after));
    const lastImps = p._events.filter(e => e.event === 'impression').slice(-12);
    ok('no stale cross-customer impressions after switch', lastImps.every(e => e.customer_idx === 477553), '');
    await p.reload(); await settle(p);
    ok('profile persists across reload', (await p.$eval('#profile-v', e => e.textContent)) === 'New shopper, 56', '');
    // restore default for later journeys
    await p.evaluate(() => localStorage.setItem('ensemble.profile', '168058'));
    ok('no console errors (home)', !p._errors.length, p._errors.join(' | '));
    ok('no failed requests (home)', !p._failed.length, p._failed.join(' | '));
    await ctx.close();
  }

  if (want('product')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE); await settle(p);
    await p.click('.hero__lead .tile__name a'); await settle(p);
    ok('product deep link from tile', /#\/product\/922037001/.test(p.url()), p.url());
    ok('click events logged (tile + product page)', p._events.some(e => e.event === 'click' && e.surface === 'for_you') && p._events.some(e => e.event === 'click' && e.surface === 'product_page'), '');
    const secs = await p.$$eval('main h2.section-title', hs => hs.map(h => h.textContent.trim()));
    ok('product sections: Complete the look, Other colours, Similar items', ['Complete the look', 'Other colours', 'Similar items'].every(s => secs.includes(s)), secs.join(' / '));
    ok('live CTL modules rendered', await p.$$eval('#pdp-look .look__slot', s => s.length) > 0, '');
    await p.click('text=Why this is in your edit');
    await p.waitForSelector('#why-dialog[open] .shap', {timeout: 10000, state: 'attached'});
    await p.keyboard.press('Escape');
    await p.click('#add-bag'); await p.waitForTimeout(200);
    ok('Add to bag is labelled demo and logs add_to_cart', p._events.some(e => e.event === 'add_to_cart') && /Demo only/.test(await p.$eval('#bag-status', e => e.textContent)), '');
    await p.screenshot({path: `${SHOTS}/product.png`, fullPage: false});
    const ctlLink = await p.$('#pdp-look .tile__name a');
    const ctlId = await ctlLink.getAttribute('data-open');
    await ctlLink.click(); await settle(p);
    ok('Complete the look item opens its product page', p.url().endsWith('/product/' + ctlId), p.url());
    ok('CTL click carries the anchor', p._events.some(e => e.event === 'click' && e.surface === 'complete_the_look' && e.anchor === 922037001), '');
    await p.click('[data-back]'); await settle(p);
    ok('Back returns to previous product', /\/product\/922037001$/.test(p.url()), p.url());
    const oc = await p.$('section[aria-labelledby="s-col"] .tile__name a');
    if (oc) { await oc.click(); await settle(p); ok('Other colours navigation', /\/product\/\d+$/.test(p.url()), p.url()); await p.goBack(); await settle(p); }
    const sim = await p.$('section[aria-labelledby="s-sim"] .tile__name a');
    if (sim) { await sim.click(); await settle(p); ok('Similar items navigation', /\/product\/\d+$/.test(p.url()), p.url()); }
    await p.goto(BASE + '#/ds'); await settle(p);
    ok('legacy #/ds deep link opens DS Studio', (await p.$eval('h1', h => h.textContent)) === 'Offline evaluation', '');
    await p.goto(BASE + '#/nope'); await settle(p);
    ok('unknown route shows not-found state', /does not exist/.test(await p.$eval('main', m => m.textContent)), '');
    ok('no console errors (product)', !p._errors.length, p._errors.join(' | '));
    ok('no failed requests (product)', !p._failed.length, p._failed.join(' | '));
    await ctx.close();
  }

  if (want('collections')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE + '#/collections/trending'); await settle(p);
    const n0 = await p.$$eval('main .grid > li:not([hidden])', l => l.length);
    await p.click('.filter[data-type]:nth-of-type(2)'); await p.waitForTimeout(200);
    const n1 = await p.$$eval('main .grid > li:not([hidden])', l => l.length);
    ok('collection filter narrows the grid', n0 === 24 && n1 > 0 && n1 < 24, `${n0} -> ${n1}`);
    ok('no console errors (collections)', !p._errors.length, p._errors.join(' | '));
    await ctx.close();
  }

  if (want('assistant')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE + '#/assistant'); await settle(p);
    await p.fill('#chat-in', 'What shoes go with wide-leg trousers?');
    await p.keyboard.press('Enter');
    ok('thinking state shown', !!(await p.waitForSelector('#pending .thinking', {timeout: 5000})), '');
    await p.screenshot({path: `${SHOTS}/assistant_thinking.png`});
    const t0 = Date.now();
    await p.waitForSelector('#pending', {state: 'detached', timeout: 600000});
    const ans = await p.$$eval('.msg--bot', m => m.at(-1).innerText);
    ok('assistant replies with grounded output', !/couldn't answer/.test(ans) && ans.length > 20, `${Math.round((Date.now() - t0) / 1000)} s; ${ans.slice(0, 160).replace(/\n/g, ' ')}`);
    ok('tool trace rendered', !!(await p.$('.msg--bot .msg__trace .tag')), '');
    await p.screenshot({path: `${SHOTS}/assistant_answer.png`});
    // session survives navigation
    await p.click('text=Discover'); await settle(p);
    await p.click('.primary-nav >> text=Style Assistant'); await settle(p);
    ok('conversation kept for this profile across navigation', (await p.$$eval('.msg--user', m => m.length)) === 1, '');
    ok('no console errors (assistant)', !p._errors.length, p._errors.join(' | '));
    ok('no failed requests (assistant)', !p._failed.length, p._failed.join(' | '));
    await ctx.close();
  }

  if (want('visual')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE + '#/assistant'); await settle(p);
    ok('photo services disabled before a photo is chosen', await p.$eval('#vs-go', e => e.disabled) && await p.$eval('#snap-go', e => e.disabled), '');
    await p.setInputFiles('#photo-in', PHOTO); await p.waitForTimeout(400);
    ok('preview with replace/remove controls', !!(await p.$('#frame img')) && !!(await p.$('#photo-remove')), '');
    const fr = await (await p.$('#frame img')).boundingBox();
    await p.mouse.move(fr.x + fr.width * .25, fr.y + fr.height * .3); await p.mouse.down();
    await p.mouse.move(fr.x + fr.width * .75, fr.y + fr.height * .65, {steps: 6}); await p.mouse.up();
    ok('drag frames a region', /Framed/.test(await p.$eval('#box-state', e => e.textContent)), '');
    await p.selectOption('#vs-cat', 'top');
    await p.click('#vs-go');
    await p.waitForSelector('#photo-results .grid, #photo-results .state', {timeout: 300000});
    const vs = await p.$$eval('#photo-results .tile', t => t.length);
    ok('visual search returns catalogue tiles', vs > 0, `${vs} matches`);
    await p.screenshot({path: `${SHOTS}/visual_search.png`});
    await p.click('#snap-go');
    await p.waitForSelector('#photo-results .notice, #photo-results .state', {timeout: 600000});
    const snapText = await p.$eval('#photo-results', e => e.innerText.slice(0, 200).replace(/\n/g, ' '));
    ok('outfit analysis returns detected garments / suggestions', /Detected:/.test(snapText), snapText);
    await p.screenshot({path: `${SHOTS}/outfit.png`});
    ok('no console errors (visual)', !p._errors.length, p._errors.join(' | '));
    ok('no failed requests (visual)', !p._failed.length, p._failed.join(' | '));
    await ctx.close();
  }

  if (want('studio')) {
    const {ctx, p} = await newPage(b);
    for (const [r, h] of [['#/studio', 'Offline evaluation'], ['#/studio/serving', 'Serving status'], ['#/studio/inspect', 'Inspect a recommendation']]) {
      await p.goto(BASE + r); await settle(p);
      const d = await p.evaluate(() => ({h: document.querySelector('main h1')?.textContent, tables: document.querySelectorAll('table.data').length,
        rows: document.querySelectorAll('table.data tbody tr').length, cur: document.querySelector('[data-exp="studio"]').getAttribute('aria-current')}));
      ok(`DS Studio ${r}`, d.h === h && d.cur === 'page' && (r.endsWith('inspect') || d.rows > 0), JSON.stringify(d));
      await p.screenshot({path: `${SHOTS}/studio_${r.split('/').pop() || 'eval'}.png`});
    }
    await p.goto(BASE + '#/studio/inspect'); await settle(p);
    await p.click('.inspect-item[data-i="2"]'); await p.waitForSelector('#inspect-detail .shap');
    ok('Inspect shows SHAP for a chosen recommendation', true, '');
    ok('no console errors (studio)', !p._errors.length, p._errors.join(' | '));
    ok('no failed requests (studio)', !p._failed.length, p._failed.join(' | '));
    await ctx.close();
  }

  if (want('label')) {
    const {ctx, p} = await newPage(b);
    let posted = null;
    // Intercept the save so this check never writes into the user's gold-label file.
    await p.route('**/api/label', route => { posted = JSON.parse(route.request().postData()); route.fulfill({status: 200, contentType: 'application/json', body: '{"ok":true}'}); });
    // All real tasks are already labelled; mark the first one pending in the response so the workflow can be exercised.
    await p.route('**/api/label/tasks**', async route => { const r = await route.fetch(); const j = await r.json();
      if (j.length) j[0].done = false; route.fulfill({response: r, json: j}); });
    await p.goto(BASE + '#/label'); await settle(p);
    const before = await p.$eval('main h1', h => h.textContent);
    if (/done/.test(before)) { ok('Label workflow (all tasks done)', true, before); }
    else {
      await p.click('.label-card[data-i="0"]'); await p.click('.label-card[data-i="2"]');
      const st = await p.$eval('.label-card[data-i="0"]', e => [e.getAttribute('aria-pressed'), e.innerText]);
      await p.click('#labsave'); await settle(p);
      ok('Label toggles (state not by colour alone) and saves', st[0] === 'true' && /relevant/i.test(st[1]) && posted && posted.relevant[0] === 1 && posted.relevant[2] === 1 && posted.relevant[1] === 0, JSON.stringify(posted));
    }
    await p.screenshot({path: `${SHOTS}/label.png`});
    ok('no console errors (label)', !p._errors.length, p._errors.join(' | '));
    await ctx.close();
  }

  if (want('keyboard')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE); await settle(p);
    await p.keyboard.press('Tab');
    ok('first Tab reaches the skip link', (await p.evaluate(() => document.activeElement.textContent)) === 'Skip to content', '');
    await p.keyboard.press('Enter');
    ok('skip link moves focus to main', (await p.evaluate(() => document.activeElement.id)) === 'main', '');
    await p.evaluate(() => document.querySelector('[data-exp="shop"]').focus());
    await p.keyboard.press('Tab');
    ok('experience switch is keyboard reachable', (await p.evaluate(() => document.activeElement.dataset.exp)) === 'studio', '');
    await p.keyboard.press('Enter'); await settle(p);
    ok('Enter on DS Studio switches experience', /#\/studio$/.test(p.url()), p.url());
    const focusVisible = await p.evaluate(() => getComputedStyle(document.activeElement).boxShadow);
    await p.evaluate(() => document.querySelector('#profile-btn').focus());
    await p.keyboard.press('Enter'); await p.waitForTimeout(200);
    ok('profile dialog opens from keyboard with focus inside', await p.evaluate(() => document.querySelector('#profile-dialog').open && document.querySelector('#profile-dialog').contains(document.activeElement)), '');
    await p.keyboard.press('Escape'); await p.waitForTimeout(200);
    ok('Escape closes profile dialog and returns focus', (await p.evaluate(() => document.activeElement.id)) === 'profile-btn', '');
    await ctx.close();
    const m = await newPage(b, [390, 844]);
    await m.p.goto(BASE); await settle(m.p);
    await m.p.evaluate(() => document.querySelector('#menu-btn').focus());
    await m.p.keyboard.press('Enter'); await m.p.waitForTimeout(250);
    ok('mobile drawer opens from keyboard', await m.p.evaluate(() => document.querySelector('#drawer').open && document.querySelector('#menu-btn').getAttribute('aria-expanded') === 'true'), '');
    await m.p.screenshot({path: `${SHOTS}/drawer_mobile.png`});
    await m.p.keyboard.press('Escape'); await m.p.waitForTimeout(250);
    ok('drawer closes on Escape, focus back on menu button', (await m.p.evaluate(() => document.activeElement.id)) === 'menu-btn', '');
    await m.p.click('#menu-btn'); await m.p.click('#drawer >> text=Style Assistant'); await settle(m.p);
    ok('drawer link navigates and closes', /assistant/.test(m.p.url()) && !(await m.p.evaluate(() => document.querySelector('#drawer').open)), '');
    ok('no console errors (keyboard)', !m.p._errors.length, m.p._errors.join(' | '));
    await m.ctx.close();
  }

  if (want('motion')) {
    for (const rm of ['reduce', 'no-preference']) {
      const {ctx, p} = await newPage(b, [1440, 900], {reducedMotion: rm});
      await p.goto(BASE); await settle(p);
      const d = await p.evaluate(() => ({motion: document.documentElement.classList.contains('motion'),
        hidden: [...document.querySelectorAll('.reveal')].filter(e => getComputedStyle(e).opacity === '0').length,
        anim: getComputedStyle(document.querySelector('.view')).animationName}));
      if (rm === 'reduce') ok('reduced motion: no reveal hiding, no view animation', !d.motion && d.hidden === 0 && d.anim === 'none', JSON.stringify(d));
      else ok('motion: reveal + view transition active', d.motion && d.anim !== 'none', JSON.stringify(d));
      await ctx.close();
    }
  }

  if (want('offline')) {
    const {ctx, p} = await newPage(b);
    await p.goto(BASE); await settle(p);
    await p.route('**/api/product/**', r => r.abort('failed'));
    await p.click('.hero__lead .tile__name a'); await p.waitForTimeout(800);
    const txt = await p.$eval('main', m => m.innerText);
    ok('backend-unavailable: friendly error with retry, nav intact', /couldn't reach/.test(txt) && !!(await p.$('[data-retry]')) && !!(await p.$('#primary-nav a')), txt.slice(0, 120).replace(/\n/g, ' '));
    await p.unroute('**/api/product/**');
    await p.click('[data-retry]'); await settle(p);
    ok('retry recovers', !!(await p.$('.pdp')), '');
    await p.route('**/api/v2/complete-the-look**', r => r.fulfill({status: 503, contentType: 'application/json', body: '{"error":{"code":"not_ready","message":"bundle not loaded"}}'}));
    await p.reload(); await settle(p);
    ok('degraded live CTL falls back to stored suggestions with a notice', /Showing stored suggestions/.test(await p.$eval('#pdp-look', e => e.innerText)), '');
    await p.screenshot({path: `${SHOTS}/degraded.png`});
    await ctx.close();
  }

  await b.close();
  const failed = results.filter(r => !r.ok);
  fs.writeFileSync(SHOTS + '/../journeys_result.json', JSON.stringify(results, null, 1));
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
