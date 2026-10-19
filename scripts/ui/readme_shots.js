// README screenshots of the real app (no mock-ups, no generated imagery).
//   PLAYWRIGHT=/path/to/playwright PHOTO_FIND=… PHOTO_LOOK=… node scripts/ui/readme_shots.js [base-url]
// Default base: the local site (`make serve`, port 8010). The shop pages show the profile PROFILE ("Shopper, 22").
// The stylist flows use PHOTO_FIND and PHOTO_LOOK: photos of the repository owner, used with her consent (defaults:
// the CC0 photo data/raw/outfit_photos/full_03.jpg, credited in CREDITS.csv). Never a third party's private photo.
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const BASE = process.argv[2] || 'http://localhost:8010/';
const OUT = 'docs/assets';
const PHOTO_FIND = process.env.PHOTO_FIND || 'data/raw/outfit_photos/full_03.jpg';
const PHOTO_LOOK = process.env.PHOTO_LOOK || 'data/raw/outfit_photos/full_03.jpg';
const PROFILE = process.env.PROFILE || '133835';   // the first "Shopper, 22" in the profile list

const ready = p => p.waitForFunction(() => [...document.images].filter(i => { const r = i.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.width > 0; })
  .every(i => i.complete && i.naturalWidth > 0), null, {timeout: 60000}).catch(() => {});
const settle = async p => { await p.waitForLoadState('networkidle', {timeout: 20000}).catch(() => {}); await p.waitForTimeout(800); await ready(p); };
const shot = async (p, name, o = {}) => { await p.mouse.move(1, 1); await settle(p); await p.screenshot({path: `${OUT}/${name}.png`, ...o}); console.log('  ' + name); };
const fresh = async (p, hash) => { await p.goto(BASE + hash); await settle(p); await p.evaluate(() => sessionStorage.clear()); await p.reload(); await settle(p); };

async function find(p) {
  await fresh(p, '#/stylist/find');
  await shot(p, 'find_photo');
  await p.setInputFiles('.dropzone input', PHOTO_FIND); await p.waitForSelector('.photo-preview img');
  await p.click('#find-next');
  await p.waitForSelector('.piece', {timeout: 300000}); await settle(p);
  // Choose the main garment as a shopper does: tap it (its box comes from clothes parsing, D-051).
  const id = await p.$$eval('.picker__box', bs => bs.map(b => b.dataset.pid).sort()[0]).catch(() => null);
  await p.click(id ? `.piece[data-pid="${id}"] .piece__pick` : '.piece__pick');
  await shot(p, 'find_choose');
  await p.click('#find-go');
  await p.waitForFunction(() => location.hash.includes('step=results'), null, {timeout: 300000}); await settle(p);
  await shot(p, 'find_results', {fullPage: true});
}

async function look(p) {
  await fresh(p, '#/stylist/look');
  await p.setInputFiles('.dropzone input', PHOTO_LOOK); await p.waitForSelector('.photo-preview img');
  await p.click('#look-detect');
  await p.waitForFunction(() => location.hash.includes('step=review'), null, {timeout: 300000}); await settle(p);
  // Keep only the dress, as a shopper would (Complete my look then finishes the outfit around it): remove every
  // other spotted piece; if the dress was called a "top", correct its category.
  const names = await p.$$eval('.garment', gs => gs.map(g => [g.querySelector('[data-field=category]').value, g.querySelector('.garment__name').textContent.toLowerCase()]));
  let keep = names.findIndex(([c, n]) => c === 'dress' || /dress/.test(n));
  if (keep < 0) keep = 0;
  for (let i = names.length - 1; i >= 0; i--) if (i !== keep) await p.click(`.garment:nth-child(${i + 1}) [data-remove]`);
  const cat = await p.$('.garment [data-field="category"]');
  if ((await cat.inputValue()) !== 'dress') await cat.selectOption('dress');
  await shot(p, 'look_review', {fullPage: true});
  await p.click('#review-next');
  await p.waitForFunction(() => location.hash.includes('step=edit'), null, {timeout: 300000}); await settle(p);
  await shot(p, 'look_edit', {fullPage: true});
}

async function ask(p) {
  await fresh(p, '#/stylist/ask');
  await p.waitForFunction(() => [...document.querySelectorAll('.reply__img')].every(i => i.complete && i.naturalWidth > 0), null, {timeout: 30000}).catch(() => {});
  await shot(p, 'ask_start');
  await p.click('.reply');
  await p.waitForSelector('#pending', {state: 'detached', timeout: 300000}); await settle(p);
  // The whole exchange (question, answer, product cards) in one picture: let the chat grow instead of scrolling.
  await p.evaluate(() => { document.querySelector('#chatlog').style.maxHeight = 'none'; });
  await p.waitForTimeout(400); await ready(p);
  await shot(p, 'ask_answer', {fullPage: true});
}

(async () => {
  const b = await chromium.launch({channel: 'chrome'});
  const ctxOpts = {viewport: {width: 1440, height: 900}};
  const desk = await (await b.newContext(ctxOpts)).newPage();
  await desk.addInitScript(id => { try { localStorage.setItem('ensemble.profile', id); } catch {} }, PROFILE);
  await desk.goto(BASE + '#/'); await settle(desk);
  await shot(desk, 'home_desktop');
  await desk.goto(BASE + '#/collections'); await desk.waitForSelector('main .grid .card', {timeout: 120000}); await settle(desk);
  await shot(desk, 'collections_desktop');
  // The product page: the first piece in this shopper's edit, opened from Collections as a shopper would.
  await desk.click('main .grid > li:first-child .card__name a'); await desk.waitForSelector('.pdp [data-add]', {timeout: 120000}); await settle(desk);
  await shot(desk, 'product_desktop');
  await desk.goto(BASE + '#/stylist'); await settle(desk);
  await shot(desk, 'stylist_hub');
  await ask(desk);
  await find(desk);
  await look(desk);
  await desk.goto(BASE + '#/studio'); await desk.waitForSelector('table.data tbody tr', {timeout: 120000}); await settle(desk);
  await shot(desk, 'studio_desktop');
  const phone = await (await b.newContext({viewport: {width: 390, height: 844}, deviceScaleFactor: 2})).newPage();
  await phone.addInitScript(id => { try { localStorage.setItem('ensemble.profile', id); } catch {} }, PROFILE);
  await phone.goto(BASE + '#/'); await settle(phone);
  await shot(phone, 'home_mobile');
  await phone.goto(BASE + '#/stylist/ask'); await settle(phone);
  await shot(phone, 'ask_mobile');
  await b.close();
  console.log('captured');
})().catch(e => { console.error(e); process.exit(1); });
