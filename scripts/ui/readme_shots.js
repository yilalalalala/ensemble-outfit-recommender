// README screenshots of the real app on the live backend (no mock-ups, no generated imagery).
//   PLAYWRIGHT=/path/to/playwright node scripts/ui/readme_shots.js [base-url]
// Default base: `make pages-preview` (the Pages build, calling the live Modal API). The stylist flows use the CC0
// photo data/raw/outfit_photos/full_03.jpg (credited in CREDITS.csv), never a private photo.
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const BASE = process.argv[2] || 'http://localhost:8040/ensemble-outfit-recommender/';
const OUT = 'docs/assets';
const PHOTO = 'data/raw/outfit_photos/full_03.jpg';
const PRODUCT = '898713001';   // Hanoi Oversize Blazer: a houndstooth, so its swatch is a fabric crop

const ready = p => p.waitForFunction(() => [...document.images].filter(i => { const r = i.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.width > 0; })
  .every(i => i.complete && i.naturalWidth > 0), null, {timeout: 60000}).catch(() => {});
const settle = async p => { await p.waitForLoadState('networkidle', {timeout: 20000}).catch(() => {}); await p.waitForTimeout(800); await ready(p); };
const shot = async (p, name, o = {}) => { await p.mouse.move(1, 1); await settle(p); await p.screenshot({path: `${OUT}/${name}.png`, ...o}); console.log('  ' + name); };
const fresh = async (p, hash) => { await p.goto(BASE + hash); await settle(p); await p.evaluate(() => sessionStorage.clear()); await p.reload(); await settle(p); };

async function find(p) {
  await fresh(p, '#/stylist/find');
  await shot(p, 'find_photo');
  await p.setInputFiles('.dropzone input', PHOTO); await p.waitForSelector('.photo-preview img');
  await p.click('#find-next');
  await p.waitForSelector('.piece', {timeout: 300000}); await settle(p);
  // Frame the shirt dress by dragging around it, as a shopper does (the detector's boxes can be loose).
  await p.$eval('#picker img', i => i.scrollIntoView({block: 'center'}));
  const r = await (await p.$('#picker img')).boundingBox();
  await p.mouse.move(r.x + r.width * .37, r.y + r.height * .2); await p.mouse.down();
  await p.mouse.move(r.x + r.width * .57, r.y + r.height * .62, {steps: 8}); await p.mouse.up();
  await shot(p, 'find_choose');
  await p.click('#find-go');
  await p.waitForFunction(() => location.hash.includes('step=results'), null, {timeout: 300000}); await settle(p);
  await shot(p, 'find_results', {fullPage: true});
}

async function look(p) {
  await fresh(p, '#/stylist/look');
  await p.setInputFiles('.dropzone input', PHOTO); await p.waitForSelector('.photo-preview img');
  await p.click('#look-detect');
  await p.waitForFunction(() => location.hash.includes('step=review'), null, {timeout: 300000}); await settle(p);
  // The model may call a shirt dress a "top": correct it in Review, as a shopper would.
  const first = await p.$('.garment:first-child [data-field="category"]');
  const desc = await p.$eval('.garment:first-child .garment__name', e => e.textContent.toLowerCase());
  if (/dress/.test(desc)) await first.selectOption('dress');
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
  const desk = await (await b.newContext({viewport: {width: 1440, height: 900}})).newPage();
  await desk.goto(BASE + '#/'); await settle(desk);
  await shot(desk, 'home_desktop');
  await desk.goto(BASE + '#/collections'); await desk.waitForSelector('main .grid .card', {timeout: 120000}); await settle(desk);
  await shot(desk, 'collections_desktop');
  await desk.goto(BASE + `#/product/${PRODUCT}`); await desk.waitForSelector('.pdp [data-add]', {timeout: 120000}); await settle(desk);
  await shot(desk, 'product_desktop');
  await desk.goto(BASE + '#/stylist'); await settle(desk);
  await shot(desk, 'stylist_hub');
  await ask(desk);
  await find(desk);
  await look(desk);
  await desk.goto(BASE + '#/studio'); await desk.waitForSelector('table.data tbody tr', {timeout: 120000}); await settle(desk);
  await shot(desk, 'studio_desktop');
  const phone = await (await b.newContext({viewport: {width: 390, height: 844}, deviceScaleFactor: 2})).newPage();
  await phone.goto(BASE + '#/'); await settle(phone);
  await shot(phone, 'home_mobile');
  await phone.goto(BASE + '#/stylist/ask'); await settle(phone);
  await shot(phone, 'ask_mobile');
  await b.close();
  console.log('captured');
})().catch(e => { console.error(e); process.exit(1); });
