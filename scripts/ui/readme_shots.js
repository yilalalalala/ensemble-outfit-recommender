// README screenshots of the real app on the live backend (no mock-ups, no generated imagery).
//   PLAYWRIGHT=/path/to/playwright node scripts/ui/readme_shots.js [base-url]
// Uses the CC0 photo data/raw/outfit_photos/full_03.jpg (credited in CREDITS.csv) for the stylist flow,
// and corrects the detected category in Review, as a shopper would.
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const BASE = process.argv[2] || 'http://localhost:8040/ensemble-outfit-recommender/';
const OUT = 'docs/assets';
const PHOTO = 'data/raw/outfit_photos/full_03.jpg';

const ready = p => p.waitForFunction(() => [...document.images].filter(i => { const r = i.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.width > 0; })
  .every(i => i.complete && i.naturalWidth > 0), null, {timeout: 60000}).catch(() => {});
const settle = async p => { await p.waitForLoadState('networkidle', {timeout: 15000}).catch(() => {}); await p.waitForTimeout(600); await ready(p); };

async function look(p, shots) {
  await p.goto(BASE + '#/stylist/look'); await settle(p);
  await p.evaluate(() => sessionStorage.clear()); await p.reload(); await settle(p);
  await p.setInputFiles('.dropzone input', PHOTO); await p.waitForSelector('.photo-preview img');
  await p.click('#look-detect');
  await p.waitForFunction(() => location.hash.includes('step=review'), null, {timeout: 300000}); await settle(p);
  // The model may call a shirt dress a "top": correct it in Review.
  const first = await p.$('.garment:first-child [data-field="category"]');
  const desc = await p.$eval('.garment:first-child .garment__name', e => e.textContent.toLowerCase());
  if (/dress/.test(desc)) await first.selectOption('dress');
  await p.mouse.move(1, 1); await settle(p);
  if (shots.review) await p.screenshot({path: shots.review});
  await p.click('#review-next'); await settle(p);
  await p.check('input[name=goal][value=complete]');
  await p.click('#dir-next');
  await p.waitForFunction(() => location.hash.includes('step=edit'), null, {timeout: 300000}); await settle(p);
  await p.mouse.move(1, 1); await settle(p);
  if (shots.edit) await p.screenshot({path: shots.edit});
  if (shots.refine) {
    await p.click('.task__actions .btn--solid'); await settle(p);
    await p.fill('#chat-in', 'Make it more formal'); await p.keyboard.press('Enter');
    await p.waitForSelector('#pending', {state: 'detached', timeout: 300000}); await settle(p);
    // Show the start of the stylist's reply (its text, then the product cards), not just the last cards.
    await p.evaluate(() => { const m = [...document.querySelectorAll('#chatlog .msg--bot')].at(-1); m.scrollIntoView({block: 'start'}); });
    await p.waitForTimeout(400); await ready(p);
    await p.screenshot({path: shots.refine});
  }
}

(async () => {
  const b = await chromium.launch({channel: 'chrome'});
  const desk = await (await b.newContext({viewport: {width: 1440, height: 900}})).newPage();
  await desk.goto(BASE + '#/'); await settle(desk); await desk.mouse.move(1, 1);
  await desk.screenshot({path: `${OUT}/home_desktop.png`});
  await look(desk, {edit: `${OUT}/assistant_desktop.png`});
  await desk.goto(BASE + '#/studio'); await desk.waitForSelector('table.data tbody tr'); await settle(desk);
  await desk.screenshot({path: `${OUT}/studio_desktop.png`});
  const phone = await (await b.newContext({viewport: {width: 390, height: 844}, deviceScaleFactor: 2})).newPage();
  await phone.goto(BASE + '#/'); await settle(phone);
  await phone.screenshot({path: `${OUT}/home_mobile.png`});
  await look(phone, {review: '/tmp/ensemble_readme_review.png', edit: '/tmp/ensemble_readme_edit.png', refine: '/tmp/ensemble_readme_refine.png'});
  await b.close();
  console.log('captured');
})().catch(e => { console.error(e); process.exit(1); });
