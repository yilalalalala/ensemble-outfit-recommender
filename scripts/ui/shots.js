// Viewport screenshots (1440×900, 1024×768, 390×844) with horizontal-overflow and console-error checks.
//   PLAYWRIGHT=/path/to/node_modules/playwright node scripts/ui/shots.js <outdir> "#/,#/studio" [base-url]
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const [out, routes] = [process.argv[2], (process.argv[3] || '#/').split(',')];
const VP = {desktop: [1440, 900], tablet: [1024, 768], mobile: [390, 844]};
(async () => {
  const b = await chromium.launch({channel: 'chrome'});
  for (const [name, [w, h]] of Object.entries(VP)) {
    const ctx = await b.newContext({viewport: {width: w, height: h}, deviceScaleFactor: 1});
    const p = await ctx.newPage();
    const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => m.type() === 'error' && errs.push(m.text()));
    for (const r of routes) {
      await p.goto((process.argv[4] || 'http://localhost:8031/') + r); await p.waitForLoadState('networkidle'); await p.waitForTimeout(900);
      const slug = r.replace(/[^a-z0-9]+/gi, '_').replace(/^_|_$/g, '') || 'home';
      await p.screenshot({path: `${out}/${slug}_${name}.png`, fullPage: false});
      await p.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += innerHeight / 2) { scrollTo(0, y); await new Promise(r => setTimeout(r, 120)); } scrollTo(0, 0); });
      await p.waitForLoadState('networkidle'); await p.waitForTimeout(800);
      await p.screenshot({path: `${out}/${slug}_${name}_full.png`, fullPage: true});
      const ov = await p.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      console.log(name, r, 'overflow', ov);
    }
    if (errs.length) console.log(name, 'ERRORS', errs);
    await ctx.close();
  }
  await b.close();
})();
