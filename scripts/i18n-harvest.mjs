/*
 * Lists every English string the site shows that has no Chinese entry yet, by opening every page at every step
 * (at three moments of each) on the dev server in harvest mode, plus every question and recap line. Strings are
 * keys as core/i18n.ts makes them: numbers and quoted spans become {}.
 *
 *   npx vite --port 5174 &     then     node scripts/i18n-harvest.mjs [route-prefix] > missing.json
 *
 * Needs Playwright (npx playwright install chromium, or PLAYWRIGHT pointing at an installed module, and
 * CHROME at a browser binary). Output: { key: { ex, routes } } for the keys not in src/locales/zh.
 */
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT ?? 'playwright')
const BASE = process.env.BASE ?? 'http://localhost:5174/'
const prefix = process.argv[2] ?? ''

const browser = await chromium.launch(process.env.CHROME ? { executablePath: process.env.CHROME } : {})
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } })
await page.addInitScript(() => { localStorage.setItem('tt-i18n-harvest', '1'); localStorage.setItem('tt-pace', 'auto') })
await page.goto(BASE + '#/start')
await page.waitForFunction(() => window.__tt)
const routes = await page.evaluate(() => [...document.querySelectorAll('.nav a[href^="#/"]')].map((a) => a.getAttribute('href').slice(2)))
for (const route of routes.filter((r) => r.startsWith(prefix))) {
  await page.evaluate((r) => { location.hash = '#/' + r }, route)
  await page.waitForTimeout(600)
  const phases = await page.evaluate(() => window.__ttPlayer?.phases.length ?? 0)
  for (let i = 0; i < phases; i++) {
    // the step's start, middle and end
    for (const f of [0.05, 0.5, 0.97]) {
      await page.evaluate(([i, f]) => {
        const p = window.__ttPlayer, ph = p.phases[i]
        p.setPlaying(false); p.t = ph.start + f * ph.dur
        window.dispatchEvent(new Event('pointermove'))
      }, [i, f])
      await page.waitForTimeout(160)
    }
  }
  await page.evaluate(() => { const d = document.querySelector('.view:last-child .steps'); if (d) d.open = true })
  await page.waitForTimeout(100)
  process.stderr.write(`${route}: ${phases} steps\n`)
}
const missing = await page.evaluate(async () => {
  const tt = window.__tt
  tt.harvest.route = 'learn'
  for (const [k, texts] of Object.entries(tt.learnTexts())) { tt.harvest.route = 'learn:' + k; texts.forEach((s) => tt.t(s)) }
  // switching re-renders the page and the check calls t() again: record neither
  tt.harvest.on = false
  tt.setLang('zh')
  const out = {}
  for (const [key, e] of tt.harvest.seen) if (tt.t(e.ex) === e.ex) out[key] = { ex: e.ex, routes: [...e.routes] }
  return out
})
console.log(JSON.stringify(missing, null, 1))
await browser.close()
