/*
 * After `vite build`: one HTML file per page, so every page has an address of its own (/anatomy/attention/) with its
 * own title, description, canonical URL and share card, and search engines can list it. Each file is dist/index.html
 * with those tags changed and the page's text written in: every page is opened in headless Chrome
 * (scripts/chrome.mjs) and its header, every step with its caption (the All steps list), its code and references
 * are read, in English. That text and a list of every page stay hidden while the app runs, which draws the page
 * instead; they are there for search engines that do not run scripts and for readers with scripts off. Opening
 * every page also checks it: a page that throws or logs an error fails the build. Also a 404.html that boots the
 * app (unknown paths go home), sitemap.xml and robots.txt. Run by `npm run build`.
 */
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { CATEGORIES, FOUNDATIONS, GLOSSARY, HOME, START, exhibitsOf, isHeading } from '../src/exhibits/registry.ts'
import { launch } from './chrome.mjs'

const SITE = 'https://tokentrails.org'
const dist = new URL('../dist/', import.meta.url)
const shell = fs.readFileSync(new URL('index.html', dist), 'utf8')
const esc = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;')
/** Tags as the rail writes them: W_E is W with a subscript E; in plain text the underscore stays out. */
const plain = (s) => s.replace(/_\{?([^}\s·]+)\}?/g, '$1')

const pages = [HOME, START, FOUNDATIONS, GLOSSARY, ...CATEGORIES.flatMap(exhibitsOf)].filter((e) => e.route && e.load)
const category = (route) => CATEGORIES.find((c) => exhibitsOf(c).some((e) => e.route === route))?.title
const href = (route) => (route === 'home' ? '/' : `/${route}/`)
const urlOf = (route) => SITE + href(route)

/* ---------- each page's text, read from the built site ---------- */

/** Runs in the page once it is drawn: its text as small, safe HTML (links, emphasis, subscripts and lists only). */
async function readView() {
  const view = document.querySelector('.main > .view')
  const steps = view.querySelector('details.steps:not(.code-d):not(.refs-d)')
  // opening All steps writes the list again from the page as it is now (a caption may count what a lazy load brought)
  if (steps && !steps.open) await new Promise((resolve) => { steps.addEventListener('toggle', resolve, { once: true }); steps.open = true })
  const KEEP = new Set(['A', 'B', 'CODE', 'DD', 'DL', 'DT', 'EM', 'H2', 'H3', 'I', 'KBD', 'LI', 'OL', 'P', 'SMALL', 'STRONG', 'SUB', 'SUP', 'UL'])
  const SKIP = 'button, canvas, input, select, svg, [aria-hidden="true"], [role="img"], .st-path .n'
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c])
  const html = (...nodes) => nodes.map((n) => {
    if (n.nodeType === Node.TEXT_NODE) return esc(n.data.replace(/[ \t\n\r]+/g, ' '))
    if (n.nodeType !== Node.ELEMENT_NODE || n.matches(SKIP)) return ''
    const inner = html(...n.childNodes), tag = n.tagName.toLowerCase()
    // an identifier with a subscript keeps its class, which keeps it out of uppercase labels
    if (n.matches('span.m')) return `<span class="m">${inner}</span>`
    if (!KEEP.has(n.tagName)) return inner
    if (!inner.trim()) return ''
    return tag === 'a' ? `<a href="${esc(n.getAttribute('href') ?? '')}">${inner}</a>` : `<${tag}>${inner}</${tag}>`
  }).join('')
  const inside = (el) => (el ? html(...el.childNodes).trim() : '')
  const all = (s) => [...view.querySelectorAll(s)]
  return {
    eyebrow: inside(view.querySelector('.head .eyebrow')),
    title: inside(view.querySelector('h1')),
    sub: inside(view.querySelector('.head .sub')),
    specs: all('.head .specs > div').map((d) => [inside(d.querySelector('dt')), inside(d.querySelector('dd'))]),
    lead: all('.home-cta > p:not(.home-credit)').map(inside),
    steps: steps ? [...steps.querySelectorAll('ol > li')].map((li) => ({ name: inside(li.querySelector('button b')), text: inside(li.querySelector(':scope > p')), shape: inside(li.querySelector(':scope > code')) })) : [],
    body: all('.start-body > :not(.st-demo)').map((el) => html(el).trim()),
    code: view.querySelector('.code-d code')?.textContent ?? '',
    refs: all('.refs-d a').map((a) => [a.textContent, a.getAttribute('href')]),
  }
}

const TYPES = { '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.woff2': 'font/woff2', '.woff': 'font/woff', '.png': 'image/png', '.svg': 'image/svg+xml', '.txt': 'text/plain', '.xml': 'application/xml' }

/** Every page's text: dist/ served on a local port as the site serves it, each page opened in headless Chrome. */
async function readPages() {
  const root = fileURLToPath(dist)
  const server = http.createServer((req, res) => {
    const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname))
    const ext = path.extname(file)
    // a page's address is the shell (the app draws the page from the path); anything else is a file
    if (!ext) { res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); res.end(shell); return }
    if (!file.startsWith(root) || !fs.statSync(file, { throwIfNoEntry: false })?.isFile()) { res.writeHead(404); res.end(); return }
    res.writeHead(200, { 'content-type': TYPES[ext] ?? 'application/octet-stream' })
    fs.createReadStream(file).pipe(res)
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const base = `http://127.0.0.1:${server.address().port}`
  // analytics never counts the build; motion is reduced, so the pages sit still instead of animating
  const chrome = await launch({ block: ['*cloudflareinsights.com*'] })
  const views = new Map(), problems = []
  const queue = [...pages]
  async function worker() {
    for (let e; (e = queue.shift());) {
      const page = await chrome.newPage({ media: [{ name: 'prefers-reduced-motion', value: 'reduce' }] })
      try {
        await page.goto(`${base}${href(e.route)}?lang=en`)
        await page.evaluate(async () => {
          for (const t0 = Date.now(); !document.querySelector('.main > .view h1'); await new Promise((r) => setTimeout(r, 50)))
            if (Date.now() - t0 > 20000) throw new Error('the page was not drawn')
        })
        await page.idle()
        views.set(e.route, await page.evaluate(readView))
      } catch (err) {
        problems.push(`${e.route}: ${err.message}`)
      }
      problems.push(...page.errors.map((m) => `${e.route}: ${m}`))
      await page.close()
    }
  }
  try {
    await Promise.all(Array.from({ length: 4 }, worker))
  } finally {
    await chrome.close()
    server.close()
  }
  if (problems.length) throw new Error(`pages failed in Chrome:\n  ${problems.join('\n  ')}`)
  return views
}

/* ---------- the page files ---------- */

/** The page's text as an article, hidden while the app runs (see .static-page in styles.css). */
function article(route, v) {
  // the real model's value is set off by a space, not only by the app's CSS
  const specs = v.specs.length ? `<dl class="specs">${v.specs.map(([dt, dd]) => `<div><dt>${dt}</dt><dd>${dd.replace(/<small>/g, ' <small>')}</dd></div>`).join('')}</dl>` : ''
  const head = `<header class="head"><div>${v.eyebrow ? `<p class="eyebrow">${v.eyebrow}</p>` : ''}<div class="titlebar"><h1>${v.title}</h1>${v.sub ? `<p class="sub">${v.sub}</p>` : ''}</div></div>${specs}</header>`
  const parts = []
  if (route === 'home') parts.push(`<section>${v.lead.map((p) => `<p>${p}</p>`).join('')}<p><a href="${href('start')}">Start the tour →</a></p></section>`)
  if (v.steps.length) parts.push(`<section><h2>All steps</h2><ol>${v.steps.map((s) => `<li><h3>${s.name}</h3>${s.text ? `<p>${s.text}</p>` : ''}${s.shape ? `<p><code>${s.shape}</code></p>` : ''}</li>`).join('')}</ol></section>`)
  parts.push(...v.body.map((b) => `<section>${b}</section>`))
  if (v.code) parts.push(`<section><h2>Code</h2><pre><code>${esc(v.code)}</code></pre></section>`)
  if (v.refs.length) parts.push(`<section><h2>Go deeper</h2><ul>${v.refs.map(([t, u]) => `<li><a href="${esc(u)}">${esc(t)}</a></li>`).join('')}</ul></section>`)
  return `<article class="static-page">${head}<div class="start-body">${parts.join('')}</div></article>`
}

/** Every page, as the rail lists them with all groups open; hidden while the app runs, which draws the rail itself. */
function railList(active) {
  const link = (e) => `<li><a class="ex" href="${href(e.route)}"${e.route === active ? ' aria-current="page"' : ''}><span>${esc(e.name)}</span></a>${e.children?.some((c) => c.route) ? `<ul class="kids">${e.children.filter((c) => c.route).map(link).join('')}</ul>` : ''}</li>`
  const groups = CATEGORIES.map((c) => `<section class="group"><p class="cat"><span class="t">${esc(c.title)}</span></p><ul>${c.entries.map((e) => (isHeading(e) ? `<li class="subh">${esc(e.heading)}</li>` : e.route ? link(e) : '')).join('')}</ul></section>`)
  return `<div class="static-nav"><ul class="start-link">${[START, FOUNDATIONS, GLOSSARY].map(link).join('')}</ul>${groups.join('')}</div>`
}

/** The shell with one page's title, description and address in every tag that names them, and its text. */
function page(route, title, description, view) {
  const url = urlOf(route)
  // puts value between the pattern's two groups
  const set = (html, re, value) => { if (!re.test(html)) throw new Error(`index.html has no ${re}`); return html.replace(re, (_, a, b) => a + value + b) }
  let html = shell
  if (route !== 'home') {
    html = set(html, /(<title>)[^<]*(<\/title>)/, esc(title))
    html = set(html, /(<meta name="description" content=")[^"]*(")/, esc(description))
    html = set(html, /(<link rel="canonical" href=")[^"]*(")/, url)
    html = set(html, /(<meta property="og:title" content=")[^"]*(")/, esc(title))
    html = set(html, /(<meta property="og:description" content=")[^"]*(")/, esc(description))
    html = set(html, /(<meta property="og:url" content=")[^"]*(")/, url)
  }
  html = set(html, /(<nav class="nav"[^>]*>)(<\/nav>)/, railList(route))
  return set(html, /(<main class="main">)(<\/main>)/, article(route, view))
}

const t0 = Date.now()
const views = await readPages()
let bytes = 0
for (const e of pages) {
  const cat = category(e.route)
  const title = [e.name, cat, 'Token Trails'].filter(Boolean).join(' · ')
  const what = e.tag ? `${e.name} (${plain(e.tag)})` : e.name
  const description = `${what}: an animated, explorable walk-through with the numbers of real models. ${cat ? `Part of ${cat}, in ` : 'From '}Token Trails, which follows the tokens through AI systems.`
  const dir = new URL(e.route === 'home' ? './' : `${e.route}/`, dist)
  const html = page(e.route, title, description, views.get(e.route))
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(new URL('index.html', dir), html)
  bytes += html.length - shell.length
}
// unknown paths: GitHub Pages serves this with a 404 status; the app starts and goes home
fs.writeFileSync(new URL('404.html', dist), shell.replace('<head>', '<head>\n    <meta name="robots" content="noindex" />'))
const today = new Date().toISOString().slice(0, 10)
fs.writeFileSync(new URL('sitemap.xml', dist), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages.map((e) => `  <url><loc>${urlOf(e.route)}</loc><lastmod>${today}</lastmod></url>`).join('\n')}
</urlset>
`)
fs.writeFileSync(new URL('robots.txt', dist), `User-agent: *\nAllow: /\nSitemap: ${SITE}/sitemap.xml\n`)
console.log(`pages: ${pages.length} page files with their text (${(bytes / 1024).toFixed(0)} KB in all, read in ${((Date.now() - t0) / 1000).toFixed(1)} s), 404.html, sitemap.xml (${pages.length} URLs), robots.txt`)
