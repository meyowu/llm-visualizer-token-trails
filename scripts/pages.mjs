/*
 * After `vite build`: one HTML file per page and language, so every page has an address of its own
 * (/anatomy/attention/, and /zh/anatomy/attention/ in Chinese) with its own title, description, canonical URL, links to
 * its other language (hreflang), share card and structured data, and search engines can list it. Each file is
 * dist/index.html with those tags changed and the page's text written in: every page is opened in headless Chrome
 * (scripts/chrome.mjs), in English and in Chinese, and its title and description (set by main.ts from
 * src/exhibits/seo.ts), header, every step with its caption (the All steps list), code, references and the rail's
 * list of pages are read. That text and list stay hidden while the app runs, which draws the page instead; they are
 * there for search engines that do not run scripts and for readers with scripts off. Opening every page also checks
 * it: a page that throws or logs an error, or has no Chinese title or description, fails the build. Also a 404.html
 * that boots the app (unknown paths go home), sitemap.xml and robots.txt. Run by `npm run build`.
 */
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { CATEGORIES, FOUNDATIONS, GLOSSARY, HOME, START, exhibitsOf } from '../src/exhibits/registry.ts'
import { SEO } from '../src/exhibits/seo.ts'
import { launch } from './chrome.mjs'

const SITE = 'https://tokentrails.org'
const LANGS = ['en', 'zh']
const dist = new URL('../dist/', import.meta.url)
const shell = fs.readFileSync(new URL('index.html', dist), 'utf8')
const esc = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;')

const pages = [HOME, START, FOUNDATIONS, GLOSSARY, ...CATEGORIES.flatMap(exhibitsOf)].filter((e) => e.route && e.load)
const unnamed = pages.filter((e) => !SEO[e.route]).map((e) => e.route)
if (unnamed.length) throw new Error(`no title and description in src/exhibits/seo.ts for ${unnamed.join(', ')}`)
/** A page's path in a language: /anatomy/ or /zh/anatomy/ (the home page is / or /zh/). */
const pathOf = (route, lang) => (lang === 'zh' ? '/zh' : '') + (route === 'home' ? '/' : `/${route}/`)
const urlOf = (route, lang) => SITE + pathOf(route, lang)

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
  const text = (el) => el?.textContent.trim() ?? ''
  // the rail's list of pages with every group open, and where this page sits in it (for the breadcrumbs)
  const nav = document.querySelector('.nav')
  const link = (a) => `<a class="ex" href="${esc(a.getAttribute('href'))}"${a.hasAttribute('aria-current') ? ' aria-current="page"' : ''}><span>${esc(text(a.querySelector('span')))}</span></a>`
  const items = (ul) => [...ul.children].map((li) => {
    if (li.classList.contains('subh')) return `<li class="subh">${esc(text(li))}</li>`
    const a = li.querySelector(':scope > a.ex'), kids = li.querySelector(':scope > ul.kids')
    return a ? `<li>${link(a)}${kids ? `<ul class="kids">${items(kids)}</ul>` : ''}</li>` : ''
  }).join('')
  const groups = [...nav.querySelectorAll(':scope > section.group')].map((g) => `<section class="group"><p class="cat"><span class="t">${esc(text(g.querySelector('.cat .t')))}</span></p><ul>${items(g.querySelector(':scope > ul'))}</ul></section>`)
  const here = nav.querySelector('a[aria-current="page"]'), parent = here?.closest('ul.kids')?.closest('li')?.querySelector(':scope > a.ex')
  return {
    title: document.title,
    description: document.querySelector('meta[name="description"]').getAttribute('content'),
    lang: document.documentElement.lang,
    rail: `<ul class="start-link">${items(nav.querySelector(':scope > ul.start-link'))}</ul>${groups.join('')}`,
    crumbs: [parent, here].filter(Boolean).map((a) => ({ name: text(a.querySelector('span')), href: a.getAttribute('href') })),
    labels: { steps: text(steps?.querySelector('summary')), code: text(view.querySelector('.code-d summary')), refs: text(view.querySelector('.refs-d summary')), start: text(view.querySelector('.home-start')) },
    eyebrow: inside(view.querySelector('.head .eyebrow')),
    h1: inside(view.querySelector('h1')),
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

/** Every page's text in both languages, by 'lang:route': dist/ served on a local port as the site serves it, each page opened in headless Chrome. */
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
  const queue = LANGS.flatMap((lang) => pages.map((e) => ({ lang, route: e.route })))
  async function worker() {
    for (let e; (e = queue.shift());) {
      const page = await chrome.newPage({ media: [{ name: 'prefers-reduced-motion', value: 'reduce' }] })
      const where = pathOf(e.route, e.lang)
      try {
        // a Chinese page's path sets its language; ?lang=en keeps a Chinese-speaking machine's browser in English
        await page.goto(`${base}${where}${e.lang === 'en' ? '?lang=en' : ''}`)
        await page.evaluate(async () => {
          for (const t0 = Date.now(); !document.querySelector('.main > .view h1'); await new Promise((r) => setTimeout(r, 50)))
            if (Date.now() - t0 > 20000) throw new Error('the page was not drawn')
        })
        await page.idle()
        const v = await page.evaluate(readView)
        if (v.lang !== (e.lang === 'zh' ? 'zh-CN' : 'en')) problems.push(`${where}: drawn in ${v.lang}`)
        views.set(`${e.lang}:${e.route}`, v)
      } catch (err) {
        problems.push(`${where}: ${err.message}`)
      }
      problems.push(...page.errors.map((m) => `${where}: ${m}`))
      await page.close()
    }
  }
  try {
    await Promise.all(Array.from({ length: 4 }, worker))
  } finally {
    await chrome.close()
    server.close()
  }
  // a Chinese page shows the English title or description when src/locales/zh/pages.ts has no entry for it
  for (const { route } of pages) {
    const en = views.get(`en:${route}`), zh = views.get(`zh:${route}`)
    if (en && zh && (en.title === zh.title || en.description === zh.description)) problems.push(`/zh${pathOf(route, 'en')}: no Chinese title or description (src/locales/zh/pages.ts)`)
  }
  if (problems.length) throw new Error(`pages failed in Chrome:\n  ${problems.join('\n  ')}`)
  return views
}

/* ---------- the page files ---------- */

/** The page's text as an article, hidden while the app runs (see .static-page in styles.css). */
function article(route, lang, v) {
  // the real model's value is set off by a space, not only by the app's CSS
  const specs = v.specs.length ? `<dl class="specs">${v.specs.map(([dt, dd]) => `<div><dt>${dt}</dt><dd>${dd.replace(/<small>/g, ' <small>')}</dd></div>`).join('')}</dl>` : ''
  const head = `<header class="head"><div>${v.eyebrow ? `<p class="eyebrow">${v.eyebrow}</p>` : ''}<div class="titlebar"><h1>${v.h1}</h1>${v.sub ? `<p class="sub">${v.sub}</p>` : ''}</div></div>${specs}</header>`
  const parts = []
  if (route === 'home') parts.push(`<section>${v.lead.map((p) => `<p>${p}</p>`).join('')}<p><a href="${pathOf('start', lang)}">${esc(v.labels.start)}</a></p></section>`)
  if (v.steps.length) parts.push(`<section><h2>${esc(v.labels.steps)}</h2><ol>${v.steps.map((s) => `<li><h3>${s.name}</h3>${s.text ? `<p>${s.text}</p>` : ''}${s.shape ? `<p><code>${s.shape}</code></p>` : ''}</li>`).join('')}</ol></section>`)
  parts.push(...v.body.map((b) => `<section>${b}</section>`))
  if (v.code) parts.push(`<section><h2>${esc(v.labels.code)}</h2><pre><code>${esc(v.code)}</code></pre></section>`)
  if (v.refs.length) parts.push(`<section><h2>${esc(v.labels.refs)}</h2><ul>${v.refs.map(([t, u]) => `<li><a href="${esc(u)}">${esc(t)}</a></li>`).join('')}</ul></section>`)
  return `<article class="static-page">${head}<div class="start-body">${parts.join('')}</div></article>`
}

/** Structured data: the site's name on the home page, the path to the page (home › exhibit › step) elsewhere. */
function jsonLd(route, lang, v) {
  const data = route === 'home'
    ? { '@context': 'https://schema.org', '@type': 'WebSite', name: 'Token Trails', url: urlOf('home', lang), inLanguage: v.lang, description: v.description }
    : {
      '@context': 'https://schema.org', '@type': 'BreadcrumbList',
      itemListElement: [{ name: 'Token Trails', href: pathOf('home', lang) }, ...v.crumbs].map((c, i) => ({ '@type': 'ListItem', position: i + 1, name: c.name, item: SITE + c.href })),
    }
  // < never ends the script early
  return `<script type="application/ld+json">${JSON.stringify(data).replace(/</g, '\\u003c')}</script>`
}

/** The shell with one page's title, description, address, language and text. */
function page(route, lang, v) {
  const url = urlOf(route, lang)
  // puts value between the pattern's two groups
  const set = (html, re, value) => { if (!re.test(html)) throw new Error(`index.html has no ${re}`); return html.replace(re, (_, a, b) => a + value + b) }
  // the same page in each language, and English for everyone else
  const alternates = [['en', 'en'], ['zh-Hans', 'zh'], ['x-default', 'en']].map(([hl, l]) => `\n    <link rel="alternate" hreflang="${hl}" href="${urlOf(route, l)}" />`).join('')
  let html = shell
  html = set(html, /(<html lang=")[^"]*(")/, lang === 'zh' ? 'zh-CN' : 'en')
  html = set(html, /(<title>)[^<]*(<\/title>)/, esc(v.title))
  html = set(html, /(<meta name="description" content=")[^"]*(")/, esc(v.description))
  html = set(html, /(<link rel="canonical" href=")[^"]*(")/, url)
  html = set(html, /(<link rel="canonical" href="[^"]*" \/>)()/, alternates)
  html = set(html, /(<meta property="og:title" content=")[^"]*(")/, esc(v.title))
  html = set(html, /(<meta property="og:description" content=")[^"]*(")/, esc(v.description))
  html = set(html, /(<meta property="og:url" content=")[^"]*(")/, url)
  html = set(html, /()(\s*<\/head>)/, '\n    ' + jsonLd(route, lang, v))
  html = set(html, /(<nav class="nav"[^>]*>)(<\/nav>)/, `<div class="static-nav">${v.rail}</div>`)
  return set(html, /(<main class="main">)(<\/main>)/, article(route, lang, v))
}

const t0 = Date.now()
const views = await readPages()
let bytes = 0
for (const lang of LANGS) for (const e of pages) {
  const dir = new URL('.' + pathOf(e.route, lang), dist)
  const html = page(e.route, lang, views.get(`${lang}:${e.route}`))
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(new URL('index.html', dir), html)
  bytes += html.length - shell.length
}
// unknown paths: GitHub Pages serves this with a 404 status; the app starts and goes home
fs.writeFileSync(new URL('404.html', dist), shell.replace('<head>', '<head>\n    <meta name="robots" content="noindex" />'))
const today = new Date().toISOString().slice(0, 10)
fs.writeFileSync(new URL('sitemap.xml', dist), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${LANGS.flatMap((lang) => pages.map((e) => `  <url><loc>${urlOf(e.route, lang)}</loc><lastmod>${today}</lastmod></url>`)).join('\n')}
</urlset>
`)
fs.writeFileSync(new URL('robots.txt', dist), `User-agent: *\nAllow: /\nSitemap: ${SITE}/sitemap.xml\n`)
const n = pages.length * LANGS.length
console.log(`pages: ${pages.length} pages × ${LANGS.length} languages with their text (${(bytes / 1024).toFixed(0)} KB in all, read in ${((Date.now() - t0) / 1000).toFixed(1)} s), 404.html, sitemap.xml (${n} URLs), robots.txt`)
