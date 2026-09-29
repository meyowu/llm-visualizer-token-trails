/*
 * After `vite build`: one HTML file per page, so every page has an address of its own (/anatomy/attention/) with its
 * own title, description, canonical URL and share card, and search engines can list it. Each file is dist/index.html
 * with those tags changed; the app then draws the page from the path. Also a 404.html that boots the app (unknown
 * paths go home), sitemap.xml and robots.txt. Run by `npm run build`.
 */
import fs from 'node:fs'
import { CATEGORIES, FOUNDATIONS, GLOSSARY, HOME, START, exhibitsOf } from '../src/exhibits/registry.ts'

const SITE = 'https://tokentrails.org'
const dist = new URL('../dist/', import.meta.url)
const shell = fs.readFileSync(new URL('index.html', dist), 'utf8')
const esc = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;')
/** Tags as the rail writes them: W_E is W with a subscript E; in plain text the underscore stays out. */
const plain = (s) => s.replace(/_\{?([^}\s·]+)\}?/g, '$1')

const pages = [HOME, START, FOUNDATIONS, GLOSSARY, ...CATEGORIES.flatMap(exhibitsOf)].filter((e) => e.route && e.load)
const category = (route) => CATEGORIES.find((c) => exhibitsOf(c).some((e) => e.route === route))?.title
const urlOf = (route) => (route === 'home' ? `${SITE}/` : `${SITE}/${route}/`)

/** The shell with one page's title, description and address in every tag that names them. */
function page(route, title, description) {
  const url = urlOf(route)
  const set = (html, re, value) => { if (!re.test(html)) throw new Error(`index.html has no ${re}`); return html.replace(re, value) }
  let html = shell
  html = set(html, /<title>[^<]*<\/title>/, `<title>${esc(title)}</title>`)
  html = set(html, /(<meta name="description" content=")[^"]*/, `$1${esc(description)}`)
  html = set(html, /(<link rel="canonical" href=")[^"]*/, `$1${url}`)
  html = set(html, /(<meta property="og:title" content=")[^"]*/, `$1${esc(title)}`)
  html = set(html, /(<meta property="og:description" content=")[^"]*/, `$1${esc(description)}`)
  html = set(html, /(<meta property="og:url" content=")[^"]*/, `$1${url}`)
  return html
}

let n = 0
for (const e of pages) {
  if (e.route === 'home') continue // dist/index.html is the home page as built
  const cat = category(e.route)
  const title = [e.name, cat, 'Token Trails'].filter(Boolean).join(' · ')
  const what = e.tag ? `${e.name} (${plain(e.tag)})` : e.name
  const description = `${what}: an animated, explorable walk-through with the numbers of real models. ${cat ? `Part of ${cat}, in ` : 'From '}Token Trails, which follows the tokens through AI systems.`
  const dir = new URL(`${e.route}/`, dist)
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(new URL('index.html', dir), page(e.route, title, description))
  n++
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
console.log(`pages: ${n} page files, 404.html, sitemap.xml (${pages.length} URLs), robots.txt`)
