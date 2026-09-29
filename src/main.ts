import './styles.css'
import { registerFonts } from './core/fonts'
import { rich } from './core/frame'
import { harvest, lang, loadLang, onLang, setLang, t } from './core/i18n'
import { openAtPhase } from './core/player'
import { pref } from './core/prefs'
import { progress } from './core/progress'
import { getParams, readAddress, routeHref, HASH_ROUTES } from './core/link'
import { poke } from './core/stage'
import { watchTheme } from './core/theme'
import { reducedMotion } from './core/util'
import { ALIASES, CATEGORIES, DEFAULT_ROUTE, FOUNDATIONS, GLOSSARY, ROUTES, START, exhibitsOf, isHeading, type Exhibit, type Mount } from './exhibits/registry'

// a Chinese reader's text arrives before anything is drawn
await loadLang(lang)

registerFonts()
// light / dark: follow the system until the reader picks one
const themeBtn = document.querySelector('.theme-btn') as HTMLButtonElement
const THEMES = ['system', 'light', 'dark'] as const
function applyTheme(t: string) {
  if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t
  else delete document.documentElement.dataset.theme
  themeBtn.textContent = tr(t === 'light' ? '☀ Light' : t === 'dark' ? '☾ Dark' : '◐ System theme')
  themeBtn.setAttribute('aria-label', tr(`Theme: ${t}. Click to change.`))
}
function tr(s: string) { return t(s) }
applyTheme(pref.get('theme') ?? 'system')
themeBtn.addEventListener('click', () => {
  const next = THEMES[(THEMES.indexOf((pref.get('theme') ?? 'system') as (typeof THEMES)[number]) + 1) % THEMES.length]
  pref.set('theme', next); applyTheme(next)
})
// ?embed=1 drops the rail and tour chrome, for an iframe in slides or a course page
if (getParams().get('embed') === '1') document.querySelector('.app')!.classList.add('embed')
// presentation mode: no rail, larger captions; the Exit button (the rail and its Present button are hidden) or Escape
// leaves it, and focus moves between the two buttons so a keyboard reader is never left on a hidden one
const presentBtn = document.querySelector('.present-btn') as HTMLButtonElement
const exitBtn = document.querySelector('.present-exit') as HTMLButtonElement
const setPresent = (on: boolean) => {
  document.querySelector('.app')!.classList.toggle('present', on)
  presentBtn.setAttribute('aria-pressed', String(on))
  ;(on ? exitBtn : presentBtn).focus()
  poke()
}
presentBtn.addEventListener('click', () => setPresent(true))
exitBtn.addEventListener('click', () => setPresent(false))
window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && document.querySelector('.app.present')) setPresent(false) })
// the skip link lands on the current page's title
;(document.querySelector('.skip') as HTMLButtonElement).addEventListener('click', () => {
  const h1 = current?.root.querySelector('h1') as HTMLElement | null
  if (h1) { h1.tabIndex = -1; h1.focus() }
})
/* ---------- language: the fixed text of the page, and the switch ---------- */
const langBtn = document.querySelector('.lang-btn') as HTMLButtonElement
/** Elements marked data-t (text), data-t-html (markup), data-t-title and data-t-aria keep their English in data-en*. */
function applyStatic() {
  document.querySelectorAll<HTMLElement>('[data-t], [data-t-html]').forEach((el) => {
    const html = el.hasAttribute('data-t-html')
    el.dataset.en ??= html ? el.innerHTML : el.textContent ?? ''
    if (html) el.innerHTML = t(el.dataset.en); else el.textContent = t(el.dataset.en)
  })
  document.querySelectorAll<HTMLElement>('[data-t-title]').forEach((el) => { el.dataset.enTitle ??= el.title; el.title = t(el.dataset.enTitle) })
  document.querySelectorAll<HTMLElement>('[data-t-aria]').forEach((el) => { el.dataset.enAria ??= el.getAttribute('aria-label') ?? ''; el.setAttribute('aria-label', t(el.dataset.enAria)) })
  // the switch names the other language, in that language
  langBtn.textContent = lang === 'en' ? '中文' : 'English'
  langBtn.lang = lang === 'en' ? 'zh-CN' : 'en'
  langBtn.setAttribute('aria-label', lang === 'en' ? '切换到中文' : 'Switch to English')
}
applyStatic()
langBtn.addEventListener('click', () => setLang(lang === 'en' ? 'zh' : 'en'))
onLang(() => {
  applyStatic()
  applyTheme(pref.get('theme') ?? 'system')
  syncRailBtn()
  remount()
})
// dev only: the harvest used to list every string a page shows (see scripts/i18n-harvest.mjs)
if (import.meta.env.DEV) import('./exhibits/learn').then(({ learnTexts }) => Object.assign(window, { __tt: { harvest, t, setLang, learnTexts } }))
if (pref.get('palette') === 'cvd') document.documentElement.dataset.palette = 'cvd'
watchTheme(poke)
// canvases drawn before a font arrived redraw with it
document.fonts?.addEventListener('loadingdone', poke)

const railNav = document.querySelector('.nav') as HTMLElement
const main = document.querySelector('.main') as HTMLElement
const app = document.querySelector('.app') as HTMLElement
// the logo opens the opening animation again
const brand = document.querySelector('.brand') as HTMLAnchorElement
brand.href = routeHref('home')
brand.addEventListener('click', (e) => { e.preventDefault(); go('home') })

/* ---------- rail: collapsible on wide screens, a drawer on phones ---------- */
const railBtn = document.querySelector('.rail-btn') as HTMLButtonElement
const phone = matchMedia('(max-width: 860px)')
function syncRailBtn() {
  const openNow = phone.matches ? app.classList.contains('menu-open') : !app.classList.contains('collapsed')
  railBtn.setAttribute('aria-expanded', String(openNow))
  railBtn.textContent = phone.matches ? t(openNow ? 'Close' : 'Menu') : openNow ? '‹' : '›'
  railBtn.setAttribute('aria-label', t(phone.matches ? (openNow ? 'Close the menu' : 'Open the menu') : openNow ? 'Collapse the sidebar' : 'Expand the sidebar'))
}
if (pref.get('rail') === 'collapsed') app.classList.add('collapsed')
railBtn.addEventListener('click', () => {
  if (phone.matches) app.classList.toggle('menu-open')
  else pref.set('rail', app.classList.toggle('collapsed') ? 'collapsed' : 'open')
  syncRailBtn()
})
phone.addEventListener('change', syncRailBtn)
syncRailBtn()

/* ---------- rail ---------- */
/** Categories the reader has opened; the one holding the current route is always open. */
const open = new Set<string>(['anatomy'])

function exhibitLink(ex: Exhibit, active: string): HTMLElement {
  const li = document.createElement('li')
  const el = document.createElement(ex.route ? 'a' : 'div')
  el.className = 'ex' + (ex.route ? '' : ' soon')
  // the name only: what the page is about is on the page (the subtitle stays as a tooltip)
  el.innerHTML = '<span></span>'
  el.querySelector('span')!.textContent = t(ex.name)
  el.title = t(ex.tag).replace(/_\{?([^}\s·]+)\}?/g, '$1')
  if (!ex.route) {
    el.insertAdjacentHTML('beforeend', `<em class="soon-pill">${t('soon')}<span class="sr-only"> ${t('(coming soon)')}</span></em>`)
    el.title = t('In progress')
  } else {
    // how far this reader got: a check when every step was seen, else a fraction
    const pg = progress(ex.route)
    if (pg.of && pg.seen >= pg.of) el.insertAdjacentHTML('beforeend', `<em class="done" title="${t('Every step seen')}">✓<span class="sr-only"> ${t('seen')}</span></em>`)
    else if (pg.seen) el.insertAdjacentHTML('beforeend', `<em class="part" title="${t('Steps seen')}">${pg.seen}/${pg.of}<span class="sr-only"> ${t('steps seen')}</span></em>`)
  }
  if (ex.route) {
    const a = el as HTMLAnchorElement
    a.href = routeHref(ex.route)
    // fetch the page's code as soon as the pointer is on its way
    a.addEventListener('pointerenter', () => { ROUTES[ex.route!]?.().catch(() => {}) }, { once: true })
    if (ex.route === active) a.setAttribute('aria-current', 'page')
    a.addEventListener('click', (e) => { e.preventDefault(); go(ex.route!) })
  }
  li.appendChild(el)
  if (ex.children?.length) {
    const kids = document.createElement('ul')
    kids.className = 'kids'
    ex.children.forEach((c) => kids.appendChild(exhibitLink(c, active)))
    li.appendChild(kids)
  }
  return li
}

function renderRail(active: string) {
  // the rail is rebuilt, so remember what had focus in it and give focus back afterwards
  const had = railNav.contains(document.activeElement) ? document.activeElement as HTMLElement : null
  const key = had?.getAttribute('href') ?? (had?.dataset.cat ? 'cat:' + had.dataset.cat : null)
  railNav.innerHTML = ''
  const start = document.createElement('ul')
  start.className = 'start-link'
  start.append(exhibitLink(START, active), exhibitLink(FOUNDATIONS, active), exhibitLink(GLOSSARY, active))
  railNav.appendChild(start)
  for (const cat of CATEGORIES) {
    const all = exhibitsOf(cat), live = all.filter((e) => e.route).length
    if (all.some((e) => e.route === active)) open.add(cat.id)
    const isOpen = open.has(cat.id)
    const sec = document.createElement('section')
    sec.className = 'group'
    const h = document.createElement('button')
    h.className = 'cat'
    h.type = 'button'
    h.setAttribute('aria-expanded', String(isOpen))
    h.innerHTML = '<span class="chev" aria-hidden="true"></span><span class="t"></span><span class="n"></span>'
    h.querySelector('.t')!.textContent = t(cat.title)
    h.querySelector('.n')!.textContent = `${live} / ${all.length}`
    h.querySelector('.n')!.setAttribute('aria-label', t(`${live} of ${all.length} live`))
    h.dataset.cat = cat.id
    h.addEventListener('click', () => { if (open.has(cat.id)) open.delete(cat.id); else open.add(cat.id); renderRail(active) })
    const ul = document.createElement('ul')
    ul.hidden = !isOpen
    for (const e of cat.entries) {
      if (isHeading(e)) {
        const li = document.createElement('li')
        li.className = 'subh'
        li.textContent = t(e.heading)
        ul.appendChild(li)
      } else ul.appendChild(exhibitLink(e, active))
    }
    sec.append(h, ul)
    railNav.appendChild(sec)
  }
  if (key) (railNav.querySelector(key.startsWith('cat:') ? `[data-cat="${key.slice(4)}"]` : `[href="${key}"]`) as HTMLElement | null)?.focus()
}

/* ---------- the tour: previous / next ---------- */
/** Reading order across pages; the steps inside the model are numbered. */
const TOUR = ['start', 'foundations', 'anatomy', 'anatomy/tokenizer', 'anatomy/embedding', 'anatomy/layernorm', 'anatomy/attention', 'anatomy/mlp', 'anatomy/unembed', 'training/loss', 'training/backprop', 'training/optimizer', 'training/tokenizer', 'training/scaling', 'training/sft', 'training/dpo', 'training/lora', 'lineage/transformer-2017', 'lineage/llama', 'lineage/mixtral', 'lineage/deepseek', 'lineage/bert', 'lineage/t5', 'lineage/vit', 'lineage/clip', 'lineage/dit', 'lineage/mamba', 'lineage/compare', 'serving/kv-cache', 'serving/flashattention', 'serving/pagedattention', 'serving/continuous-batching', 'serving/speculative-decoding', 'serving/quantization', 'agents/in-context', 'agents/react', 'agents/tool-calling', 'agents/rag', 'agents/multi-agent']
const STEPS = TOUR.filter((r) => r.startsWith('anatomy/'))
const nameOf = (route: string) => [START, FOUNDATIONS, GLOSSARY, ...CATEGORIES.flatMap(exhibitsOf)].find((e) => e.route === route)?.name ?? route
const tourStep = (route: string, dir: number) => { const i = TOUR.indexOf(route); return i < 0 ? null : TOUR[i + dir] ?? null }

/** Top right of the page header: the step number and the previous and next pages of the tour. */
function chapterNav(route: string, root: HTMLElement) {
  const head = root.querySelector('.head')
  if (!head || !TOUR.includes(route)) return
  const row = document.createElement('nav')
  row.className = 'chapnav'
  row.setAttribute('aria-label', t('Tour'))
  const n = STEPS.indexOf(route)
  if (n >= 0) { const c = document.createElement('span'); c.className = 'count'; c.textContent = t(`step ${n + 1} of ${STEPS.length}`); row.appendChild(c) }
  for (const [dir, label] of [[-1, (t: string) => `‹ ${t}`], [1, (t: string) => `${t} ›`]] as const) {
    const to = tourStep(route, dir)
    if (!to) continue
    const b = document.createElement('button')
    b.type = 'button'
    b.className = dir < 0 ? 'chap prev' : 'chap next'
    b.textContent = label(t(nameOf(to)))
    b.title = t(dir < 0 ? 'Previous page (Shift + ←)' : 'Next page (Shift + →)')
    b.addEventListener('click', () => go(to))
    row.appendChild(b)
  }
  head.prepend(row)
  if (route.startsWith('anatomy')) pipeline(route, head)
}

/** Where this step sits in the forward pass: a clickable strip under the title. */
const PIPE: [string, string][] = [['tokenize', 'anatomy/tokenizer'], ['embed', 'anatomy/embedding'], ['ln', 'anatomy/layernorm'], ['attn', 'anatomy/attention'], ['ln', 'anatomy/layernorm'], ['mlp', 'anatomy/mlp'], ['× 12', 'anatomy'], ['ln_f · unembed · sample', 'anatomy/unembed']]
function pipeline(route: string, head: Element) {
  const strip = document.createElement('nav')
  strip.className = 'pipeline'
  strip.setAttribute('aria-label', t('Where this step sits in the forward pass'))
  PIPE.forEach(([label, to], i) => {
    if (i) strip.insertAdjacentHTML('beforeend', '<span class="sep" aria-hidden="true">→</span>')
    const b = document.createElement('button')
    b.type = 'button'
    b.innerHTML = rich(t(label))
    if (to === route && !(label === 'ln' && i === 4)) b.setAttribute('aria-current', 'step')
    b.addEventListener('click', () => go(to))
    strip.appendChild(b)
  })
  head.appendChild(strip)
}

window.addEventListener('keydown', (e) => {
  const tag = (e.target as HTMLElement).tagName
  if (!e.shiftKey || e.altKey || e.ctrlKey || e.metaKey || tag === 'INPUT' || tag === 'TEXTAREA' || !current) return
  const to = e.key === 'ArrowRight' ? tourStep(current.route, 1) : e.key === 'ArrowLeft' ? tourStep(current.route, -1) : null
  if (to) { e.preventDefault(); go(to) }
})

/** "Save frame": the drawing as it is now, as a PNG. */
function saveButton(route: string, root: HTMLElement) {
  const cv = root.querySelector('.stage canvas') as HTMLCanvasElement | null, meta = root.querySelector('.controls .meta')
  // the single-file preview runs where pages may not download files, so it has no Save button
  if (!cv || !meta || import.meta.env.MODE === 'artifact') return
  const b = document.createElement('button')
  b.type = 'button'; b.className = 'cycle'; b.textContent = t('Save frame'); b.title = t('Download the drawing as a PNG')
  b.addEventListener('click', () => cv.toBlob((blob) => {
    if (!blob) return
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = `token-trails-${route.replace(/\//g, '-')}.png`
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }))
  meta.appendChild(b)
}

/* ---------- router ---------- */
let current: { route: string; root: HTMLElement; destroy: () => void } | null = null

/** The route (and ?query) in the address; old #/ links, old prefixes and unknown routes are rewritten in place. */
function parse(): string {
  let { route: h, qs } = readAddress()
  let fix = !HASH_ROUTES && location.hash.startsWith('#/')
  for (const [from, to] of ALIASES) {
    if (h !== from && !h.startsWith(from + '/')) continue
    h = to + h.slice(from.length); fix = true
  }
  if (!h) h = DEFAULT_ROUTE
  if (!ROUTES[h]) { h = DEFAULT_ROUTE; qs = ''; fix = true }
  if (fix) try { history.replaceState(null, '', routeHref(h, qs)) } catch { /* sandboxed frames may refuse */ }
  return h + (qs ? '?' + qs : '')
}

const depth = (r: string) => r.split('/').length

let navTicket = 0
/** Open a route, e.g. 'anatomy/unembed' or 'anatomy/unembed?phase=sample' to start at a phase. */
async function go(target: string, origin?: { x: number; y: number }, push = true) {
  const [route, qs] = target.split('?')
  const load = ROUTES[route]
  if (!load) return
  if (current?.route === route) { current.root.dispatchEvent(new Event('tt-restart')); return }
  if (push) {
    // "← Forward pass" and the like: if that is where we came from, go back instead of stacking history
    if (!qs && history.state?.prev === route) { try { history.back(); return } catch { /* fall through */ } }
    try { history.pushState({ prev: current?.route ?? null }, '', routeHref(route, qs)) } catch { /* sandboxed frames may refuse */ }
  }
  // each page is its own chunk: wait for its code; a newer navigation that starts meanwhile wins
  const ticket = ++navTicket
  let mount: Mount | null = null
  try { mount = await load() } catch (err) {
    // after a deploy the old chunk names are gone: reload once for the new ones
    if (reloadOnce()) return
    console.error(err)
  }
  if (ticket !== navTicket) return
  openAtPhase(new URLSearchParams(qs ?? '').get('phase'))
  harvest.route = route
  const old = current
  const root = document.createElement('div')
  root.className = 'view'
  main.appendChild(root)
  let destroy: () => void
  try {
    if (!mount) throw new Error(`no code for ${route}`)
    destroy = mount(root, go)
    try { sessionStorage.removeItem('tt-reloaded') } catch { /* storage may be blocked */ }
  } catch (err) {
    // a page that fails to start says so instead of leaving a blank screen
    console.error(err)
    root.innerHTML = `<header class="head"><div><p class="eyebrow">${t('Something went wrong')}</p><div class="titlebar"><h1>${t('This page did not load')}</h1></div><p class="start-body">${t('Try reloading the page; the error is in the browser console.')}</p></div></header>`
    destroy = () => {}
  }
  openAtPhase(null)
  chapterNav(route, root)
  saveButton(route, root)
  current = { route, root, destroy }
  const cat = CATEGORIES.find((c) => exhibitsOf(c).some((e) => e.route === route))?.title
  document.title = route === 'home' ? 'Token Trails' : [t(nameOf(route)), cat && t(cat), 'Token Trails'].filter(Boolean).join(' · ')
  // the opening animation has the whole window: no rail
  app.classList.toggle('home', route === 'home')
  app.classList.remove('menu-open'); syncRailBtn()
  renderRail(route)
  // the next page of the tour is the likeliest next click: fetch its code while the reader reads this one
  const next = tourStep(route, 1)
  if (next) (window.requestIdleCallback ?? ((f: () => void) => setTimeout(f, 1500)))(() => { ROUTES[next]?.().catch(() => {}) })
  if (old) {
    old.root.inert = true // the leaving view can no longer take focus or be read
    // move focus to the new page's title so keyboard and screen-reader users land on it
    const h1 = root.querySelector('h1')
    if (h1 && push) { h1.tabIndex = -1; h1.focus({ preventScroll: true }) }
    transition(old, root, depth(route) >= depth(old.route) ? 'in' : 'out', origin)
  }
}

/** Zoom the old stage toward the clicked point (or away from it) while the views crossfade. */
function transition(old: { root: HTMLElement; destroy: () => void }, next: HTMLElement, dir: 'in' | 'out', origin?: { x: number; y: number }) {
  const done = () => { old.destroy(); old.root.remove() }
  if (reducedMotion()) { done(); return }
  old.root.classList.add('leaving')
  const ease = 'cubic-bezier(.45,0,.2,1)'
  const oldStage = old.root.querySelector('.stage') as HTMLElement | null
  const newStage = next.querySelector('.stage') as HTMLElement | null
  if (oldStage) {
    const r = oldStage.getBoundingClientRect()
    const ox = origin ? origin.x - r.left : r.width / 2, oy = origin ? origin.y - r.top : r.height / 2
    oldStage.style.transformOrigin = `${ox}px ${oy}px`
    oldStage.animate([{ transform: 'scale(1)' }, { transform: dir === 'in' ? 'scale(2.6)' : 'scale(0.86)' }], { duration: 520, easing: ease, fill: 'forwards' })
  }
  old.root.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 420, easing: ease, fill: 'forwards' })
  next.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 460, delay: 160, easing: ease, fill: 'backwards' })
  newStage?.animate([{ transform: dir === 'in' ? 'scale(0.94)' : 'scale(1.12)' }, { transform: 'scale(1)' }], { duration: 620, delay: 120, easing: ease, fill: 'backwards' })
  setTimeout(done, 540)
}

/** Rebuild the current page where it is (after a language switch): same route, same step. */
function remount() {
  if (!current) return
  const { route, qs } = readAddress(), target = (route || current.route) + (qs ? '?' + qs : '')
  const old = current
  current = null
  old.destroy(); old.root.remove()
  go(target, undefined, false)
}

/** Reload the page once (per tab session) when a page's code fails to load, as after a deploy renamed the chunks. */
function reloadOnce(): boolean {
  try {
    if (sessionStorage.getItem('tt-reloaded')) return false
    sessionStorage.setItem('tt-reloaded', '1')
  } catch { return false }
  location.reload()
  return true
}
window.addEventListener('popstate', () => go(parse(), undefined, false))
go(parse(), undefined, false)
