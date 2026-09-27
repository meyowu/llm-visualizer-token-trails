import './styles.css'
import { rich } from './core/frame'
import { openAtPhase } from './core/player'
import { poke } from './core/stage'
import { watchTheme } from './core/theme'
import { reducedMotion } from './core/util'
import { ALIASES, CATEGORIES, DEFAULT_ROUTE, ROUTES, START, exhibitsOf, isHeading, type Exhibit } from './exhibits/registry'

watchTheme(poke)

const railNav = document.querySelector('.nav') as HTMLElement
const main = document.querySelector('.main') as HTMLElement

/* ---------- rail ---------- */
/** Categories the reader has opened; the one holding the current route is always open. */
const open = new Set<string>(['anatomy'])

function exhibitLink(ex: Exhibit, active: string): HTMLElement {
  const li = document.createElement('li')
  const el = document.createElement(ex.route ? 'a' : 'div')
  el.className = 'ex' + (ex.route ? '' : ' soon')
  el.innerHTML = '<span></span><small></small>'
  el.querySelector('span')!.textContent = ex.name
  el.querySelector('small')!.innerHTML = rich(ex.tag)
  if (ex.route) {
    const a = el as HTMLAnchorElement
    a.href = '#/' + ex.route
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
  railNav.innerHTML = ''
  const start = document.createElement('ul')
  start.className = 'start-link'
  start.appendChild(exhibitLink(START, active))
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
    h.querySelector('.t')!.textContent = cat.title
    h.querySelector('.n')!.textContent = `${live} / ${all.length}`
    h.addEventListener('click', () => { if (open.has(cat.id)) open.delete(cat.id); else open.add(cat.id); renderRail(active) })
    const ul = document.createElement('ul')
    ul.hidden = !isOpen
    for (const e of cat.entries) {
      if (isHeading(e)) {
        const li = document.createElement('li')
        li.className = 'subh'
        li.textContent = e.heading
        ul.appendChild(li)
      } else ul.appendChild(exhibitLink(e, active))
    }
    sec.append(h, ul)
    railNav.appendChild(sec)
  }
}

/* ---------- the tour: previous / next ---------- */
/** Reading order across pages; the Anatomy steps are numbered. */
const TOUR = ['start', 'anatomy', 'anatomy/tokenizer', 'anatomy/embedding', 'anatomy/layernorm', 'anatomy/attention', 'anatomy/mlp', 'anatomy/unembed', 'lineage/llama']
const STEPS = TOUR.filter((r) => r.startsWith('anatomy/'))
const nameOf = (route: string) => [START, ...CATEGORIES.flatMap(exhibitsOf)].find((e) => e.route === route)?.name ?? route
const tourStep = (route: string, dir: number) => { const i = TOUR.indexOf(route); return i < 0 ? null : TOUR[i + dir] ?? null }

/** Top right of the page header: the step number and the previous and next pages of the tour. */
function chapterNav(route: string, root: HTMLElement) {
  const head = root.querySelector('.head')
  if (!head || !TOUR.includes(route)) return
  const row = document.createElement('nav')
  row.className = 'chapnav'
  row.setAttribute('aria-label', 'Tour')
  const n = STEPS.indexOf(route)
  if (n >= 0) { const c = document.createElement('span'); c.className = 'count'; c.textContent = `step ${n + 1} of ${STEPS.length}`; row.appendChild(c) }
  for (const [dir, label] of [[-1, (t: string) => `‹ ${t}`], [1, (t: string) => `${t} ›`]] as const) {
    const to = tourStep(route, dir)
    if (!to) continue
    const b = document.createElement('button')
    b.type = 'button'
    b.className = 'chap'
    b.textContent = label(nameOf(to))
    b.title = `${dir < 0 ? 'Previous' : 'Next'} page (Shift + ${dir < 0 ? '←' : '→'})`
    b.addEventListener('click', () => go(to))
    row.appendChild(b)
  }
  head.prepend(row)
}

window.addEventListener('keydown', (e) => {
  const tag = (e.target as HTMLElement).tagName
  if (!e.shiftKey || e.altKey || e.ctrlKey || e.metaKey || tag === 'INPUT' || tag === 'TEXTAREA' || !current) return
  const to = e.key === 'ArrowRight' ? tourStep(current.route, 1) : e.key === 'ArrowLeft' ? tourStep(current.route, -1) : null
  if (to) { e.preventDefault(); go(to) }
})

/* ---------- router ---------- */
let current: { route: string; root: HTMLElement; destroy: () => void } | null = null

/** The route (and ?query) in the address, with old prefixes rewritten. */
function parse(): string {
  let [h, qs] = location.hash.replace(/^#\/?/, '').split('?')
  for (const [from, to] of ALIASES) {
    if (h !== from && !h.startsWith(from + '/')) continue
    h = to + h.slice(from.length)
    try { history.replaceState(null, '', '#/' + h + (qs ? '?' + qs : '')) } catch { /* sandboxed frames may refuse */ }
  }
  return ROUTES[h] ? h + (qs ? '?' + qs : '') : DEFAULT_ROUTE
}

const depth = (r: string) => r.split('/').length

/** Open a route, e.g. 'anatomy/unembed' or 'anatomy/unembed?phase=sample' to start at a phase. */
function go(target: string, origin?: { x: number; y: number }, push = true) {
  const [route, qs] = target.split('?')
  if (!ROUTES[route] || current?.route === route) return
  if (push) {
    try { history.pushState(null, '', '#/' + target) } catch { /* sandboxed frames may refuse */ }
  }
  openAtPhase(new URLSearchParams(qs ?? '').get('phase'))
  const old = current
  const root = document.createElement('div')
  root.className = 'view'
  main.appendChild(root)
  const destroy = ROUTES[route](root, go)
  openAtPhase(null)
  chapterNav(route, root)
  current = { route, root, destroy }
  renderRail(route)
  if (old) transition(old, root, depth(route) >= depth(old.route) ? 'in' : 'out', origin)
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

window.addEventListener('popstate', () => go(parse(), undefined, false))
go(parse(), undefined, false)
