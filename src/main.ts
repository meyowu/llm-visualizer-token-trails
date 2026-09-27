import './styles.css'
import { watchTheme } from './core/theme'
import { reducedMotion } from './core/util'
import { CATEGORIES, DEFAULT_ROUTE, ROUTES } from './exhibits/registry'

watchTheme()

const railNav = document.querySelector('.nav') as HTMLElement
const main = document.querySelector('.main') as HTMLElement

/* ---------- rail ---------- */
function renderRail(active: string) {
  railNav.innerHTML = ''
  for (const cat of CATEGORIES) {
    const sec = document.createElement('section')
    sec.className = 'group'
    const h = document.createElement('h2')
    h.textContent = cat.title
    const ul = document.createElement('ul')
    for (const ex of cat.items) {
      const li = document.createElement('li')
      const el = document.createElement(ex.route ? 'a' : 'div')
      el.className = 'ex' + (ex.route ? '' : ' soon')
      el.innerHTML = '<span></span><small></small>'
      el.querySelector('span')!.textContent = ex.name
      el.querySelector('small')!.textContent = ex.tag
      if (ex.route) {
        const a = el as HTMLAnchorElement
        a.href = '#/' + ex.route
        if (ex.route === active) a.setAttribute('aria-current', 'page')
        a.addEventListener('click', (e) => { e.preventDefault(); go(ex.route!) })
      }
      li.appendChild(el)
      ul.appendChild(li)
    }
    sec.append(h, ul)
    railNav.appendChild(sec)
  }
}

/* ---------- router ---------- */
let current: { route: string; root: HTMLElement; destroy: () => void } | null = null

function parse(): string {
  const h = location.hash.replace(/^#\/?/, '')
  return ROUTES[h] ? h : DEFAULT_ROUTE
}

const depth = (r: string) => r.split('/').length

function go(route: string, origin?: { x: number; y: number }, push = true) {
  if (!ROUTES[route] || current?.route === route) return
  if (push) {
    try { history.pushState(null, '', '#/' + route) } catch { /* sandboxed frames may refuse */ }
  }
  const old = current
  const root = document.createElement('div')
  root.className = 'view'
  main.appendChild(root)
  const destroy = ROUTES[route](root, go)
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
