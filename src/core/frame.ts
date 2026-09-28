import { withTerms } from './glossary'
import { t } from './i18n'
import { legendList } from './legend'
import { pref } from './prefs'
import { C, type RGB } from './theme'

/** The shared page frame of an exhibit: header, stage, formula strip, caption line and control bar. */
export interface Spec {
  label: string
  value: string
  /** The real model's value, shown faintly after the drawn one, as "GPT-2 768". */
  real?: string
  /** What `real` refers to; defaults to GPT-2, '' for a plain note. */
  realLabel?: string
}

export interface FrameOptions {
  eyebrow: string
  title: string
  subtitle: string
  specs: Spec[]
  back?: { label: string; onClick: () => void }
  /** Reserve a strip under the stage for the focused cell's formula and note (detail views). */
  formula?: boolean
  /** How to inspect this view, shown faintly in the formula strip. */
  formulaHint?: string
}

/** One run of the formula line: text, its colour (C.ink / C.ink2 / C.mute or a token hue), optional alpha. */
export type FormulaSeg = [string, RGB, number?]

export interface Frame {
  stageHost: HTMLElement
  controls: HTMLElement
  /** title: phase name; sub: optional short form (hidden when equal to title). */
  setCaption(title: string, sub: string, text: string, shape: string): void
  setSubtitle(s: string): void
  /** Show a formula line and note under the stage (null clears it); `announce` also reads it to screen readers. */
  setFormula(segs: FormulaSeg[] | null, note?: string, announce?: boolean): void
  /** Replace the hint under the formula strip ('' hides it), e.g. per phase. */
  setHint(s: string): void
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
/**
 * Escape text and render identifiers like W_Q or d_model with a real subscript. Greek letters and
 * math signs are wrapped too, so uppercase labels (timeline, specs) never turn σ into Σ.
 */
export const rich = (s: string) =>
  esc(s).replace(/([A-Za-z]+)_([A-Za-z0-9]+)/g, '<span class="m">$1<sub>$2</sub></span>').replace(/(?:[A-Za-z]\u0302|[\u0370-\u03ff\u221a\u2211\u1d40])[A-Za-z0-9\u0302\u0370-\u03ff\u1d40]*/g, '<span class="m">$&</span>')

const ROLE = (c: RGB) => (c === C.ink ? 'f-ink' : c === C.ink2 ? 'f-ink2' : c === C.mute ? 'f-mute' : '')

export function createFrame(root: HTMLElement, o: FrameOptions): Frame {
  root.innerHTML = `
    <header class="head">
      <div>
        ${o.back ? `<button class="back" type="button">← ${esc(t(o.back.label))}</button>` : ''}
        <p class="eyebrow">${rich(t(o.eyebrow))}</p>
        <div class="titlebar"><h1>${esc(t(o.title))}</h1><p class="sub">${rich(t(o.subtitle))}</p></div>
      </div>
      <dl class="specs">
        ${o.specs.map((s) => `<div><dt>${rich(t(s.label))}</dt><dd>${rich(t(s.value))}${s.real ? `<small>${rich([t(s.realLabel ?? 'GPT-2'), t(s.real)].filter(Boolean).join(' '))}</small>` : ''}</dd></div>`).join('')}
      </dl>
    </header>
    <section class="stage"></section>
    ${o.formula ? `<section class="formula"><div class="f-line"></div><div class="f-note"></div><p class="f-hint"${o.formulaHint === '' ? ' hidden' : ''}>${esc(t(o.formulaHint ?? 'Hover a result cell to see how it is computed; click or tap it to pin it.'))}</p></section>` : ''}
    <section class="caption" aria-live="polite" aria-atomic="true">
      <div class="cap-title"><b></b><em></em></div>
      <p class="cap-text"></p>
      <code class="cap-shape" aria-hidden="true"></code>
    </section>
    <section class="controls" aria-label="${t('Playback')}"></section>`
  if (o.back) root.querySelector('.back')!.addEventListener('click', o.back.onClick)
  // how to read the pictures, under the controls; open by itself on a first visit
  const legend = document.createElement('details')
  legend.className = 'legend-d'
  legend.innerHTML = `<summary>${t('How to read the pictures')}</summary>${legendList()}<label class="legend-opts"><input type="checkbox"> ${t('Colour-blind-safe token colours')}</label><p class="legend-note">${t('In the header, a plain number is what the drawing uses and “GPT-2 768” is the real model’s size.')}</p>`
  root.querySelector('.controls')!.appendChild(legend)
  const cvd = legend.querySelector('input')!
  cvd.checked = document.documentElement.dataset.palette === 'cvd'
  cvd.addEventListener('change', () => {
    if (cvd.checked) document.documentElement.dataset.palette = 'cvd'
    else delete document.documentElement.dataset.palette
    pref.set('palette', cvd.checked ? 'cvd' : '')
  })
  // what a keyboard or touch inspection finds, read out to screen readers
  const live = document.createElement('p')
  live.className = 'sr-live'
  live.setAttribute('aria-live', 'polite')
  root.appendChild(live)
  if (!pref.get('legend-seen')) { legend.open = true; pref.set('legend-seen', '1') }
  const q = <T extends HTMLElement>(s: string) => root.querySelector(s) as T
  const title = q('.cap-title b'), short = q('.cap-title em'), text = q('.cap-text'), shape = q('.cap-shape'), sub = q('.titlebar .sub')
  const fLine = root.querySelector('.f-line'), fNote = root.querySelector('.f-note'), fHint = root.querySelector<HTMLElement>('.f-hint')
  let last = '', lastF = '', lastFAt = 0
  let pendingF: (() => void) | null = null
  return {
    stageHost: q('.stage'),
    controls: q('.controls'),
    setCaption(a, b, c, d) {
      a = t(a); b = t(b); c = t(c); d = t(d)
      const key = a + b + c + d
      if (key === last) return
      last = key
      title.innerHTML = rich(a)
      short.innerHTML = b === a ? '' : rich(b)
      text.innerHTML = withTerms(rich(c))
      shape.innerHTML = rich(d)
    },
    setHint(s) {
      s = t(s)
      if (!fHint || (fHint.textContent === s && fHint.hidden === !s)) return
      fHint.textContent = s
      fHint.hidden = !s
    },
    setSubtitle(s) {
      const html = rich(t(s))
      if (sub.innerHTML !== html) sub.innerHTML = html
    },
    setFormula(segs, note = '', announce = false) {
      segs = segs && segs.map(([x, c, a]) => [t(x), c, a] as FormulaSeg)
      note = t(note)
      if (announce && segs) live.textContent = segs.map((g) => g[0]).join('') + '. ' + note
      if (!fLine || !fNote) return
      const line = (segs ?? []).map(([t, c, a]) => {
        const role = ROLE(c), op = a !== undefined && a < 1 ? `opacity:${a.toFixed(2)};` : ''
        return `<span${role ? ` class="${role}"` : ''}${role && !op ? '' : ` style="${role ? '' : `color:rgb(${c.map((v) => v | 0).join(',')});`}${op}"`}>${rich(t)}</span>`
      }).join('')
      const key = line + '|' + note
      if (key === lastF) { pendingF = null; return }
      // during fast sweeps the focused cell changes every frame: show at most ~6 formulas a second
      const now = performance.now(), apply = () => { lastF = key; lastFAt = performance.now(); pendingF = null; fLine.innerHTML = line; fNote.innerHTML = segs ? rich(note) : '' }
      if (announce || now - lastFAt > 160) { apply(); return }
      if (!pendingF) setTimeout(() => pendingF?.(), 170)
      pendingF = apply
    },
  }
}

/** A small segmented toggle placed in the player's meta slot. */
export function toggle(parent: HTMLElement, label: string, options: string[], value: number, onChange: (i: number) => void) {
  const el = document.createElement('div')
  el.className = 'toggle'
  el.setAttribute('role', 'group')
  el.setAttribute('aria-label', t(label))
  const buttons = options.map((o, i) => {
    const b = document.createElement('button')
    b.textContent = t(o)
    b.setAttribute('aria-pressed', String(i === value))
    b.addEventListener('click', () => set(i, true))
    el.appendChild(b)
    return b
  })
  function set(i: number, notify: boolean) {
    buttons.forEach((b, k) => b.setAttribute('aria-pressed', String(k === i)))
    if (notify) onChange(i)
  }
  parent.prepend(el)
  return { set: (i: number) => set(i, false) }
}

/** A ‹ label n › stepper placed in the player's meta slot; wraps around at both ends. */
export function stepper(parent: HTMLElement, label: string, count: number, value: number, onChange: (i: number) => void) {
  const el = document.createElement('div')
  el.className = 'stepper'
  el.setAttribute('role', 'group')
  el.setAttribute('aria-label', t(label))
  el.innerHTML = `<button type="button" aria-label="${esc(t(`Previous ${label}`))}">‹</button><output></output><button type="button" aria-label="${esc(t(`Next ${label}`))}">›</button>`
  const [prev, next] = el.querySelectorAll('button'), out = el.querySelector('output')!
  let v = value
  function set(i: number, notify: boolean) {
    v = (i + count) % count
    out.textContent = t(`${label} ${v + 1}`)
    if (notify) onChange(v)
  }
  prev.addEventListener('click', () => set(v - 1, true))
  next.addEventListener('click', () => set(v + 1, true))
  set(value, false)
  parent.prepend(el)
  return { set: (i: number) => set(i, false) }
}
