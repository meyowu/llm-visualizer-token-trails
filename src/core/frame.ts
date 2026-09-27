import { withTerms } from './glossary'
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
  /** Show a formula line and note under the stage (null clears it). */
  setFormula(segs: FormulaSeg[] | null, note?: string): void
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
        ${o.back ? `<button class="back" type="button">← ${esc(o.back.label)}</button>` : ''}
        <p class="eyebrow">${rich(o.eyebrow)}</p>
        <h1>${esc(o.title)}<span class="sub">${rich(o.subtitle)}</span></h1>
      </div>
      <dl class="specs">
        ${o.specs.map((s) => `<div><dt>${rich(s.label)}</dt><dd>${rich(s.value)}${s.real ? `<small>${rich([s.realLabel ?? 'GPT-2', s.real].filter(Boolean).join(' '))}</small>` : ''}</dd></div>`).join('')}
      </dl>
    </header>
    <section class="stage"></section>
    ${o.formula ? `<section class="formula"><div class="f-line"></div><div class="f-note"></div><p class="f-hint">${esc(o.formulaHint ?? 'Hover a result cell to see how it is computed; click or tap it to pin it.')}</p></section>` : ''}
    <section class="caption" aria-live="polite">
      <div class="cap-title"><b></b><em></em></div>
      <p class="cap-text"></p>
      <code class="cap-shape"></code>
    </section>
    <section class="controls" aria-label="Playback"></section>`
  if (o.back) root.querySelector('.back')!.addEventListener('click', o.back.onClick)
  // how to read the pictures, under the controls; open by itself on a first visit
  const legend = document.createElement('details')
  legend.className = 'legend-d'
  legend.innerHTML = `<summary>How to read the pictures</summary>${legendList()}<p class="legend-note">In the header, a plain number is what the drawing uses and “GPT-2 768” is the real model’s size.</p>`
  root.querySelector('.controls')!.appendChild(legend)
  if (!pref.get('legend-seen')) { legend.open = true; pref.set('legend-seen', '1') }
  const q = <T extends HTMLElement>(s: string) => root.querySelector(s) as T
  const title = q('.cap-title b'), short = q('.cap-title em'), text = q('.cap-text'), shape = q('.cap-shape'), sub = q('h1 .sub')
  const fLine = root.querySelector('.f-line'), fNote = root.querySelector('.f-note')
  let last = '', lastF = ''
  return {
    stageHost: q('.stage'),
    controls: q('.controls'),
    setCaption(a, b, c, d) {
      const key = a + b + c + d
      if (key === last) return
      last = key
      title.innerHTML = rich(a)
      short.innerHTML = b === a ? '' : rich(b)
      text.innerHTML = withTerms(rich(c))
      shape.innerHTML = rich(d)
    },
    setSubtitle(s) {
      const html = rich(s)
      if (sub.innerHTML !== html) sub.innerHTML = html
    },
    setFormula(segs, note = '') {
      if (!fLine || !fNote) return
      const line = (segs ?? []).map(([t, c, a]) => {
        const role = ROLE(c), op = a !== undefined && a < 1 ? `opacity:${a.toFixed(2)};` : ''
        return `<span${role ? ` class="${role}"` : ''}${role && !op ? '' : ` style="${role ? '' : `color:rgb(${c.map((v) => v | 0).join(',')});`}${op}"`}>${rich(t)}</span>`
      }).join('')
      const key = line + '|' + note
      if (key === lastF) return
      lastF = key
      fLine.innerHTML = line
      fNote.innerHTML = segs ? rich(note) : ''
    },
  }
}

/** A small segmented toggle placed in the player's meta slot. */
export function toggle(parent: HTMLElement, label: string, options: string[], value: number, onChange: (i: number) => void) {
  const el = document.createElement('div')
  el.className = 'toggle'
  el.setAttribute('role', 'group')
  el.setAttribute('aria-label', label)
  const buttons = options.map((t, i) => {
    const b = document.createElement('button')
    b.textContent = t
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
  el.setAttribute('aria-label', label)
  el.innerHTML = `<button type="button" aria-label="Previous ${esc(label)}">‹</button><output></output><button type="button" aria-label="Next ${esc(label)}">›</button>`
  const [prev, next] = el.querySelectorAll('button'), out = el.querySelector('output')!
  let v = value
  function set(i: number, notify: boolean) {
    v = (i + count) % count
    out.textContent = `${label} ${v + 1}`
    if (notify) onChange(v)
  }
  prev.addEventListener('click', () => set(v - 1, true))
  next.addEventListener('click', () => set(v + 1, true))
  set(value, false)
  parent.prepend(el)
  return { set: (i: number) => set(i, false) }
}
