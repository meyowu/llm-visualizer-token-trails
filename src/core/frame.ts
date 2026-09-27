/** The shared page frame of an exhibit: header, stage, caption line and control bar. */
export interface Spec {
  label: string
  value: string
  /** Real-model value, shown faintly after a toy value. */
  real?: string
}

export interface FrameOptions {
  eyebrow: string
  title: string
  subtitle: string
  specs: Spec[]
  back?: { label: string; onClick: () => void }
}

export interface Frame {
  stageHost: HTMLElement
  controls: HTMLElement
  /** title: phase name; sub: optional short form (hidden when equal to title). */
  setCaption(title: string, sub: string, text: string, shape: string): void
  setSubtitle(s: string): void
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
/** Escape text and render identifiers like W_Q or d_model with a real subscript. */
export const rich = (s: string) => esc(s).replace(/([A-Za-z]+)_([A-Za-z0-9]+)/g, '<span class="m">$1<sub>$2</sub></span>')

export function createFrame(root: HTMLElement, o: FrameOptions): Frame {
  root.innerHTML = `
    <header class="head">
      <div>
        ${o.back ? `<button class="back" type="button">← ${esc(o.back.label)}</button>` : ''}
        <p class="eyebrow">${rich(o.eyebrow)}</p>
        <h1>${esc(o.title)}<span class="sub">${rich(o.subtitle)}</span></h1>
      </div>
      <dl class="specs">
        ${o.specs.map((s) => `<div><dt>${rich(s.label)}</dt><dd>${rich(s.value)}${s.real ? `<small>/ ${rich(s.real)}</small>` : ''}</dd></div>`).join('')}
      </dl>
    </header>
    <section class="stage"></section>
    <section class="caption" aria-live="polite">
      <div class="cap-title"><b></b><em></em></div>
      <p class="cap-text"></p>
      <code class="cap-shape"></code>
    </section>
    <section class="controls" aria-label="Playback"></section>`
  if (o.back) root.querySelector('.back')!.addEventListener('click', o.back.onClick)
  const q = <T extends HTMLElement>(s: string) => root.querySelector(s) as T
  const title = q('.cap-title b'), short = q('.cap-title em'), text = q('.cap-text'), shape = q('.cap-shape'), sub = q('h1 .sub')
  let last = ''
  return {
    stageHost: q('.stage'),
    controls: q('.controls'),
    setCaption(a, b, c, d) {
      const key = a + b + c + d
      if (key === last) return
      last = key
      title.innerHTML = rich(a)
      short.innerHTML = b === a ? '' : rich(b)
      text.innerHTML = rich(c)
      shape.innerHTML = rich(d)
    },
    setSubtitle(s) {
      const html = rich(s)
      if (sub.innerHTML !== html) sub.innerHTML = html
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
