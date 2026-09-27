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
  setCaption(zh: string, en: string, text: string, shape: string): void
  setSubtitle(s: string): void
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)

export function createFrame(root: HTMLElement, o: FrameOptions): Frame {
  root.innerHTML = `
    <header class="head">
      <div>
        ${o.back ? `<button class="back" type="button">← ${esc(o.back.label)}</button>` : ''}
        <p class="eyebrow">${esc(o.eyebrow)}</p>
        <h1>${esc(o.title)}<span class="sub">${esc(o.subtitle)}</span></h1>
      </div>
      <dl class="specs">
        ${o.specs.map((s) => `<div><dt>${esc(s.label)}</dt><dd>${esc(s.value)}${s.real ? `<small>/ ${esc(s.real)}</small>` : ''}</dd></div>`).join('')}
      </dl>
    </header>
    <section class="stage"></section>
    <section class="caption" aria-live="polite">
      <div class="cap-title"><b></b><em></em></div>
      <p class="cap-text"></p>
      <code class="cap-shape"></code>
    </section>
    <section class="controls" aria-label="播放控制"></section>`
  if (o.back) root.querySelector('.back')!.addEventListener('click', o.back.onClick)
  const q = <T extends HTMLElement>(s: string) => root.querySelector(s) as T
  const zh = q('.cap-title b'), en = q('.cap-title em'), text = q('.cap-text'), shape = q('.cap-shape'), sub = q('h1 .sub')
  let last = ''
  return {
    stageHost: q('.stage'),
    controls: q('.controls'),
    setCaption(a, b, c, d) {
      const key = a + b + c + d
      if (key === last) return
      last = key
      zh.textContent = a
      en.textContent = b
      text.textContent = c
      shape.textContent = d
    },
    setSubtitle(s) {
      if (sub.textContent !== s) sub.textContent = s
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
