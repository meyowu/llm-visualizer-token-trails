import { rich } from './frame'
import { withTerms } from './glossary'
import { pref } from './prefs'
import { clamp } from './util'

export interface Phase {
  id: string
  /** Full name, shown in the caption. */
  name: string
  /** Timeline label; defaults to name. */
  short?: string
  dur: number
}
export interface TimedPhase extends Phase {
  start: number
  end: number
}

const PAUSE_ICON = 'M4 3h3v10H4zM9 3h3v10H9z'
const PLAY_ICON = 'M4.5 2.5v11l9-5.5z'
const SPEEDS = [0.25, 0.5, 1, 2]

/** The phase the next Player should open on, set by the router from a link like #/anatomy/unembed?phase=sample. */
let pendingPhase: string | null = null
export const openAtPhase = (id: string | null) => { pendingPhase = id }

/**
 * Owns the phase timeline of one exhibit: time, play/pause, step buttons, speed, pacing, the
 * scrubbable segmented timeline, and the Space / ← / → shortcuts. Everything a view draws is a
 * function of `t`. In step pacing (the default) playback holds at the end of every phase until
 * the reader continues.
 */
export class Player {
  readonly phases: TimedPhase[]
  readonly total: number
  t = 0
  speed = Number(pref.get('speed')) || 1
  playing = true
  /** Step pacing: hold at the end of each phase. */
  guided = pref.get('pace') !== 'auto'
  /** Holding at the end of a phase; playing again moves on to the next one. */
  held = false
  /** Called when t reaches the end. Without it the player stops at the last frame. */
  onEnd: (() => void) | null = null
  /** Caption text and shape of phase i, for the list of all steps. */
  describe: ((i: number) => [string, string]) | null = null
  /** Slot for exhibit-specific controls (head switcher, temperature…). */
  readonly meta: HTMLElement
  private byId: Record<string, TimedPhase> = {}
  private segs: { el: HTMLButtonElement; fill: HTMLElement }[] = []
  private steps: HTMLDetailsElement
  private shown = -1
  private btn: HTMLButtonElement
  private nextBtn: HTMLButtonElement
  private icon: SVGPathElement
  private offKeys: () => void

  constructor(phases: Phase[], controls: HTMLElement, opts: { playing?: boolean } = {}) {
    let acc = 0
    this.phases = phases.map((p) => {
      const tp = { ...p, start: acc, end: acc + p.dur }
      acc += p.dur
      this.byId[p.id] = tp
      return tp
    })
    this.total = acc

    this.btn = document.createElement('button')
    this.btn.className = 'play'
    this.btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor"/></svg>'
    this.icon = this.btn.querySelector('path')!
    this.btn.addEventListener('click', () => this.setPlaying(!this.playing))
    const stepBtn = (label: string, path: string, dir: number) => {
      const b = document.createElement('button')
      b.className = 'step'
      b.type = 'button'
      b.setAttribute('aria-label', label)
      b.title = label
      b.innerHTML = `<svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="${path}"/></svg>`
      b.addEventListener('click', () => this.step(dir))
      return b
    }
    const prevBtn = stepBtn('Replay this step (←)', 'M3 2.5h2v11H3zM14 2.5v11L6 8z', -1)
    this.nextBtn = stepBtn('Next step (→)', 'M11 2.5h2v11h-2zM2 2.5v11L10 8z', 1)

    const tl = document.createElement('div')
    tl.className = 'tl'
    tl.setAttribute('role', 'group')
    tl.setAttribute('aria-label', 'Phases')
    this.phases.forEach((p) => {
      const b = document.createElement('button')
      b.className = 'seg'
      b.style.flex = `${p.dur} 1 0`
      b.innerHTML = `<span class="track"><span class="fill"></span></span><span class="lbl"></span>`
      b.querySelector('.lbl')!.innerHTML = rich(p.short ?? p.name)
      b.setAttribute('aria-label', p.name.replace(/_/g, ' '))
      b.addEventListener('click', (e) => { if (e.detail === 0) this.t = p.start + 0.001 })
      tl.appendChild(b)
      this.segs.push({ el: b, fill: b.querySelector('.fill') as HTMLElement })
    })
    let dragging = false
    tl.addEventListener('pointerdown', (e) => { dragging = true; tl.setPointerCapture(e.pointerId); this.seekFromX(e.clientX) })
    tl.addEventListener('pointermove', (e) => { if (dragging) this.seekFromX(e.clientX) })
    tl.addEventListener('pointerup', () => { dragging = false })
    tl.addEventListener('pointercancel', () => { dragging = false })

    this.meta = document.createElement('div')
    this.meta.className = 'meta'
    if (!SPEEDS.includes(this.speed)) this.speed = 1
    // speed and pacing as single buttons that cycle, to keep the control bar on one line
    const cycle = (label: string, show: () => [string, string], next: () => void) => {
      const b = document.createElement('button')
      b.type = 'button'
      b.className = 'cycle'
      const sync = () => { const [t, title] = show(); b.textContent = t; b.title = title; b.setAttribute('aria-label', `${label}: ${title}`) }
      b.addEventListener('click', () => { next(); sync() })
      sync()
      return b
    }
    const pace = cycle('Pacing', () => (this.guided ? ['Step ⏸', 'Pause after each step (click for Auto)'] : ['Auto ▶', 'Play straight through (click for Step)']), () => {
      this.guided = !this.guided
      pref.set('pace', this.guided ? 'step' : 'auto')
    })
    const speed = cycle('Speed', () => [this.speed + '×', `Speed ${this.speed}× (click to change)`], () => {
      this.speed = SPEEDS[(SPEEDS.indexOf(this.speed) + 1) % SPEEDS.length]
      pref.set('speed', String(this.speed))
    })
    this.meta.append(pace, speed)
    const transport = document.createElement('div')
    transport.className = 'transport'
    transport.append(prevBtn, this.btn, this.nextBtn)
    // every step with its explanation, to read at leisure or jump to
    this.steps = document.createElement('details')
    this.steps.className = 'steps'
    this.steps.innerHTML = '<summary>All steps</summary><ol></ol>'
    this.steps.addEventListener('toggle', () => this.renderSteps())
    controls.append(transport, tl, this.meta, this.steps)
    // a pointer click leaves focus on the button, where Space would click it again instead of play / pause
    controls.addEventListener('click', (e) => { if (e.detail > 0) (e.target as HTMLElement).closest('button')?.blur() })
    if (pendingPhase) {
      const p = this.byId[pendingPhase]
      if (p) this.t = p.start + 0.001
      pendingPhase = null
    }

    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (e.defaultPrevented || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
      if (e.code === 'Space' && tag !== 'BUTTON' && tag !== 'SUMMARY' && tag !== 'A') { e.preventDefault(); this.setPlaying(!this.playing) }
      else if (e.key === 'ArrowRight') { e.preventDefault(); this.step(1) }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); this.step(-1) }
      // , and . nudge time while paused, for a frame-by-frame look
      else if ((e.key === ',' || e.key === '.') && !this.playing) { this.setHeld(false); this.t = clamp(this.t + (e.key === '.' ? 0.1 : -0.1), 0, this.total - 0.001) }
    }
    document.addEventListener('keydown', onKey)
    this.offKeys = () => document.removeEventListener('keydown', onKey)
    this.setPlaying(opts.playing ?? true)
  }

  prog(id: string) {
    const p = this.byId[id]
    return clamp((this.t - p.start) / p.dur)
  }
  start(id: string) {
    return this.byId[id].start
  }
  curIndex() {
    for (let i = this.phases.length - 1; i >= 0; i--) if (this.t >= this.phases[i].start) return i
    return 0
  }
  cur() {
    return this.phases[this.curIndex()]
  }

  setPlaying(v: boolean) {
    if (v && this.held) {
      const ci = this.curIndex()
      this.t = ci === this.phases.length - 1 ? this.total : this.phases[ci + 1].start + 0.001
    }
    if (v && !this.onEnd && this.t >= this.total) this.t = 0
    this.setHeld(false)
    this.playing = v
    this.icon.setAttribute('d', v ? PAUSE_ICON : PLAY_ICON)
    this.btn.setAttribute('aria-label', v ? 'Pause' : 'Play')
  }

  private setHeld(v: boolean) {
    this.held = v
    this.btn.classList.toggle('held', v)
    this.nextBtn.classList.toggle('held', v)
  }

  /** Jump to the next step, or back to the start of this one (the previous one when already there), and play it. */
  step(dir: number) {
    this.setHeld(false)
    this.seekPhase(dir)
    if (this.t < this.total || this.onEnd) this.setPlaying(true)
  }

  seekPhase(dir: number) {
    this.setHeld(false)
    const ci = this.curIndex()
    if (dir > 0) this.t = ci === this.phases.length - 1 ? this.total : this.phases[ci + 1].start + 0.001
    else this.t = this.t - this.phases[ci].start > 0.4 ? this.phases[ci].start + 0.001 : this.phases[Math.max(0, ci - 1)].start + 0.001
  }

  private seekFromX(cx: number) {
    this.setHeld(false)
    const rects = this.segs.map((s) => s.el.getBoundingClientRect())
    if (cx <= rects[0].left) { this.t = 0; return }
    for (let i = 0; i < this.phases.length; i++) {
      const r = rects[i]
      const nextL = i < this.phases.length - 1 ? rects[i + 1].left : Infinity
      if (cx < nextL) {
        this.t = Math.min(this.total - 0.001, this.phases[i].start + clamp((cx - r.left) / r.width) * this.phases[i].dur)
        return
      }
    }
  }

  tick(dt: number) {
    if (this.playing) {
      const end = this.phases[this.curIndex()].end, nt = this.t + dt * this.speed
      if (this.guided && this.t < end && nt >= end) {
        this.t = end - 1e-4
        this.setPlaying(false)
        this.setHeld(true)
        return
      }
      this.t = nt
    }
    if (this.t >= this.total) {
      if (this.onEnd) this.onEnd()
      else { this.t = this.total; if (this.playing) this.setPlaying(false) }
    }
  }

  updateUI() {
    const ci = this.curIndex()
    this.segs.forEach((s, i) => {
      const p = this.phases[i]
      s.fill.style.transform = `scaleX(${clamp((this.t - p.start) / p.dur)})`
      s.el.classList.toggle('on', i === ci)
    })
    if (ci !== this.shown) {
      this.shown = ci
      if (this.steps.open) this.renderSteps()
      this.linkPhase(this.phases[ci].id)
    }
  }

  /** Keep the address pointing at the current step, so a copied link opens here. */
  private linkPhase(id: string) {
    if (!this.btn.isConnected || this.btn.closest('.leaving')) return
    const [route] = location.hash.replace(/^#\/?/, '').split('?')
    const want = `#/${route}?phase=${id}`
    if (location.hash !== want) try { history.replaceState(history.state, '', want) } catch { /* sandboxed frames may refuse */ }
  }

  private renderSteps() {
    if (!this.steps.open) return
    const ol = this.steps.querySelector('ol')!, ci = this.curIndex()
    ol.innerHTML = ''
    this.phases.forEach((p, i) => {
      const [text, shape] = this.describe?.(i) ?? ['', '']
      const li = document.createElement('li')
      if (i === ci) li.setAttribute('aria-current', 'step')
      li.innerHTML = `<button type="button"><b>${rich(p.name)}</b></button><p>${withTerms(rich(text))}</p>${shape ? `<code>${rich(shape)}</code>` : ''}`
      li.querySelector('button')!.addEventListener('click', () => { this.setHeld(false); this.t = p.start + 0.001 })
      ol.appendChild(li)
    })
  }

  destroy() {
    this.offKeys()
  }
}
