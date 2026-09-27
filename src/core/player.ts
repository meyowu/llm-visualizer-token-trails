import { rich } from './frame'
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

/**
 * Owns the phase timeline of one exhibit: time, play/pause, speed, the scrubbable segmented
 * timeline, and the Space / ← / → shortcuts. Everything a view draws is a function of `t`.
 */
export class Player {
  readonly phases: TimedPhase[]
  readonly total: number
  t = 0
  speed = 1
  playing = true
  /** Called when t reaches the end. Without it the player stops at the last frame. */
  onEnd: (() => void) | null = null
  /** Slot for exhibit-specific controls (head switcher, temperature…). */
  readonly meta: HTMLElement
  private byId: Record<string, TimedPhase> = {}
  private segs: { el: HTMLButtonElement; fill: HTMLElement }[] = []
  private btn: HTMLButtonElement
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
    const speed = document.createElement('div')
    speed.className = 'toggle'
    speed.setAttribute('role', 'group')
    speed.setAttribute('aria-label', 'Speed')
    ;[0.5, 1, 2].forEach((s) => {
      const b = document.createElement('button')
      b.textContent = s + '×'
      b.setAttribute('aria-pressed', String(s === 1))
      b.addEventListener('click', () => {
        this.speed = s
        speed.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)))
      })
      speed.appendChild(b)
    })
    this.meta.appendChild(speed)
    controls.append(this.btn, tl, this.meta)

    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return
      if (e.code === 'Space' && tag !== 'BUTTON') { e.preventDefault(); this.setPlaying(!this.playing) }
      else if (e.key === 'ArrowRight') { e.preventDefault(); this.seekPhase(1) }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); this.seekPhase(-1) }
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
    if (v && !this.onEnd && this.t >= this.total) this.t = 0
    this.playing = v
    this.icon.setAttribute('d', v ? PAUSE_ICON : PLAY_ICON)
    this.btn.setAttribute('aria-label', v ? 'Pause' : 'Play')
  }

  seekPhase(dir: number) {
    const ci = this.curIndex()
    if (dir > 0) this.t = ci === this.phases.length - 1 ? this.total : this.phases[ci + 1].start + 0.001
    else this.t = this.t - this.phases[ci].start > 0.4 ? this.phases[ci].start + 0.001 : this.phases[Math.max(0, ci - 1)].start + 0.001
  }

  private seekFromX(cx: number) {
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
    if (this.playing) this.t += dt * this.speed
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
  }

  destroy() {
    this.offKeys()
  }
}
