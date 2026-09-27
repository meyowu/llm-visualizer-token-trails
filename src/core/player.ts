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
const SPEEDS = [0.25, 0.5, 1, 2]

/** Reader preferences shared by every exhibit, kept in this browser when storage is available. */
const pref = {
  get(key: string): string | null { try { return localStorage.getItem('tt-' + key) } catch { return null } },
  set(key: string, v: string) { try { localStorage.setItem('tt-' + key, v) } catch { /* private mode */ } },
}

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
  /** Slot for exhibit-specific controls (head switcher, temperature…). */
  readonly meta: HTMLElement
  private byId: Record<string, TimedPhase> = {}
  private segs: { el: HTMLButtonElement; fill: HTMLElement }[] = []
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
    const group = (label: string, options: [string, string][], on: string, pick: (v: string) => void) => {
      const g = document.createElement('div')
      g.className = 'toggle'
      g.setAttribute('role', 'group')
      g.setAttribute('aria-label', label)
      for (const [v, text, title] of options.map(([v, t]) => [v, ...t.split('|')])) {
        const b = document.createElement('button')
        b.type = 'button'
        b.textContent = text
        if (title) b.title = title
        b.setAttribute('aria-pressed', String(v === on))
        b.addEventListener('click', () => { pick(v); g.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))) })
        g.appendChild(b)
      }
      return g
    }
    const pace = group('Pacing', [['step', 'Step|Pause after each step'], ['auto', 'Auto|Play continuously']], this.guided ? 'step' : 'auto', (v) => {
      this.guided = v === 'step'
      pref.set('pace', v)
    })
    const speed = group('Speed', SPEEDS.map((s) => [String(s), s + '×']), String(this.speed), (v) => { this.speed = +v; pref.set('speed', v) })
    this.meta.append(pace, speed)
    const transport = document.createElement('div')
    transport.className = 'transport'
    transport.append(prevBtn, this.btn, this.nextBtn)
    controls.append(transport, tl, this.meta)

    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
      if (e.code === 'Space' && tag !== 'BUTTON') { e.preventDefault(); this.setPlaying(!this.playing) }
      else if (e.key === 'ArrowRight') { e.preventDefault(); this.step(1) }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); this.step(-1) }
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
  }

  destroy() {
    this.offKeys()
  }
}
