import { rich } from './frame'
import { withTerms } from './glossary'
import { t as tr } from './i18n'
import { setParams } from './link'
import { markSeen } from './progress'
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
/** A predict-then-reveal question asked when playback reaches the start of `phase`. */
export interface Check {
  phase: string
  q: string
  options: string[]
  answer: number
  why: string
}
/** Lines of real code for this view; `at` lists the lines each phase runs. */
export interface Code {
  lines: string[]
  at: Record<string, number[]>
}

export interface TimedPhase extends Phase {
  start: number
  end: number
}

const PAUSE_ICON = 'M4 3h3v10H4zM9 3h3v10H9z'
const PLAY_ICON = 'M4.5 2.5v11l9-5.5z'
const SPEEDS = [0.25, 0.5, 1, 2]
/** Questions already answered right in this browser are not asked again. */
const answered = (c: Check) => { try { return !!JSON.parse(pref.get('checks') ?? '{}')[c.q] } catch { return false } }

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
  /** Questions to answer before a phase plays. */
  checks: Check[] = []
  /** Three takeaways shown when the last step ends. */
  recap: string[] = []
  /** Slot for exhibit-specific controls (head switcher, temperature…). */
  readonly meta: HTMLElement
  private byId: Record<string, TimedPhase> = {}
  private segs: { el: HTMLButtonElement; fill: HTMLElement }[] = []
  private steps: HTMLDetailsElement
  private shown = -1
  private card: HTMLElement
  private codeEl: HTMLDetailsElement | null = null
  private code: Code | null = null
  private asked = new Set<string>()
  private controls: HTMLElement
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
      b.setAttribute('aria-label', tr(label))
      b.title = tr(label)
      b.innerHTML = `<svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="${path}"/></svg>`
      b.addEventListener('click', () => this.step(dir))
      return b
    }
    const prevBtn = stepBtn('Replay this step (←)', 'M3 2.5h2v11H3zM14 2.5v11L6 8z', -1)
    this.nextBtn = stepBtn('Next step (→)', 'M11 2.5h2v11h-2zM2 2.5v11L10 8z', 1)

    const tl = document.createElement('div')
    tl.className = 'tl'
    tl.setAttribute('role', 'group')
    tl.setAttribute('aria-label', tr('Phases'))
    this.phases.forEach((p) => {
      const b = document.createElement('button')
      b.className = 'seg'
      b.style.flex = `${p.dur} 1 0`
      b.innerHTML = `<span class="track"><span class="fill"></span><span class="knob"></span></span><span class="lbl"></span>`
      b.querySelector('.lbl')!.innerHTML = rich(tr(p.short ?? p.name))
      // the accessible name starts with the visible label
      const short = tr(p.short ?? p.name).replace(/_/g, ' '), full = tr(p.name).replace(/_/g, ' ')
      b.setAttribute('aria-label', short === full ? full : `${short}: ${full}`)
      b.title = full
      b.addEventListener('click', (e) => { if (e.detail === 0) this.t = p.start + 0.001 })
      tl.appendChild(b)
      this.segs.push({ el: b, fill: b.querySelector('.fill') as HTMLElement })
    })
    // a label jumps to the start of its step; the track seeks and drags, paused while dragging
    let dragging = false, wasPlaying = false
    tl.addEventListener('pointerdown', (e) => {
      const seg = (e.target as HTMLElement).closest('.seg'), lbl = (e.target as HTMLElement).closest('.lbl')
      if (lbl && seg) { const p = this.phases[this.segs.findIndex((s) => s.el === seg)]; this.setHeld(false); this.t = p.start + 0.001; return }
      dragging = true; wasPlaying = this.playing; this.setPlaying(false)
      tl.setPointerCapture(e.pointerId); this.seekFromX(e.clientX)
    })
    tl.addEventListener('pointermove', (e) => { if (dragging) this.seekFromX(e.clientX) })
    const endDrag = () => { if (dragging && wasPlaying) this.setPlaying(true); dragging = false }
    tl.addEventListener('pointerup', endDrag)
    tl.addEventListener('pointercancel', endDrag)

    this.meta = document.createElement('div')
    this.meta.className = 'meta'
    if (!SPEEDS.includes(this.speed)) this.speed = 1
    // speed and pacing as single buttons that cycle, to keep the control bar on one line
    const cycle = (label: string, show: () => [string, string], next: () => void) => {
      const b = document.createElement('button')
      b.type = 'button'
      b.className = 'cycle'
      const sync = () => { const [t, title] = show().map(tr); b.textContent = t; b.title = title; b.setAttribute('aria-label', `${tr(label)}: ${title}`) }
      b.addEventListener('click', () => { next(); sync() })
      sync()
      return b
    }
    const pace = cycle('Pacing', () => (this.guided ? ['Step ⏸', 'Pause after each step (click for Auto)'] : ['Auto ▶', 'Play straight through (click for Step)']), () => {
      this.guided = !this.guided
      pref.set('pace', this.guided ? 'step' : 'auto')
      this.syncLive()
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
    this.steps.innerHTML = `<summary>${tr('All steps')}</summary><ol></ol>`
    this.steps.addEventListener('toggle', () => this.renderSteps())
    // a card above the controls for questions and the end-of-page recap
    this.card = document.createElement('section')
    this.card.className = 'coach'
    this.card.hidden = true
    this.card.setAttribute('aria-live', 'polite')
    this.controls = controls
    controls.append(this.card, transport, tl, this.meta, this.steps)
    // a pointer click leaves focus on the button, where Space would click it again instead of play / pause
    controls.addEventListener('click', (e) => { if (e.detail > 0) (e.target as HTMLElement).closest('button')?.blur() })
    if (pendingPhase) {
      const p = this.byId[pendingPhase]
      if (p) this.t = p.start + 0.001
      pendingPhase = null
    }
    // clicking the current page in the rail starts it over
    controls.closest('.view')?.addEventListener('tt-restart', () => { this.card.hidden = true; this.setHeld(false); this.t = 0; this.setPlaying(true) })

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
    // dev only: the translation harvest (scripts/i18n-harvest.mjs) steps the newest player directly
    if (import.meta.env.DEV) Object.assign(window, { __ttPlayer: this })
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
      const q = ci < this.phases.length - 1 && this.pending(this.phases[ci + 1].id)
      if (q) { this.setHeld(false); this.ask(q); return }
    }
    if (v && !this.onEnd && this.t >= this.total) this.t = 0
    this.setHeld(false)
    this.playing = v
    this.syncLive()
    this.icon.setAttribute('d', v ? PAUSE_ICON : PLAY_ICON)
    this.btn.setAttribute('aria-label', tr(v ? 'Pause' : 'Play'))
  }

  /** Captions are announced when the reader is in charge (paused, or stepping), not during Auto playback. */
  private syncLive() {
    this.btn.closest('.view')?.querySelector('.caption')?.setAttribute('aria-live', this.playing && !this.guided ? 'off' : 'polite')
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
    const q = dir > 0 && this.pending(this.cur().id)
    if (q) { this.setPlaying(false); this.ask(q); return }
    if (this.t < this.total || this.onEnd) this.setPlaying(true)
  }
  /** The question waiting at a phase, if it has not been asked here or answered before. */
  private pending(phase: string) {
    return this.checks.find((c) => c.phase === phase && !this.asked.has(phase) && !answered(c)) ?? null
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
      // a question waits at the start of its phase
      const crossing = this.checks.find((c) => { const p = this.byId[c.phase]; return p && this.t < p.start && nt >= p.start })
      const q = crossing && this.pending(crossing.phase)
      if (q) { this.t = this.byId[q.phase].start + 0.001; this.setPlaying(false); this.ask(q); return }
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
    // the recap shows once the last step has played to its end
    const atEnd = !this.onEnd && this.recap.length > 0 && ci === this.phases.length - 1 && (this.held || (!this.playing && this.t >= this.total - 0.01))
    if (atEnd && this.card.hidden) this.showRecap()
    else if (!atEnd && this.card.dataset.kind === 'recap' && !this.card.hidden) this.card.hidden = true
    this.segs.forEach((s, i) => {
      const p = this.phases[i], f = clamp((this.t - p.start) / p.dur)
      s.fill.style.transform = `scaleX(${f})`
      s.el.classList.toggle('on', i === ci)
      if (i === ci) { s.el.setAttribute('aria-current', 'step'); (s.el.querySelector('.knob') as HTMLElement).style.left = `${f * 100}%` }
      else s.el.removeAttribute('aria-current')
    })
    if (ci !== this.shown) {
      this.shown = ci
      if (this.steps.open) this.renderSteps()
      this.linkPhase(this.phases[ci].id)
      this.markCode()
    }
  }

  /** Keep the address pointing at the current step, so a copied link opens here. */
  private linkPhase(id: string) {
    if (!this.btn.isConnected || this.btn.closest('.leaving')) return
    setParams({ phase: id })
    markSeen(location.hash.replace(/^#\/?/, '').split('?')[0], id, this.phases.length)
  }

  /** Show a question card; playback waits until it is answered or skipped. */
  private ask(c: Check) {
    this.asked.add(c.phase)
    this.card.dataset.kind = 'check'
    this.card.hidden = false
    this.card.innerHTML = `<p class="coach-k">${tr('Predict first')}</p><p class="coach-q">${withTerms(rich(tr(c.q)))}</p><div class="coach-opts"></div><p class="coach-why" hidden></p><div class="coach-go"><button type="button" class="chap">${tr('Skip')}</button></div>`
    const opts = this.card.querySelector('.coach-opts')!, why = this.card.querySelector('.coach-why') as HTMLElement, go = this.card.querySelector('.coach-go button') as HTMLButtonElement
    c.options.forEach((o, i) => {
      const b = document.createElement('button')
      b.type = 'button'
      b.innerHTML = rich(tr(o))
      b.addEventListener('click', () => {
        const ok = i === c.answer
        opts.querySelectorAll('button').forEach((x, k) => { (x as HTMLButtonElement).disabled = true; if (k === c.answer) x.classList.add('right'); else if (x === b) x.classList.add('wrong') })
        why.hidden = false
        why.innerHTML = `${tr(ok ? 'Right.' : 'Not quite.')} ${withTerms(rich(tr(c.why)))}`
        go.textContent = tr('Watch it ›')
        go.focus()
        if (ok) try { const m = JSON.parse(pref.get('checks') ?? '{}'); m[c.q] = 1; pref.set('checks', JSON.stringify(m)) } catch { /* ignore */ }
      })
      opts.appendChild(b)
    })
    go.addEventListener('click', () => { this.card.hidden = true; this.setPlaying(true) })
    ;(opts.firstElementChild as HTMLElement | null)?.focus({ preventScroll: true })
  }

  private showRecap() {
    this.card.dataset.kind = 'recap'
    this.card.hidden = false
    this.card.innerHTML = `<p class="coach-k">${tr('Recap')}</p><ul>${this.recap.map((r) => `<li>${withTerms(rich(tr(r)))}</li>`).join('')}</ul><div class="coach-go"><button type="button" class="chap replay">${tr('↺ Replay')}</button></div>`
    const go = this.card.querySelector('.coach-go')!
    this.card.querySelector('.replay')!.addEventListener('click', () => { this.card.hidden = true; this.t = 0; this.setPlaying(true) })
    const next = this.btn.closest('.view')?.querySelector('.chapnav .chap.next') as HTMLButtonElement | null
    if (next) {
      const b = document.createElement('button')
      b.type = 'button'; b.className = 'chap primary'; b.textContent = next.textContent ?? 'Next ›'
      b.addEventListener('click', () => next.click())
      go.appendChild(b)
    }
  }

  /** Further reading for this page, in a drawer next to the code. */
  setRefs(refs: [string, string][]) {
    if (!refs.length) return
    const d = document.createElement('details')
    d.className = 'steps refs-d'
    d.innerHTML = `<summary>${tr('Go deeper')}</summary><ul>${refs.map(([t, u]) => `<li><a href="${u}" target="_blank" rel="noopener">${t.replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[ch]!)}</a></li>`).join('')}</ul>`
    this.controls.appendChild(d)
  }

  /** Real code for this view in a drawer; the current step's lines are marked. */
  setCode(code: Code) {
    this.code = code
    this.codeEl = document.createElement('details')
    this.codeEl.className = 'steps code-d'
    this.codeEl.innerHTML = `<summary>${tr('Code')}</summary><pre><code>${code.lines.map((l) => `<span>${l.replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[ch]!)}</span>`).join('\n')}</code></pre>`
    this.controls.appendChild(this.codeEl)
    this.markCode()
  }
  private markCode() {
    if (!this.code || !this.codeEl) return
    const on = new Set(this.code.at[this.phases[this.curIndex()].id] ?? [])
    this.codeEl.querySelectorAll('pre span').forEach((el, i) => el.classList.toggle('on', on.has(i)))
  }

  private renderSteps() {
    if (!this.steps.open) return
    const ol = this.steps.querySelector('ol')!, ci = this.curIndex()
    ol.innerHTML = ''
    this.phases.forEach((p, i) => {
      const [text, shape] = this.describe?.(i) ?? ['', '']
      const li = document.createElement('li')
      if (i === ci) li.setAttribute('aria-current', 'step')
      li.innerHTML = `<button type="button"><b>${rich(tr(p.name))}</b></button><p>${withTerms(rich(tr(text)))}</p>${shape ? `<code>${rich(tr(shape))}</code>` : ''}`
      li.querySelector('button')!.addEventListener('click', () => { this.setHeld(false); this.t = p.start + 0.001 })
      ol.appendChild(li)
    })
  }

  destroy() {
    this.offKeys()
  }
}
