import { localizeCanvas, t } from './i18n'
import { C, rgba } from './theme'

/**
 * A DPR-aware canvas that fills its host element. The drawing surface never shrinks below
 * minW × minH: a narrower host shows it scaled down to fit the width, and on phones a toggle
 * shows it at full size, scrolling sideways.
 */
export class Stage {
  readonly canvas: HTMLCanvasElement
  readonly ctx: CanvasRenderingContext2D
  W = 0
  H = 0
  dpr = 1
  onResize: (() => void) | null = null
  private ro: ResizeObserver
  private scroll: HTMLElement
  private zoomBtn: HTMLButtonElement
  /** On narrow screens: true shows the drawing at full size (scrolls), false scales it to fit. */
  private full = false

  constructor(private host: HTMLElement, private minW: number, private minH: number, label: string) {
    this.scroll = document.createElement('div')
    this.scroll.className = 'stage-scroll'
    this.canvas = document.createElement('canvas')
    this.canvas.setAttribute('role', 'img')
    this.canvas.setAttribute('aria-label', t(label))
    this.scroll.appendChild(this.canvas)
    this.zoomBtn = document.createElement('button')
    this.zoomBtn.type = 'button'
    this.zoomBtn.className = 'stage-zoom'
    this.zoomBtn.hidden = true
    this.zoomBtn.addEventListener('click', () => { this.full = !this.full; this.resize() })
    host.append(this.scroll, this.zoomBtn)
    this.scroll.addEventListener('scroll', () => this.edges(), { passive: true })
    this.ctx = this.canvas.getContext('2d')!
    localizeCanvas(this.ctx)
    this.ro = new ResizeObserver(() => this.resize())
    this.ro.observe(host)
    this.watchDpr()
    this.resize()
  }

  resize() {
    // layout size, not getBoundingClientRect: that includes the transforms of the page transition
    const r = { width: this.host.clientWidth, height: this.host.clientHeight }
    // narrower than the drawing: scale it down to fit; on phones (under 3/4 size) offer a full-size view that scrolls
    const k = r.width > 0 && r.width < this.minW ? r.width / this.minW : 1
    const small = k < 0.75, s = small && this.full ? 1 : k
    this.W = s < 1 ? this.minW : Math.max(this.minW, Math.floor(r.width))
    this.H = Math.max(this.minH, Math.floor(r.height / s))
    // the page scrolls instead of clipping the bottom of the drawing
    const mh = Math.ceil(this.minH * s) + 'px'
    if (this.host.style.minHeight !== mh) this.host.style.minHeight = mh
    this.dpr = Math.min(2, window.devicePixelRatio || 1)
    this.canvas.width = this.W * this.dpr
    this.canvas.height = this.H * this.dpr
    this.canvas.style.width = this.W * s + 'px'
    this.canvas.style.height = this.H * s + 'px'
    this.zoomBtn.hidden = !small
    this.zoomBtn.textContent = t(this.full ? 'Fit to screen' : 'Full size ⤢')
    this.edges()
    poke()
    this.onResize?.()
  }

  /** Re-render at the new resolution when the window moves to a screen with another pixel ratio. */
  private watchDpr() {
    const mq = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`)
    mq.addEventListener('change', () => { if (this.host.isConnected) { this.resize(); this.watchDpr() } }, { once: true })
  }

  /** Fade the edge the drawing continues past, while it scrolls sideways. */
  private edges() {
    const s = this.scroll, more = s.scrollWidth - s.clientWidth
    s.classList.toggle('more-r', more > 2 && s.scrollLeft < more - 2)
    s.classList.toggle('more-l', more > 2 && s.scrollLeft > 2)
  }

  /** Reset the transform and paint the background plus the faint dot grid. */
  begin() {
    const { ctx, W, H } = this
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0)
    ctx.fillStyle = rgba(C.bg)
    ctx.fillRect(0, 0, W, H)
    ctx.fillStyle = rgba(C.ink, C.dark ? 0.05 : 0.08)
    for (let x = 16; x < W; x += 24) for (let y = 16; y < H; y += 24) ctx.fillRect(x, y, 1, 1)
  }

  /** Pointer position in drawing coordinates (the canvas may be scaled to fit). */
  local(e: { clientX: number; clientY: number }): [number, number] {
    const r = this.canvas.getBoundingClientRect(), k = r.width ? this.W / r.width : 1
    return [(e.clientX - r.left) * k, (e.clientY - r.top) * k]
  }

  destroy() {
    this.ro.disconnect()
  }
}

/** Time of the last pointer, key, wheel, resize or theme event: anything that can change a still frame. */
let lastPoke = performance.now()
export const poke = () => { lastPoke = performance.now() }
for (const ev of ['pointermove', 'pointerdown', 'pointerup', 'keydown', 'wheel', 'resize', 'focusin', 'input', 'change']) window.addEventListener(ev, poke, { passive: true, capture: true })

/**
 * requestAnimationFrame loop with a capped dt; returns a stop function. While `idle()` is true
 * (e.g. paused) and nothing has been poked for a moment, frames are skipped instead of redrawn.
 */
export function runLoop(step: (dt: number, now: number) => void, idle: () => boolean = () => false): () => void {
  let last = performance.now()
  let id = 0, reported = false
  const tick = (now: number) => {
    // the next frame is booked first: an error in one frame must not stop the page for good
    id = requestAnimationFrame(tick)
    // the first frame's timestamp can precede performance.now() at start: never step backwards
    const dt = Math.max(0, Math.min(0.05, (now - last) / 1000))
    last = now
    try {
      if (!idle() || now - lastPoke < 700) step(dt, now)
    } catch (err) {
      if (!reported) { reported = true; console.error(err) }
    }
  }
  id = requestAnimationFrame(tick)
  return () => cancelAnimationFrame(id)
}
