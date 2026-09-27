import { C, rgba } from './theme'

/**
 * A DPR-aware canvas that fills its host element. The drawing surface never shrinks below
 * minW × minH; narrower hosts scroll horizontally instead of squeezing the diagram.
 */
export class Stage {
  readonly canvas: HTMLCanvasElement
  readonly ctx: CanvasRenderingContext2D
  W = 0
  H = 0
  dpr = 1
  onResize: (() => void) | null = null
  private ro: ResizeObserver

  constructor(private host: HTMLElement, private minW: number, private minH: number, label: string) {
    const scroll = document.createElement('div')
    scroll.className = 'stage-scroll'
    this.canvas = document.createElement('canvas')
    this.canvas.setAttribute('role', 'img')
    this.canvas.setAttribute('aria-label', label)
    scroll.appendChild(this.canvas)
    host.appendChild(scroll)
    this.ctx = this.canvas.getContext('2d')!
    this.ro = new ResizeObserver(() => this.resize())
    this.ro.observe(host)
    this.resize()
  }

  resize() {
    const r = this.host.getBoundingClientRect()
    this.W = Math.max(this.minW, Math.floor(r.width))
    this.H = Math.max(this.minH, Math.floor(r.height))
    this.dpr = Math.min(2, window.devicePixelRatio || 1)
    this.canvas.width = this.W * this.dpr
    this.canvas.height = this.H * this.dpr
    this.canvas.style.width = this.W + 'px'
    this.canvas.style.height = this.H + 'px'
    this.onResize?.()
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

  local(e: { clientX: number; clientY: number }): [number, number] {
    const r = this.canvas.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top]
  }

  destroy() {
    this.ro.disconnect()
  }
}

/** requestAnimationFrame loop with a capped dt; returns a stop function. */
export function runLoop(step: (dt: number, now: number) => void): () => void {
  let last = performance.now()
  let id = 0
  const tick = (now: number) => {
    const dt = Math.min(0.05, (now - last) / 1000)
    last = now
    step(dt, now)
    id = requestAnimationFrame(tick)
  }
  id = requestAnimationFrame(tick)
  return () => cancelAnimationFrame(id)
}
