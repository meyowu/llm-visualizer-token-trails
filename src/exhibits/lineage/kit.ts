import { F, fillRich, plate, rr, serifAt, spaced, upper, useCtx, type TokLike } from '../../core/draw'
import { createFrame, type Frame, type FrameOptions } from '../../core/frame'
import { MatrixKit } from '../../core/matrix'
import { Player, type Phase } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, rgba, type RGB } from '../../core/theme'
import { reducedMotion } from '../../core/util'
import { teach } from '../learn'
import type { Nav } from '../registry'

/*
 * The shared frame of a Lineage page: an architecture drawn as a diff against GPT-2, one scene per
 * phase, a Compare button that opens the GPT-2 part each phase changes, and pill labels on the
 * canvas that jump to a phase.
 */

/** Plain word tokens for a toy sentence; token i gets hue i. */
export const words = (ws: string[]): TokLike[] => ws.map((text, c) => ({ text, c }))

/** Canvas helpers shared by the Lineage pages; they draw on the page's stage. */
export interface Kit {
  /** A dark label that jumps to `phase` when clicked. */
  pill(t: string, x: number, y: number, phase: string, a: number): void
  /** The ⊕ of a residual add. */
  addNode(x: number, y: number, a: number): void
  caption(t: string, x: number, y: number, a: number, col?: RGB, align?: CanvasTextAlign): void
  /** An uppercase mono group label. */
  title(t: string, x: number, y: number, a: number, align?: CanvasTextAlign): void
  /** A model's name in the serif, with a small note under it. */
  rowName(name: string, sub: string, x: number, y: number, a: number): void
  /** A polyline ending in an arrowhead. */
  arrow(pts: [number, number][], a: number, dash?: boolean, col?: RGB): void
  /** One token lane, a straight line in the token's hue. */
  lane(x0: number, x1: number, y: number, col: RGB, a: number, w?: number): void
  /** A glass plate that fades with the rest of the scene. */
  glass(x: number, y0: number, y1: number, act: number, o: { w?: number; d?: number; hatch?: number }, a: number): void
}

export interface Env {
  stage: Stage
  ctx: CanvasRenderingContext2D
  player: Player
  mk: MatrixKit
  k: Kit
  frame: Frame
  nav: Nav
  /** Run fn every frame after drawing, for page-specific controls. */
  onFrame(fn: () => void): void
}

export interface LineageOptions {
  frame: FrameOptions
  /** Minimum drawing size. */
  size: [number, number]
  /** What the drawing shows, for screen readers. */
  aria: string
  phases: Phase[]
  /** The page's key in learn.ts. */
  learn: string
  /** Tokens the MatrixKit colours rows and columns by. */
  tokens: TokLike[]
  /** Per phase: the GPT-2 part it changes, as [label, route]. */
  compare: Record<string, [string, string]>
  /** Per phase: caption text and its short shape line. */
  caps: Record<string, [string, string]>
  /** Builds the scenes, one per phase id, each drawing at progress p ∈ [0, 1]. */
  scenes: (env: Env) => Record<string, (p: number) => void>
  /** Where to open when motion is reduced: a phase and seconds into it. */
  still: [string, number]
}

export function mountLineage(root: HTMLElement, nav: Nav, o: LineageOptions): () => void {
  const reduced = reducedMotion()
  const frame = createFrame(root, { formula: true, ...o.frame })
  const stage = new Stage(frame.stageHost, o.size[0], o.size[1], o.aria)
  const ctx = stage.ctx
  const player = new Player(o.phases, frame.controls, { playing: !reduced })
  teach(player, o.learn)
  const mk = new MatrixKit(stage, o.tokens, frame.setFormula)

  const compare = document.createElement('button')
  compare.type = 'button'; compare.className = 'compare'
  compare.addEventListener('click', () => nav(o.compare[player.cur().id][1]))
  player.meta.prepend(compare)

  let pills: { x: number; y: number; w: number; h: number; phase: string }[] = []
  let hoverPill = ''
  const k: Kit = {
    pill(t, x, y, phase, a) {
      ctx.font = F.mono(11, 500)
      const w = ctx.measureText(t).width + 16
      rr(x - w / 2, y - 10, w, 20, 10)
      ctx.fillStyle = rgba(C.ink, (hoverPill === phase ? 1 : 0.9) * a); ctx.fill()
      ctx.fillStyle = rgba(C.bg, a); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
      ctx.fillText(t, x, y + 0.5)
      if (a > 0.3) pills.push({ x: x - w / 2, y: y - 10, w, h: 20, phase })
    },
    addNode(x, y, a) {
      ctx.fillStyle = rgba(C.bg, a); ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.fill()
      ctx.strokeStyle = rgba(C.ink, 0.8 * a); ctx.lineWidth = 1.2; ctx.stroke()
      ctx.beginPath(); ctx.moveTo(x - 4.5, y); ctx.lineTo(x + 4.5, y); ctx.moveTo(x, y - 4.5); ctx.lineTo(x, y + 4.5); ctx.stroke()
    },
    caption(t, x, y, a, col = C.mute, align = 'center') {
      ctx.font = F.small; ctx.textAlign = align; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(col, a); fillRich(t, x, y)
    },
    title(t, x, y, a, align = 'left') {
      ctx.font = F.label; spaced(true); ctx.textAlign = align; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
      fillRich(upper(t), x, y); spaced(false)
    },
    rowName(name, sub, x, y, a) {
      ctx.font = serifAt(24); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(name, x, y + 2)
      ctx.font = F.small; ctx.fillStyle = rgba(C.mute, a); fillRich(sub, x, y + 20)
    },
    arrow(pts, a, dash = false, col = C.ink) {
      if (a <= 0) return
      ctx.strokeStyle = rgba(col, 0.55 * a); ctx.lineWidth = 1
      if (dash) ctx.setLineDash([3, 4])
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke()
      ctx.setLineDash([])
      const [x1, y1] = pts[pts.length - 1], [x0, y0] = pts[pts.length - 2], ang = Math.atan2(y1 - y0, x1 - x0)
      ctx.beginPath()
      ctx.moveTo(x1 - 6 * Math.cos(ang - 0.45), y1 - 6 * Math.sin(ang - 0.45)); ctx.lineTo(x1, y1); ctx.lineTo(x1 - 6 * Math.cos(ang + 0.45), y1 - 6 * Math.sin(ang + 0.45))
      ctx.stroke()
    },
    lane(x0, x1, y, col, a, w = 1.3) {
      if (a <= 0 || x1 <= x0) return
      ctx.strokeStyle = rgba(col, 0.8 * a); ctx.lineWidth = w
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke()
    },
    glass(x, y0, y1, act, po, a) {
      if (a <= 0.01) return
      ctx.globalAlpha = a; plate(x, y0, y1, act, po); ctx.globalAlpha = 1
    },
  }

  const hooks: (() => void)[] = []
  const env: Env = { stage, ctx, player, mk, k, frame, nav, onFrame: (fn) => hooks.push(fn) }
  const scenes = o.scenes(env)
  function draw() {
    useCtx(ctx)
    stage.begin()
    mk.begin()
    pills = []
    let id = o.phases[0].id
    for (const ph of o.phases) if (player.prog(ph.id) > 0) id = ph.id
    scenes[id](player.prog(id))
    mk.drawFormula()
  }

  const pillAt = (e: PointerEvent | MouseEvent) => {
    const [x, y] = stage.local(e)
    return pills.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h)
  }
  stage.canvas.addEventListener('pointermove', (e) => {
    const hit = pillAt(e)
    hoverPill = hit?.phase ?? ''
    if (hit) stage.canvas.style.cursor = 'pointer'
  })
  stage.canvas.addEventListener('click', (e) => {
    const hit = pillAt(e)
    if (hit) player.t = player.start(hit.phase) + 0.001
  })

  if (reduced && player.t === 0) player.t = player.start(o.still[0]) + o.still[1]
  player.describe = (i) => o.caps[o.phases[i].id]
  const stop = runLoop((dt) => {
    player.tick(dt)
    draw()
    player.updateUI()
    for (const fn of hooks) fn()
    const cur = player.cur(), cmp = `Compare: ${o.compare[cur.id][0]} ↗`
    if (compare.textContent !== cmp) compare.textContent = cmp
    const [t, s] = o.caps[cur.id]
    frame.setCaption(cur.name, cur.short ?? cur.name, t, s)
  }, () => !player.playing)
  return () => { stop(); player.destroy(); stage.destroy() }
}
