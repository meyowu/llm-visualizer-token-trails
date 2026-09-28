import { F, ctx, drawChip, tokText } from '../../core/draw'
import { C, rgba, type RGB } from '../../core/theme'
import type { Kit } from '../kit'

/* Drawing pieces the Agents pages share: next-token bars, prompts as rows of chips, the model as a stack of plates. */

/** One row of a next-token distribution. */
export interface Bar { text: string; p: number }
export const bars = (top: [string, number][]): Bar[] => top.map(([text, p]) => ({ text, p }))
/** A probability as a percentage, as the overview writes it. */
export const pct = (p: number) => (p < 0.001 ? '<0.1%' : (p * 100).toFixed(p < 0.1 ? 1 : 0) + '%')

/**
 * A next-token distribution in the overview's style: token, bar, percentage. `grow` (0–1) scales the bars in;
 * `mark` picks rows to draw at full strength (the pick, or the right answer).
 */
export function distRows(rows: Bar[], x: number, y: number, w: number, a: number, o: { rowH?: number; labelW?: number; grow?: number; mark?: (r: Bar, i: number) => boolean; col?: RGB; tag?: (r: Bar, i: number) => string | undefined } = {}) {
  if (a <= 0) return
  const rowH = o.rowH ?? 24, lw = o.labelW ?? 100, g = o.grow ?? 1, bx0 = x + lw, bx1 = x + w - 52
  rows.forEach((r, i) => {
    const yy = y + i * rowH, m = o.mark?.(r, i) ?? false, col = m && o.col ? o.col : C.ink
    tokText(x, yy, r.text, a * (m ? 1 : 0.8), m ? col : null)
    ctx.fillStyle = rgba(C.ink, 0.07 * a); ctx.fillRect(bx0, yy - 3, bx1 - bx0, 6)
    const bw = Math.max(1, r.p * g * (bx1 - bx0))
    ctx.fillStyle = rgba(col, (m ? 0.9 : 0.45) * a); ctx.fillRect(bx0, yy - 3, bw, 6)
    const tag = o.tag?.(r, i)
    if (tag) { ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, a * g); ctx.fillText(tag, bx0 + bw + 8, yy) }
    ctx.font = F.small; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(m ? C.ink : C.mute, a)
    ctx.fillText(pct(r.p), x + w, yy)
  })
}

/**
 * A prompt as rows of token chips, a new row after each newline token (Ċ). Token i gets hue c0 + i.
 * `alpha(i)` fades single tokens, `hl(i)` highlights them. Returns each chip's box.
 */
export function chipLines(toks: string[], x: number, y: number, a: number, o: { rowH?: number; c0?: number; alpha?: (i: number) => number; hl?: (i: number) => boolean; h?: number } = {}) {
  const rowH = o.rowH ?? 30, h = o.h ?? 22, boxes: { x: number; y: number; w: number }[] = []
  let cx = x, cy = y
  toks.forEach((t, i) => {
    const ta = a * (o.alpha?.(i) ?? 1), w = ta > 0 ? drawChip(cx, cy, { text: t, c: (o.c0 ?? 0) + i }, ta, h, o.hl?.(i) ?? false) : 0
    boxes.push({ x: cx, y: cy, w })
    cx += w + 5
    if (t === 'Ċ' || t.endsWith('\n')) { cx = x; cy += rowH }
  })
  return boxes
}

/** The model as a small stack of glass plates with its name under it. */
export function modelGlyph(k: Kit, x: number, yc: number, name: string, sub: string, a: number, act = 0.3) {
  for (let i = 0; i < 4; i++) k.glass(x - 18 + i * 12, yc - 34, yc + 34, act, { w: 7, d: 8 }, a)
  ctx.font = F.mono(11, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a)
  ctx.fillText(name, x, yc + 62)
  ctx.font = F.small; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(sub, x, yc + 78)
}

/**
 * Keep JetBrains Mono from joining symbols (<| into a triangle, -> into an arrow) in special tokens and code:
 * any letter spacing switches ligatures off in Chrome's canvas.
 */
export function noLigatures(on: boolean) {
  if ('letterSpacing' in ctx) (ctx as unknown as { letterSpacing: string }).letterSpacing = on ? '0.01px' : '0px'
}

/** Runs of text in one line, each in its own colour; returns the x after the last run. */
export function runs(parts: [string, RGB][], x: number, y: number, a: number, font = F.mono(12)) {
  ctx.font = font; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
  noLigatures(true)
  for (const [t, col] of parts) { ctx.fillStyle = rgba(col, a); ctx.fillText(t, x, y); x += ctx.measureText(t).width }
  noLigatures(false)
  return x
}
