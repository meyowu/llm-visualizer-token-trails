import { C, rgba, type RGB } from './theme'

/** Shared canvas primitives. Call useCtx() with the stage's context at the start of a frame. */
export let ctx: CanvasRenderingContext2D
export function useCtx(c: CanvasRenderingContext2D) {
  ctx = c
}

export const F = {
  serif: 'italic 400 26px Newsreader, Georgia, serif',
  serifM: 'italic 400 19px Newsreader, Georgia, serif',
  chip: '500 12px "Geist Mono", ui-monospace, Menlo, monospace',
  small: '400 10.5px "Geist Mono", ui-monospace, Menlo, monospace',
  label: '500 10px "Geist Mono", ui-monospace, Menlo, monospace',
  body: '400 12px Geist, "PingFang SC", "Hiragino Sans GB", "Noto Sans SC", sans-serif',
  mono: (px: number, w = 400) => `${w} ${px}px "Geist Mono", ui-monospace, Menlo, monospace`,
}

export interface TokLike {
  text: string
  c: number
}

/** GPT-2 byte-level BPE display: Ġ marks a leading space, Ċ a newline. */
export const tokLabel = (t: string) => (t === '\n' ? 'Ċ' : t.startsWith(' ') ? 'Ġ' + t.slice(1) : t)
export const tokDisp = (t: string) => (t === '\n' ? ' ↵' : t)
export const tokCol = (t: TokLike): RGB => C.tok[t.c % 7]

export function spaced(on: boolean) {
  if ('letterSpacing' in ctx) (ctx as unknown as { letterSpacing: string }).letterSpacing = on ? '1.2px' : '0px'
}

export function rr(x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r)
  else ctx.rect(x, y, w, h)
}

export function chipW(text: string, font = F.chip) {
  ctx.font = font
  return Math.ceil(ctx.measureText(tokLabel(text)).width) + 14
}

/** Token text with a dimmed Ġ prefix. */
export function tokText(x: number, y: number, text: string, a: number, col?: RGB | null, font = F.chip) {
  const lab = tokLabel(text)
  ctx.font = font
  ctx.textBaseline = 'middle'
  ctx.textAlign = 'left'
  if (lab[0] === 'Ġ') {
    ctx.fillStyle = rgba(C.mute, a * 0.8)
    ctx.fillText('Ġ', x, y + 0.5)
    ctx.fillStyle = rgba(col || C.ink, a)
    ctx.fillText(lab.slice(1), x + ctx.measureText('Ġ').width, y + 0.5)
  } else {
    ctx.fillStyle = rgba(col || C.ink, a)
    ctx.fillText(lab, x, y + 0.5)
  }
}

/** Token chip, left edge at x, vertically centred on y. Returns its width. */
export function drawChip(x: number, y: number, tok: TokLike, a: number, h: number, hl = false, font = F.chip) {
  const w = chipW(tok.text, font)
  const col = tokCol(tok)
  rr(x, y - h / 2, w, h, 5)
  ctx.fillStyle = rgba(col, (hl ? 0.28 : 0.14) * a)
  ctx.fill()
  ctx.lineWidth = 1
  ctx.strokeStyle = rgba(col, (hl ? 1 : 0.55) * a)
  ctx.stroke()
  tokText(x + 7, y, tok.text, a, null, font)
  return w
}

/** A layer drawn as a tilted glass pane standing in the stream (the 2.5D motif). */
export function plate(x: number, y0: number, y1: number, act: number, o: { w?: number; d?: number; hatch?: number } = {}) {
  const w = o.w || 8, d = o.d || 9
  ctx.beginPath()
  ctx.moveTo(x - w, y0 + d)
  ctx.lineTo(x + w, y0 - d)
  ctx.lineTo(x + w, y1 - d)
  ctx.lineTo(x - w, y1 + d)
  ctx.closePath()
  ctx.fillStyle = rgba(C.ink, (C.dark ? 0.03 : 0.045) + 0.07 * act)
  ctx.fill()
  ctx.lineWidth = 1
  ctx.strokeStyle = rgba(C.ink, 0.14 + 0.5 * act)
  ctx.stroke()
  if (o.hatch) {
    ctx.strokeStyle = rgba(C.ink, 0.07 + 0.12 * act)
    for (let k = 1; k < o.hatch; k++) {
      const f = k / o.hatch
      ctx.beginPath()
      ctx.moveTo(x - w, y0 + d + (y1 - y0) * f)
      ctx.lineTo(x + w, y0 - d + (y1 - y0) * f)
      ctx.stroke()
    }
  }
  ctx.beginPath()
  ctx.moveTo(x - w, y0 + d)
  ctx.lineTo(x + w, y0 - d)
  ctx.strokeStyle = rgba(C.ink, 0.3 + 0.6 * act)
  ctx.stroke()
}

export function subLabel(t: string, x: number, y: number, on: boolean) {
  ctx.font = F.small
  ctx.textAlign = 'center'
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = rgba(on ? C.ink : C.mute, on ? 1 : 0.85)
  ctx.fillText(t, x, y)
}

/** Uppercase mono group label with a thin bracket underneath, spanning [a, b]. */
export function bracketLabel(t: string, a: number, b: number, y: number, on: boolean) {
  ctx.strokeStyle = rgba(on ? C.ink : C.faint, on ? 0.9 : 0.8)
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(a + 0.5, y + 14)
  ctx.lineTo(a + 0.5, y + 8.5)
  ctx.lineTo(b - 0.5, y + 8.5)
  ctx.lineTo(b - 0.5, y + 14)
  ctx.stroke()
  ctx.font = F.label
  ctx.textAlign = 'left'
  ctx.textBaseline = 'alphabetic'
  spaced(true)
  ctx.fillStyle = rgba(on ? C.ink : C.mute, 1)
  ctx.fillText(t.toUpperCase(), a, y)
  spaced(false)
}

/** A run of italic math text; parts marked `true` are drawn as subscripts. Returns the width drawn. */
export function mathRun(parts: [string, boolean][], x: number, y: number, a: number, size = 19, col: RGB = C.ink) {
  ctx.textAlign = 'left'
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = rgba(col, a)
  let w = 0
  for (const [t, isSub] of parts) {
    ctx.font = `italic 400 ${isSub ? Math.round(size * 0.62) : size}px Newsreader, Georgia, serif`
    ctx.fillText(t, x + w, isSub ? y + size * 0.22 : y)
    w += ctx.measureText(t).width + (isSub ? 1 : 0)
  }
  return w
}

/** Italic math name with `_` subscripts, e.g. "W_Q" → W with subscript Q. Returns the width drawn. */
export function mathName(name: string, x: number, y: number, a: number, size = 19, col: RGB = C.ink) {
  const [base, sub] = name.split('_')
  ctx.textAlign = 'left'
  ctx.textBaseline = 'alphabetic'
  ctx.font = `italic 400 ${size}px Newsreader, Georgia, serif`
  ctx.fillStyle = rgba(col, a)
  ctx.fillText(base, x, y)
  let w = ctx.measureText(base).width
  if (sub) {
    ctx.font = `italic 400 ${Math.round(size * 0.62)}px Newsreader, Georgia, serif`
    ctx.fillText(sub, x + w + 1, y + size * 0.22)
    w += ctx.measureText(sub).width + 1
  }
  return w
}
