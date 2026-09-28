import { raw, t as tr } from './i18n'
import { C, rgba, type RGB } from './theme'

/** Shared canvas primitives. Call useCtx() with the stage's context at the start of a frame. */
export let ctx: CanvasRenderingContext2D
export function useCtx(c: CanvasRenderingContext2D) {
  ctx = c
}

export const MONO = '"JetBrains Mono", ui-monospace, Menlo, monospace'
export const SERIF = '"EB Garamond", Georgia, serif'
/** Italic serif for names and math. EB Garamond runs small, so sizes are scaled up 10%. */
export const serifAt = (px: number) => `italic 400 ${Math.round(px * 1.1 * 10) / 10}px ${SERIF}`

export const F = {
  serif: serifAt(26),
  serifM: serifAt(19),
  chip: `500 12.5px ${MONO}`,
  small: `400 11px ${MONO}`,
  label: `500 10.5px ${MONO}`,
  body: '400 12px Geist, "PingFang SC", "Hiragino Sans GB", "Noto Sans SC", sans-serif',
  /** Mono at a given size; anything under 11px gets +0.5px, since small JetBrains Mono needs it to stay crisp. */
  mono: (px: number, w = 400) => `${w} ${px < 11 ? px + 0.5 : px}px ${MONO}`,
}

export interface TokLike {
  text: string
  c: number
}

/** GPT-2 byte-level BPE display: Ġ marks a leading space, Ċ a newline. */
/** GPT-2's spelling of a token: a leading space shows as Ġ, a newline as Ċ. */
export const tokLabel = (t: string) => (t.startsWith(' ') ? 'Ġ' + t.slice(1) : t).replace(/\n/g, 'Ċ')
/** A token as running text, with newlines shown as ↵. */
export const tokDisp = (t: string) => t.replace(/\n/g, ' ↵')
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
  return Math.ceil(raw(() => ctx.measureText(tokLabel(text)).width)) + 14
}

/** Token text with a dimmed Ġ prefix. */
export function tokText(x: number, y: number, text: string, a: number, col?: RGB | null, font = F.chip) {
  raw(() => tokTextRaw(x, y, text, a, col, font))
}
function tokTextRaw(x: number, y: number, text: string, a: number, col?: RGB | null, font = F.chip) {
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
  fillRich(t, x, y)
}

/* ---------- subscripts: "W_Q", "ln_f", "d_model" render as W with a subscript Q, and so on ---------- */

/** An identifier with a subscript; not inside special tokens such as <|im_start|>. */
/** x_1, W_up, x_{t−1}: a name, then a run of letters and digits or anything in braces as its subscript. */
const SUB_RE = /(?<![|A-Za-z\u0391-\u03c9\u1f00-\u1fff\u0300-\u036f])([A-Za-z\u0391-\u03c9\u1f00-\u1fff][A-Za-z\u0391-\u03c9\u1f00-\u1fff\u0300-\u036f]*)_(?:\{([^}]+)\}|([A-Za-z0-9\u2205]+))/g
/** Split text into [run, isSubscript] pieces. */
export function richSegs(t: string): [string, boolean][] {
  const out: [string, boolean][] = []
  let last = 0
  for (const m of t.matchAll(SUB_RE)) {
    const i = m.index!
    out.push([t.slice(last, i) + m[1], false], [m[2] ?? m[3], true])
    last = i + m[0].length
  }
  out.push([t.slice(last), false])
  return out.filter(([s]) => s.length > 0)
}
const subFont = (font: string) => font.replace(/(\d+(?:\.\d+)?)px/, (_, n) => `${Math.round(parseFloat(n) * 0.72 * 10) / 10}px`)
const fontPx = (font: string) => parseFloat(/(\d+(?:\.\d+)?)px/.exec(font)?.[1] ?? '12')
/** Width of text as fillRich would draw it with the current font (translated first, like fillRich). */
export function measureRich(t: string): number {
  t = tr(t)
  return raw(() => measureRichRaw(t))
}
function measureRichRaw(t: string): number {
  const main = ctx.font, sf = subFont(main)
  let w = 0
  for (const [s, sub] of richSegs(t)) { ctx.font = sub ? sf : main; w += ctx.measureText(s).width + (sub ? 0.5 : 0) }
  ctx.font = main
  return w
}
/** fillText with real subscripts; honours the current textAlign. Returns the width drawn. The whole text is translated first. */
export function fillRich(t: string, x: number, y: number): number {
  t = tr(t)
  return raw(() => fillRichRaw(t, x, y))
}
function fillRichRaw(t: string, x: number, y: number): number {
  if (!t.includes('_')) { ctx.fillText(t, x, y); return ctx.measureText(t).width }
  const main = ctx.font, sf = subFont(main), align = ctx.textAlign, w = measureRichRaw(t), dy = fontPx(main) * 0.28
  let cx = align === 'center' ? x - w / 2 : align === 'right' || align === 'end' ? x - w : x
  ctx.textAlign = 'left'
  for (const [s, sub] of richSegs(t)) {
    ctx.font = sub ? sf : main
    ctx.fillText(s, cx, sub ? y + dy : y)
    cx += ctx.measureText(s).width + (sub ? 0.5 : 0)
  }
  ctx.font = main
  ctx.textAlign = align
  return w
}

/** Uppercase mono group label with a thin bracket underneath, spanning [a, b]. */
/** Uppercase Latin letters only, so Greek (σ, μ) and math keep their case in uppercase labels. */
export const upper = (t: string) => t.replace(/[a-z]+/g, (m) => m.toUpperCase())

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
  ctx.fillText(upper(t), a, y)
  spaced(false)
}

/** A run of italic math text; parts marked `true` are drawn as subscripts. Returns the width drawn. */
export function mathRun(parts: [string, boolean][], x: number, y: number, a: number, size = 19, col: RGB = C.ink) {
  return raw(() => mathRunRaw(parts, x, y, a, size, col))
}
function mathRunRaw(parts: [string, boolean][], x: number, y: number, a: number, size: number, col: RGB) {
  ctx.textAlign = 'left'
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = rgba(col, a)
  let w = 0
  for (const [t, isSub] of parts) {
    ctx.font = serifAt(isSub ? size * 0.62 : size)
    ctx.fillText(t, x + w, isSub ? y + size * 0.22 : y)
    w += ctx.measureText(t).width + (isSub ? 1 : 0)
  }
  return w
}

/** Italic math name with `_` subscripts, e.g. "W_Q" → W with subscript Q. Returns the width drawn. */
export function mathName(name: string, x: number, y: number, a: number, size = 19, col: RGB = C.ink) {
  ctx.textAlign = 'left'
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = rgba(col, a)
  let w = 0
  raw(() => {
    for (const [s, sub] of richSegs(name)) {
      ctx.font = serifAt(sub ? size * 0.62 : size)
      ctx.fillText(s, x + w + (sub ? 1 : 0), sub ? y + size * 0.22 : y)
      w += ctx.measureText(s).width + (sub ? 1 : 0)
    }
  })
  return w
}
