import { F, chipW, drawChip, fillRich, mathName, tokCol, tokLabel, type TokLike } from './draw'
import type { Stage } from './stage'
import { C, rgba, type RGB } from './theme'
import { clamp, lerp } from './util'

/*
 * Matrices and GEMMs for detail views. A GEMM C = A·B is laid out with A on the left, B above
 * and C where A's row meets B's column, so row i of A lines up with row i of C and column j of B
 * with column j of C. `bT` draws B transposed instead (row j of Bᵀ is column j of B), which puts
 * both operands of a dot product on the same horizontal axis.
 */

export type M = number[][]
export interface Rect { x: number; y: number; c: number }
export const lr = (a: Rect, b: Rect, t: number): Rect => ({ x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), c: lerp(a.c, b.c, t) })

export type Kind = 'row' | 'col' | 'w' | 'score' | 'attn'
export type Seg = [string, RGB, number?]
export interface Gemm { rev: (i: number, j: number) => number; cur: { i: number; j: number; k: number } | null }
export interface Focus { key: string; i: number; j: number; k: number }
export interface Cell { key: string; i: number; j: number }

/** Compact numbers for cells: .57, −1.3, 12. */
export const fmt = (v: number) => {
  if (!isFinite(v)) return v < 0 ? '−∞' : '∞'
  const s = v < 0 ? '−' : '', a = Math.abs(v)
  if (a >= 9.95) return s + a.toFixed(0)
  if (a >= 0.995) return s + a.toFixed(1)
  return s + a.toFixed(2).slice(1)
}
export const vmaxOf = (m: M) => Math.max(1e-6, ...m.flat().filter((v) => isFinite(v)).map(Math.abs))

/**
 * Cell schedule of one GEMM over progress p ∈ [0,1].
 * slow: the first cell accumulates term by term, the rest of row 0 follows, then rows sweep.
 * fast: rows sweep from the start.
 */
export function gemm(p: number, m: number, n: number, K: number, mode: 'slow' | 'fast'): Gemm {
  const td = (i: number, j: number) => mode === 'slow'
    ? i === 0 && j === 0 ? 0.34 : i === 0 ? 0.34 + j * (0.18 / Math.max(1, n - 1)) : 0.52 + (i - 1 + (j + 1) / n) * (0.44 / Math.max(1, m - 1))
    : ((i + (j + 1) / n) / m) * 0.96
  let cur: Gemm['cur'] = null, best = Infinity
  if (p > 0) for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) { const t = td(i, j); if (t > p && t < best) { best = t; cur = { i, j, k: K } } }
  if (cur && mode === 'slow' && cur.i === 0 && cur.j === 0) cur.k = clamp(p / 0.34) * K
  return { rev: (i, j) => (p >= td(i, j) ? 1 : 0), cur }
}

export interface MatOpts {
  r: Rect
  vals: M
  kind: Kind
  alpha: number
  name: string
  shape: string
  real?: string
  label?: 'top' | 'bottom' | 'none'
  labelAlpha?: number
  reveal?: (i: number, j: number) => number
  rowCols?: RGB[]
  colToks?: boolean
  /** Hide numbers even when cells are large enough. */
  noText?: boolean
  /** Custom cell painter; returns fill alpha. */
  paint?: (x: number, y: number, c: number, i: number, j: number, a: number) => number
  text?: (i: number, j: number) => string
}

export interface GemmView {
  A: Rect
  Av: M
  B: Rect
  /** B's values; when bT is set, pass Bᵀ (row j = column j of B). */
  Bv: M
  bT?: boolean
  C: Rect
  f: Focus
  names: [string, string, string]
  note: string
  bias?: number[]
  /** Where the bias row is drawn, to highlight its cell once the dot product is done. */
  biasR?: Rect
}

/** Stateful helper for one detail view: draws matrices, tracks hover targets and the formula line. */
export class MatrixKit {
  hits: { key: string; r: Rect; rows: number; cols: number }[] = []
  formula: { segs: Seg[]; note: string } | null = null
  hover: Cell | null = null

  constructor(private stage: Stage, private tokens: TokLike[], private pad: number) {
    const cv = stage.canvas
    cv.addEventListener('pointermove', (e) => {
      const [x, y] = stage.local(e)
      this.hover = null
      for (const h of this.hits) {
        const j = Math.floor((x - h.r.x) / h.r.c), i = Math.floor((y - h.r.y) / h.r.c)
        if (i >= 0 && j >= 0 && i < h.rows && j < h.cols) { this.hover = { key: h.key, i, j }; break }
      }
      cv.style.cursor = this.hover ? 'crosshair' : 'default'
    })
    cv.addEventListener('pointerleave', () => { this.hover = null })
  }

  private get ctx() { return this.stage.ctx }
  tokRGB = (i: number) => tokCol(this.tokens[i])
  tl = (i: number) => tokLabel(this.tokens[i].text)
  hovered = (...keys: string[]) => (this.hover && keys.includes(this.hover.key) ? this.hover : null)

  /** Reset per-frame state. */
  begin() {
    this.hits = []
    this.formula = null
  }
  hit(key: string, r: Rect, rows: number, cols: number) {
    this.hits.push({ key, r, rows, cols })
  }

  /** The 2.5D slab under a matrix: a right and a bottom face. */
  slab(x: number, y: number, w: number, h: number, a: number) {
    const ctx = this.ctx, d = 5
    ctx.fillStyle = rgba(C.ink, 0.07 * a)
    ctx.beginPath(); ctx.moveTo(x + w, y); ctx.lineTo(x + w + d, y + d); ctx.lineTo(x + w + d, y + h + d); ctx.lineTo(x + w, y + h); ctx.closePath(); ctx.fill()
    ctx.fillStyle = rgba(C.ink, 0.045 * a)
    ctx.beginPath(); ctx.moveTo(x, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w + d, y + h + d); ctx.lineTo(x + d, y + h + d); ctx.closePath(); ctx.fill()
  }
  /** Filled = positive, outlined = negative. Returns the fill alpha so text can pick a contrasting colour. */
  paintCell(x: number, y: number, c: number, v: number, vmax: number, hue: RGB | null, strength: number, a: number) {
    const ctx = this.ctx, m = Math.min(1, Math.abs(v) / vmax)
    if (v < 0) {
      ctx.fillStyle = rgba(C.neg, (0.03 + 0.14 * m) * a); ctx.fillRect(x + 0.5, y + 0.5, c - 1, c - 1)
      if (c >= 6) { ctx.strokeStyle = rgba(C.neg, (0.18 + 0.5 * m) * a); ctx.lineWidth = 1; ctx.strokeRect(x + 2.5, y + 2.5, c - 5, c - 5) }
      return 0
    }
    const fa = hue ? 0.1 + 0.72 * m : 0.05 + strength * m
    ctx.fillStyle = rgba(hue ?? C.ink, fa * a); ctx.fillRect(x + 0.5, y + 0.5, c - 1, c - 1)
    return fa
  }
  paintAttn(x: number, y: number, c: number, v: number, hue: RGB, a: number) {
    const fa = 0.05 + 0.9 * Math.sqrt(Math.max(0, v))
    this.ctx.fillStyle = rgba(hue, fa * a); this.ctx.fillRect(x + 0.5, y + 0.5, c - 1, c - 1)
    return fa
  }
  hatch(x: number, y: number, c: number, a: number) {
    const ctx = this.ctx
    ctx.save(); ctx.beginPath(); ctx.rect(x + 0.5, y + 0.5, c - 1, c - 1); ctx.clip()
    ctx.fillStyle = rgba(C.ink, 0.025 * a); ctx.fillRect(x, y, c, c)
    ctx.strokeStyle = rgba(C.ink, 0.13 * a); ctx.lineWidth = 1
    for (let d = -c; d < c; d += 5) { ctx.beginPath(); ctx.moveTo(x + d, y + c); ctx.lineTo(x + d + c, y); ctx.stroke() }
    ctx.restore()
  }
  cellText(t: string, x: number, y: number, c: number, fa: number, a: number, weak = false) {
    if (!t || c < 17) return
    const ctx = this.ctx
    ctx.font = F.mono(clamp(c * 0.3, 8, 11)); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    ctx.fillStyle = rgba(fa > 0.55 ? C.bg : weak ? C.ink2 : C.ink, (weak ? 0.7 : 0.9) * a)
    ctx.fillText(t, x + c / 2, y + c / 2 + 0.5)
  }

  drawMat(o: MatOpts) {
    const ctx = this.ctx, { r, vals, alpha } = o
    if (alpha <= 0.01) return
    const rows = vals.length, cols = vals[0].length, c = r.c, w = cols * c, h = rows * c
    const vmax = vmaxOf(vals)
    this.slab(r.x, r.y, w, h, alpha)
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const x = r.x + j * c, y = r.y + i * c, rv = o.reveal ? o.reveal(i, j) : 1
      if (rv <= 0) { ctx.strokeStyle = rgba(C.ink, 0.1 * alpha); ctx.lineWidth = 1; ctx.strokeRect(x + 1.5, y + 1.5, c - 3, c - 3); continue }
      const a = alpha * rv, v = vals[i][j]
      let fa: number
      if (o.paint) fa = o.paint(x, y, c, i, j, a)
      else if (o.kind === 'attn') fa = this.paintAttn(x, y, c, v, this.tokRGB(j), a)
      else {
        const hue = o.kind === 'row' ? o.rowCols?.[i] ?? this.tokRGB(i) : o.kind === 'col' ? this.tokRGB(j) : null
        fa = this.paintCell(x, y, c, v, vmax, hue, o.kind === 'w' ? 0.3 : 0.45, a)
      }
      if (!o.noText) this.cellText(o.text ? o.text(i, j) : fmt(v), x, y, c, fa, a, o.kind === 'w')
    }
    ctx.strokeStyle = rgba(C.ink, 0.22 * alpha); ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, w - 1, h - 1)
    if (o.colToks) {
      ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'
      for (let j = 0; j < cols; j++) { ctx.fillStyle = rgba(this.tokRGB(j), alpha); ctx.fillText(this.tl(j), r.x + (j + 0.5) * c, r.y - 8) }
    }
    if (o.label !== 'none') {
      const la = alpha * (o.labelAlpha ?? 1)
      const ly = o.label === 'bottom' ? r.y + h + 24 : r.y - (o.colToks ? 28 : 11)
      this.label(o.name, o.shape, o.real, r.x, ly, la)
    }
  }
  /** Italic name, then the toy shape and (fainter) the real shape. */
  label(name: string, shape: string, real: string | undefined, x: number, y: number, a: number) {
    const ctx = this.ctx
    const w1 = mathName(name, x, y, a, 18)
    ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
    ctx.fillStyle = rgba(C.mute, a); ctx.fillText(shape, x + w1 + 8, y)
    if (real) { const w2 = ctx.measureText(shape).width; ctx.fillStyle = rgba(C.mute, 0.55 * a); ctx.fillText('· ' + real, x + w1 + 8 + w2 + 6, y) }
  }
  /** Token chips to the left of a row-per-token matrix. */
  rowChips(r: Rect, a: number, hl = -1) {
    if (a <= 0.01) return
    const h = Math.min(20, r.c * 0.72)
    this.tokens.forEach((t, i) => drawChip(r.x - 10 - chipW(t.text), r.y + (i + 0.5) * r.c, t, a * (hl < 0 || hl === i ? 1 : 0.45), h, hl === i))
  }

  /** The hovered cell if it belongs to one of these GEMMs and is computed; otherwise the animated one. */
  resolve(states: Record<string, { g: Gemm; K: number }>): Focus | null {
    const h = this.hover
    if (h && states[h.key] && states[h.key].g.rev(h.i, h.j)) return { key: h.key, i: h.i, j: h.j, k: states[h.key].K }
    for (const [key, s] of Object.entries(states)) if (s.g.cur) return { key, ...s.g.cur }
    return null
  }

  private gemmSegs(v: GemmView, a: number[], b: number[]): Seg[] {
    const [cN, aN, bN] = v.names, { i, j, k } = v.f
    const K = a.length, done = k >= K, cur = Math.min(K - 1, Math.floor(k)), upto = done ? K : cur + 1
    const bias = v.bias?.[j]
    const segs: Seg[] = [[`${cN}[${i},${j}]`, C.ink], ['  =  ', C.mute], [`Σₖ ${aN}[${i},k]·${bN}[k,${j}]${bias !== undefined ? ' + b[' + j + ']' : ''}`, C.mute], ['  =  ', C.mute]]
    const idx = [...Array(upto).keys()]
    const shown = idx.length > 6 ? [...idx.slice(0, 3), -1, ...idx.slice(-2)] : idx
    shown.forEach((t, n) => {
      if (n) segs.push([' + ', C.mute])
      if (t < 0) segs.push(['…', C.mute])
      else segs.push([`${fmt(a[t])}×${fmt(b[t])}`, !done && t === cur ? C.ink : C.ink2])
    })
    let sum = 0
    for (let t = 0; t < upto; t++) sum += a[t] * b[t]
    if (bias !== undefined && done) { segs.push([' + ', C.mute], [fmt(bias), C.ink2]); sum += bias }
    segs.push(['  =  ', C.mute], [fmt(sum) + (done ? '' : ' …'), done ? C.ink : C.mute])
    return segs
  }

  /** Highlight A's row and B's column, the active term pair, and the C cell being written. */
  gemmOverlay(v: GemmView) {
    const ctx = this.ctx, { i, j, k } = v.f, K = v.Av[0].length, done = k >= K, kc = Math.min(K - 1, Math.floor(k))
    const aRow = v.Av[i], bCol = v.bT ? v.Bv[j] : v.Bv.map((r) => r[j])
    const Ac = v.A.c, Bc = v.B.c, Cc = v.C.c
    const ay = v.A.y + (i + 0.5) * Ac
    const cx = v.C.x + j * Cc, cy = v.C.y + i * Cc
    ctx.setLineDash([2, 4]); ctx.strokeStyle = rgba(C.ink, 0.4); ctx.lineWidth = 1
    ctx.beginPath()
    if (v.C.x > v.A.x) { ctx.moveTo(v.A.x + K * Ac + 3, ay); ctx.lineTo(cx - 3, ay) }
    else { ctx.moveTo(v.A.x - 3, ay); ctx.lineTo(cx + Cc + 3, ay) }
    ctx.stroke()
    ctx.beginPath()
    if (v.bT) {
      const by = v.B.y + (j + 0.5) * Bc, tx = cx + Cc / 2
      ctx.moveTo(v.B.x - 3, by); ctx.quadraticCurveTo(tx, by, tx, cy - 3)
    } else {
      const bx = v.B.x + (j + 0.5) * Bc
      ctx.moveTo(bx, v.B.y + K * Bc + 3); ctx.lineTo(bx, cy - 3)
    }
    ctx.stroke()
    ctx.setLineDash([])
    ctx.strokeStyle = rgba(C.ink, 0.9); ctx.lineWidth = 1.5
    ctx.strokeRect(v.A.x - 1, v.A.y + i * Ac - 1, K * Ac + 2, Ac + 2)
    if (v.bT) ctx.strokeRect(v.B.x - 1, v.B.y + j * Bc - 1, K * Bc + 2, Bc + 2)
    else ctx.strokeRect(v.B.x + j * Bc - 1, v.B.y - 1, Bc + 2, K * Bc + 2)
    if (!done) {
      const ax = v.A.x + kc * Ac, ayy = v.A.y + i * Ac
      const bx = v.bT ? v.B.x + kc * Bc : v.B.x + j * Bc, by = v.bT ? v.B.y + j * Bc : v.B.y + kc * Bc
      ctx.fillStyle = rgba(C.ink, 0.22); ctx.fillRect(ax, ayy, Ac, Ac); ctx.fillRect(bx, by, Bc, Bc)
      ctx.lineWidth = 2; ctx.strokeRect(ax, ayy, Ac, Ac); ctx.strokeRect(bx, by, Bc, Bc)
      if (v.bT) { ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(bx + Bc / 2, by + Bc); ctx.lineTo(ax + Ac / 2, ayy); ctx.stroke() }
    } else if (v.biasR) {
      ctx.lineWidth = 1.5; ctx.strokeRect(v.biasR.x + j * v.biasR.c - 1, v.biasR.y - 1, v.biasR.c + 2, v.biasR.c + 2)
    }
    if (!done) {
      let part = 0
      for (let t = 0; t <= kc; t++) part += aRow[t] * bCol[t]
      ctx.fillStyle = rgba(C.ink, 0.1); ctx.fillRect(cx, cy, Cc, Cc)
      this.cellText(fmt(part), cx, cy, Cc, 0, 1)
    }
    ctx.lineWidth = 2; ctx.strokeStyle = rgba(C.ink, 1); ctx.strokeRect(cx, cy, Cc, Cc)
    this.formula = { segs: this.gemmSegs(v, aRow, bCol), note: v.note }
  }

  drawFormula() {
    if (!this.formula) return
    const ctx = this.ctx, y = this.stage.H - 46
    let x = this.pad
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.font = F.mono(12)
    for (const [t, col, a] of this.formula.segs) { ctx.fillStyle = rgba(col, a ?? 1); x += fillRich(t, x, y) }
    if (this.formula.note) { ctx.font = F.body; ctx.fillStyle = rgba(C.mute); fillRich(this.formula.note, this.pad, y + 21) }
  }
}
