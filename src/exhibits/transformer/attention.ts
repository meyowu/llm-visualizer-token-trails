import { createFrame, toggle } from '../../core/frame'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, blend, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import { F, chipW, drawChip, mathName, mathRun, plate, subLabel, tokLabel, useCtx } from '../../core/draw'
import type { Nav } from '../registry'
import { TOY, attention, promptTokens, transpose, type M } from './model'

/*
 * Attention, opened up. One head of block 1 at toy scale (d_model 8, d_head 4), with every
 * matrix product drawn as a GEMM: A on the left, B above, C where A's row meets B's column.
 * Everything drawn is a pure function of the player's time, so the timeline scrubs freely.
 */

const PHASES = [
  { id: 'qkv', name: 'Projections', short: 'Q · K · V', dur: 8 },
  { id: 'scores', name: 'Scores', short: 'QKᵀ', dur: 6.5 },
  { id: 'scale', name: 'Scale', short: '÷ √d', dur: 2 },
  { id: 'mask', name: 'Causal mask', short: 'Mask', dur: 2 },
  { id: 'softmax', name: 'Softmax', dur: 3.5 },
  { id: 'av', name: 'Weighted sum', short: 'A · V', dur: 7.5 },
  { id: 'out', name: 'Output projection', short: 'W_O', dur: 6 },
]

interface Rect { x: number; y: number; c: number }
const lr = (a: Rect, b: Rect, t: number): Rect => ({ x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), c: lerp(a.c, b.c, t) })

type Kind = 'row' | 'col' | 'w' | 'score' | 'attn'
type Seg = [string, RGB, number?]
interface Gemm { rev: (i: number, j: number) => number; cur: { i: number; j: number; k: number } | null }
interface Focus { key: string; i: number; j: number; k: number }

/** Compact numbers for cells: .57, −1.3, 12. */
const fmt = (v: number) => {
  if (!isFinite(v)) return v < 0 ? '−∞' : '∞'
  const s = v < 0 ? '−' : '', a = Math.abs(v)
  if (a >= 9.95) return s + a.toFixed(0)
  if (a >= 0.995) return s + a.toFixed(1)
  return s + a.toFixed(2).slice(1)
}
const SUBS = '₀₁₂₃₄₅₆₇₈₉'

/**
 * Cell schedule of one GEMM over progress p ∈ [0,1].
 * slow: the first cell accumulates term by term, the rest of row 0 follows, then rows sweep.
 * fast: rows sweep from the start.
 */
function gemm(p: number, m: number, n: number, K: number, mode: 'slow' | 'fast'): Gemm {
  const td = (i: number, j: number) => mode === 'slow'
    ? i === 0 && j === 0 ? 0.34 : i === 0 ? 0.34 + j * (0.18 / Math.max(1, n - 1)) : 0.52 + (i - 1 + (j + 1) / n) * (0.44 / Math.max(1, m - 1))
    : ((i + (j + 1) / n) / m) * 0.96
  let cur: Gemm['cur'] = null, best = Infinity
  if (p > 0) for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) { const t = td(i, j); if (t > p && t < best) { best = t; cur = { i, j, k: K } } }
  if (cur && mode === 'slow' && cur.i === 0 && cur.j === 0) cur.k = clamp(p / 0.34) * K
  return { rev: (i, j) => (p >= td(i, j) ? 1 : 0), cur }
}

export function mountAttention(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const seq = promptTokens(), N = seq.length, R = attention(seq)
  const S = { head: 0, hover: null as null | { key: string; i: number; j: number } }

  const frame = createFrame(root, {
    eyebrow: 'Transformer · Attention',
    title: 'Attention',
    subtitle: 'causal self-attention · block 1 · head 1',
    back: { label: 'Forward pass', onClick: () => nav('transformer') },
    specs: [
      { label: 'shown', value: 'toy', real: 'GPT-2 small' },
      { label: 'tokens', value: String(N) },
      { label: 'd_model', value: String(TOY.d), real: '768' },
      { label: 'd_head', value: String(TOY.dh), real: '64' },
      { label: 'heads', value: String(TOY.heads), real: '12' },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 520, 'Step-by-step attention: X times W_Q, W_K and W_V gives Q, K and V; Q times K transposed gives scores, which are scaled, masked and softmaxed into attention weights A; A times V is projected by W_O and added back to the residual stream.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  toggle(player.meta, 'Attention head', ['head 1', 'head 2'], 0, (i) => {
    S.head = i
    frame.setSubtitle(`causal self-attention · block 1 · head ${i + 1}`)
  })
  const prog = (id: string) => player.prog(id)
  const hd = () => R.heads[S.head]
  const tokRGB = (i: number) => C.tok[seq[i].c % 7]
  const tl = (i: number) => tokLabel(seq[i].text)

  /* ---------- layouts, one per scene ---------- */
  const pad = 36, tokW = 76, top = 60, bot = 96
  const G = {
    qkv: { lx: 0, lnX: 0, X: r0(), Wq: r0(), Wk: r0(), Wv: r0(), Q: r0(), K: r0(), V: r0() },
    scores: { Q: r0(), KT: r0(), S: r0(), Vp: r0(), opsX: 0 },
    av: { A: r0(), V: r0(), O: r0(), px: 0 },
    out: { CC: r0(), Wo: r0(), Out: r0() },
  }
  function r0(): Rect { return { x: 0, y: 0, c: 30 } }
  function geom() {
    const { W, H } = stage
    const fit = (cols: number, rows: number, exW: number, exH: number) =>
      Math.floor(clamp(Math.min((W - 2 * pad - tokW - exW) / cols, (H - top - bot - exH) / rows), 18, 40))
    const cx = (tw: number) => Math.max(pad, (W - tw) / 2)
    const cy = (th: number, exTop: number) => top + exTop + Math.max(0, (H - top - bot - th) / 2)
    {
      const lw = 84, c = fit(22.4, 14, lw, 56), lx = cx(lw + tokW + 22.4 * c), x0 = lx + lw + tokW, y0 = cy(14 * c + 56, 24)
      const col = (k: number) => x0 + (9 + k * 4.7) * c
      Object.assign(G.qkv, {
        lx, lnX: lx + 42, X: { x: x0, y: y0 + 9 * c, c },
        Wq: { x: col(0), y: y0, c }, Wk: { x: col(1), y: y0, c }, Wv: { x: col(2), y: y0, c },
        Q: { x: col(0), y: y0 + 9 * c, c }, K: { x: col(1), y: y0 + 9 * c, c }, V: { x: col(2), y: y0 + 9 * c, c },
      })
    }
    {
      const c = fit(14, 10, 214, 70), x0 = cx(tokW + 14 * c + 214) + tokW, y0 = cy(10 * c + 40, 36)
      Object.assign(G.scores, {
        Q: { x: x0, y: y0 + 5 * c, c }, KT: { x: x0 + 5 * c, y: y0, c }, S: { x: x0 + 5 * c, y: y0 + 5 * c, c },
        Vp: { x: x0 + 10 * c + 214, y: y0 + 5 * c, c }, opsX: x0 + 10 * c + 64,
      })
    }
    {
      const c = fit(10, 11, 300, 70), x0 = cx(tokW + 10 * c + 300) + tokW, y0 = cy(11 * c + 40, 24)
      Object.assign(G.av, { V: { x: x0 + 6 * c, y: y0, c }, A: { x: x0, y: y0 + 6 * c, c }, O: { x: x0 + 6 * c, y: y0 + 6 * c, c }, px: x0 + 10 * c + 52 })
    }
    {
      const c = fit(17, 14, 250, 56), x0 = cx(tokW + 17 * c + 250) + tokW, y0 = cy(14 * c + 56, 24)
      Object.assign(G.out, { CC: { x: x0, y: y0 + 9 * c, c }, Wo: { x: x0 + 9 * c, y: y0, c }, Out: { x: x0 + 9 * c, y: y0 + 9 * c, c } })
    }
  }
  stage.onResize = geom
  geom()

  /* ---------- matrix drawing ---------- */
  let hits: { key: string; r: Rect; rows: number; cols: number }[] = []
  let formula: { segs: Seg[]; note: string } | null = null

  /** The 2.5D slab under a matrix: a right and a bottom face. */
  function slab(x: number, y: number, w: number, h: number, a: number) {
    const d = 5
    ctx.fillStyle = rgba(C.ink, 0.07 * a)
    ctx.beginPath(); ctx.moveTo(x + w, y); ctx.lineTo(x + w + d, y + d); ctx.lineTo(x + w + d, y + h + d); ctx.lineTo(x + w, y + h); ctx.closePath(); ctx.fill()
    ctx.fillStyle = rgba(C.ink, 0.045 * a)
    ctx.beginPath(); ctx.moveTo(x, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w + d, y + h + d); ctx.lineTo(x + d, y + h + d); ctx.closePath(); ctx.fill()
  }
  /** Filled = positive, outlined = negative. Returns the fill alpha so text can pick a contrasting colour. */
  function paintCell(x: number, y: number, c: number, v: number, vmax: number, hue: RGB | null, strength: number, a: number) {
    const m = Math.min(1, Math.abs(v) / vmax)
    if (v < 0) {
      ctx.fillStyle = rgba(C.neg, (0.03 + 0.14 * m) * a); ctx.fillRect(x + 0.5, y + 0.5, c - 1, c - 1)
      ctx.strokeStyle = rgba(C.neg, (0.18 + 0.5 * m) * a); ctx.lineWidth = 1; ctx.strokeRect(x + 2.5, y + 2.5, c - 5, c - 5)
      return 0
    }
    const fa = hue ? 0.1 + 0.72 * m : 0.05 + strength * m
    ctx.fillStyle = rgba(hue ?? C.ink, fa * a); ctx.fillRect(x + 0.5, y + 0.5, c - 1, c - 1)
    return fa
  }
  function paintAttn(x: number, y: number, c: number, v: number, hue: RGB, a: number) {
    const fa = 0.05 + 0.9 * Math.sqrt(Math.max(0, v))
    ctx.fillStyle = rgba(hue, fa * a); ctx.fillRect(x + 0.5, y + 0.5, c - 1, c - 1)
    return fa
  }
  function hatch(x: number, y: number, c: number, a: number) {
    ctx.save(); ctx.beginPath(); ctx.rect(x + 0.5, y + 0.5, c - 1, c - 1); ctx.clip()
    ctx.fillStyle = rgba(C.ink, 0.025 * a); ctx.fillRect(x, y, c, c)
    ctx.strokeStyle = rgba(C.ink, 0.13 * a); ctx.lineWidth = 1
    for (let d = -c; d < c; d += 5) { ctx.beginPath(); ctx.moveTo(x + d, y + c); ctx.lineTo(x + d + c, y); ctx.stroke() }
    ctx.restore()
  }
  function cellText(t: string, x: number, y: number, c: number, fa: number, a: number, weak = false) {
    if (!t || c < 17) return
    ctx.font = F.mono(clamp(c * 0.3, 8, 11)); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    ctx.fillStyle = rgba(fa > 0.55 ? C.bg : weak ? C.ink2 : C.ink, (weak ? 0.7 : 0.9) * a)
    ctx.fillText(t, x + c / 2, y + c / 2 + 0.5)
  }

  interface MatOpts {
    r: Rect; vals: M; kind: Kind; alpha: number
    name: string; shape: string; real?: string
    label?: 'top' | 'bottom' | 'none'; labelAlpha?: number
    reveal?: (i: number, j: number) => number
    rowCols?: RGB[]
    colToks?: boolean
    /** Custom cell painter; returns fill alpha. */
    paint?: (x: number, y: number, c: number, i: number, j: number, a: number) => number
    text?: (i: number, j: number) => string
  }
  function drawMat(o: MatOpts) {
    const { r, vals, alpha } = o
    if (alpha <= 0.01) return
    const rows = vals.length, cols = vals[0].length, c = r.c, w = cols * c, h = rows * c
    const vmax = Math.max(1e-6, ...vals.flat().filter((v) => isFinite(v)).map(Math.abs))
    slab(r.x, r.y, w, h, alpha)
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const x = r.x + j * c, y = r.y + i * c, rv = o.reveal ? o.reveal(i, j) : 1
      if (rv <= 0) { ctx.strokeStyle = rgba(C.ink, 0.1 * alpha); ctx.lineWidth = 1; ctx.strokeRect(x + 1.5, y + 1.5, c - 3, c - 3); continue }
      const a = alpha * rv, v = vals[i][j]
      let fa: number
      if (o.paint) fa = o.paint(x, y, c, i, j, a)
      else if (o.kind === 'attn') fa = paintAttn(x, y, c, v, tokRGB(j), a)
      else {
        const hue = o.kind === 'row' ? o.rowCols?.[i] ?? tokRGB(i) : o.kind === 'col' ? tokRGB(j) : null
        fa = paintCell(x, y, c, v, vmax, hue, o.kind === 'w' ? 0.3 : 0.45, a)
      }
      cellText(o.text ? o.text(i, j) : fmt(v), x, y, c, fa, a, o.kind === 'w')
    }
    ctx.strokeStyle = rgba(C.ink, 0.22 * alpha); ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, w - 1, h - 1)
    if (o.colToks) {
      ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'
      for (let j = 0; j < cols; j++) { ctx.fillStyle = rgba(tokRGB(j), alpha); ctx.fillText(tl(j), r.x + (j + 0.5) * c, r.y - 8) }
    }
    if (o.label !== 'none') {
      const la = alpha * (o.labelAlpha ?? 1)
      const ly = o.label === 'bottom' ? r.y + h + 24 : r.y - (o.colToks ? 28 : 11)
      const w1 = mathName(o.name, r.x, ly, la, 18)
      ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
      ctx.fillStyle = rgba(C.mute, la); ctx.fillText(o.shape, r.x + w1 + 8, ly)
      if (o.real) { const w2 = ctx.measureText(o.shape).width; ctx.fillStyle = rgba(C.mute, 0.55 * la); ctx.fillText('· ' + o.real, r.x + w1 + 8 + w2 + 6, ly) }
    }
  }
  /** Token chips to the left of a row-per-token matrix. */
  function rowChips(r: Rect, a: number, hl = -1) {
    if (a <= 0.01) return
    const h = Math.min(20, r.c * 0.72)
    seq.forEach((t, i) => drawChip(r.x - 10 - chipW(t.text), r.y + (i + 0.5) * r.c, t, a * (hl < 0 || hl === i ? 1 : 0.45), h, hl === i))
  }

  /* ---------- GEMM focus ---------- */
  function resolve(states: Record<string, { g: Gemm; K: number }>): Focus | null {
    const h = S.hover
    if (h && states[h.key] && states[h.key].g.rev(h.i, h.j)) return { key: h.key, i: h.i, j: h.j, k: states[h.key].K }
    for (const [key, s] of Object.entries(states)) if (s.g.cur) return { key, ...s.g.cur }
    return null
  }
  function gemmSegs(cN: string, aN: string, bN: string, i: number, j: number, a: number[], b: number[], k: number): Seg[] {
    const K = a.length, done = k >= K, cur = Math.min(K - 1, Math.floor(k)), upto = done ? K : cur + 1
    const segs: Seg[] = [[`${cN}[${i},${j}]`, C.ink], ['  =  ', C.mute], [`Σₖ ${aN}[${i},k]·${bN}[k,${j}]`, C.mute], ['  =  ', C.mute]]
    const idx = [...Array(upto).keys()]
    const shown = idx.length > 6 ? [...idx.slice(0, 3), -1, ...idx.slice(-2)] : idx
    shown.forEach((t, n) => {
      if (n) segs.push([' + ', C.mute])
      if (t < 0) segs.push(['…', C.mute])
      else segs.push([`${fmt(a[t])}×${fmt(b[t])}`, !done && t === cur ? C.ink : C.ink2])
    })
    let sum = 0
    for (let t = 0; t < upto; t++) sum += a[t] * b[t]
    segs.push(['  =  ', C.mute], [fmt(sum) + (done ? '' : ' …'), done ? C.ink : C.mute])
    return segs
  }
  /** Highlight A's row and B's column, the active term pair, and the C cell being written. */
  function gemmOverlay(o: { A: Rect; Av: M; B: Rect; Bv: M; C: Rect; f: Focus; names: [string, string, string]; note: string }) {
    const { i, j, k } = o.f, K = o.Av[0].length, done = k >= K, kc = Math.min(K - 1, Math.floor(k))
    const aRow = o.Av[i], bCol = o.Bv.map((r) => r[j])
    const Ac = o.A.c, Bc = o.B.c, Cc = o.C.c
    const ay = o.A.y + (i + 0.5) * Ac, bx = o.B.x + (j + 0.5) * Bc
    ctx.setLineDash([2, 4]); ctx.strokeStyle = rgba(C.ink, 0.4); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(o.A.x + K * Ac + 3, ay); ctx.lineTo(o.C.x + j * Cc - 3, ay); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(bx, o.B.y + K * Bc + 3); ctx.lineTo(bx, o.C.y + i * Cc - 3); ctx.stroke()
    ctx.setLineDash([])
    ctx.strokeStyle = rgba(C.ink, 0.9); ctx.lineWidth = 1.5
    ctx.strokeRect(o.A.x - 1, o.A.y + i * Ac - 1, K * Ac + 2, Ac + 2)
    ctx.strokeRect(o.B.x + j * Bc - 1, o.B.y - 1, Bc + 2, K * Bc + 2)
    if (!done) {
      ctx.fillStyle = rgba(C.ink, 0.22)
      ctx.fillRect(o.A.x + kc * Ac, o.A.y + i * Ac, Ac, Ac); ctx.fillRect(o.B.x + j * Bc, o.B.y + kc * Bc, Bc, Bc)
      ctx.lineWidth = 2
      ctx.strokeRect(o.A.x + kc * Ac, o.A.y + i * Ac, Ac, Ac); ctx.strokeRect(o.B.x + j * Bc, o.B.y + kc * Bc, Bc, Bc)
    }
    const cx = o.C.x + j * Cc, cy = o.C.y + i * Cc
    if (!done) {
      let part = 0
      for (let t = 0; t <= kc; t++) part += aRow[t] * bCol[t]
      ctx.fillStyle = rgba(C.ink, 0.1); ctx.fillRect(cx, cy, Cc, Cc)
      cellText(fmt(part), cx, cy, Cc, 0, 1)
    }
    ctx.lineWidth = 2; ctx.strokeStyle = rgba(C.ink, 1); ctx.strokeRect(cx, cy, Cc, Cc)
    formula = { segs: gemmSegs(...o.names, i, j, aRow, bCol, k), note: o.note }
  }
  function drawFormula() {
    if (!formula) return
    const y = stage.H - 46
    let x = pad
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.font = F.mono(12)
    for (const [t, col, a] of formula.segs) { ctx.fillStyle = rgba(col, a ?? 1); ctx.fillText(t, x, y); x += ctx.measureText(t).width }
    if (formula.note) { ctx.font = F.body; ctx.fillStyle = rgba(C.mute); ctx.fillText(formula.note, pad, y + 21) }
  }

  /* ---------- colour of a token after attention (matches the overview's lanes) ---------- */
  const hues = () => seq.map((_, i) => tokRGB(i))
  const oCols = (A: M) => A.map((row) => blend(hues(), row))
  const laneCols = () => R.heads[0].A.map((row, i) => blend(hues(), row.map((w, k) => 0.55 * w + (k === i ? 0.45 : 0))))

  /* ---------- scene 1: X · W_Q, W_K, W_V ---------- */
  function sceneQKV(p: number) {
    const L = G.qkv, h = hd(), c = L.X.c, pin = eout(clamp(p / 0.06)), pw = eout(clamp((p - 0.02) / 0.07))
    seq.forEach((t, i) => {
      const y = L.X.y + (i + 0.5) * c
      ctx.strokeStyle = rgba(tokRGB(i), 0.85 * pin); ctx.lineWidth = 1.6
      ctx.beginPath(); ctx.moveTo(L.lx, y); ctx.lineTo(L.lnX - 9, y); ctx.moveTo(L.lnX + 9, y); ctx.lineTo(L.X.x - 18 - chipW(t.text), y); ctx.stroke()
    })
    plate(L.lnX, L.X.y - 2, L.X.y + N * c + 2, p > 0 && p < 0.08 ? 1 : 0, { w: 7, d: 8 })
    subLabel('ln_1', L.lnX, L.X.y - 16, false)
    ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, pin); ctx.fillText('h', L.lx, L.X.y - 16)
    rowChips(L.X, pin)
    drawMat({ r: L.X, vals: R.X, kind: 'row', alpha: pin, name: 'X', shape: '5 × 8', real: 'N × 768' })
    const st = {
      Q: { g: gemm((p - 0.08) / 0.52, N, 4, 8, 'slow'), K: 8 },
      K: { g: gemm((p - 0.6) / 0.18, N, 4, 8, 'fast'), K: 8 },
      V: { g: gemm((p - 0.78) / 0.18, N, 4, 8, 'fast'), K: 8 },
    }
    const W = { Q: h.Wq, K: h.Wk, V: h.Wv }, Rm = { Q: h.Q, K: h.K, V: h.V }
    const Wr = { Q: L.Wq, K: L.Wk, V: L.Wv }, Rr = { Q: L.Q, K: L.K, V: L.V }
    ;(['Q', 'K', 'V'] as const).forEach((k) => {
      drawMat({ r: Wr[k], vals: W[k], kind: 'w', alpha: pw, name: 'W_' + k, shape: '8 × 4', real: '768 × 64' })
      drawMat({ r: Rr[k], vals: Rm[k], kind: 'row', alpha: pw, name: k, shape: '5 × 4', real: 'N × 64', label: 'bottom', reveal: st[k].g.rev })
      hits.push({ key: k, r: Rr[k], rows: N, cols: 4 })
    })
    const f = resolve(st)
    if (f) {
      const k = f.key as 'Q' | 'K' | 'V', role = { Q: 'query', K: 'key', V: 'value' }[k]
      gemmOverlay({ A: L.X, Av: R.X, B: Wr[k], Bv: W[k], C: Rr[k], f, names: [k, 'X', 'W_' + k], note: `${tl(f.i)}'s ${role} vector, dim ${f.j}: row ${f.i} of X times column ${f.j} of W_${k}, term by term, summed.` })
    }
  }

  /* ---------- scene 2: Q · Kᵀ, then scale, mask, softmax ---------- */
  function transposeCells(K: M, from: Rect, to: Rect, t: number) {
    const e = eio(t), vmax = Math.max(...K.flat().map(Math.abs))
    K.forEach((row, i) => row.forEach((v, j) => {
      const c = lerp(from.c, to.c, e)
      const x = lerp(from.x + j * from.c, to.x + i * to.c, e), y = lerp(from.y + i * from.c, to.y + j * to.c, e)
      const fa = paintCell(x, y, c, v, vmax, tokRGB(i), 0, 1)
      cellText(fmt(v), x, y, c, fa, 1)
    }))
  }
  function drawOps(x: number, y0: number, c: number, steps: [string, number, string][], a: number) {
    steps.forEach(([name, p, note], k) => {
      const y = y0 + 10 + k * Math.max(34, c * 1.1), done = p >= 1, on = p > 0 && p < 1
      ctx.beginPath(); ctx.arc(x + 4, y - 4, 3.5, 0, 7)
      if (done) { ctx.fillStyle = rgba(C.ink2, a); ctx.fill() } else { ctx.strokeStyle = rgba(on ? C.ink : C.faint, a); ctx.lineWidth = 1.2; ctx.stroke() }
      ctx.font = F.mono(12, on ? 500 : 400); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
      ctx.fillStyle = rgba(on ? C.ink : done ? C.ink2 : C.mute, a); ctx.fillText(name, x + 16, y)
      if (on || done) { ctx.font = F.body; ctx.fillStyle = rgba(C.mute, a * (on ? 1 : 0.7)); ctx.fillText(note, x + 16, y + 16) }
    })
  }
  function sceneScores(ps: number, pc: number, pm: number, pso: number) {
    const Lq = G.qkv, L = G.scores, h = hd(), tr = eio(clamp(ps / 0.2)), c = L.S.c
    if (tr < 1) {
      const fa = 1 - tr
      drawMat({ r: Lq.X, vals: R.X, kind: 'row', alpha: fa, name: 'X', shape: '5 × 8' })
      drawMat({ r: Lq.Wq, vals: h.Wq, kind: 'w', alpha: fa, name: 'W_Q', shape: '8 × 4' })
      drawMat({ r: Lq.Wk, vals: h.Wk, kind: 'w', alpha: fa, name: 'W_K', shape: '8 × 4' })
      drawMat({ r: Lq.Wv, vals: h.Wv, kind: 'w', alpha: fa, name: 'W_V', shape: '8 × 4' })
      transposeCells(h.K, Lq.K, L.KT, tr)
    } else drawMat({ r: L.KT, vals: transpose(h.K), kind: 'col', alpha: 1, name: 'Kᵀ', shape: '4 × 5', real: '64 × N', colToks: true })
    rowChips(lr(Lq.X, L.Q, tr), 1)
    drawMat({ r: lr(Lq.Q, L.Q, tr), vals: h.Q, kind: 'row', alpha: 1, name: 'Q', shape: '5 × 4', real: 'N × 64', labelAlpha: tr })
    drawMat({ r: lr(Lq.V, L.Vp, tr), vals: h.V, kind: 'row', alpha: lerp(1, 0.35, tr), name: 'V', shape: '5 × 4 · used later', labelAlpha: tr })

    const g = gemm((ps - 0.22) / 0.75, N, N, 4, 'slow')
    const scaleT = eio(clamp(pc / 0.6))
    const vals = h.S.map((r, i) => r.map((v, j) => lerp(v, h.Ss[i][j], scaleT)))
    const vmax = Math.max(...vals.flat().map(Math.abs))
    const maskP = (i: number, j: number) => (j > i ? clamp((pm - (j - i - 1) * 0.12) / 0.4) : 0)
    const softR = (i: number) => clamp((pso - i * 0.13) / 0.35)
    const named = pso > 0.5 ? 'A' : scaleT > 0.5 ? 'S′' : 'S'
    drawMat({
      r: L.S, vals, kind: 'score', alpha: tr, name: named, shape: '5 × 5', real: 'N × N', reveal: g.rev,
      paint: (x, y, cc, i, j, a) => {
        const m = maskP(i, j), sr = softR(i)
        let fa = 0
        if (m < 1) fa = paintCell(x, y, cc, vals[i][j], vmax, null, 0.45, a * (1 - sr) * (1 - m))
        if (sr > 0 && j <= i) fa = Math.max(fa, paintAttn(x, y, cc, h.A[i][j], tokRGB(j), a * sr))
        if (m > 0) hatch(x, y, cc, a * m)
        return m > 0.5 ? 0 : sr > 0.5 ? 0.05 + 0.9 * Math.sqrt(h.A[i][j]) : fa
      },
      text: (i, j) => (maskP(i, j) > 0.5 ? (softR(i) > 0.5 ? '0' : '−∞') : softR(i) > 0.5 ? fmt(h.A[i][j]) : fmt(vals[i][j])),
    })
    hits.push({ key: 'S', r: L.S, rows: N, cols: N })
    // row sums once a row has been normalised
    for (let i = 0; i < N; i++) {
      const a = clamp((softR(i) - 0.8) / 0.2)
      if (a <= 0) continue
      ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
      ctx.fillStyle = rgba(C.mute, a); ctx.fillText('Σ 1.00', L.S.x + N * c + 12, L.S.y + (i + 0.5) * c)
    }
    drawOps(L.opsX, L.S.y, c, [
      ['Q · Kᵀ', clamp((ps - 0.22) / 0.75), 'one dot product per cell'],
      ['÷ √d_head', pc, `÷ ${Math.sqrt(TOY.dh)}  (GPT-2: ÷ 8)`],
      ['mask', pm, 'set j > i to −∞'],
      ['softmax', pso, 'each row sums to 1'],
    ], tr)

    const f = resolve({ S: { g, K: 4 } })
    if (f && pc === 0) {
      gemmOverlay({ A: L.Q, Av: h.Q, B: L.KT, Bv: transpose(h.K), C: L.S, f, names: ['S', 'Q', 'Kᵀ'], note: `${tl(f.i)}'s query · ${tl(f.j)}'s key: the higher the score, the more ${tl(f.i)} attends to ${tl(f.j)}.` })
    } else if (pso > 0) {
      const i = S.hover?.key === 'S' ? S.hover.i : Math.min(N - 1, Math.floor(clamp(pso / 0.87) * N))
      const row = h.Ss[i].slice(0, i + 1), arow = h.A[i].slice(0, i + 1)
      formula = {
        segs: [[`A[${i}]`, C.ink], ['  =  softmax( ', C.mute], [row.map(fmt).join(', '), C.ink2], [' )  =  ', C.mute], [arow.map((v) => v.toFixed(2)).join(', '), C.ink]],
        note: `The ${i + 1} tokens ${tl(i)} can see: exponentiate, divide by the sum. The largest score takes most of the weight.`,
      }
      ctx.strokeStyle = rgba(C.ink, 0.9); ctx.lineWidth = 1.5; ctx.strokeRect(L.S.x - 1, L.S.y + i * c - 1, N * c + 2, c + 2)
    } else if (pm > 0) {
      formula = { segs: [['S′[i,j]  =  −∞', C.ink], ['    when j > i', C.mute]], note: 'Token i sees only itself and earlier tokens. exp(−∞) = 0, so these weights become 0.' }
    } else if (pc > 0) {
      formula = { segs: [['S′  =  S / √d_head  =  S / ', C.ink], [String(Math.sqrt(TOY.dh)), C.ink]], note: `After scaling, the ${tl(N - 1)} row goes from [${h.S[N - 1].map(fmt).join(', ')}] to [${h.Ss[N - 1].map(fmt).join(', ')}].` }
    }
  }

  /* ---------- scene 3: A · V ---------- */
  function attnPaint(A: M) {
    return (x: number, y: number, c: number, i: number, j: number, a: number) => {
      if (j > i) { hatch(x, y, c, a); return 0 }
      return paintAttn(x, y, c, A[i][j], tokRGB(j), a)
    }
  }
  const attnText = (A: M) => (i: number, j: number) => (j > i ? '' : fmt(A[i][j]))
  function drawPanel(i: number, a: number) {
    const L = G.av, h = hd(), c = L.O.c, rh = clamp(c * 0.82, 22, 30), s = Math.min(rh * 0.68, 16)
    const x = L.px, y0 = L.A.y, vmax = Math.max(...h.V.flat().map(Math.abs))
    mathRun([['o', false], [String(i), true], [' = Σ', false], ['j', true], [' a', false], [`${i}j`, true], [' · v', false], ['j', true]], x, y0 - 12, a)
    const amax = Math.max(...h.A[i])
    for (let j = 0; j <= i; j++) {
      const y = y0 + j * rh + rh / 2, w = h.A[i][j], wa = a * (0.25 + 0.75 * (w / amax))
      drawChip(x, y, seq[j], wa, Math.min(18, rh * 0.75), false, F.mono(11, 500))
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
      ctx.fillStyle = rgba(C.ink2, wa); ctx.fillText('× ' + w.toFixed(2), x + 60, y)
      for (let k = 0; k < TOY.dh; k++) paintCell(x + 118 + k * (s + 2), y - s / 2, s, h.V[j][k] * w / amax, vmax, tokRGB(j), 0, a)
    }
    const ys = y0 + (i + 1) * rh + 6, xr = x + 118 + TOY.dh * (s + 2)
    ctx.strokeStyle = rgba(C.ink, 0.5 * a); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x + 118, ys); ctx.lineTo(xr - 2, ys); ctx.stroke()
    const oc = oCols(h.A)[i], omax = Math.max(...h.O.flat().map(Math.abs))
    ctx.font = F.mono(11); ctx.fillStyle = rgba(C.mute, a); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    ctx.fillText(`o${SUBS[i]} =`, x + 110, ys + 6 + s / 2)
    for (let k = 0; k < TOY.dh; k++) paintCell(x + 118 + k * (s + 2), ys + 6, s, h.O[i][k], omax, oc, 0, a)
  }
  function sceneAV(pav: number) {
    const Ls = G.scores, L = G.av, h = hd(), tr = eio(clamp(pav / 0.15))
    if (tr < 1) {
      drawMat({ r: Ls.Q, vals: h.Q, kind: 'row', alpha: 1 - tr, name: 'Q', shape: '5 × 4' })
      drawMat({ r: Ls.KT, vals: transpose(h.K), kind: 'col', alpha: 1 - tr, name: 'Kᵀ', shape: '4 × 5', colToks: true })
    }
    rowChips(lr(Ls.Q, L.A, tr), 1, S.hover && (S.hover.key === 'A' || S.hover.key === 'O') ? S.hover.i : -1)
    drawMat({ r: lr(Ls.S, L.A, tr), vals: h.A, kind: 'attn', alpha: 1, name: 'A', shape: '5 × 5', real: 'N × N', colToks: tr > 0.5, paint: attnPaint(h.A), text: attnText(h.A) })
    const vr = lr(Ls.Vp, L.V, tr)
    drawMat({ r: vr, vals: h.V, kind: 'row', alpha: lerp(0.35, 1, tr), name: 'V', shape: '5 × 4', real: 'N × 64' })
    ctx.font = F.mono(10); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    for (let j = 0; j < N; j++) { ctx.fillStyle = rgba(tokRGB(j), tr); ctx.fillText(tl(j), vr.x - 8, vr.y + (j + 0.5) * vr.c) }
    const g = gemm((pav - 0.17) / 0.78, N, TOY.dh, N, 'slow')
    drawMat({ r: L.O, vals: h.O, kind: 'row', alpha: tr, name: 'O', shape: '5 × 4', real: 'N × 64', reveal: g.rev, rowCols: oCols(h.A) })
    hits.push({ key: 'O', r: L.O, rows: N, cols: TOY.dh }, { key: 'A', r: L.A, rows: N, cols: N })
    const f = resolve({ O: { g, K: N } })
    if (f) gemmOverlay({ A: L.A, Av: h.A, B: L.V, Bv: h.V, C: L.O, f, names: ['O', 'A', 'V'], note: `${tl(f.i)}'s output, dim ${f.j}: dim ${f.j} of every value, weighted by ${tl(f.i)}'s attention.` })
    const row = S.hover && (S.hover.key === 'A' || S.hover.key === 'O') ? S.hover.i : f ? f.i : pav >= 0.95 ? N - 1 : -1
    if (row >= 0) drawPanel(row, tr)
  }

  /* ---------- scene 4: concat heads · W_O, + residual ---------- */
  function sceneOut(po: number) {
    const La = G.av, L = G.out, h = hd(), tr = eio(clamp(po / 0.15)), c = L.CC.c
    if (tr < 1) {
      drawMat({ r: La.A, vals: h.A, kind: 'attn', alpha: 1 - tr, name: 'A', shape: '5 × 5', paint: attnPaint(h.A), text: attnText(h.A), colToks: true })
      drawMat({ r: La.V, vals: h.V, kind: 'row', alpha: 1 - tr, name: 'V', shape: '5 × 4' })
    }
    rowChips(lr(La.A, L.CC, tr), 1)
    const slot = (k: number): Rect => ({ x: L.CC.x + k * TOY.dh * c, y: L.CC.y, c })
    const other = 1 - S.head
    drawMat({ r: slot(other), vals: R.heads[other].O, kind: 'row', alpha: tr, name: 'O', shape: '', label: 'none', rowCols: oCols(R.heads[other].A) })
    drawMat({ r: lr(La.O, slot(S.head), tr), vals: h.O, kind: 'row', alpha: 1, name: 'O', shape: '', label: 'none', rowCols: oCols(h.A) })
    // head brackets + concat label
    ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'
    for (let k = 0; k < TOY.heads; k++) {
      const s = slot(k)
      ctx.fillStyle = rgba(k === S.head ? C.ink2 : C.mute, tr); ctx.fillText(`head ${k + 1}`, s.x + (TOY.dh * c) / 2, s.y - 8)
    }
    const la = clamp((tr - 0.6) / 0.4)
    const w1 = mathName('concat', L.CC.x, L.CC.y - 26, la, 18)
    ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, la); ctx.fillText('5 × 8', L.CC.x + w1 + 8, L.CC.y - 26)
    ctx.fillStyle = rgba(C.mute, 0.55 * la); ctx.fillText('· N × 768', L.CC.x + w1 + 8 + ctx.measureText('5 × 8').width + 6, L.CC.y - 26)

    const pw = clamp((po - 0.1) / 0.1)
    drawMat({ r: L.Wo, vals: R.Wo, kind: 'w', alpha: pw, name: 'W_O', shape: '8 × 8', real: '768 × 768' })
    const g = gemm((po - 0.18) / 0.5, N, TOY.d, TOY.d, 'fast')
    const mixed = R.heads[0].A.map((_, i) => blend(hues(), hues().map((__, k) => (R.heads[0].A[i][k] + R.heads[1].A[i][k]) / 2)))
    drawMat({ r: L.Out, vals: R.out, kind: 'row', alpha: pw, name: 'attn_out', shape: '5 × 8', real: 'N × 768', reveal: g.rev, rowCols: mixed })
    hits.push({ key: 'Out', r: L.Out, rows: N, cols: TOY.d })

    // residual add, then back into the stream
    const pr = clamp((po - 0.72) / 0.22)
    if (pr > 0) {
      const xp = L.Out.x + TOY.d * c + 24, xEnd = Math.min(stage.W - pad, xp + 230), lc = laneCols()
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, pr)
      ctx.fillText('+ h', xp, L.Out.y - 10)
      for (let i = 0; i < N; i++) {
        const y = L.Out.y + (i + 0.5) * c, e = eio(clamp(pr * 1.3 - i * 0.06))
        ctx.strokeStyle = rgba(C.ink, 0.35 * pr); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(L.Out.x + TOY.d * c + 6, y); ctx.lineTo(xp - 7, y); ctx.stroke()
        ctx.beginPath(); ctx.arc(xp, y, 7, 0, 7); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(xp - 3.5, y); ctx.lineTo(xp + 3.5, y); ctx.moveTo(xp, y - 3.5); ctx.lineTo(xp, y + 3.5); ctx.stroke()
        ctx.strokeStyle = rgba(lc[i], 0.9 * e); ctx.lineWidth = 1.8
        ctx.beginPath(); ctx.moveTo(xp + 8, y); ctx.lineTo(lerp(xp + 8, xEnd, e), y); ctx.stroke()
      }
      ctx.font = F.small; ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.mute, clamp(pr * 2 - 1))
      ctx.fillText('→ ln_2 · MLP', xEnd, L.Out.y - 10)
    }

    const f = resolve({ Out: { g, K: TOY.d } })
    if (f) gemmOverlay({ A: L.CC, Av: R.concat, B: L.Wo, Bv: R.Wo, C: L.Out, f, names: ['out', 'concat', 'W_O'], note: `${tl(f.i)}'s attention output, dim ${f.j}: W_O mixes what the two heads found.` })
    else if (pr > 0) formula = { segs: [['h′  =  h + attn_out', C.ink]], note: "The result is added to the residual stream, not substituted for it. Each row's color is what that token gathered from the others." }
  }

  /* ---------- frame ---------- */
  function draw() {
    useCtx(ctx)
    stage.begin()
    hits = []
    formula = null
    const ps = prog('scores'), pav = prog('av'), po = prog('out')
    if (ps <= 0) sceneQKV(prog('qkv'))
    else if (pav <= 0) sceneScores(ps, prog('scale'), prog('mask'), prog('softmax'))
    else if (po <= 0) sceneAV(pav)
    else sceneOut(po)
    drawFormula()
  }

  const CAPS: Record<string, [string, string]> = {
    qkv: ['X (after ln_1) is multiplied by W_Q, W_K and W_V to give every token a query, key and value. Each cell of a result is one row of X dotted with one column of W. GPT-2 fuses the three into a single GEMM: X · W_qkv.', 'GPT-2 [N×768]·[768×2304] · 17.7 MFLOPs'],
    scores: ["Q times K transposed. Row i, column j is the dot product of token i's query with token j's key: how much i should attend to j.", 'GPT-2 12 × [N×64]·[64×N]'],
    scale: ['Divide by √d_head. Dot products grow with dimension; scaling keeps softmax from saturating from the start.', 'toy ÷ 2 · GPT-2 ÷ 8'],
    mask: ['Set the upper triangle to −∞. When predicting token i, the model cannot see the tokens after it.', 'causal: j > i → −∞'],
    softmax: ["Softmax each row, turning scores into weights that sum to 1; −∞ becomes 0 after exp. Each row is one token's attention distribution.", 'A = softmax(S / √d + mask)'],
    av: ['Weight the rows of V by attention and sum them. Output row i blends the values of every visible token, and its color blends with them. The panel on the right breaks down the current row.', 'GPT-2 12 × [N×N]·[N×64]'],
    out: ["Concatenate the heads' outputs, multiply by W_O to mix them, and add the result back to the residual stream for the MLP.", 'GPT-2 [N×768]·[768×768] · 5.9 MFLOPs'],
  }

  /* ---------- pointer ---------- */
  const cv = stage.canvas
  cv.addEventListener('pointermove', (e) => {
    const [x, y] = stage.local(e)
    S.hover = null
    for (const hsp of hits) {
      const j = Math.floor((x - hsp.r.x) / hsp.r.c), i = Math.floor((y - hsp.r.y) / hsp.r.c)
      if (i >= 0 && j >= 0 && i < hsp.rows && j < hsp.cols) { S.hover = { key: hsp.key, i, j }; break }
    }
    cv.style.cursor = S.hover ? 'crosshair' : 'default'
  })
  cv.addEventListener('pointerleave', () => { S.hover = null })

  if (reduced) player.t = player.start('av') + 4
  const stop = runLoop((dt) => {
    player.tick(dt)
    draw()
    player.updateUI()
    const cur = player.cur(), [t, s] = CAPS[cur.id]
    frame.setCaption(cur.name, cur.short ?? cur.name, t, s)
  })
  return () => { stop(); player.destroy(); stage.destroy() }
}
