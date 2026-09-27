import { chipW, drawChip, F, fillRich, mathName, serifAt, spaced, useCtx } from '../../core/draw'
import { createFrame, toggle } from '../../core/frame'
import { MatrixKit, fmt, lr, type M, type Rect } from '../../core/matrix'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, rgba } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import type { Nav } from '../registry'
import { TOY, TOY_PROMPTS, posEmb, tokEmb, toyTokens, type Tok } from './model'

/*
 * Embedding as a GEMM: one-hot(ids) [N × V] · W_E [V × d] picks rows of W_E. The vocabulary
 * axis (50,257 rows) is compressed on a log scale, which is also where this prompt's small ids
 * are readable. Then the first N rows of W_P are added: h = E + P.
 */

const PHASES = [
  { id: 'onehot', name: 'One-hot ids', short: 'One-hot', dur: 4 },
  { id: 'lookup', name: 'Row lookup', short: 'onehot · W_E', dur: 7 },
  { id: 'pos', name: 'Add positions', short: '+ W_P', dur: 6 },
  { id: 'stream', name: 'Into the stream', short: 'Stream', dur: 3.5 },
]
const V = 50257, CTX = 1024
const logPos = (id: number) => Math.log10(Math.max(1, id)) / Math.log10(V)

export function mountEmbedding(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const D = TOY.d, N = 5 // every toy prompt has five tokens
  const seq: Tok[] = []
  let E: M = [], P: M = [], Hm: M = [], promptK = 0
  function setPrompt(k: number) {
    promptK = k
    seq.length = 0; seq.push(...toyTokens(k))
    E = seq.map((t) => tokEmb(t)); P = seq.map((_, i) => posEmb(i))
    Hm = E.map((r, i) => r.map((v, q) => v + P[i][q]))
  }
  setPrompt(0)

  const frame = createFrame(root, {
    formula: true,
    eyebrow: 'Anatomy · Embedding',
    title: 'Embedding',
    subtitle: 'token + position · GPT-2',
    back: { label: 'Forward pass', onClick: () => nav('anatomy') },
    specs: [
      { label: 'shown', value: 'toy scale' },
      { label: 'vocab', value: '50,257' },
      { label: 'd_model', value: String(D), real: '768' },
      { label: 'W_E params', value: '38.6M' },
      { label: 'W_P params', value: '0.79M' },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 470, 'Embedding: each token id becomes a one-hot row that selects one row of the embedding matrix W_E; the position embedding for each slot is then added to form the residual stream.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  const prog = (id: string) => player.prog(id)
  const mk = new MatrixKit(stage, seq, frame.setFormula)
  const { tokRGB, tl } = mk
  toggle(player.meta, 'Prompt', TOY_PROMPTS.map((p) => p.label), 0, setPrompt)
  /** What this prompt shows about positions, for the notes. */
  const orderNote = () => {
    const dup = seq.findIndex((t, i) => seq.findIndex((u) => u.id === t.id) !== i)
    if (dup >= 0) return `${tl(seq.findIndex((t) => t.id === seq[dup].id))} appears twice: both E rows are identical, and only their P rows tell the two apart.`
    if (promptK >= 2) return `The same five E rows as “${TOY_PROMPTS[promptK === 2 ? 3 : 2].label}”, in another order: without P the model could not tell who bit whom.`
    return 'The same token in another slot would get a different P row, so order is not lost.'
  }

  /* ---------- layout ---------- */
  const pad = 36, tokW = 124, top = 60, bot = 46
  const L = { A: r0(), Lv: 200, B: { x: 0, y: 0, w: 0, h: 0 }, C: r0(), E2: r0(), P2: r0(), H2: r0(), WP: { x: 0, y: 0, w: 0, h: 0 } }
  function r0(): Rect { return { x: 0, y: 0, c: 30 } }
  function geom() {
    const { W, H } = stage, avail = H - top - bot
    const c = Math.floor(clamp(Math.min(avail / 12, (W - 2 * pad - tokW - 60) / 27), 20, 34))
    const Lv = Math.round(clamp(avail - 5 * c - 70, 130, 300))
    const total = tokW + Lv + 30 + D * c
    const x0 = Math.max(pad + tokW, (W - total) / 2 + tokW), y0 = top + 26 + Math.max(0, (avail - (Lv + 24 + 5 * c + 50)) / 2)
    const cy = y0 + Lv + 24
    L.Lv = Lv
    L.A = { x: x0, y: cy, c }
    L.B = { x: x0 + Lv + 30, y: y0, w: D * c, h: Lv }
    L.C = { x: x0 + Lv + 30, y: cy, c }
    const t2 = tokW + 26 * c, px0 = Math.max(pad + tokW, (W - t2) / 2 + tokW)
    L.E2 = { x: px0, y: cy, c }; L.P2 = { x: px0 + 9 * c, y: cy, c }; L.H2 = { x: px0 + 18 * c, y: cy, c }
    L.WP = { x: px0 + 9 * c, y: y0 + 40, w: D * c, h: Math.max(80, Lv - 60) }
  }
  stage.onResize = geom
  geom()

  /* ---------- pieces ---------- */
  /** Chips plus ids to the left of a row-per-token rect. */
  function chips(r: Rect, a: number, idA: number) {
    seq.forEach((t, i) => {
      const y = r.y + (i + 0.5) * r.c, h = Math.min(20, r.c * 0.72)
      drawChip(r.x - 58 - chipW(t.text), y, t, a, h, mk.hovered('A', 'E', 'H')?.i === i)
      ctx.font = F.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
      ctx.fillStyle = rgba(C.mute, a * idA); ctx.fillText(String(t.id), r.x - 10, y + 0.5)
    })
  }
  /** One-hot rows over the log-compressed vocabulary axis. */
  function drawOneHot(a: number, grow: number, lit: number) {
    const { A, Lv } = L, c = A.c
    for (let i = 0; i < N; i++) {
      const y = A.y + i * c
      ctx.strokeStyle = rgba(C.ink, 0.14 * a); ctx.lineWidth = 1; ctx.strokeRect(A.x + 0.5, y + 1.5, Lv * grow - 1, c - 3)
      ctx.fillStyle = rgba(C.ink, 0.1 * a)
      for (let x = A.x + 6; x < A.x + Lv * grow - 3; x += 7) ctx.fillRect(x, y + c / 2, 1, 1)
      const la = clamp(lit * 1.4 - i * 0.08)
      if (la > 0) {
        const x = A.x + logPos(seq[i].id) * Lv
        ctx.fillStyle = rgba(tokRGB(i), a * la); ctx.fillRect(x - 3, y + 2, 6, c - 4)
        ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
        ctx.fillStyle = rgba(C.bg, a * la); ctx.fillText('1', x, y + c / 2 + 0.5)
      }
    }
    ctx.strokeStyle = rgba(C.ink, 0.22 * a); ctx.strokeRect(A.x + 0.5, A.y + 0.5, Lv * grow - 1, N * c - 1)
    // log axis under the one-hot rows
    ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, a * grow)
    for (const v of [1, 10, 100, 1000, 50257]) ctx.fillText(v === 50257 ? '50,257' : v >= 1000 ? v / 1000 + 'k' : String(v), A.x + logPos(v) * Lv, A.y + N * c + 6)
    const w1 = mathName('onehot', A.x, A.y - 11, a, 18)
    ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
    ctx.fillText('5 × 50,257 · log-scaled ids', A.x + w1 + 8, A.y - 11)
  }
  /** W_E as a tall slab: 50,257 rows compressed on the same log axis, 8 columns. */
  function drawWE(a: number, bands: number[]) {
    const { B } = L
    mk.slab(B.x, B.y, B.w, B.h, a)
    ctx.fillStyle = rgba(C.ink, 0.035 * a); ctx.fillRect(B.x, B.y, B.w, B.h)
    ctx.strokeStyle = rgba(C.ink, 0.06 * a); ctx.lineWidth = 1
    for (let k = 1; k < 40; k++) { const y = B.y + (k / 40) * B.h; ctx.beginPath(); ctx.moveTo(B.x, y); ctx.lineTo(B.x + B.w, y); ctx.stroke() }
    for (let j = 1; j < D; j++) { const x = B.x + (j / D) * B.w; ctx.beginPath(); ctx.moveTo(x, B.y); ctx.lineTo(x, B.y + B.h); ctx.stroke() }
    ctx.strokeStyle = rgba(C.ink, 0.22 * a); ctx.strokeRect(B.x + 0.5, B.y + 0.5, B.w - 1, B.h - 1)
    ctx.fillStyle = rgba(C.ink, 0.12 * a); ctx.fillRect(B.x, B.y, B.w, logPos(256) * B.h)
    // id labels beside their bands, pushed apart where ids sit close on the log axis
    const labY: number[] = []
    seq.map((t, i) => ({ i, y: B.y + logPos(t.id) * B.h })).sort((p, q) => p.y - q.y).forEach(({ i, y }, n, all) => {
      labY[i] = n ? Math.max(y, labY[all[n - 1].i] + 13) : y
    })
    bands.forEach((b, i) => {
      if (b <= 0) return
      const y = B.y + logPos(seq[i].id) * B.h, ly = labY[i]
      ctx.fillStyle = rgba(tokRGB(i), a * b); ctx.fillRect(B.x - 2, y - 1.5, B.w + 4, 3)
      ctx.strokeStyle = rgba(tokRGB(i), 0.6 * a * b); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(B.x + B.w + 3, y); ctx.lineTo(B.x + B.w + 9, ly); ctx.stroke()
      ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(tokRGB(i), a * b)
      ctx.fillText(`${seq[i].id} ${tl(i)}`, B.x + B.w + 12, ly)
    })
    const w1 = mathName('W_E', B.x, B.y - 11, a, 18)
    ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
    ctx.fillText('50,257 × 8', B.x + w1 + 8, B.y - 11)
    ctx.fillStyle = rgba(C.mute, 0.55 * a); ctx.fillText('· 50,257 × 768', B.x + w1 + 8 + ctx.measureText('50,257 × 8').width + 6, B.y - 11)
  }
  /** A row pulled out of a weight slab, travelling to its slot in a result matrix. */
  function pulledRow(vals: number[], hue: number, x0: number, y0: number, h0: number, to: Rect, row: number, t: number, a: number) {
    const e = eio(t), c = lerp(L.B.w / D, to.c, e), y = lerp(y0 - h0 / 2, to.y + row * to.c, e), x = lerp(x0, to.x, e), hh = lerp(h0, to.c, e)
    const vmax = Math.max(...E.flat().map(Math.abs))
    vals.forEach((v, k) => {
      ctx.save(); ctx.translate(x + k * c, y); ctx.scale(1, hh / c)
      const fa = mk.paintCell(0, 0, c, v, vmax, C.tok[seq[hue].c % 7], 0, a)
      ctx.restore()
      if (hh > 16) mk.cellText(fmt(v), x + k * c, y, c, fa, a)
    })
  }

  /* ---------- scenes ---------- */
  function sceneOneHot(p: number) {
    const grow = eout(clamp(p / 0.35)), lit = clamp((p - 0.3) / 0.5)
    chips(L.A, eout(clamp(p / 0.15)), eout(clamp(p / 0.2)))
    drawOneHot(1, grow, lit)
    drawWE(eout(clamp((p - 0.4) / 0.3)), [])
    mk.drawMat({ r: L.C, vals: E, kind: 'row', alpha: eout(clamp((p - 0.5) / 0.3)), name: 'E', shape: '5 × 8', real: 'N × 768', label: 'bottom', reveal: () => 0 })
    mk.formula = { segs: [[`onehot(${seq[1].id})`, C.ink], ['  =  [0, 0, …, 0, ', C.mute], ['1', C.ink], [', 0, …, 0]', C.mute], [`   ·   1 at position ${seq[1].id} of 50,257`, C.ink2]], note: 'Each id becomes a row with a single 1. The axis is log-scaled so the small ids of common words are visible.' }
  }

  function sceneLookup(p: number) {
    const per = 0.9 / N
    chips(L.A, 1, 1)
    drawOneHot(1, 1, 1)
    const band = (i: number) => clamp((p - i * per) / (per * 0.35))
    drawWE(1, seq.map((_, i) => band(i)))
    const done = (i: number) => p >= (i + 1) * per
    mk.drawMat({ r: L.C, vals: E, kind: 'row', alpha: 1, name: 'E', shape: '5 × 8', real: 'N × 768', label: 'bottom', reveal: (i) => (done(i) ? 1 : 0) })
    mk.hit('E', L.C, N, D)
    const cur = Math.min(N - 1, Math.floor(p / per)), hv = mk.hovered('E')
    const i = hv && done(hv.i) ? hv.i : cur
    const lp = hv && done(hv.i) ? 1 : clamp((p - i * per) / per)
    // the one term that survives: lit cell → its W_E row
    const lx = L.A.x + logPos(seq[i].id) * L.Lv, ly = L.A.y + i * L.A.c, by = L.B.y + logPos(seq[i].id) * L.B.h
    ctx.setLineDash([2, 4]); ctx.strokeStyle = rgba(C.ink, 0.5); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(lx, ly - 3); ctx.quadraticCurveTo(lx, by, L.B.x - 4, by); ctx.stroke(); ctx.setLineDash([])
    ctx.strokeStyle = rgba(C.ink, 0.9); ctx.lineWidth = 1.5; ctx.strokeRect(L.A.x - 1, ly - 1, L.Lv + 2, L.A.c + 2)
    if (lp < 1 && !hv) {
      const t = clamp((lp - 0.3) / 0.6)
      if (t > 0) pulledRow(E[i], i, L.B.x, by, 3, L.C, i, t, 1)
    } else { ctx.strokeStyle = rgba(C.ink); ctx.lineWidth = 2; ctx.strokeRect(L.C.x - 1, L.C.y + i * L.C.c - 1, D * L.C.c + 2, L.C.c + 2) }
    mk.formula = {
      segs: [[`E[${i}]`, C.ink], ['  =  onehot(', C.mute], [String(seq[i].id), C.ink], [') · W_E  =  W_E[', C.mute], [String(seq[i].id), C.ink], [']  =  ', C.mute], [`[${E[i].map(fmt).join(', ')}]`, C.ink2]],
      note: `Only the 1 at ${seq[i].id} survives the multiply, so the product is just row ${seq[i].id}. Real code just indexes the row, skipping about 193M multiply-adds (≈386M FLOPs) for these 5 tokens.`,
    }
  }

  function scenePos(p: number) {
    const tr = eio(clamp(p / 0.2))
    if (tr < 1) { drawOneHot(1 - tr, 1, 1); drawWE(1 - tr, seq.map(() => 1)) }
    const er = lr(L.C, L.E2, tr)
    chips(er, 1, 1 - tr)
    mk.drawMat({ r: er, vals: E, kind: 'row', alpha: 1, name: 'E', shape: '5 × 8', real: 'N × 768', label: 'bottom' })
    // W_P slab with its first N rows highlighted
    const wa = eout(clamp((p - 0.15) / 0.2)), WP = L.WP
    if (wa > 0) {
      mk.slab(WP.x, WP.y, WP.w, WP.h, wa)
      ctx.fillStyle = rgba(C.ink, 0.035 * wa); ctx.fillRect(WP.x, WP.y, WP.w, WP.h)
      // all 1,024 rows as a faint texture: each column is a wave with its own frequency
      const cw = WP.w / D
      for (let yy = 0; yy < WP.h; yy += 2) {
        const row = posEmb(Math.floor((yy / WP.h) * CTX))
        row.forEach((v, q) => { ctx.fillStyle = rgba(v >= 0 ? C.ink : C.neg, (0.04 + 0.3 * Math.abs(v)) * wa); ctx.fillRect(WP.x + q * cw + 1, WP.y + yy, cw - 2, 2) })
      }
      ctx.strokeStyle = rgba(C.ink, 0.22 * wa); ctx.lineWidth = 1; ctx.strokeRect(WP.x + 0.5, WP.y + 0.5, WP.w - 1, WP.h - 1)
      const bh = Math.max(6, (N / CTX) * WP.h)
      ctx.fillStyle = rgba(C.ink, 0.5 * wa); ctx.fillRect(WP.x - 2, WP.y, WP.w + 4, bh)
      ctx.font = F.mono(10); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, wa)
      ctx.fillText(`rows 0–${N - 1}`, WP.x + WP.w + 10, WP.y + bh / 2)
      ctx.fillText(`${CTX - 1}`, WP.x + WP.w + 10, WP.y + WP.h - 4)
      const w1 = mathName('W_P', WP.x, WP.y - 11, wa, 18)
      ctx.font = F.mono(10.5); ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, wa); ctx.fillText('1,024 × 8', WP.x + w1 + 8, WP.y - 11)
      ctx.fillStyle = rgba(C.mute, 0.55 * wa); ctx.fillText('· 1,024 × 768', WP.x + w1 + 8 + ctx.measureText('1,024 × 8').width + 6, WP.y - 11)
    }
    const pp = clamp((p - 0.3) / 0.25)
    mk.drawMat({ r: L.P2, vals: P, kind: 'w', alpha: pp, name: 'P', shape: '5 × 8', real: 'N × 768', label: 'bottom', reveal: (i) => (pp * N > i ? 1 : 0) })
    const hp = clamp((p - 0.6) / 0.35)
    mk.drawMat({ r: L.H2, vals: Hm, kind: 'row', alpha: clamp(hp * 3), name: 'h', shape: '5 × 8', real: 'N × 768', label: 'bottom', reveal: (i, j) => (hp * N * D > i * D + j ? 1 : 0) })
    mk.hit('H', L.H2, N, D)
    // + and = signs
    ctx.font = serifAt(26); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    const my = L.E2.y + (N * L.E2.c) / 2
    ctx.fillStyle = rgba(C.ink2, pp); ctx.fillText('+', L.E2.x + 8.5 * L.E2.c, my)
    ctx.fillStyle = rgba(C.ink2, clamp(hp * 3)); ctx.fillText('=', L.P2.x + 8.5 * L.P2.c, my)
    const hv = mk.hovered('H')
    const f = hv ?? (hp > 0 && hp < 1 ? { i: Math.floor((hp * N * D) / D), j: Math.floor(hp * N * D) % D } : null)
    if (f) {
      const c = L.H2.c
      ctx.strokeStyle = rgba(C.ink); ctx.lineWidth = 2
      for (const r of [L.E2, L.P2, L.H2]) ctx.strokeRect(r.x + f.j * c, r.y + f.i * c, c, c)
      mk.formula = { segs: [[`h[${f.i},${f.j}]`, C.ink], ['  =  E + P  =  ', C.mute], [`${fmt(E[f.i][f.j])} + ${fmt(P[f.i][f.j])}`, C.ink2], ['  =  ', C.mute], [fmt(Hm[f.i][f.j]), C.ink]], note: `${tl(f.i)} in slot ${f.i}. ${orderNote()}` }
    } else mk.formula = { segs: [['P  =  W_P[0 : 5]', C.ink]], note: `Positions 0–4 take the first five rows of W_P (GPT-2 learns them; the toy rows follow waves). ${orderNote()}` }
  }

  function sceneStream(p: number) {
    const fade = 1 - eout(clamp(p / 0.25)), hr = lr(L.H2, L.E2, eio(clamp(p / 0.35))), c = hr.c
    chips(L.E2, fade, 0)
    mk.drawMat({ r: L.E2, vals: E, kind: 'row', alpha: fade, name: 'E', shape: '5 × 8' })
    mk.drawMat({ r: L.P2, vals: P, kind: 'w', alpha: fade, name: 'P', shape: '5 × 8' })
    mk.rowChips(hr, 1 - fade)
    mk.drawMat({ r: hr, vals: Hm, kind: 'row', alpha: 1, name: 'h', shape: '5 × 8', real: 'N × 768', label: 'bottom' })
    mk.hit('H', hr, N, D)
    const x0 = hr.x + D * c + 8, x1 = stage.W - pad, e = eio(clamp((p - 0.3) / 0.65))
    for (let i = 0; i < N; i++) {
      const y = L.H2.y + (i + 0.5) * c
      ctx.strokeStyle = rgba(tokRGB(i), 0.9); ctx.lineWidth = 1.8
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(lerp(x0, x1, eio(clamp(e * 1.2 - i * 0.05))), y); ctx.stroke()
    }
    ctx.font = F.small; ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, clamp(e * 2 - 1))
    fillRich('→ block 1 · ln_1', x1, hr.y - 10)
    ctx.font = F.label; spaced(true); ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, e)
    ctx.fillText('RESIDUAL STREAM', x0 + 16, hr.y - 10); spaced(false)
    mk.formula = { segs: [['h₀  =  W_E[ids] + W_P[0 : N]', C.ink]], note: 'Each row is now one lane of the residual stream. At the very end, the same W_E is reused, transposed, to score every vocabulary token.' }
  }

  function draw() {
    useCtx(ctx)
    stage.begin()
    mk.begin()
    const pl = prog('lookup'), pp = prog('pos'), ps = prog('stream')
    if (pl <= 0) sceneOneHot(prog('onehot'))
    else if (pp <= 0) sceneLookup(pl)
    else if (ps <= 0) scenePos(pp)
    else sceneStream(ps)
    mk.drawFormula()
  }

  const CAPS: Record<string, [string, string]> = {
    onehot: ['Each token id becomes a one-hot row: 50,257 zeros with a single 1 at the id. Stacked, the prompt is a 5 × 50,257 matrix.', 'toy [5 × 50,257]'],
    lookup: ['Multiplying the one-hot rows by W_E selects one row of W_E per token: a GEMM on paper, a table lookup in practice. Each row is a learned 768-number description of its token, and tokens used alike get similar rows: in GPT-2, Ġcat is closer to Ġdog (cosine 0.55) and Ġkitten (0.50) than to Ġon (0.21).', 'GPT-2 W_E [50,257 × 768] · 38.6M params'],
    pos: ['Attention on its own ignores order, so each slot adds its own row of the position matrix W_P. Slot i always gets row i. Try the other prompts: a repeated word, and the same words in two orders.', 'GPT-2 W_P [1,024 × 768] · context 1,024'],
    stream: ['The sum is the residual stream that enters block 1: one 768-wide lane per token, still unmixed.', 'h₀ [N × 768]'],
  }

  if (reduced && player.t === 0) player.t = player.start('lookup') + 3
  player.describe = (i) => CAPS[PHASES[i].id]
  const stop = runLoop((dt) => {
    player.tick(dt)
    draw()
    player.updateUI()
    const cur = player.cur(), [t, s] = CAPS[cur.id]
    frame.setCaption(cur.name, cur.short ?? cur.name, t, s)
  }, () => !player.playing)
  return () => { stop(); player.destroy(); stage.destroy() }
}
