import { createFrame } from '../../core/frame'
import { MatrixKit, fmt, gemm, type Rect } from '../../core/matrix'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, blend, rgba } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import { chipW, F, mathName, plate, serifAt, subLabel, useCtx } from '../../core/draw'
import type { Nav } from '../registry'
import { GELU_MIN, TOY, TOY_FF, attention, gelu, mlp, promptTokens, transpose } from './model'

/*
 * The MLP of block 1, opened up at toy scale (d_model 8, d_ff 32 — the same 4× as GPT-2's
 * 768 → 3,072). One layout serves every phase: X sits bottom-left, the weights sit above the
 * wide activation matrix, and the output Y lands back in X's slot. W_proj is drawn transposed
 * so each output dimension is a row directly above the activations it multiplies.
 */

const PHASES = [
  { id: 'up', name: 'Up-projection', short: 'X · W_fc', dur: 8.5 },
  { id: 'gelu', name: 'GELU', dur: 6.5 },
  { id: 'down', name: 'Down-projection', short: '· W_proj', dur: 8.5 },
  { id: 'resid', name: 'Residual add', short: '+ h', dur: 4.5 },
]

export function mountMlp(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const seq = promptTokens(), N = seq.length, att = attention(seq), R = mlp(seq, att)
  const WprojT = transpose(R.Wproj)

  const frame = createFrame(root, {
    eyebrow: 'Transformer · MLP',
    title: 'MLP',
    subtitle: 'feed-forward · block 1',
    back: { label: 'Forward pass', onClick: () => nav('transformer') },
    specs: [
      { label: 'shown', value: 'toy', real: 'GPT-2 small' },
      { label: 'tokens', value: String(N) },
      { label: 'd_model', value: String(TOY.d), real: '768' },
      { label: 'd_ff', value: String(TOY_FF), real: '3,072' },
      { label: 'params', value: String(TOY.d * TOY_FF * 2 + TOY_FF + TOY.d), real: '4.7M' },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 520, 'Step-by-step MLP: X is multiplied by W_fc and widened four times, GELU is applied to every cell, the result is projected back down by W_proj, and the output is added to the residual stream.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  const prog = (id: string) => player.prog(id)

  /* ---------- layout ---------- */
  const pad = 36, tokW = 76, lw = 84, top = 60, bot = 96
  const L = { lx: 0, lnX: 0, X: r0(), W: r0(), bfc: r0(), H: r0(), bproj: r0(), panel: { x0: 0, x1: 0, y0: 0, y1: 0 } }
  function r0(): Rect { return { x: 0, y: 0, c: 24 } }
  function geom() {
    const { W, H } = stage, exW = lw + 110, exH = 56 // 110: room for the b_fc label on the right
    const c = Math.floor(clamp(Math.min((W - 2 * pad - tokW - exW) / 41, (H - top - bot - exH) / 15), 12, 40))
    const lx = Math.max(pad, (W - (exW + tokW + 41 * c)) / 2), x0 = lx + lw + tokW
    const y0 = top + 24 + Math.max(0, (H - top - bot - (15 * c + exH)) / 2)
    Object.assign(L, {
      lx, lnX: lx + 42,
      X: { x: x0, y: y0 + 10 * c, c }, W: { x: x0 + 9 * c, y: y0, c },
      bfc: { x: x0 + 9 * c, y: y0 + 8.5 * c, c }, H: { x: x0 + 9 * c, y: y0 + 10 * c, c },
      bproj: { x: x0, y: y0 + 8.5 * c, c },
      panel: { x0: lx + 14, x1: x0 + 8 * c, y0: y0 + 6, y1: y0 + 7.6 * c },
    })
  }
  stage.onResize = geom
  geom()

  /* ---------- helpers ---------- */
  const mk = new MatrixKit(stage, seq, pad)
  const { tl } = mk
  // The residual stream after attention already carries mixed colours; the MLP never mixes positions.
  const laneCols = () => att.heads[0].A.map((row, i) => blend(seq.map((_, k) => mk.tokRGB(k)), row.map((w, k) => 0.55 * w + (k === i ? 0.45 : 0))))

  /** Bias vector as a one-row slab, labelled on the given side. */
  function drawBias(r: Rect, vals: number[], name: string, shape: string, real: string, a: number, side: 'left' | 'right') {
    if (a <= 0.01) return
    mk.drawMat({ r, vals: [vals], kind: 'w', alpha: a, name, shape, label: 'none', noText: true })
    ctx.font = serifAt(18)
    const [base, sub] = name.split('_')
    const bw = ctx.measureText(base).width
    ctx.font = serifAt(11)
    const w = bw + ctx.measureText(sub).width + 1
    const y = r.y + r.c * 0.72
    if (side === 'right') {
      const x = r.x + vals.length * r.c + 12, w1 = mathName(name, x, y, a, 18)
      ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(shape, x + w1 + 8, y)
      ctx.fillStyle = rgba(C.mute, 0.55 * a); ctx.fillText('· ' + real, x + w1 + 8 + ctx.measureText(shape).width + 6, y)
    } else mathName(name, r.x - 12 - w, y, a, 18)
  }

  /** Residual lanes entering from the left, through ln_2, into the chips. */
  function drawInput(a: number, lnAct: number) {
    const c = L.X.c, lc = laneCols()
    seq.forEach((t, i) => {
      const y = L.X.y + (i + 0.5) * c
      ctx.strokeStyle = rgba(lc[i], 0.85 * a); ctx.lineWidth = 1.6
      ctx.beginPath(); ctx.moveTo(L.lx, y); ctx.lineTo(L.lnX - 9, y); ctx.moveTo(L.lnX + 9, y); ctx.lineTo(L.X.x - 18 - chipW(t.text), y); ctx.stroke()
    })
    plate(L.lnX, L.X.y - 2, L.X.y + N * c + 2, lnAct, { w: 7, d: 8 })
    subLabel('ln_2', L.lnX, L.X.y - 16, false)
    ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, a); ctx.fillText('h′', L.lx, L.X.y - 16)
    mk.rowChips(L.X, a)
  }

  /** The GELU curve with each activated cell plotted on it. */
  function drawGeluPanel(a: number, rowP: (i: number) => number, focus: { i: number; k: number } | null) {
    if (a <= 0.01) return
    const P = L.panel, xmin = -4, xmax = 4.5, ymin = -0.8, ymax = 4.5
    const X = (v: number) => P.x0 + ((v - xmin) / (xmax - xmin)) * (P.x1 - P.x0)
    const Y = (v: number) => P.y1 - ((v - ymin) / (ymax - ymin)) * (P.y1 - P.y0)
    ctx.save(); ctx.beginPath(); ctx.rect(P.x0 - 2, P.y0 - 2, P.x1 - P.x0 + 4, P.y1 - P.y0 + 4); ctx.clip()
    // axes
    ctx.strokeStyle = rgba(C.ink, 0.22 * a); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(P.x0, Y(0)); ctx.lineTo(P.x1, Y(0)); ctx.moveTo(X(0), P.y0); ctx.lineTo(X(0), P.y1); ctx.stroke()
    // ReLU for reference
    ctx.setLineDash([2, 4]); ctx.strokeStyle = rgba(C.ink, 0.25 * a)
    ctx.beginPath(); ctx.moveTo(X(xmin), Y(0)); ctx.lineTo(X(0), Y(0)); ctx.lineTo(X(ymax), Y(ymax)); ctx.stroke()
    // floor at −0.17
    ctx.strokeStyle = rgba(C.ink, 0.3 * a)
    ctx.beginPath(); ctx.moveTo(X(-3.2), Y(GELU_MIN)); ctx.lineTo(X(0.6), Y(GELU_MIN)); ctx.stroke(); ctx.setLineDash([])
    // curve
    ctx.strokeStyle = rgba(C.ink, 0.9 * a); ctx.lineWidth = 1.6
    ctx.beginPath()
    for (let s = 0; s <= 120; s++) { const v = xmin + ((xmax - xmin) * s) / 120; if (s) ctx.lineTo(X(v), Y(gelu(v))); else ctx.moveTo(X(v), Y(gelu(v))) }
    ctx.stroke()
    // activations
    const lc = laneCols()
    for (let i = 0; i < N; i++) {
      const rp = rowP(i)
      if (rp <= 0) continue
      for (let k = 0; k < TOY_FF; k++) {
        const h = R.H[i][k]
        ctx.fillStyle = rgba(lc[i], a * rp * 0.85); ctx.beginPath(); ctx.arc(X(h), Y(gelu(h)), 2.3, 0, 7); ctx.fill()
      }
    }
    ctx.restore()
    if (focus) {
      const h = R.H[focus.i][focus.k], g = gelu(h), fx = X(h), fy = Y(g)
      ctx.setLineDash([2, 3]); ctx.strokeStyle = rgba(C.ink, 0.6 * a); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(fx, Y(0)); ctx.lineTo(fx, fy); ctx.lineTo(X(0), fy); ctx.stroke(); ctx.setLineDash([])
      ctx.fillStyle = rgba(lc[focus.i], a); ctx.beginPath(); ctx.arc(fx, fy, 4.5, 0, 7); ctx.fill()
      ctx.strokeStyle = rgba(C.ink, a); ctx.lineWidth = 1.5; ctx.stroke()
    }
    // labels
    ctx.font = serifAt(17); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
    ctx.fillStyle = rgba(C.ink, a); ctx.fillText('GELU(x) = x · Φ(x)', P.x0, P.y0 + 14)
    ctx.font = F.mono(9.5); ctx.fillStyle = rgba(C.mute, a); ctx.textBaseline = 'top'
    ctx.textAlign = 'center'
    for (const v of [-4, -2, 2, 4]) ctx.fillText(String(v).replace('-', '−'), X(v), Y(0) + 4)
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    ctx.fillText('4', X(0) - 5, Y(4)); ctx.fillText('2', X(0) - 5, Y(2))
    ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, 0.8 * a)
    ctx.fillText('min −0.17', X(0.7), Y(GELU_MIN) + 1)
    ctx.fillText('ReLU', X(3.3), Y(3.3) - 12)
  }

  /* ---------- scenes ---------- */
  function sceneUp(p: number) {
    const pin = eout(clamp(p / 0.06)), pw = eout(clamp((p - 0.02) / 0.07)), lc = laneCols()
    drawInput(pin, p > 0 && p < 0.08 ? 1 : 0)
    mk.drawMat({ r: L.X, vals: R.X, kind: 'row', alpha: pin, name: 'X', shape: '5 × 8', real: 'N × 768', rowCols: lc })
    mk.drawMat({ r: L.W, vals: R.Wfc, kind: 'w', alpha: pw, name: 'W_fc', shape: '8 × 32', real: '768 × 3072', noText: true })
    drawBias(L.bfc, R.bfc, 'b_fc', '32', '3072', pw, 'right')
    const g = gemm((p - 0.09) / 0.87, N, TOY_FF, TOY.d, 'slow')
    mk.drawMat({ r: L.H, vals: R.H, kind: 'row', alpha: pw, name: 'H', shape: '5 × 32', real: 'N × 3072', label: 'bottom', reveal: g.rev, rowCols: lc })
    mk.hit('H', L.H, N, TOY_FF)
    const f = mk.resolve({ H: { g, K: TOY.d } })
    if (f) mk.gemmOverlay({
      A: L.X, Av: R.X, B: L.W, Bv: R.Wfc, C: L.H, f, names: ['H', 'X', 'W_fc'], bias: R.bfc, biasR: L.bfc,
      note: `Neuron ${f.j} for ${tl(f.i)}: row ${f.i} of X dotted with column ${f.j} of W_fc, plus its bias.`,
    })
  }

  function sceneGelu(p: number) {
    const lc = laneCols(), c = L.H.c
    const rowP = (i: number) => clamp((p - 0.1 - i * 0.15) / 0.22)
    const vals = R.H.map((r, i) => r.map((v, k) => lerp(v, R.G[i][k], eio(rowP(i)))))
    drawInput(1, 0)
    mk.drawMat({ r: L.X, vals: R.X, kind: 'row', alpha: 1, name: 'X', shape: '5 × 8', real: 'N × 768', rowCols: lc })
    const dimW = 1 - 0.6 * eout(clamp(p / 0.1))
    mk.drawMat({ r: L.W, vals: R.Wfc, kind: 'w', alpha: dimW, name: 'W_fc', shape: '8 × 32', real: '768 × 3072', noText: true })
    drawBias(L.bfc, R.bfc, 'b_fc', '32', '3072', dimW, 'right')
    mk.drawMat({ r: L.H, vals, kind: 'row', alpha: 1, name: p > 0.55 ? 'G' : 'H', shape: '5 × 32', real: 'N × 3072', label: 'bottom', rowCols: lc })
    mk.hit('G', L.H, N, TOY_FF)
    const cur = [...Array(N).keys()].find((i) => rowP(i) > 0 && rowP(i) < 1)
    if (cur !== undefined) { ctx.strokeStyle = rgba(C.ink, 0.9); ctx.lineWidth = 1.5; ctx.strokeRect(L.H.x - 1, L.H.y + cur * c - 1, TOY_FF * c + 2, c + 2) }
    const hv = mk.hovered('G')
    const focus = hv && rowP(hv.i) > 0 ? { i: hv.i, k: hv.j } : null
    drawGeluPanel(eout(clamp(p / 0.08)), rowP, focus)
    if (focus) {
      const h = R.H[focus.i][focus.k], g = R.G[focus.i][focus.k]
      ctx.strokeStyle = rgba(C.ink); ctx.lineWidth = 2; ctx.strokeRect(L.H.x + focus.k * c, L.H.y + focus.i * c, c, c)
      mk.formula = {
        segs: [[`G[${focus.i},${focus.k}]`, C.ink], ['  =  GELU(', C.mute], [fmt(h), C.ink2], [')  =  ', C.mute], [fmt(g), C.ink]],
        note: h < 0 ? `Neuron ${focus.k} for ${tl(focus.i)}: a negative input, squeezed toward 0.` : `Neuron ${focus.k} for ${tl(focus.i)}: a positive input passes through almost unchanged.`,
      }
    } else {
      const neg = R.H.flat().filter((v) => v < 0).length
      mk.formula = {
        segs: [['GELU(x)  ≈  0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))', C.ink]],
        note: `GPT-2 uses this tanh approximation. ${neg} of ${N * TOY_FF} inputs here are negative; GELU pushes all of them into [−0.17, 0).`,
      }
    }
  }

  function sceneDown(p: number) {
    const tr = eio(clamp(p / 0.12)), lc = laneCols()
    if (tr < 1) {
      drawInput(1 - tr, 0)
      mk.drawMat({ r: L.X, vals: R.X, kind: 'row', alpha: 1 - tr, name: 'X', shape: '5 × 8', rowCols: lc })
      mk.drawMat({ r: L.W, vals: R.Wfc, kind: 'w', alpha: 0.4 * (1 - tr), name: 'W_fc', shape: '8 × 32', noText: true })
      drawBias(L.bfc, R.bfc, 'b_fc', '32', '3072', 0.4 * (1 - tr), 'right')
      drawGeluPanel(1 - tr, () => 1, null)
    }
    mk.rowChips(L.X, tr)
    mk.drawMat({ r: L.W, vals: WprojT, kind: 'w', alpha: tr, name: 'W_projᵀ', shape: '8 × 32', real: '768 × 3072', noText: true })
    drawBias(L.bproj, R.bproj, 'b_proj', '8', '768', tr, 'left')
    mk.drawMat({ r: L.H, vals: R.G, kind: 'row', alpha: 1, name: 'G', shape: '5 × 32', real: 'N × 3072', label: 'bottom', rowCols: lc })
    const g = gemm((p - 0.14) / 0.84, N, TOY.d, TOY_FF, 'slow')
    mk.drawMat({ r: L.X, vals: R.Y, kind: 'row', alpha: tr, name: 'Y', shape: '5 × 8', real: 'N × 768', label: 'bottom', reveal: g.rev, rowCols: lc })
    mk.hit('Y', L.X, N, TOY.d)
    const f = mk.resolve({ Y: { g, K: TOY_FF } })
    if (f) mk.gemmOverlay({
      A: L.H, Av: R.G, B: L.W, Bv: WprojT, bT: true, C: L.X, f, names: ['Y', 'G', 'W_proj'], bias: R.bproj, biasR: L.bproj,
      note: `Output dim ${f.j} for ${tl(f.i)}: all 32 of its activations, weighted by row ${f.j} of W_projᵀ, plus the bias.`,
    })
  }

  function sceneResid(p: number) {
    const tr = eio(clamp(p / 0.25)), lc = laneCols(), c = L.X.c
    mk.rowChips(L.X, 1)
    mk.drawMat({ r: L.W, vals: WprojT, kind: 'w', alpha: 1 - tr, name: 'W_projᵀ', shape: '8 × 32', noText: true })
    drawBias(L.bproj, R.bproj, 'b_proj', '8', '768', 1 - tr, 'left')
    mk.drawMat({ r: L.H, vals: R.G, kind: 'row', alpha: 1 - tr, name: 'G', shape: '5 × 32', label: 'bottom', rowCols: lc })
    mk.drawMat({ r: L.X, vals: R.Y, kind: 'row', alpha: 1, name: 'Y', shape: '5 × 8', real: 'N × 768', label: 'bottom', rowCols: lc })
    mk.hit('Y', L.X, N, TOY.d)
    const pr = clamp((p - 0.15) / 0.6)
    if (pr > 0) {
      const xp = L.X.x + TOY.d * c + 24, xEnd = stage.W - pad
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, pr)
      ctx.fillText('+ h′', xp, L.X.y - 10)
      for (let i = 0; i < N; i++) {
        const y = L.X.y + (i + 0.5) * c, e = eio(clamp(pr * 1.3 - i * 0.06))
        ctx.strokeStyle = rgba(C.ink, 0.35 * pr); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(L.X.x + TOY.d * c + 6, y); ctx.lineTo(xp - 7, y); ctx.stroke()
        ctx.beginPath(); ctx.arc(xp, y, 7, 0, 7); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(xp - 3.5, y); ctx.lineTo(xp + 3.5, y); ctx.moveTo(xp, y - 3.5); ctx.lineTo(xp, y + 3.5); ctx.stroke()
        ctx.strokeStyle = rgba(lc[i], 0.9 * e); ctx.lineWidth = 1.8
        ctx.beginPath(); ctx.moveTo(xp + 8, y); ctx.lineTo(lerp(xp + 8, xEnd, e), y); ctx.stroke()
      }
      ctx.font = F.small; ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.mute, clamp(pr * 2 - 1))
      ctx.fillText('→ block 2', xEnd, L.X.y - 10)
    }
    const hv = mk.hovered('Y')
    mk.formula = hv
      ? { segs: [[`h″[${hv.i},${hv.j}]`, C.ink], ['  =  h′ + Y  =  ', C.mute], [`${fmt(R.h[hv.i][hv.j])} + ${fmt(R.Y[hv.i][hv.j])}`, C.ink2], ['  =  ', C.mute], [fmt(R.out[hv.i][hv.j]), C.ink]], note: `${tl(hv.i)}, dim ${hv.j}: the MLP's contribution is added to what was already in the stream.` }
      : { segs: [['h″  =  h′ + MLP(ln_2(h′))', C.ink]], note: 'Each token went through the MLP on its own; the colors do not mix here. The stream flows on to block 2.' }
  }

  /* ---------- frame ---------- */
  function draw() {
    useCtx(ctx)
    stage.begin()
    mk.begin()
    const pg = prog('gelu'), pd = prog('down'), pr = prog('resid')
    if (pg <= 0) sceneUp(prog('up'))
    else if (pd <= 0) sceneGelu(pg)
    else if (pr <= 0) sceneDown(pd)
    else sceneResid(pr)
    mk.drawFormula()
  }

  const CAPS: Record<string, [string, string]> = {
    up: ["X (after ln_2) is multiplied by W_fc, widening every token from 8 numbers to 32 (768 → 3,072 in GPT-2), and the bias b_fc is added. Each column is one neuron; each cell is a row of X dotted with that neuron's weights.", 'GPT-2 [N×768]·[768×3072] · 23.6 MFLOPs'],
    gelu: ['GELU is applied to every cell on its own. Negative inputs are squeezed toward 0, never below −0.17; positive inputs pass through almost unchanged. So only some neurons stay active for each token.', 'elementwise · 5 × 32 cells'],
    down: ['The 32 activations are projected back down to 8 (3,072 → 768), and the bias b_proj is added. W_proj is drawn transposed, so each output dimension is a row sitting directly above the activations it multiplies.', 'GPT-2 [N×3072]·[3072×768] · 23.6 MFLOPs'],
    resid: ['The MLP output is added to the residual stream. Positions never exchanged information in this sub-layer; each token was processed on its own. Next is block 2.', "h″ = h′ + MLP(ln_2(h′))"],
  }

  if (reduced) player.t = player.start('gelu') + 4
  const stop = runLoop((dt) => {
    player.tick(dt)
    draw()
    player.updateUI()
    const cur = player.cur(), [t, s] = CAPS[cur.id]
    frame.setCaption(cur.name, cur.short ?? cur.name, t, s)
  })
  return () => { stop(); player.destroy(); stage.destroy() }
}
