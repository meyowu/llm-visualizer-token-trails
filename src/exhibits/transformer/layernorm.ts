import { F, fillRich, mathName, plate, rr, subLabel, useCtx } from '../../core/draw'
import { createFrame } from '../../core/frame'
import { MatrixKit, fmt, type Rect } from '../../core/matrix'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, blend, pop, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import type { Nav } from '../registry'
import { LN, LN_EPS, TOY, attention, embedRow, lnStats, promptTokens } from './model'

/*
 * Pre-LN residual stream: the stream runs straight through each block, and every sub-layer reads a
 * normalised copy (ln_1, ln_2) and adds its result back. Then ln_1 of block 1 is opened up on the
 * real toy numbers: each token's 8 features as dots on a number line, centred, scaled, then γ, β.
 */

const PHASES = [
  { id: 'stream', name: 'Residual stream', short: 'Stream', dur: 6.5 },
  { id: 'mean', name: 'Subtract the mean', short: '− μ', dur: 4 },
  { id: 'scale', name: 'Divide by the std', short: '÷ σ', dur: 4 },
  { id: 'affine', name: 'Scale and shift', short: 'γ · x̂ + β', dur: 5.5 },
]
const XR = 4 // number-line range ±4

export function mountLayerNorm(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const seq = promptTokens(), N = seq.length, D = TOY.d
  const h = seq.map((t, i) => embedRow(t, i)), st = h.map((r) => lnStats(r, LN.ln1)), X = st.map((s) => s.out)
  const att = attention(seq)

  const frame = createFrame(root, {
    eyebrow: 'Anatomy · LayerNorm & Residual',
    title: 'LayerNorm & Residual',
    subtitle: 'pre-LN · ln_1 of block 1',
    back: { label: 'Forward pass', onClick: () => nav('anatomy') },
    specs: [
      { label: 'shown', value: 'toy', real: 'GPT-2 small' },
      { label: 'features', value: String(D), real: '768' },
      { label: 'LayerNorms', value: '25', real: '2 per block + ln_f' },
      { label: 'params each', value: String(2 * D), real: '1,536' },
      { label: 'ε', value: '1e-5' },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 520, 'LayerNorm and the residual stream: the stream flows straight through a block while each sub-layer reads a normalised copy and adds its output back; LayerNorm centres each token vector, scales it to unit variance, then applies a learned scale and shift.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  const prog = (id: string) => player.prog(id)
  const mk = new MatrixKit(stage, seq, 36)
  const { tokRGB, tl } = mk

  /* ---------- layout ---------- */
  const pad = 36, tokW = 76, top = 60, bot = 96
  const L = { H: r0(), X: r0(), nx0: 0, nx1: 0, statX: 0, gb: r0(), sy: 0, sx0: 0, sx1: 0, ln1: 0, attn: 0, add1: 0, ln2: 0, mlp: 0, add2: 0, by: 0 }
  function r0(): Rect { return { x: 0, y: 0, c: 32 } }
  function geom() {
    const { W, H } = stage, avail = H - top - bot
    const c = Math.floor(clamp(Math.min((avail - 90) / (N + 2.5), (W - 2 * pad - tokW - 460) / (2 * D)), 22, 40))
    const nl = clamp(W - 2 * pad - tokW - 2 * D * c - 170, 260, 420)
    const total = tokW + 2 * D * c + nl + 170, x0 = Math.max(pad + tokW, (W - total) / 2 + tokW)
    const y0 = top + 2.4 * c + Math.max(0, (avail - (N + 2.4) * c - 60) / 2)
    L.H = { x: x0, y: y0, c }
    L.nx0 = x0 + D * c + 40; L.nx1 = L.nx0 + nl; L.statX = L.nx1 + 18
    L.X = { x: L.statX + 110, y: y0, c }
    L.gb = { x: L.X.x, y: y0 - 2.3 * c, c }
    // stream schematic
    L.sx0 = pad + 40; L.sx1 = W - pad - 60; L.sy = top + avail * 0.32; L.by = L.sy + Math.min(150, avail * 0.42)
    const w = L.sx1 - L.sx0
    L.ln1 = L.sx0 + w * 0.2; L.attn = L.sx0 + w * 0.32; L.add1 = L.sx0 + w * 0.46
    L.ln2 = L.sx0 + w * 0.6; L.mlp = L.sx0 + w * 0.72; L.add2 = L.sx0 + w * 0.86
  }
  stage.onResize = geom
  geom()

  const nxPos = (v: number) => L.nx0 + ((clamp(v, -XR, XR) + XR) / (2 * XR)) * (L.nx1 - L.nx0)
  const hues = seq.map((_, i) => tokRGB(i))
  const afterAttn = (): RGB[] => att.heads[0].A.map((row, i) => blend(hues, row.map((w, k) => 0.55 * w + (k === i ? 0.45 : 0))))

  /* ---------- scene 1: the residual stream through one block ---------- */
  let hoverPlate = ''
  function sceneStream(p: number) {
    const lanes = (i: number) => L.sy + (i - (N - 1) / 2) * 12
    const blanes = (i: number) => L.by + (i - (N - 1) / 2) * 8
    const mixed = afterAttn(), t = player.t
    const grow = eio(clamp(p / 0.35))
    // main stream: straight through, colour changes only where attention adds in
    for (let i = 0; i < N; i++) {
      const y = lanes(i), x1 = lerp(L.sx0, L.sx1, grow)
      const g = ctx.createLinearGradient(L.add1 - 10, 0, L.add1 + 10, 0)
      g.addColorStop(0, rgba(hues[i], 0.9)); g.addColorStop(1, rgba(mixed[i], 0.9))
      ctx.strokeStyle = g; ctx.lineWidth = 1.8
      ctx.beginPath(); ctx.moveTo(L.sx0, y); ctx.lineTo(x1, y); ctx.stroke()
    }
    // two branches: copy down, normalise, sub-layer, add back
    const branch = (xs: number, xa: number, cols: RGB[], ba: number) => {
      if (ba <= 0) return
      for (let i = 0; i < N; i++) {
        const y0 = lanes(i), y1 = blanes(i)
        ctx.strokeStyle = rgba(cols[i], 0.55 * ba); ctx.lineWidth = 1.1
        ctx.beginPath(); ctx.moveTo(xs, y0)
        ctx.bezierCurveTo(xs + 40, y0, xs + 20, y1, xs + 70, y1)
        ctx.lineTo(xa - 70, y1)
        ctx.bezierCurveTo(xa - 20, y1, xa - 30, y0, xa - 9, y0)
        ctx.stroke()
      }
    }
    const b1 = eout(clamp((p - 0.25) / 0.3)), b2 = eout(clamp((p - 0.45) / 0.3))
    branch(L.ln1 - 70, L.add1, hues, b1)
    branch(L.ln2 - 70, L.add2, mixed, b2)
    const py0 = L.by - 30, py1 = L.by + 30
    const pa = (x: number, b: number, lab: string, key: string) => {
      if (b <= 0) return
      const hv = hoverPlate === key
      plate(x, py0, py1, hv ? 0.9 : 0.35 * b, { w: 8, d: 9 })
      subLabel(hv ? lab + ' ↗' : lab, x, py1 + 26, hv)
    }
    pa(L.ln1, b1, 'ln_1', ''); pa(L.attn, b1, 'attn', 'attn'); pa(L.ln2, b2, 'ln_2', ''); pa(L.mlp, b2, 'mlp', 'mlp')
    for (const [x, b] of [[L.add1, b1], [L.add2, b2]] as const) {
      if (b <= 0) continue
      ctx.strokeStyle = rgba(C.ink, 0.8 * b); ctx.lineWidth = 1.2
      ctx.beginPath(); ctx.arc(x, L.sy, 10, 0, 7); ctx.fillStyle = rgba(C.bg, b); ctx.fill(); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(x - 5, L.sy); ctx.lineTo(x + 5, L.sy); ctx.moveTo(x, L.sy - 5); ctx.lineTo(x, L.sy + 5); ctx.stroke()
    }
    // pulses along the main stream
    if (!reduced) for (let k = 0; k < 6; k++) {
      const x = L.sx0 + ((t * 90 + k * 170) % (L.sx1 - L.sx0))
      if (x > lerp(L.sx0, L.sx1, grow)) continue
      for (let i = 0; i < N; i++) { ctx.fillStyle = rgba(pop(x > L.add1 ? mixed[i] : hues[i]), 0.9); ctx.beginPath(); ctx.arc(x, lanes(i), 1.6, 0, 7); ctx.fill() }
    }
    // stream labels
    const la = clamp((p - 0.6) / 0.25)
    const lab = (x: number, name: string, eq: string) => {
      mathName(name, x, L.sy - 44, la, 20)
      ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, la); fillRich(eq, x, L.sy - 26)
    }
    lab(L.sx0, 'h', 'from the embedding')
    lab(L.add1 + 22, 'h′', '= h + attn(ln_1(h))')
    lab(L.add2 + 22, 'h″', '= h′ + mlp(ln_2(h′))')
    ctx.font = F.small; ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.mute, la); ctx.fillText('→ block 2', L.sx1, L.sy + 40)
    if (hoverPlate) {
      const x = hoverPlate === 'attn' ? L.attn : L.mlp, t2 = 'Open details ↗'
      ctx.font = F.body; const w = ctx.measureText(t2).width + 16
      rr(x - w / 2, py0 - 40, w, 20, 10); ctx.fillStyle = rgba(C.ink, 0.92); ctx.fill()
      ctx.fillStyle = rgba(C.bg); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(t2, x, py0 - 29.5)
    }
    mk.formula = { segs: [['h″  =  h + attn(ln_1(h)) + mlp(ln_2(h′))', C.ink]], note: 'Pre-LN: sub-layers read a normalised copy and write back by addition, so the stream itself is never normalised or overwritten inside a block.' }
  }

  /* ---------- scenes 2–4: ln_1 on the real numbers ---------- */
  function numberLines(pos: (i: number, k: number) => number, band: ((i: number) => [number, number]) | null, a: number, marks: ((i: number) => number) | null) {
    const c = L.H.c, hv = mk.hovered('H', 'X')
    // axis ticks
    ctx.font = F.mono(9.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, a)
    for (const v of [-4, -2, -1, 0, 1, 2, 4]) ctx.fillText(String(v).replace('-', '−'), nxPos(v), L.H.y + N * c + 6)
    ctx.strokeStyle = rgba(C.ink, 0.2 * a); ctx.setLineDash([2, 3]); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(nxPos(0), L.H.y - 4); ctx.lineTo(nxPos(0), L.H.y + N * c + 2); ctx.stroke(); ctx.setLineDash([])
    for (let i = 0; i < N; i++) {
      const y = L.H.y + (i + 0.5) * c
      ctx.strokeStyle = rgba(C.ink, 0.18 * a); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(L.nx0, y); ctx.lineTo(L.nx1, y); ctx.stroke()
      if (band) {
        const [b0, b1] = band(i)
        ctx.fillStyle = rgba(tokRGB(i), 0.12 * a); ctx.fillRect(nxPos(b0), y - c * 0.32, nxPos(b1) - nxPos(b0), c * 0.64)
      }
      if (marks) {
        const m = nxPos(marks(i))
        ctx.strokeStyle = rgba(C.ink, 0.9 * a); ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.moveTo(m, y - c * 0.36); ctx.lineTo(m, y + c * 0.36); ctx.stroke()
      }
      for (let k = 0; k < D; k++) {
        const x = nxPos(pos(i, k)), on = hv && hv.j === k && hv.i === i
        ctx.fillStyle = rgba(tokRGB(i), 0.9 * a); ctx.beginPath(); ctx.arc(x, y, on ? 5 : 3.4, 0, 7); ctx.fill()
        if (on) { ctx.strokeStyle = rgba(C.ink, a); ctx.lineWidth = 1.5; ctx.stroke() }
      }
    }
  }
  function stats(showMu: number, showSig: number) {
    const c = L.H.c
    ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
    for (let i = 0; i < N; i++) {
      const y = L.H.y + (i + 0.5) * c
      if (showMu > 0) { ctx.fillStyle = rgba(C.ink2, showMu); ctx.fillText(`μ ${fmt(st[i].mu)}`, L.statX, y - (showSig > 0 ? 7 : 0)) }
      if (showSig > 0) { ctx.fillStyle = rgba(C.ink2, showSig); ctx.fillText(`σ ${fmt(st[i].sigma)}`, L.statX, y + 7) }
    }
  }
  function leftMatrix(a = 1) {
    mk.rowChips(L.H, a)
    mk.drawMat({ r: L.H, vals: h, kind: 'row', alpha: a, name: 'h', shape: '5 × 8', real: 'N × 768' })
    mk.hit('H', L.H, N, D)
  }

  function sceneMean(p: number) {
    const fin = eout(clamp(p / 0.15)), e = eio(clamp((p - 0.3) / 0.55))
    leftMatrix(fin)
    numberLines((i, k) => h[i][k] - e * st[i].mu, null, fin, (i) => st[i].mu * (1 - e))
    stats(clamp((p - 0.1) / 0.2), 0)
    const hv = mk.hovered('H'), i = hv?.i ?? 0
    mk.formula = { segs: [[`μ[${i}]`, C.ink], ['  =  mean(h[', C.mute], [String(i), C.ink], [', :])  =  ', C.mute], [`(${h[i].map(fmt).join(' + ')}) / 8`, C.ink2], ['  =  ', C.mute], [fmt(st[i].mu), C.ink]], note: `Each token is normalised over its own ${D} features (768 in GPT-2). Tokens never mix here.` }
  }

  function sceneScale(p: number) {
    const e = eio(clamp((p - 0.3) / 0.55)), sig = (i: number) => lerp(1, st[i].sigma, e)
    leftMatrix()
    numberLines((i, k) => (h[i][k] - st[i].mu) / sig(i), (i) => [-st[i].sigma / sig(i), st[i].sigma / sig(i)], 1, null)
    stats(1, clamp(p / 0.25))
    const hv = mk.hovered('H'), i = hv?.i ?? 0
    mk.formula = { segs: [[`σ[${i}]`, C.ink], ['  =  √( mean((h − μ)²) + ε )  =  ', C.mute], [fmt(st[i].sigma), C.ink], ['     x̂  =  (h − μ) / σ', C.ink2]], note: `The shaded band is ±σ. Dividing by σ makes every token's features spread the same amount (ε = ${LN_EPS} guards against σ = 0).` }
  }

  function sceneAffine(p: number) {
    const e = eio(clamp((p - 0.25) / 0.45)), fill = clamp((p - 0.45) / 0.45)
    leftMatrix()
    numberLines((i, k) => lerp(st[i].xhat[k], X[i][k], e), () => [-1, 1], 1, null)
    stats(1, 1)
    const ga = eout(clamp(p / 0.2)), c = L.X.c
    mk.drawMat({ r: L.gb, vals: [LN.ln1.gamma], kind: 'w', alpha: ga, name: 'γ', shape: '8', real: '768' })
    mk.drawMat({ r: { x: L.gb.x, y: L.gb.y + c + 4, c }, vals: [LN.ln1.beta], kind: 'w', alpha: ga, name: 'β', shape: '8', label: 'none' })
    ctx.save(); ctx.globalAlpha = ga
    mathName('β', L.gb.x - 22, L.gb.y + 1.7 * c + 4, 1, 18)
    ctx.restore()
    mk.drawMat({ r: L.X, vals: X, kind: 'row', alpha: 1, name: 'X', shape: '5 × 8', real: 'N × 768', label: 'bottom', reveal: (i, j) => (fill * N * D > i * D + j ? 1 : 0) })
    mk.hit('X', L.X, N, D)
    const hv = mk.hovered('X', 'H')
    if (hv) {
      const { i, j } = hv, s = st[i]
      for (const r of [L.H, L.X]) { ctx.strokeStyle = rgba(C.ink); ctx.lineWidth = 2; ctx.strokeRect(r.x + j * c, r.y + i * c, c, c) }
      ctx.strokeStyle = rgba(C.ink, 0.9); ctx.lineWidth = 1.5
      ctx.strokeRect(L.gb.x + j * c - 1, L.gb.y - 1, c + 2, c + 2); ctx.strokeRect(L.gb.x + j * c - 1, L.gb.y + c + 3, c + 2, c + 2)
      mk.formula = { segs: [[`X[${i},${j}]`, C.ink], ['  =  γ[j] · (h − μ) / σ + β[j]  =  ', C.mute], [`${fmt(LN.ln1.gamma[j])} · (${fmt(h[i][j])} − ${fmt(s.mu)}) / ${fmt(s.sigma)} + ${fmt(LN.ln1.beta[j])}`, C.ink2], ['  =  ', C.mute], [fmt(X[i][j]), C.ink]], note: `${tl(i)}, feature ${j}. This X is exactly what attention multiplies by W_Q, W_K and W_V.` }
    } else mk.formula = { segs: [['X  =  γ ⊙ x̂ + β', C.ink]], note: 'γ and β are learned per feature, so the model can undo the normalisation where it helps. ln_2 and ln_f work the same way with their own γ and β.' }
  }

  function draw() {
    useCtx(ctx)
    stage.begin()
    mk.begin()
    const pm = prog('mean'), ps = prog('scale'), pa = prog('affine')
    if (pm <= 0) sceneStream(prog('stream'))
    else if (ps <= 0) sceneMean(pm)
    else if (pa <= 0) sceneScale(ps)
    else sceneAffine(pa)
    mk.drawFormula()
  }

  // attn and mlp plates in the stream schematic open their detail views
  const plateAt = (x: number, y: number) => {
    if (prog('mean') > 0 || Math.abs(y - L.by) > 44) return ''
    if (Math.abs(x - L.attn) < 14) return 'attn'
    if (Math.abs(x - L.mlp) < 14) return 'mlp'
    return ''
  }
  stage.canvas.addEventListener('pointermove', (e) => {
    const [x, y] = stage.local(e)
    hoverPlate = plateAt(x, y)
    if (hoverPlate) stage.canvas.style.cursor = 'pointer'
  })
  stage.canvas.addEventListener('pointerleave', () => { hoverPlate = '' })
  stage.canvas.addEventListener('click', (e) => {
    const [x, y] = stage.local(e), pl = plateAt(x, y)
    if (pl) nav(pl === 'attn' ? 'anatomy/attention' : 'anatomy/mlp', { x: e.clientX, y: e.clientY })
  })

  const CAPS: Record<string, [string, string]> = {
    stream: ['The residual stream runs straight through every block. Each sub-layer reads a normalised copy of it and adds its result back, so information is only ever added. Click attn or mlp to open them.', 'GPT-2: pre-LN, 12 blocks'],
    mean: ['LayerNorm works on one token at a time, over its features. First the mean of the token\'s features is subtracted, so they centre on 0.', 'μ over 768 features'],
    scale: ['Then everything is divided by the standard deviation, so every token\'s features have the same spread no matter how large the stream has grown.', 'σ = √(var + ε)'],
    affine: ['Finally each feature is scaled by γ and shifted by β, both learned. The result X is what the attention layer receives.', 'X = γ ⊙ x̂ + β'],
  }

  if (reduced) player.t = player.start('scale') + 3
  const stop = runLoop((dt) => {
    player.tick(dt)
    draw()
    player.updateUI()
    const cur = player.cur(), [t, s] = CAPS[cur.id]
    frame.setCaption(cur.name, cur.short ?? cur.name, t, s)
  })
  return () => { stop(); player.destroy(); stage.destroy() }
}
