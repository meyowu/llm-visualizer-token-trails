import { chipW, drawChip, F, plate, rr, serifAt, spaced, subLabel, tokDisp, tokLabel, tokText, useCtx } from '../../core/draw'
import { createFrame, toggle } from '../../core/frame'
import { MatrixKit, fmt, gemm, type M, type Rect } from '../../core/matrix'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, blend, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import { VOCAB, mixing, nextDist, presets, type Next } from '../../lib/gpt2/data'
import type { Nav } from '../registry'

/*
 * The last position's vector goes through ln_f and is dotted with every row of W_E (GPT-2 ties
 * the LM head to the embedding), with the numbers of a real GPT-2 small run. The GEMM draws 8 of
 * the 768 dimensions for the six most likely tokens; each logit is the full 768-term sum. Then
 * temperature and softmax over the whole vocabulary, and a sampling strategy.
 */

const PHASES = [
  { id: 'lnf', name: 'Final LayerNorm', short: 'ln_f', dur: 3.5 },
  { id: 'logits', name: 'Logits', short: 'x · W_Eᵀ', dur: 7.5 },
  { id: 'temp', name: 'Temperature', short: '÷ T', dur: 4 },
  { id: 'softmax', name: 'Softmax', dur: 4 },
  { id: 'sample', name: 'Sample', dur: 5.5 },
]
/** Candidates drawn in the GEMM and as bars; dimensions of x drawn; tokens whose logits are stored exactly. */
const K6 = 6, D = 8, TOPN = 256
const STRATS = ['sample', 'greedy', 'top-k 3', 'top-p 0.9'] as const
const PROMPT_LABELS: Record<string, string> = { cat: '“The cat sat…”', france: '“capital of France”', count: '“one two three…”' }
/** The logit chart shows gaps to the top logit down to this value. */
const GAP_MIN = -12

interface Tok { text: string; id: number; c: number }
interface Run {
  seq: Tok[]
  h: Float32Array
  xf: Float32Array
  mu: number
  sigma: number
  cands: { text: string; id: number; z: number }[]
  xs: M
  WT: M
  Z: M
  next: Next
  lanes: RGB[]
}

function load(k: number, seq: Tok[]): Run {
  const pass = presets()[k].passes[0], u = pass.unembed!
  seq.length = 0
  pass.texts.forEach((text, i) => seq.push({ text, id: pass.ids[i], c: i % 7 }))
  const dot = (a: Float32Array, b: Float32Array, from = 0) => { let s = 0; for (let t = from; t < a.length; t++) s += a[t] * b[t]; return s }
  const mu = u.h.reduce((s, v) => s + v, 0) / u.h.length
  const sigma = Math.sqrt(u.h.reduce((s, v) => s + (v - mu) ** 2, 0) / u.h.length + 1e-5)
  const cands = u.rows.map((r) => ({ text: r.text, id: r.id, z: dot(u.xf, r.v) }))
  const mix = mixing(pass.att)[11]
  const hues = seq.map((t) => C.tok[t.c])
  return {
    seq, h: u.h, xf: u.xf, mu, sigma, cands,
    xs: [Array.from(u.xf.subarray(0, D))],
    WT: Array.from({ length: D }, (_, d) => u.rows.map((r) => r.v[d])),
    Z: [cands.map((c) => c.z)],
    next: pass.next,
    lanes: mix.map((row) => blend(hues, row)),
  }
}
/** More decimals than fmt(): GPT-2's logits sit near −80 and differ in the first decimal. */
const fz = (v: number) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2)
const pctS = (p: number) => (p < 0.001 ? '<0.1%' : (p * 100).toFixed(p < 0.1 ? 1 : 0) + '%')

export function mountUnembed(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const seq: Tok[] = []
  const S = { T: 0.8, strat: 0, u: 0.22, preset: 0 }
  let R = load(S.preset, seq)

  const frame = createFrame(root, {
    eyebrow: 'Anatomy · Unembed & Sampling',
    title: 'Unembed & Sampling',
    subtitle: 'LM head · tied to W_E · real run',
    back: { label: 'Forward pass', onClick: () => nav('anatomy') },
    specs: [
      { label: 'shown', value: 'real GPT-2' },
      { label: 'd_model', value: '768', real: `${D} drawn` },
      { label: 'vocab', value: '50,257' },
      { label: 'LM head', value: 'W_Eᵀ', real: 'tied' },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 520, 'Unembedding and sampling with numbers from a real GPT-2 small run: the last position passes through ln_f, is scored against every vocabulary embedding to give logits, which are divided by the temperature, turned into probabilities by softmax, and sampled.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  const prog = (id: string) => player.prog(id)
  const mk = new MatrixKit(stage, seq, 36)
  const nextCol = () => C.tok[seq.length % 7]
  /** A new draw is shown from the start of the Sample phase, so a changed setting never silently swaps the token. */
  const redraw = () => { if (player.t > player.start('sample')) player.t = player.start('sample') + 0.001 }

  // controls: prompt, strategy, resample, temperature
  const temp = document.createElement('label')
  temp.className = 'temp'; temp.htmlFor = 'un-temp'
  temp.innerHTML = '<small>temperature</small><span>T</span><input id="un-temp" type="range" min="0.2" max="2" step="0.05" value="0.8"><output>0.80</output>'
  player.meta.append(temp)
  const tIn = temp.querySelector('input')!, tOut = temp.querySelector('output')!
  tIn.addEventListener('input', () => { S.T = +tIn.value; tOut.textContent = S.T.toFixed(2); redraw() })
  const resample = document.createElement('button')
  resample.className = 'resample'; resample.textContent = 'Resample'
  resample.addEventListener('click', () => { S.u = Math.random(); if (player.t < player.start('sample')) player.t = player.start('sample') + 0.001; else redraw() })
  player.meta.prepend(resample)
  toggle(player.meta, 'Sampling strategy', [...STRATS], 0, (i) => { S.strat = i; redraw() })
  toggle(player.meta, 'Prompt', presets().map((p) => PROMPT_LABELS[p.key] ?? p.text), 0, (i) => { S.preset = i; R = load(i, seq) })

  /* ---------- the distribution under the current strategy ---------- */
  function sampling(T: number) {
    const d = nextDist(R.next, T, TOPN), st = STRATS[S.strat], p = d.rows.map((r) => r.p)
    let keep = p.slice(), tail = d.rest
    if (st === 'greedy') { keep = p.map((_, k) => (k === 0 ? 1 : 0)); tail = 0 }
    else if (st === 'top-k 3') { keep = p.map((v, k) => (k < 3 ? v : 0)); tail = 0 }
    else if (st === 'top-p 0.9') {
      let c = 0
      keep = p.map((v) => { const k = c < 0.9; c += v; return k ? v : 0 })
      tail = c < 0.9 ? Math.min(d.rest, 0.9 - c) : 0 // the nucleus reaches past the 256 stored tokens
    }
    const kz = keep.reduce((a, b) => a + b, 0) + tail
    keep = keep.map((v) => v / kz); tail /= kz
    // inverse CDF: lay the kept probabilities end to end on [0, 1) and take the slice u lands in
    let pick = -1, lo = 0
    if (st === 'greedy') pick = 0
    else for (let k = 0, c = 0; k < keep.length; k++) { if (S.u < c + keep[k]) { pick = k; lo = c; break } c += keep[k] }
    if (pick < 0) lo = 1 - tail
    const hi = pick < 0 ? 1 : lo + keep[pick]
    const tok = pick < 0 ? null : { text: d.rows[pick].text, id: d.rows[pick].id }
    return { d, p, keep, tail, pick, lo, hi, tok }
  }

  /* ---------- layout ---------- */
  const pad = 36, top = 60, bot = 96, tokW = 150
  const L = { lx: 0, lnX: 0, xr: r0(), B: r0(), C: r0(), restW: 60, ch: { x0: 0, x1: 0, y0: 0, y1: 0, slot: 40 }, cdf: { x0: 0, x1: 0, y: 0, h: 22 }, sentY: 0 }
  function r0(): Rect { return { x: 0, y: 0, c: 26 } }
  function geom() {
    const { W, H } = stage, avail = H - top - bot
    const c = Math.floor(clamp(Math.min((avail - 130) / (D + 2.5), (W - 2 * pad - tokW - 520) / (D + K6 + 3)), 18, 30))
    const x0 = pad + tokW + 10, y0 = top + 52
    L.sentY = top + 16
    L.xr = { x: x0, y: y0 + (D + 1.2) * c, c }
    L.B = { x: x0 + (D + 1.2) * c, y: y0, c }
    L.C = { x: L.B.x, y: L.xr.y, c }
    L.restW = 2.6 * c
    L.lx = pad; L.lnX = x0 - 44
    const cx0 = L.B.x + K6 * c + L.restW + 60
    L.ch = { x0: cx0, x1: W - pad, y0: y0 + 16, y1: L.C.y + c, slot: 0 }
    L.ch.slot = Math.min(46, (L.ch.x1 - L.ch.x0) * 0.5 / K6)
    L.cdf = { x0: pad, x1: W - pad, y: Math.min(H - bot - 36, L.C.y + c + 58), h: 22 }
  }
  stage.onResize = geom
  geom()

  /* ---------- pieces ---------- */
  function sentence(appendA: number, text: string) {
    ctx.font = serifAt(26); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
    let xx0 = pad
    seq.forEach((t) => { const s = tokDisp(t.text); ctx.fillStyle = rgba(C.ink); ctx.fillText(s, xx0, L.sentY); xx0 += ctx.measureText(s).width })
    if (appendA > 0) { ctx.fillStyle = rgba(nextCol(), appendA); ctx.fillText(tokDisp(text), xx0, L.sentY); xx0 += ctx.measureText(tokDisp(text)).width }
    ctx.fillStyle = rgba(C.ink, 0.5 + 0.5 * Math.sin(performance.now() / 180)); ctx.fillRect(xx0 + 4, L.sentY - 19, 1.5, 23)
    return xx0
  }
  /** Logit of the token at a rank (1-based): exact for the top 256, from 64 quantiles beyond. */
  function zAtRank(rank: number) {
    const n = R.next
    if (rank <= TOPN) return n.top[rank - 1].z
    const q = ((rank - TOPN - 1) / (n.tailCount - 1)) * (n.tailQuantiles.length - 1), q0 = Math.floor(q)
    return lerp(n.tailQuantiles[q0], n.tailQuantiles[Math.min(q0 + 1, n.tailQuantiles.length - 1)], q - q0)
  }
  /** Top-6 bars in fixed slots, then ranks 7 → 50,257 on a log scale. mode 0 = gaps to the top logit ÷ T, 1 = probabilities. */
  function chart(mode: number, T: number, grow: (k: number) => number, tailA: number, a: number) {
    const { ch } = L, sm = sampling(T), zTop = R.next.zmax, p0 = sm.p[0]
    const yL = (g: number) => lerp(ch.y1, ch.y0, (clamp(g, GAP_MIN, 0) - GAP_MIN) / -GAP_MIN)
    const yP = (p: number) => lerp(ch.y1, ch.y0, clamp(p))
    const y = (g: number, p: number) => lerp(yL(g), yP(p), mode), base = ch.y1
    ctx.strokeStyle = rgba(C.ink, 0.25 * a); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(ch.x0, ch.y1); ctx.lineTo(ch.x1, ch.y1); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, a)
    if (mode < 0.5) for (const v of [0, -4, -8, -12]) ctx.fillText(String(v).replace('-', '−'), ch.x0 - 6, yL(v))
    else for (const v of [0, 0.5, 1]) ctx.fillText(String(v), ch.x0 - 6, yP(v))
    ctx.font = F.label; spaced(true); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
    ctx.fillText(mode < 0.5 ? `${T === 1 ? 'LOGIT − TOP' : `(LOGIT − TOP) ÷ ${T.toFixed(2)}`} · TOP = ${fz(zTop)}` : 'PROBABILITIES', ch.x0, ch.y0 - 22); spaced(false)
    const settled = prog('sample') > 0.45
    R.cands.forEach((cd, k) => {
      const g = grow(k)
      if (g <= 0) return
      const bx = ch.x0 + 8 + k * ch.slot, bw = ch.slot * 0.5, gap = (R.next.top[k].z - zTop) / T, p = sm.p[k]
      const yt = lerp(base, y(gap, p), eout(g))
      const chosen = settled && sm.pick === k
      ctx.fillStyle = chosen ? rgba(nextCol(), a) : rgba(C.ink, (sm.keep[k] > 0 || mode < 1 ? 0.75 : 0.2) * a)
      ctx.fillRect(bx, Math.min(yt, base), bw, Math.max(1, Math.abs(base - yt)))
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.ink2, a * g)
      ctx.fillText(mode < 0.5 ? fmt(gap) : pctS(p), bx + bw / 2, Math.min(yt, base) - 3)
      ctx.save(); ctx.translate(bx + bw / 2, ch.y1 + 8); ctx.rotate(-Math.PI / 4)
      tokText(-chipW(cd.text, F.mono(10.5)) + 10, 0, cd.text, a * g, chosen ? nextCol() : null, F.mono(10.5)); ctx.restore()
    })
    if (tailA > 0) {
      const tx0 = ch.x0 + 8 + K6 * ch.slot + 14, tx1 = ch.x1, l0 = Math.log(K6 + 1), l1 = Math.log(VOCAB)
      const rankX = (r: number) => lerp(tx0, tx1, (Math.log(r) - l0) / (l1 - l0))
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.faint, a * tailA); ctx.fillText('⋯', tx0 - 8, ch.y1 - 8)
      ctx.fillStyle = rgba(C.ink, 0.35 * a * tailA)
      for (let xpx = tx0; xpx < tx1; xpx += 2) {
        const r = Math.round(Math.exp(lerp(l0, l1, (xpx - tx0) / (tx1 - tx0)))), g = (zAtRank(r) - zTop) / T
        const yy = y(g, Math.exp(g) * p0)
        if (base - yy >= 0.5) ctx.fillRect(xpx, yy, 1, base - yy)
      }
      // tick where the 256 exactly stored logits end
      const x256 = rankX(TOPN)
      ctx.fillStyle = rgba(C.mute, a * tailA); ctx.fillRect(x256, base, 1, 4)
      ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'top'
      ctx.fillText('7', tx0, base + 6); ctx.textAlign = 'center'; ctx.fillText('256', x256, base + 6)
      ctx.textAlign = 'right'; ctx.fillText('50,257', tx1, base + 6)
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'
      const rest = 1 - sm.p.slice(0, K6).reduce((s, v) => s + v, 0)
      ctx.fillText(mode < 0.5 ? 'rank, log scale' : `50,251 others · ${(rest * 100).toFixed(rest < 0.01 ? 2 : 1)}% together`, tx0, ch.y0 - 2)
      // a sampled token outside the top 6
      if (settled && sm.pick !== 0 && (sm.pick >= K6 || sm.pick < 0)) {
        const px = sm.pick < 0 ? (rankX(TOPN) + tx1) / 2 : rankX(sm.pick + 1)
        ctx.fillStyle = rgba(nextCol(), a); ctx.beginPath(); ctx.moveTo(px, base - 2); ctx.lineTo(px - 5, base - 11); ctx.lineTo(px + 5, base - 11); ctx.closePath(); ctx.fill()
        if (sm.pick < 0) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText('rank > 256', px, base - 14) }
        else tokText(px - chipW(sm.tok!.text, F.mono(10.5)) / 2 + 6, base - 20, sm.tok!.text, a, nextCol(), F.mono(10.5))
      }
    }
    return sm
  }

  /* ---------- scenes ---------- */
  function lanesIn(a: number, lnAct: number) {
    const yl = L.xr.y + L.xr.c / 2, N = seq.length
    for (let i = 0; i < N; i++) {
      const y = yl + (i - (N - 1)) * 10, last = i === N - 1
      ctx.strokeStyle = rgba(R.lanes[i], (last ? 0.9 : 0.25) * a); ctx.lineWidth = last ? 1.8 : 1
      ctx.beginPath(); ctx.moveTo(L.lx, y); ctx.lineTo(last ? L.lnX - 8 : L.lnX - 30, y); ctx.stroke()
      if (last) { ctx.beginPath(); ctx.moveTo(L.lnX + 8, y); ctx.lineTo(L.xr.x - 4, y); ctx.stroke() }
    }
    plate(L.lnX, yl - 6 - (N - 1) * 10, yl + 12, lnAct, { w: 7, d: 8 })
    subLabel('ln_f', L.lnX, yl - 24 - (N - 1) * 10, lnAct > 0.5)
    drawChip(L.lx, yl + 26, seq[N - 1], a, 20)
    ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, a)
    ctx.fillText('last position only', L.lx + chipW(seq[N - 1].text) + 8, yl + 26)
  }
  function drawX(a: number, reveal?: (i: number, j: number) => number) {
    mk.drawMat({ r: L.xr, vals: R.xs, kind: 'row', alpha: a, name: 'x', shape: '1 × 768', real: `${D} drawn`, label: 'bottom', rowCols: [mk.tokRGB(seq.length - 1)], reveal })
  }
  function drawB(a: number) {
    mk.drawMat({ r: L.B, vals: R.WT, kind: 'w', alpha: a, name: 'W_Eᵀ', shape: '768 × 50,257' })
    const rx = L.B.x + K6 * L.B.c + 6, rh = D * L.B.c
    mk.slab(rx, L.B.y, L.restW, rh, a)
    ctx.fillStyle = rgba(C.ink, 0.04 * a); ctx.fillRect(rx, L.B.y, L.restW, rh)
    ctx.strokeStyle = rgba(C.ink, 0.08 * a); ctx.lineWidth = 1
    for (let xx = rx + 3; xx < rx + L.restW; xx += 3) { ctx.beginPath(); ctx.moveTo(xx, L.B.y); ctx.lineTo(xx, L.B.y + rh); ctx.stroke() }
    ctx.strokeStyle = rgba(C.ink, 0.22 * a); ctx.strokeRect(rx + 0.5, L.B.y + 0.5, L.restW - 1, rh - 1)
    ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, a)
    ctx.fillText('+50,251', rx + L.restW / 2, L.B.y + rh + 8)
    R.cands.forEach((cd, j) => {
      ctx.save(); ctx.translate(L.B.x + (j + 0.5) * L.B.c, L.B.y - 30); ctx.rotate(-Math.PI / 4)
      tokText(0, 0, cd.text, a, null, F.mono(10.5)); ctx.restore()
    })
  }
  const drawZ = (a: number, reveal?: (i: number, j: number) => number) =>
    mk.drawMat({ r: L.C, vals: R.Z, kind: 'score', alpha: a, name: 'z', shape: '1 × 50,257', real: 'logits', label: 'bottom', reveal, text: (_, j) => R.Z[0][j].toFixed(1) })

  function sceneLnf(p: number) {
    const a = eout(clamp(p / 0.2))
    sentence(0, '')
    lanesIn(a, p > 0.2 && p < 0.6 ? 1 : 0)
    drawX(eout(clamp((p - 0.35) / 0.3)))
    mk.formula = { segs: [['x  =  ln_f(h[', C.mute], [String(seq.length - 1), C.ink], ['])  =  γ ⊙ (h − ', C.mute], [fmt(R.mu), C.ink2], [') / ', C.mute], [fmt(R.sigma), C.ink2], [' + β', C.mute]], note: `h is ${tokLabel(seq[seq.length - 1].text)}'s stream after all 12 blocks. The other positions were still needed: attention read their keys and values.` }
  }

  function sceneLogits(p: number) {
    sentence(0, '')
    lanesIn(1 - eout(clamp(p / 0.15)), 0)
    drawX(1)
    drawB(eout(clamp(p / 0.12)))
    const g = gemm((p - 0.1) / 0.72, 1, K6, D, 'slow')
    drawZ(1, g.rev)
    const restA = clamp((p - 0.84) / 0.12), rx = L.C.x + K6 * L.C.c + 6
    ctx.fillStyle = rgba(C.ink, 0.12 * restA); ctx.fillRect(rx, L.C.y + 1, L.restW * restA, L.C.c - 2)
    ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, restA)
    ctx.fillText('…', rx + L.restW / 2, L.C.y + L.C.c / 2)
    mk.hit('Z', L.C, 1, K6)
    chart(0, 1, (k) => (g.rev(0, k) ? 1 : 0), restA, eout(clamp((p - 0.15) / 0.2)))
    const f = mk.resolve({ Z: { g, K: D } })
    if (f) {
      const cd = R.cands[f.j]
      mk.gemmOverlay({
        A: L.xr, Av: R.xs, B: L.B, Bv: R.WT, C: L.C, f, names: ['z', 'x', 'W_Eᵀ'], fmtC: fz,
        more: { n: 768 - D, sum: (_, j) => { let s = 0; const v = presets()[S.preset].passes[0].unembed!.rows[j].v; for (let t = D; t < 768; t++) s += R.xf[t] * v[t]; return s } },
        note: `Logit of ${tokLabel(cd.text)} (id ${cd.id}): x dotted with row ${cd.id} of W_E, the same row the embedding looks up.`,
      })
    } else if (restA > 0) mk.formula = { segs: [['z  =  x · W_Eᵀ', C.ink], ['   ·   1 × 768 · 768 × 50,257  →  50,257 logits', C.ink2]], note: 'One GEMM scores every token: 768 × 50,257 ≈ 38.6M multiply-adds, or 77M FLOPs, per generated token.' }
  }

  function staticGemm(a = 1, withSentence = true) {
    if (withSentence) sentence(0, '')
    drawX(a); drawB(a); drawZ(a)
    mk.hit('Z', L.C, 1, K6)
  }

  function sceneTemp(p: number) {
    staticGemm()
    const T = lerp(1, S.T, eio(clamp(p / 0.5)))
    chart(0, T, () => 1, 1, 1)
    const g1 = R.next.top[1].z - R.next.zmax
    mk.formula = { segs: [['z / T', C.ink], ['   ·   T = ', C.mute], [S.T.toFixed(2), C.ink], ['   ·   gap to ', C.mute], [`${tokLabel(R.cands[1].text)}: ${fmt(g1)} → ${fmt(g1 / S.T)}`, C.ink2]], note: 'Only gaps between logits matter. T < 1 stretches them (more decisive; as T → 0 this becomes greedy), T > 1 shrinks them. Drag T below.' }
  }

  function sceneSoftmax(p: number) {
    staticGemm()
    const sm = chart(eio(clamp(p / 0.6)), S.T, () => 1, 1, 1), rest = 1 - sm.p.slice(0, K6).reduce((s, v) => s + v, 0)
    mk.formula = { segs: [['p  =  softmax(z / T)  =  exp(z / T) / Σ exp(z / T)', C.ink], ['   ·   ', C.mute], [`${tokLabel(R.cands[0].text)} ${pctS(sm.p[0])}`, C.ink2]], note: `The sum runs over all 50,257 tokens. The 50,251 outside the top 6 hold ${(rest * 100).toFixed(1)}% together: each is tiny, but together they are the long tail.` }
  }

  function sceneSample(p: number) {
    staticGemm(0.5, false)
    const sm = chart(1, S.T, () => 1, 1, 1)
    const { cdf } = L, w = cdf.x1 - cdf.x0, settled = p > 0.45, col = nextCol()
    ctx.font = F.label; spaced(true); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute)
    ctx.fillText(`${STRATS[S.strat].toUpperCase()} · KEPT, RENORMALISED, END TO END`, (cdf.x0 + cdf.x1) / 2, cdf.y + cdf.h + 6); spaced(false)
    let cx = cdf.x0
    sm.keep.forEach((q, k) => {
      if (q <= 0) return
      const sw = q * w, chosen = settled && sm.pick === k
      if (k < K6 || chosen) {
        rr(cx + 0.5, cdf.y, Math.max(1, sw - 1), cdf.h, 3)
        ctx.fillStyle = chosen ? rgba(col, 0.9) : rgba(C.ink, 0.14 + 0.1 * (k % 2)); ctx.fill()
        if (sw > 44 && k < K6) { ctx.save(); ctx.beginPath(); ctx.rect(cx, cdf.y, sw, cdf.h); ctx.clip(); tokText(cx + 6, cdf.y + cdf.h / 2, sm.d.rows[k].text, 1, chosen ? C.bg : null, F.mono(11)); ctx.restore() }
      } else { ctx.fillStyle = rgba(C.ink, 0.07 + 0.06 * (k % 2)); ctx.fillRect(cx, cdf.y, sw, cdf.h) }
      cx += sw
    })
    if (sm.tail > 0) {
      const sw = sm.tail * w, chosen = settled && sm.pick < 0
      ctx.fillStyle = chosen ? rgba(col, 0.5) : rgba(C.ink, 0.04); ctx.fillRect(cx, cdf.y, sw, cdf.h)
      ctx.strokeStyle = rgba(C.ink, 0.1); ctx.lineWidth = 1
      ctx.save(); ctx.beginPath(); ctx.rect(cx, cdf.y, sw, cdf.h); ctx.clip()
      for (let xx = cx - cdf.h; xx < cx + sw; xx += 5) { ctx.beginPath(); ctx.moveTo(xx, cdf.y + cdf.h); ctx.lineTo(xx + cdf.h, cdf.y); ctx.stroke() }
      if (sw > 90) { ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute); ctx.fillText('rank > 256', cx + 6, cdf.y + cdf.h / 2) }
      ctx.restore()
    }
    ctx.font = F.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute)
    ctx.fillText('0', cdf.x0, cdf.y + cdf.h + 5); ctx.textAlign = 'right'; ctx.fillText('1', cdf.x1, cdf.y + cdf.h + 5)
    // u falls onto the bar
    const greedy = STRATS[S.strat] === 'greedy'
    const ux = greedy ? cdf.x0 + (sm.keep[0] * w) / 2 : cdf.x0 + S.u * w, drop = eout(clamp(p / 0.4))
    const uy = lerp(cdf.y - 60, cdf.y - 2, drop)
    ctx.fillStyle = rgba(C.ink); ctx.beginPath(); ctx.moveTo(ux, uy); ctx.lineTo(ux - 5, uy - 9); ctx.lineTo(ux + 5, uy - 9); ctx.closePath(); ctx.fill()
    ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.ink2)
    ctx.fillText(greedy ? 'argmax' : `u = ${S.u.toFixed(2)}`, ux, uy - 11)
    // the chosen token flies up to the end of the prompt
    const text = sm.tok ? sm.tok.text : ' …'
    const endX = sentence(clamp((p - 0.85) / 0.1), text)
    if (p > 0.5 && p < 0.95) {
      const f = eio(clamp((p - 0.5) / 0.35)), tx = endX + 12, ty = L.sentY - 8
      const qx = lerp(ux, tx, f), qy = lerp(cdf.y, ty, f) - Math.sin(f * Math.PI) * 60
      drawChip(qx - 20, qy, { text, c: seq.length % 7 }, 1 - clamp((p - 0.85) / 0.1), 20, true)
    }
    const segs: [string, RGB][] = [[STRATS[S.strat], C.ink], ['   →   ', C.mute]]
    if (sm.tok) segs.push([tokLabel(sm.tok.text), C.ink], ['   ·   p = ', C.mute], [pctS(sm.p[sm.pick]), C.ink2], ['   ·   id ', C.mute], [String(sm.tok.id), C.ink2])
    else segs.push(['a token outside the top 256', C.ink])
    const why = greedy ? 'Greedy skips the draw and takes the top token.' : `u = ${S.u.toFixed(2)} lands in the slice [${sm.lo.toFixed(sm.hi - sm.lo < 0.01 ? 4 : 2)}, ${sm.hi.toFixed(sm.hi - sm.lo < 0.01 ? 4 : 2)}).`
    const tail = sm.tok ? '' : ` Only the top 256 logits are stored here; the other 50,001 tokens hold ${pctS(sm.d.rest)} together.`
    mk.formula = { segs, note: `${why}${tail} Press Resample, or change the strategy or T.` }
  }

  function draw() {
    useCtx(ctx)
    stage.begin()
    mk.begin()
    const pl = prog('logits'), pt = prog('temp'), ps = prog('softmax'), pS = prog('sample')
    if (pl <= 0) sceneLnf(prog('lnf'))
    else if (pt <= 0) sceneLogits(pl)
    else if (ps <= 0) sceneTemp(pt)
    else if (pS <= 0) sceneSoftmax(ps)
    else sceneSample(pS)
    mk.drawFormula()
  }

  function caption(id: string): [string, string] {
    switch (id) {
      case 'lnf': return ['Only the last position predicts the next token. Its vector, after all 12 blocks, goes through the final LayerNorm ln_f. The numbers are from a real GPT-2 small run.', '[1 × 768]']
      case 'logits': return ['The vector is dotted with every row of the embedding matrix (GPT-2 reuses W_E as the LM head), giving one logit per token. The grid draws 8 of the 768 dimensions for the 6 likeliest tokens; each logit sums all 768.', '[1×768]·[768×50,257] · 77M FLOPs']
      case 'temp': return ['Logits are divided by the temperature T before softmax. Only the gaps between them matter: lower T stretches the gaps and sharpens the distribution, higher T flattens it.', 'z / T']
      case 'softmax': return ['Softmax turns the scaled logits into probabilities over all 50,257 tokens, summing to 1. A few tokens lead, and the long tail together often holds most of the mass.', 'p = softmax(z / T)']
      default: return ['A strategy picks the next token: greedy takes the likeliest, sampling lays the probabilities end to end and a random u picks one, and top-k or top-p drop the tail first. Generation repeats until <|endoftext|> or a length limit.', 'sample · greedy · top-k · top-p']
    }
  }

  if (reduced) player.t = player.start('softmax') + 3
  const stop = runLoop((dt) => {
    player.tick(dt)
    draw()
    player.updateUI()
    const cur = player.cur(), [t, s] = caption(cur.id)
    frame.setCaption(cur.name, cur.short ?? cur.name, t, s)
  })
  return () => { stop(); player.destroy(); stage.destroy() }
}
