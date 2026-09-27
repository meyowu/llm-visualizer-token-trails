import { F, chipW, drawChip, plate, rr, spaced, subLabel, tokDisp, tokLabel, tokText, useCtx } from '../../core/draw'
import { createFrame, toggle } from '../../core/frame'
import { MatrixKit, fmt, gemm, type M, type Rect } from '../../core/matrix'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, blend, rgba } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import type { Nav } from '../registry'
import { LN, PROMPT, TAIL, TAIL_LOGIT, TOY, attention, lnStats, mlp, promptTokens, tokEmb, tokId, topLogits } from './model'

/*
 * The last position's vector goes through ln_f and is dotted with every row of W_E (GPT-2 ties
 * the LM head to the embedding). The six real candidates are drawn as a GEMM; the remaining
 * 50,251 columns share one toy logit. Then temperature, softmax, and a sampling strategy.
 * Toy W_E rows for the candidates are chosen so their dot products equal the canned logits.
 */

const PHASES = [
  { id: 'lnf', name: 'Final LayerNorm', short: 'ln_f', dur: 3.5 },
  { id: 'logits', name: 'Logits', short: 'x · W_Eᵀ', dur: 7.5 },
  { id: 'temp', name: 'Temperature', short: '÷ T', dur: 4 },
  { id: 'softmax', name: 'Softmax', dur: 4 },
  { id: 'sample', name: 'Sample', dur: 5.5 },
]
const V = 50257, K6 = 6, OTHERS = V - K6
const STRATS = ['sample', 'greedy', 'top-k 3', 'top-p 0.9'] as const

export function mountUnembed(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const seq = promptTokens(), N = seq.length, D = TOY.d
  const att = attention(seq), R = mlp(seq, att)
  const x = lnStats(R.out[N - 1], LN.lnf).out
  const cands = topLogits(seq[N - 1].text).map(([text, logit]) => ({ text, logit, id: tokId(text) }))
  // toy W_E rows for the candidates: a seeded row, nudged along x so that x · e = the target logit
  const xx = x.reduce((s, v) => s + v * v, 0)
  const rows: M = cands.map((cd) => {
    const r = tokEmb(cd).map((v) => v * 0.5), d = x.reduce((s, v, k) => s + v * r[k], 0)
    return r.map((v, k) => v + ((cd.logit - d) / xx) * x[k])
  })
  const WT: M = Array.from({ length: D }, (_, k) => rows.map((r) => r[k])) // W_Eᵀ restricted to the candidates
  const Z: M = [cands.map((cd) => cd.logit)]

  const S = { T: 0.8, strat: 0, u: 0.22 }
  const frame = createFrame(root, {
    eyebrow: 'Transformer · Unembed & Sampling',
    title: 'Unembed & Sampling',
    subtitle: 'LM head · tied to W_E',
    back: { label: 'Forward pass', onClick: () => nav('transformer') },
    specs: [
      { label: 'shown', value: 'toy', real: 'GPT-2 small' },
      { label: 'd_model', value: String(D), real: '768' },
      { label: 'vocab', value: '50,257' },
      { label: 'LM head', value: 'W_Eᵀ', real: 'tied' },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 520, 'Unembedding and sampling: the last position passes through ln_f, is scored against every vocabulary embedding to give logits, which are divided by the temperature, turned into probabilities by softmax, and sampled.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  const prog = (id: string) => player.prog(id)
  const mk = new MatrixKit(stage, seq, 36)
  const nextCol = C.tok[N % 7]

  // controls: strategy, resample, temperature
  toggle(player.meta, 'Sampling strategy', [...STRATS], 0, (i) => { S.strat = i })
  const resample = document.createElement('button')
  resample.className = 'resample'; resample.textContent = 'Resample'
  resample.addEventListener('click', () => { S.u = Math.random(); if (player.t < player.start('sample')) player.t = player.start('sample') + 0.001 })
  const temp = document.createElement('label')
  temp.className = 'temp'; temp.htmlFor = 'un-temp'
  temp.innerHTML = '<span>T</span><input id="un-temp" type="range" min="0.2" max="2" step="0.05" value="0.8"><output>0.80</output>'
  player.meta.prepend(resample)
  player.meta.append(temp)
  const tIn = temp.querySelector('input')!, tOut = temp.querySelector('output')!
  tIn.addEventListener('input', () => { S.T = +tIn.value; tOut.textContent = S.T.toFixed(2) })

  /* ---------- distribution under the current strategy ---------- */
  function distNow(T: number) {
    const ex = cands.map((cd) => Math.exp(cd.logit / T)), rest = OTHERS * Math.exp(TAIL_LOGIT / T)
    const Zs = ex.reduce((a, b) => a + b, 0) + rest
    const probs = [...ex.map((e) => e / Zs), rest / Zs] // 6 candidates + others
    let keep = probs.map(() => true)
    const st = STRATS[S.strat]
    if (st === 'greedy') keep = probs.map((_, k) => k === 0)
    else if (st === 'top-k 3') keep = probs.map((_, k) => k < 3)
    else if (st === 'top-p 0.9') { let c = 0; keep = probs.map((p) => { const k = c < 0.9; c += p; return k }) }
    const kz = probs.reduce((s, p, k) => s + (keep[k] ? p : 0), 0)
    const kept = probs.map((p, k) => (keep[k] ? p / kz : 0))
    let pick = 0
    if (st !== 'greedy') { let c = 0; pick = kept.length - 1; for (let k = 0; k < kept.length; k++) { c += kept[k]; if (S.u < c) { pick = k; break } } }
    else pick = 0
    return { probs, keep, kept, pick, text: pick < K6 ? cands[pick].text : TAIL[0] }
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
    L.ch = { x0: cx0, x1: W - pad, y0: y0 + 10, y1: L.C.y + c, slot: 0 }
    L.ch.slot = Math.min(46, (L.ch.x1 - L.ch.x0) * 0.5 / K6)
    L.cdf = { x0: pad, x1: W - pad, y: Math.min(H - bot - 36, L.C.y + c + 58), h: 22 }
  }
  stage.onResize = geom
  geom()

  /* ---------- pieces ---------- */
  function sentence(appendA: number, text: string) {
    ctx.font = 'italic 400 26px Newsreader, Georgia, serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
    let xx0 = pad
    PROMPT.forEach((w) => { ctx.fillStyle = rgba(C.ink); ctx.fillText(w, xx0, L.sentY); xx0 += ctx.measureText(w).width })
    if (appendA > 0) { ctx.fillStyle = rgba(nextCol, appendA); ctx.fillText(tokDisp(text), xx0, L.sentY); xx0 += ctx.measureText(tokDisp(text)).width }
    ctx.fillStyle = rgba(C.ink, 0.5 + 0.5 * Math.sin(performance.now() / 180)); ctx.fillRect(xx0 + 4, L.sentY - 19, 1.5, 23)
    return xx0
  }
  /** Top-6 bars in fixed slots, then the tail of 50,251 others. mode 0 = logits, 1 = probabilities. */
  function chart(mode: number, T: number, grow: (k: number) => number, tailA: number, a: number) {
    const { ch } = L, d = distNow(T), zmin = -15, zmax = 8
    const yL = (z: number) => lerp(ch.y1, ch.y0, (clamp(z, zmin, zmax) - zmin) / (zmax - zmin))
    const yP = (p: number) => lerp(ch.y1, ch.y0, clamp(p))
    const y = (z: number, p: number) => lerp(yL(z), yP(p), mode), base = lerp(yL(0), ch.y1, mode)
    // axes
    ctx.strokeStyle = rgba(C.ink, 0.25 * a); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(ch.x0, ch.y1); ctx.lineTo(ch.x1, ch.y1); ctx.stroke()
    if (mode < 1) { ctx.setLineDash([2, 3]); ctx.strokeStyle = rgba(C.ink, 0.25 * a * (1 - mode)); ctx.beginPath(); ctx.moveTo(ch.x0, yL(0)); ctx.lineTo(ch.x1, yL(0)); ctx.stroke(); ctx.setLineDash([]) }
    ctx.font = F.mono(9.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    if (mode < 0.5) { ctx.fillStyle = rgba(C.mute, a); for (const v of [5, 0, -5, -10, -15]) ctx.fillText(String(v).replace('-', '−'), ch.x0 - 6, yL(v)) }
    else { ctx.fillStyle = rgba(C.mute, a); for (const v of [0, 0.5, 1]) ctx.fillText(String(v), ch.x0 - 6, yP(v)) }
    ctx.font = F.label; spaced(true); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
    ctx.fillText(mode < 0.5 ? (T === 1 ? 'LOGITS, SORTED' : `LOGITS ÷ ${T.toFixed(2)}`) : 'PROBABILITIES', ch.x0, ch.y0 - 14); spaced(false)
    const settled = prog('sample') > 0.45
    // top 6
    cands.forEach((cd, k) => {
      const g = grow(k)
      if (g <= 0) return
      const bx = ch.x0 + 8 + k * ch.slot, bw = ch.slot * 0.5, z = cd.logit / T, p = d.probs[k]
      const yt = lerp(base, y(z, p), eout(g))
      const chosen = settled && d.pick === k
      ctx.fillStyle = chosen ? rgba(nextCol, a) : rgba(C.ink, (d.keep[k] || mode < 1 ? 0.75 : 0.2) * a)
      ctx.fillRect(bx, Math.min(yt, base), bw, Math.abs(base - yt))
      ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.ink2, a * g)
      ctx.fillText(mode < 0.5 ? fmt(z) : p < 0.001 ? '<.1%' : (p * 100).toFixed(p < 0.1 ? 1 : 0) + '%', bx + bw / 2, Math.min(yt, base) - 3)
      ctx.save(); ctx.translate(bx + bw / 2, ch.y1 + 8); ctx.rotate(-Math.PI / 4)
      tokText(-chipW(cd.text, F.mono(10.5)) + 10, 0, cd.text, a * g, chosen ? nextCol : null, F.mono(10.5)); ctx.restore()
    })
    // tail: 50,251 tokens, one toy logit, a little texture so it reads as many bars
    if (tailA > 0) {
      const tx0 = ch.x0 + 8 + K6 * ch.slot + 14, tx1 = ch.x1, pt = d.probs[K6] / OTHERS
      ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.faint, a * tailA); ctx.fillText('⋯', tx0 - 8, ch.y1 - 8)
      ctx.fillStyle = rgba(C.ink, 0.35 * a * tailA)
      for (let xpx = tx0; xpx < tx1; xpx += 2) {
        const jit = Math.sin(xpx * 12.9898) * 0.35, yy = y((TAIL_LOGIT + jit) / T, pt)
        ctx.fillRect(xpx, Math.min(yy, base), 1, Math.max(1, Math.abs(base - yy)))
      }
      ctx.font = F.mono(10); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.mute, a * tailA)
      const ty = mode < 0.5 ? base - 6 : ch.y1 - 6
      ctx.fillText(mode < 0.5 ? `50,251 others · ${fmt(TAIL_LOGIT / T)} each` : `50,251 others · ${(d.probs[K6] * 100).toFixed(d.probs[K6] < 0.01 ? 2 : 1)}% together`, tx0, ty)
    }
    return d
  }

  /* ---------- scenes ---------- */
  function lanesIn(a: number, lnAct: number) {
    const cols = att.heads[0].A.map((row, i) => blend(seq.map((_, k) => mk.tokRGB(k)), row.map((w, k) => 0.55 * w + (k === i ? 0.45 : 0))))
    const yl = L.xr.y + L.xr.c / 2
    for (let i = 0; i < N; i++) {
      const y = yl + (i - (N - 1)) * 10, last = i === N - 1
      ctx.strokeStyle = rgba(cols[i], (last ? 0.9 : 0.25) * a); ctx.lineWidth = last ? 1.8 : 1
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
    mk.drawMat({ r: L.xr, vals: [x], kind: 'row', alpha: a, name: 'x', shape: '1 × 8', real: '1 × 768', label: 'bottom', rowCols: [mk.tokRGB(N - 1)], reveal })
  }
  function drawB(a: number) {
    mk.drawMat({ r: L.B, vals: WT, kind: 'w', alpha: a, name: 'W_Eᵀ', shape: '8 × 50,257', real: '768 × 50,257' })
    // the rest of the vocabulary as a hatched slab
    const rx = L.B.x + K6 * L.B.c + 6, rh = D * L.B.c
    mk.slab(rx, L.B.y, L.restW, rh, a)
    ctx.fillStyle = rgba(C.ink, 0.04 * a); ctx.fillRect(rx, L.B.y, L.restW, rh)
    ctx.strokeStyle = rgba(C.ink, 0.08 * a); ctx.lineWidth = 1
    for (let xx = rx + 3; xx < rx + L.restW; xx += 3) { ctx.beginPath(); ctx.moveTo(xx, L.B.y); ctx.lineTo(xx, L.B.y + rh); ctx.stroke() }
    ctx.strokeStyle = rgba(C.ink, 0.22 * a); ctx.strokeRect(rx + 0.5, L.B.y + 0.5, L.restW - 1, rh - 1)
    ctx.font = F.mono(9.5); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, a)
    ctx.textBaseline = 'top'; ctx.fillText('+50,251', rx + L.restW / 2, L.B.y + rh + 8)
    // candidate labels above their columns
    cands.forEach((cd, j) => {
      ctx.save(); ctx.translate(L.B.x + (j + 0.5) * L.B.c, L.B.y - 30); ctx.rotate(-Math.PI / 4)
      tokText(0, 0, cd.text, a, null, F.mono(10.5)); ctx.restore()
    })
  }

  function sceneLnf(p: number) {
    const a = eout(clamp(p / 0.2))
    sentence(0, '')
    lanesIn(a, p > 0.2 && p < 0.6 ? 1 : 0)
    drawX(eout(clamp((p - 0.35) / 0.3)))
    const s = lnStats(R.out[N - 1], LN.lnf)
    mk.formula = { segs: [['x  =  ln_f(h[', C.mute], [String(N - 1), C.ink], ['])  =  γ ⊙ (h − ', C.mute], [fmt(s.mu), C.ink2], [') / ', C.mute], [fmt(s.sigma), C.ink2], [' + β', C.mute]], note: `Only the last position (${tokLabel(seq[N - 1].text)}) is needed to predict the next token. Toy: after 1 block; GPT-2 runs 12 first.` }
  }

  function sceneLogits(p: number) {
    sentence(0, '')
    lanesIn(1 - eout(clamp(p / 0.15)), 0)
    drawX(1)
    drawB(eout(clamp(p / 0.12)))
    const g = gemm((p - 0.1) / 0.72, 1, K6, D, 'slow')
    mk.drawMat({ r: L.C, vals: Z, kind: 'score', alpha: 1, name: 'z', shape: '1 × 50,257', real: 'logits', label: 'bottom', reveal: g.rev })
    const restA = clamp((p - 0.84) / 0.12), rx = L.C.x + K6 * L.C.c + 6
    ctx.fillStyle = rgba(C.ink, 0.12 * restA); ctx.fillRect(rx, L.C.y + 1, L.restW * restA, L.C.c - 2)
    ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, restA)
    ctx.fillText(fmt(TAIL_LOGIT) + ' …', rx + L.restW / 2, L.C.y + L.C.c / 2)
    mk.hit('Z', L.C, 1, K6)
    chart(0, 1, (k) => (g.rev(0, k) ? 1 : 0), restA, eout(clamp((p - 0.15) / 0.2)))
    const f = mk.resolve({ Z: { g, K: D } })
    if (f) mk.gemmOverlay({ A: L.xr, Av: [x], B: L.B, Bv: WT, C: L.C, f, names: ['z', 'x', 'W_Eᵀ'], note: `Logit of ${tokLabel(cands[f.j].text)} (id ${cands[f.j].id}): x dotted with row ${cands[f.j].id} of W_E, the same row the embedding looks up.` })
    else if (restA > 0) mk.formula = { segs: [['z  =  x · W_Eᵀ', C.ink], ['   ·   1 × 768 · 768 × 50,257  →  50,257 logits', C.ink2]], note: 'One GEMM scores every vocabulary token at once: 2 × 768 × 50,257 ≈ 77M multiply-adds per generated token.' }
  }

  function staticGemm(a = 1, withSentence = true) {
    if (withSentence) sentence(0, '')
    drawX(a); drawB(a)
    mk.drawMat({ r: L.C, vals: Z, kind: 'score', alpha: a, name: 'z', shape: '1 × 50,257', real: 'logits', label: 'bottom' })
    mk.hit('Z', L.C, 1, K6)
  }

  function sceneTemp(p: number) {
    staticGemm()
    const T = lerp(1, S.T, eio(clamp(p / 0.5)))
    chart(0, T, () => 1, 1, 1)
    mk.formula = { segs: [['z / T', C.ink], ['   ·   T = ', C.mute], [S.T.toFixed(2), C.ink], ['   ·   ', C.mute], [`${tokLabel(cands[0].text)}: ${fmt(cands[0].logit)} → ${fmt(cands[0].logit / S.T)}`, C.ink2]], note: 'Dividing by T < 1 stretches the gaps between logits (more decisive); T > 1 shrinks them (more random). Drag T below.' }
  }

  function sceneSoftmax(p: number) {
    staticGemm()
    const d = chart(eio(clamp(p / 0.6)), S.T, () => 1, 1, 1)
    mk.formula = { segs: [['p  =  softmax(z / T)  =  exp(z / T) / Σ exp(z / T)', C.ink], ['   ·   ', C.mute], [`${tokLabel(cands[0].text)} ${(d.probs[0] * 100).toFixed(1)}%`, C.ink2]], note: `The 50,251 other tokens share ${(d.probs[K6] * 100).toFixed(2)}% between them: each is tiny, but together they are the long tail.` }
  }

  function sceneSample(p: number) {
    staticGemm(0.5, false)
    const d = chart(1, S.T, () => 1, 1, 1)
    // cumulative bar over the kept candidates
    const { cdf } = L, w = cdf.x1 - cdf.x0
    let cx = cdf.x0
    ctx.font = F.label; spaced(true); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute)
    ctx.fillText(`${STRATS[S.strat].toUpperCase()} · KEPT, RENORMALISED`, cdf.x0, cdf.y - 10); spaced(false)
    const settled = p > 0.45
    d.kept.forEach((q, k) => {
      if (q <= 0) return
      const sw = q * w, chosen = settled && d.pick === k
      rr(cx + 0.5, cdf.y, Math.max(1, sw - 1), cdf.h, 3)
      ctx.fillStyle = chosen ? rgba(nextCol, 0.9) : rgba(C.ink, k === K6 ? 0.08 : 0.14 + 0.1 * (k % 2)); ctx.fill()
      if (sw > 44) {
        ctx.save(); ctx.beginPath(); ctx.rect(cx, cdf.y, sw, cdf.h); ctx.clip()
        if (k < K6) tokText(cx + 6, cdf.y + cdf.h / 2, cands[k].text, 1, chosen ? C.bg : null, F.mono(11))
        else { ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute); ctx.fillText('others', cx + 6, cdf.y + cdf.h / 2) }
        ctx.restore()
      }
      cx += sw
    })
    // the random draw u falls onto the bar
    const greedy = STRATS[S.strat] === 'greedy'
    const ux = greedy ? cdf.x0 + (d.kept[0] * w) / 2 : cdf.x0 + S.u * w, drop = eout(clamp(p / 0.4))
    const uy = lerp(cdf.y - 60, cdf.y - 2, drop)
    ctx.fillStyle = rgba(C.ink); ctx.beginPath(); ctx.moveTo(ux, uy); ctx.lineTo(ux - 5, uy - 9); ctx.lineTo(ux + 5, uy - 9); ctx.closePath(); ctx.fill()
    ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = rgba(C.ink2)
    ctx.fillText(greedy ? 'argmax' : `u = ${S.u.toFixed(2)}`, ux, uy - 11)
    // the chosen token flies up to the end of the prompt
    const endX = sentence(clamp((p - 0.85) / 0.1), d.text)
    if (p > 0.5 && p < 0.95) {
      const f = eio(clamp((p - 0.5) / 0.35)), sx = ux, sy = cdf.y, tx = endX + 12, ty = L.sentY - 8
      const qx = lerp(sx, tx, f), qy = lerp(sy, ty, f) - Math.sin(f * Math.PI) * 60
      drawChip(qx - 20, qy, { text: d.text, c: N % 7 }, 1 - clamp((p - 0.85) / 0.1), 20, true)
    }
    const pk = d.pick < K6 ? d.probs[d.pick] : d.probs[K6]
    mk.formula = { segs: [[STRATS[S.strat], C.ink], ['   →   ', C.mute], [tokLabel(d.text), C.ink], ['   ·   p = ', C.mute], [`${(pk * 100).toFixed(1)}%`, C.ink2], ['   ·   next token id ', C.mute], [String(tokId(d.text)), C.ink2]], note: 'Greedy always takes the top token. Top-k and top-p cut the long tail, renormalise, then draw. Press Resample or change the strategy.' }
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

  const CAPS: Record<string, [string, string]> = {
    lnf: ['Only the last position predicts the next token. Its vector goes through the final LayerNorm, ln_f.', '[1 × 768]'],
    logits: ['The vector is dotted with every row of the embedding matrix; GPT-2 reuses W_E as the LM head. Tokens whose embeddings point the same way as x score high.', 'GPT-2 [1×768]·[768×50,257] · 77 MFLOPs'],
    temp: ['The logits are divided by the temperature T before softmax. Lower T sharpens the distribution, higher T flattens it.', `T = ${S.T.toFixed(2)}`],
    softmax: ['Softmax turns the scaled logits into probabilities that sum to 1. The top few tokens take most of the mass; 50,251 others share the rest.', 'p = softmax(z / T)'],
    sample: ['A strategy picks the next token: greedy takes the most likely, sampling draws at random by probability, and top-k or top-p first drop the unlikely tail.', 'sample · greedy · top-k · top-p'],
  }

  if (reduced) player.t = player.start('softmax') + 3
  const stop = runLoop((dt) => {
    player.tick(dt)
    draw()
    player.updateUI()
    const cur = player.cur(), [t, s] = CAPS[cur.id]
    frame.setCaption(cur.name, cur.short ?? cur.name, t, cur.id === 'temp' ? `T = ${S.T.toFixed(2)}` : s)
  })
  return () => { stop(); player.destroy(); stage.destroy() }
}
