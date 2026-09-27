import { F, rr, serifAt, type TokLike } from '../../core/draw'
import { fmt, fmtF } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eout } from '../../core/util'
import { quant } from '../../lib/quant/data'
import { mountExhibit, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * Quantization: storing weights (and sometimes activations) in fewer bits. The weight column, the
 * activation vector, the losses and the next-token guesses are all real GPT-2 small, quantized and
 * rerun offline (scripts/quant-export.ts).
 */

const PHASES = [
  { id: 'formats', name: 'Fewer bits per number', short: 'Formats', dur: 9 },
  { id: 'round', name: 'Rounding onto a grid', short: 'absmax', dur: 11 },
  { id: 'scales', name: 'One scale or many', short: 'Scales', dur: 10 },
  { id: 'outliers', name: 'Outliers', short: 'Outliers', dur: 10 },
  { id: 'effect', name: 'What it costs the model', short: 'Effect', dur: 10 },
]

const COL = quant.column, ACT = quant.activation, SHOW = 24
const RES = (name: string) => quant.results.find((r) => r.name === name)!
const levels = (bits: number) => (1 << (bits - 1)) - 1
const qz = (v: number, scale: number, q: number) => Math.max(-q, Math.min(q, Math.round(v / scale)))
const TOKS: TokLike[] = [{ text: 'w', c: 0 }]

const COMPARE: Record<string, [string, string]> = {
  formats: ['the KV cache’s size', 'serving/kv-cache?phase=size'], round: ['GPT-2’s MLP weights', 'anatomy/mlp?phase=up'], scales: ['GPT-2’s MLP weights', 'anatomy/mlp?phase=up'],
  outliers: ['GPT-2’s LayerNorm', 'anatomy/layernorm?phase=affine'], effect: ['GPT-2’s next-token guesses', 'anatomy/unembed?phase=softmax'],
}

const CAPS: Record<string, [string, string]> = {
  formats: ['Models train in 32- or 16-bit floating point. Serving them in 8 or 4 bits makes the weights 2 to 4 times smaller, so a model fits on fewer GPUs and each decode step, which is limited by reading the weights, runs faster.', '32 → 16 → 8 → 4 bits per weight'],
  round: ['The simplest scheme, absmax: divide by a scale so the largest weight lands on the largest integer, round, and multiply back when computing. int8 has 255 levels, int4 only 15, so the rounding error is much larger. A real column of GPT-2’s weights.', 'q = round(w / s) · s = max|w| / (2^(b−1) − 1)'],
  scales: ['One scale for a whole matrix is set by its single largest weight, which makes the grid coarse for everything else. A scale per output channel, or per group of 128 weights, follows the local range. Real GPT-2, all linear layers quantized and rerun.', 'per tensor · per channel · per group of 128'],
  outliers: ['Activations are harder than weights: a few dimensions are far larger than the rest. In this real GPT-2 activation one number is 30 times the typical size, and with one int4 scale most values round to zero. LLM.int8 keeps such dimensions in 16 bits; most 4-bit serving quantizes only the weights.', 'one large value stretches the whole grid'],
  effect: ['The real cost for GPT-2 small. int8 with a scale per channel is nearly free; int4 in groups of 128 costs a little; int4 with a single scale per matrix breaks the model. Methods like GPTQ and AWQ choose the rounding more cleverly and keep 4-bit models close to full precision.', 'perplexity on a paragraph · next-token guesses'],
}

export function mountQuantization(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: '',
      eyebrow: 'Serving · Memory',
      title: 'Quantization',
      subtitle: 'weights in 8 and 4 bits, on the real GPT-2',
      specs: [
        { label: 'model', value: 'GPT-2 small', real: '124M weights', realLabel: '' },
        { label: 'fp32', value: '498 MB' },
        { label: 'int8', value: '124 MB' },
        { label: 'int4', value: '62 MB', real: '+ scales', realLabel: '' },
        { label: 'perplexity', value: `${RES('fp32').ppl} → ${RES('int4 · groups of 128').ppl}`, real: 'fp32 → int4 g128', realLabel: '' },
      ],
    },
    size: [1040, 480],
    aria: 'Quantization: storing a model’s weights in 8 or 4 bits by rounding them onto a grid with a scale; real GPT-2 weights, activations and results show that int8 is nearly free, int4 needs small groups, and a few large activations make activations hard to quantize.',
    phases: PHASES, learn: 'quantization', tokens: TOKS, compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['round', 10],
    hints: {
      round: 'Hover a weight to see its code and rounding error; click or tap to pin it.',
    },
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { caption, title } = k
  const pad = 36, top = 56, bot = 46
  const BLUE: RGB = C.tok[0], ORANGE: RGB = C.tok[3]

  /* ---------- scene 1: formats ---------- */
  function sceneFormats(p: number) {
    const { W } = stage
    const rows: [string, number[], string, string][] = [
      ['fp32', [1, 8, 23], 'sign · exponent · fraction', 'training'],
      ['bf16', [1, 8, 7], 'fp32’s range, less precision', 'training, serving'],
      ['fp16', [1, 5, 10], 'more precision, less range', 'serving'],
      ['fp8 E4M3', [1, 4, 3], 'H100 and later', 'serving'],
      ['int8', [8], '255 evenly spaced levels', 'weights, activations'],
      ['int4', [4], '15 levels', 'weights'],
    ]
    const x0 = pad + 110, bw = 12, y0 = top + 30, rh = 52
    rows.forEach(([name, parts, note], r) => {
      const a = eout(clamp((p - 0.03 - r * 0.07) / 0.1)), y = y0 + r * rh
      if (a <= 0) return
      ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(name, pad, y + 10)
      let x = x0
      const cols: RGB[] = parts.length === 3 ? [C.ink, BLUE, ORANGE] : [C.tok[4]]
      parts.forEach((n, i) => {
        for (let b = 0; b < n; b++) { rr(x + b * (bw - 2), y, bw - 4, 20, 2); ctx.fillStyle = rgba(cols[i], (parts.length === 3 && i === 0 ? 0.5 : 0.75) * a); ctx.fill() }
        x += n * (bw - 2) + 8
      })
      caption(note, x0 + 32 * (bw - 2) + 30, y + 14, a, C.ink2, 'left')
    })
    const sa = eout(clamp((p - 0.55) / 0.12)), sx = W - pad - 250
    if (sa > 0) {
      title('llama 3 8b weights', sx, y0 + 4, sa)
      ;[['fp32', 32], ['16-bit', 16], ['int8', 8], ['int4', 4]].forEach(([n, bits], i) => {
        const gb = (8.03 * (bits as number)) / 8, y = y0 + 30 + i * 40
        ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink2, sa); ctx.fillText(n as string, sx, y + 12)
        ctx.fillStyle = rgba(C.ink, 0.7 * sa); ctx.fillRect(sx + 70, y, (gb / 32.1) * 120, 16)
        ctx.fillStyle = rgba(C.ink, sa); ctx.fillText(`${gb.toFixed(0)} GB`, sx + 76 + (gb / 32.1) * 120, y + 12)
      })
      caption('at 4 bits it fits a 24 GB gaming card', sx, y0 + 200, sa, C.mute, 'left')
    }
    mk.formula = { segs: [['bytes = weights × bits / 8', C.ink], ['     ·     ', C.mute], ['8.03B × 4 / 8 = 4.0 GB, plus about 3% for int4’s scales', C.ink2]], note: 'Floating-point formats spend bits on an exponent, so they cover a wide range with relative precision. Integer formats are evenly spaced, so they need a scale that maps the weights onto their range; that scale is stored alongside.' }
  }

  /* ---------- scene 2: absmax rounding of a real weight column ---------- */
  function sceneRound(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const max = Math.max(...COL.map(Math.abs)), vals = COL.slice(0, SHOW)
    const x0 = pad + 70, x1 = W - pad - 260, cw = (x1 - x0) / SHOW, mid = top + avail * 0.45, hs = (avail * 0.36) / 0.2
    const Y = (v: number) => mid - v * hs
    const bits = p < 0.55 ? 8 : 4, q = levels(bits), sc = max / q, a8 = eout(clamp((p - 0.12) / 0.12)), a4 = eout(clamp((p - 0.55) / 0.1))
    title(`block 6 mlp · w_fc column 0 · first ${SHOW} of 768 weights`, x0, top + 14, 1)
    ctx.strokeStyle = rgba(C.ink, 0.25); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, mid); ctx.lineTo(x1, mid); ctx.stroke()
    // the grid of levels
    const ga = bits === 8 ? a8 : a4
    for (let l = -q; l <= q; l++) {
      const y = Y(l * sc)
      if (y < top + 20 || y > H - bot) continue
      ctx.strokeStyle = rgba(bits === 8 ? BLUE : ORANGE, (bits === 8 ? 0.12 : 0.35) * ga); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke()
    }
    let errSum = 0
    vals.forEach((v, i) => {
      const x = x0 + (i + 0.5) * cw, dq = qz(v, sc, q) * sc
      ctx.fillStyle = rgba(C.ink, 0.35); ctx.fillRect(x - cw * 0.3, Math.min(mid, Y(v)), cw * 0.6, Math.abs(Y(v) - mid))
      if (ga > 0) {
        ctx.fillStyle = rgba(bits === 8 ? BLUE : ORANGE, ga); ctx.beginPath(); ctx.arc(x, Y(dq), 3.5, 0, 7); ctx.fill()
        ctx.strokeStyle = rgba(bits === 8 ? BLUE : ORANGE, 0.7 * ga); ctx.beginPath(); ctx.moveTo(x, Y(v)); ctx.lineTo(x, Y(dq)); ctx.stroke()
      }
      errSum += (dq - v) ** 2
    })
    mk.hit('w', { x: x0, y: top + 20, c: cw }, Math.ceil((avail - 40) / cw), SHOW)
    // the numbers
    const rx = x1 + 30
    ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(bits === 8 ? BLUE : ORANGE, 1)
    ctx.fillText(`int${bits}`, rx, top + 40)
    ctx.font = F.mono(12); ctx.fillStyle = rgba(C.ink2, 1)
    ctx.fillText(`max |w| = ${fmtF(max)}`, rx, top + 66)
    ctx.fillText(`scale = ${fmtF(max)} / ${q} = ${sc.toExponential(2)}`, rx, top + 88)
    ctx.fillText(`levels: ${2 * q + 1}`, rx, top + 110)
    const rms = Math.sqrt(errSum / SHOW), wrms = Math.sqrt(vals.reduce((s, v) => s + v * v, 0) / SHOW)
    ctx.fillStyle = rgba(C.ink, 1); ctx.fillText(`rounding error: ${Math.round((rms / wrms) * 100)}%`, rx, top + 140)
    caption('of the weights’ typical size', rx, top + 158, 1, C.mute, 'left')
    caption('bars: the real weights', rx, top + 196, 1, C.mute, 'left')
    caption('dots: after rounding', rx, top + 212, 1, C.mute, 'left')
    const f = mk.focus?.key === 'w' ? mk.focus : null
    if (f && f.j < SHOW) {
      const v = vals[f.j], code = qz(v, sc, q)
      mk.formula = { segs: [[`w = ${fmtF(v)}`, C.ink], ['  →  ', C.mute], [`round(${fmtF(v)} / ${sc.toExponential(2)}) = ${code}`, C.ink2], ['  →  ', C.mute], [`${code} × scale = ${fmtF(code * sc)}`, C.ink]], note: `Stored as the ${bits}-bit integer ${code}; the error is ${fmtF(code * sc - v)}.` }
    } else mk.formula = { segs: [['code = round(w / s)', C.ink2], ['   ·   ', C.mute], ['w ≈ code × s', C.ink2], ['   ·   ', C.mute], [`s = max|w| / ${q}`, C.ink]], note: 'The scale is set by the largest weight of the whole column (768 weights), not just the ones drawn. With 4 bits the grid is so coarse that many weights share a level or round to zero.' }
  }

  /* ---------- scene 3: granularity ---------- */
  function sceneScales(p: number) {
    // schematic: a matrix with its scales
    const mx = pad + 20, my = top + 50, cs = 9, rows = 16, cols = 16
    const schemes: [string, string, (r: number, c: number) => number][] = [['per tensor', '1 scale', () => 0], ['per channel', '1 per column', (_, c) => c], ['groups of 128', '1 per column per 128 rows', (r, c) => c * 2 + (r >= 8 ? 1 : 0)]]
    schemes.forEach(([name, sub, id], s) => {
      const a = eout(clamp((p - 0.04 - s * 0.1) / 0.1)), x = mx + s * (cols * cs + 40)
      if (a <= 0) return
      title(name, x, my - 16, a)
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const g = id(r, c), col = C.tok[g % 7]
        ctx.fillStyle = rgba(col, (s === 0 ? 0.35 : 0.2 + 0.25 * ((g % 3) / 2)) * a); ctx.fillRect(x + c * cs, my + r * cs, cs - 1, cs - 1)
      }
      caption(sub, x, my + rows * cs + 18, a, C.mute, 'left')
    })
    // real results
    const tx = mx, ty = my + rows * cs + 62, rs = ['int8 · per tensor', 'int8 · per channel', 'int4 · per tensor', 'int4 · per channel', 'int4 · groups of 128']
    const ra = eout(clamp((p - 0.4) / 0.1))
    if (ra > 0) {
      title('real gpt-2 · all linear weights quantized', tx, ty - 12, ra)
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, ra)
      ctx.fillText('weight error', tx + 200, ty + 10); ctx.fillText(`perplexity (fp32: ${RES('fp32').ppl})`, tx + 400, ty + 10)
      rs.forEach((n, i) => {
        const r = RES(n), y = ty + 32 + i * 24, a = ra * eout(clamp((p - 0.45 - i * 0.05) / 0.1))
        ctx.font = F.mono(12, r.name.startsWith('int4 · per tensor') ? 500 : 400); ctx.fillStyle = rgba(C.ink, a); ctx.fillText(n, tx, y)
        ctx.fillStyle = rgba(r.bits === 8 ? BLUE : ORANGE, 0.8 * a); ctx.fillRect(tx + 200, y - 11, Math.min(160, r.relErr * 200), 13)
        ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(`${Math.round(r.relErr * 100)}%`, tx + 206 + Math.min(160, r.relErr * 200), y)
        ctx.fillStyle = rgba(r.ppl > 1000 ? C.ink : C.ink2, a); ctx.fillText(r.ppl.toLocaleString('en-US'), tx + 400, y)
      })
    }
    mk.formula = { segs: [['more scales → finer grids where the weights are small', C.ink], ['     ·     ', C.mute], ['cost: one 16-bit scale per 128 weights = 0.125 extra bits each', C.ink2]], note: `Perplexity on a ${quant.tokens}-token paragraph (lower is better). A weight-only int4 model with groups of 128 is the common format for running models locally (GPTQ, AWQ and llama.cpp’s Q4 formats all use small groups).` }
  }

  /* ---------- scene 4: activation outliers ---------- */
  function sceneOutliers(p: number) {
    const { W } = stage
    const n = ACT.length, max = Math.max(...ACT.map(Math.abs)), x0 = pad + 20, x1 = W - pad - 20, cw = (x1 - x0) / n, mid = top + 130, hs = 90 / max
    const med = [...ACT].map(Math.abs).sort((a, b) => a - b)[n >> 1]
    title('block 6 · input to the mlp · last token of “the cat sat on the” · 768 numbers', x0, top + 14, 1)
    ctx.strokeStyle = rgba(C.ink, 0.25); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, mid); ctx.lineTo(x1, mid); ctx.stroke()
    ACT.forEach((v, i) => { const big = Math.abs(v) > 10 * med; ctx.fillStyle = rgba(big ? ORANGE : C.ink, big ? 0.95 : 0.45); ctx.fillRect(x0 + i * cw, Math.min(mid, mid - v * hs), Math.max(1, cw), Math.abs(v * hs)) })
    const top4 = [...ACT.keys()].sort((a, b) => Math.abs(ACT[b]) - Math.abs(ACT[a])).slice(0, 2)
    top4.forEach((d) => caption(`dim ${d}: ${fmt(ACT[d])}`, x0 + d * cw + 6, mid - ACT[d] * hs + (ACT[d] > 0 ? -6 : 16), 1, C.ink, 'left'))
    caption(`typical size (median |x|): ${fmtF(med)} · largest: ${fmt(max)}, ${Math.round(max / med)} × larger`, x0, mid + 110, 1, C.ink2, 'left')
    // what one scale does to the rest
    const ra = eout(clamp((p - 0.35) / 0.12))
    if (ra > 0) {
      const y = mid + 150
      ;[8, 4].forEach((bits, i) => {
        const q = levels(bits), sc = max / q, zero = ACT.filter((v) => qz(v, sc, q) === 0).length, a = ra * eout(clamp((p - 0.35 - i * 0.15) / 0.12))
        ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(bits === 8 ? BLUE : ORANGE, a)
        ctx.fillText(`int${bits}, one scale:`, x0, y + i * 34)
        ctx.fillStyle = rgba(C.ink, a); ctx.fillText(`${zero} of ${n} values round to 0 (${Math.round((zero / n) * 100)}%)`, x0 + 190, y + i * 34)
      })
      const wo = ACT.filter((_, i) => !top4.includes(i)), m2 = Math.max(...wo.map(Math.abs)), z2 = wo.filter((v) => qz(v, m2 / 7, 7) === 0).length
      const la = eout(clamp((p - 0.7) / 0.12))
      ctx.font = F.mono(13, 500); ctx.fillStyle = rgba(C.tok[4], la); ctx.fillText('int4, the 2 outliers kept in 16-bit:', x0, y + 68)
      ctx.fillStyle = rgba(C.ink, la); ctx.fillText(`${z2} of ${wo.length} round to 0 (${Math.round((z2 / wo.length) * 100)}%)`, x0 + 330, y + 68)
    }
    mk.formula = { segs: [['one scale per vector: s = max|x| / q', C.ink2], ['     ·     ', C.mute], ['a single large x makes s large for everyone', C.ink]], note: 'Real GPT-2 values. Larger models have far stronger outliers (LLM.int8 found them in every layer beyond about 6.7B parameters), which is why most 4-bit serving quantizes weights only and keeps activations in 16 bits, and why methods like SmoothQuant shift the range from activations into the weights.' }
  }

  /* ---------- scene 5: the effect on the model ---------- */
  function sceneEffect(p: number) {
    const { W } = stage
    const shown = ['fp32', 'int8 · per channel', 'int4 · groups of 128', 'int4 · per tensor'], cw = (W - 2 * pad) / shown.length
    title(`gpt-2 small · next token after “${quant.prompt}” · perplexity on a ${quant.tokens}-token paragraph`, pad, top + 14, 1)
    shown.forEach((n, i) => {
      const r = RES(n), a = eout(clamp((p - 0.05 - i * 0.14) / 0.12)), x = pad + i * cw + 6
      if (a <= 0) return
      ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(r.bits === 32 ? C.ink : r.bits === 8 ? BLUE : ORANGE, a); ctx.fillText(n, x, top + 50)
      ctx.font = serifAt(20); ctx.fillStyle = rgba(C.ink, a); ctx.fillText(`${r.ppl.toLocaleString('en-US')}`, x, top + 82)
      caption('perplexity', x + ctx.measureText(`${r.ppl.toLocaleString('en-US')}`).width + 10, top + 80, a, C.mute, 'left')
      r.top.forEach((t, j) => {
        const y = top + 112 + j * 28
        ctx.font = F.mono(12, j === 0 ? 500 : 400); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(j === 0 ? C.ink : C.ink2, a); ctx.fillText(t.s, x, y)
        ctx.fillStyle = rgba(C.ink, 0.6 * a); ctx.fillRect(x + 84, y - 6, Math.max(1.5, t.p * 700), 12)
        ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(`${(t.p * 100).toFixed(1)}%`, x + 90 + t.p * 700, y)
      })
      caption(r.bits === 32 ? 'the reference' : `KL from fp32: ${r.kl < 0.01 ? r.kl.toFixed(4) : fmt(r.kl)}`, x, top + 112 + 6 * 28 + 10, a, C.ink2, 'left')
      caption(`${r.bits === 32 ? 498 : r.bits === 8 ? 124 : 62} MB of weights`, x, top + 112 + 6 * 28 + 28, a, C.mute, 'left')
    })
    mk.formula = { segs: [['perplexity = e^(mean −log p(next token))', C.ink2], ['     ·     ', C.mute], ['KL = Σ p_fp32 log(p_fp32 / p_quantized)', C.ink2]], note: 'Round-to-nearest with absmax scales, the simplest method. GPTQ adjusts the remaining weights to compensate for each rounding error, and AWQ scales up the few weight channels that matter most before rounding; both keep 4-bit models much closer to full precision.' }
  }

  return { formats: sceneFormats, round: sceneRound, scales: sceneScales, outliers: sceneOutliers, effect: sceneEffect }
}
