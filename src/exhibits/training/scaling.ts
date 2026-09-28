import { F, ctx } from '../../core/draw'
import { C, rgba } from '../../core/theme'
import { clamp, eout, lerp } from '../../core/util'
import { Gpt2Bpe } from '../../lib/gpt2/bpe'
import { scaling as S } from '../../lib/scaling/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * Scaling laws: loss falls smoothly, as a power law, with model size, data and compute. A real three-point experiment
 * (GPT-2 small, medium and large on text none of them saw), then the Chinchilla fit L(N, D) with its published
 * constants, compute-optimal sizes for a budget, and where real models sit.
 */

const PHASES = [
  { id: 'size', name: 'Bigger model, lower loss', short: 'Size', dur: 10 },
  { id: 'tokens', name: 'Which tokens got easier', short: 'Per token', dur: 9 },
  { id: 'fit', name: 'Parameters and data together', short: 'L(N, D)', dur: 10 },
  { id: 'compute', name: 'Spending a compute budget', short: 'Compute', dur: 11 },
  { id: 'models', name: 'Where real models sit', short: 'Models', dur: 10 },
]

const Mo = S.models
/** Least-squares slope of log L against log N over the three GPT-2 sizes: L ∝ N^−a. */
const fitA = (() => {
  const xs = Mo.map((m) => Math.log(m.nonEmbedding)), ys = Mo.map((m) => Math.log(m.loss)), mx = xs.reduce((a, b) => a + b) / 3, my = ys.reduce((a, b) => a + b) / 3
  const b = xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0) / xs.reduce((a, x) => a + (x - mx) ** 2, 0)
  return { a: -b, c: Math.exp(my - b * mx) }
})()
/* Chinchilla's parametric fit (Hoffmann et al. 2022, approach 3), loss in nats per token on their data */
const E = 1.69, A = 406.4, B = 410.7, AL = 0.34, BE = 0.28
const Lnd = (N: number, D: number) => E + A / N ** AL + B / D ** BE
/** The best N for a budget of C FLOPs (C ≈ 6 N D), by searching log N. */
function optimal(C: number) {
  let best = { N: 0, D: 0, L: Infinity }
  for (let e = 7; e <= 13; e += 0.005) { const N = 10 ** e, D = C / (6 * N), l = Lnd(N, D); if (l < best.L) best = { N, D, L: l } }
  return best
}
const BUDGETS = [1e21, 1e22, 1e23, 1e24]
/** Published sizes and training tokens. */
const REAL: [string, number, number][] = [['GPT-3', 175e9, 300e9], ['Gopher', 280e9, 300e9], ['Chinchilla', 70e9, 1.4e12], ['LLaMA 7B', 6.7e9, 1e12], ['LLaMA 2 7B', 6.7e9, 2e12], ['LLaMA 3 8B', 8e9, 15e12], ['LLaMA 3 70B', 70e9, 15e12]]
const big = (v: number) => (v >= 1e12 ? `${+(v / 1e12).toFixed(1)}T` : v >= 1e9 ? `${+(v / 1e9).toFixed(1)}B` : `${+(v / 1e6).toFixed(0)}M`)
const sci = (v: number) => { const e = Math.floor(Math.log10(v)); return `10${[...String(e)].map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('')}` }

let bpe: Gpt2Bpe | null = null
let loading: Promise<void> | null = null
const loadBpe = () => (loading ??= import('../../lib/gpt2/merges.txt?raw').then((m) => { bpe = new Gpt2Bpe(m.default) }))

const COMPARE: Record<string, [string, string]> = {
  size: ['the loss being measured', 'training/loss?phase=loss'],
  tokens: ['per-position loss', 'training/loss?phase=predict'],
  fit: ['LLaMA 3 8B’s shape', 'lineage/llama'],
  compute: ['FLOPs of a forward pass', 'serving/kv-cache?phase=bound'],
  models: ['LLaMA, drawn as a diff to GPT-2', 'lineage/llama'],
}

const CAPS: Record<string, [string, string]> = {
  size: [`GPT-2 small, medium and large were trained the same way on the same data. On ${S.tokens} tokens of this site’s glossary, which none of them saw, their loss falls ${Mo.map((m) => m.loss.toFixed(2)).join(' → ')} as size grows 8×. On a log scale the three points sit close to a straight line: a power law.`, `L ∝ N^−${fitA.a.toFixed(3)} on these three`],
  tokens: [`The average hides where the gain comes from. Most tokens change little; a few that need knowledge or longer context get much easier for the large model.`, 'loss(small) − loss(large), per token'],
  fit: [`Loss depends on both the number of parameters N and training tokens D. Chinchilla (Hoffmann et al. 2022) fit L = E + A/N^α + B/D^β to hundreds of runs. Each term shrinks as its budget grows, down to E, the loss no model removes.`, 'L(N, D) = 1.69 + 406/N^0.34 + 411/D^0.28'],
  compute: [`Training compute is about 6 N D FLOPs. For a fixed budget a bigger model sees fewer tokens, and loss is lowest in between. By the fit, the best size grows about as √C, with roughly 20 tokens per parameter.`, 'C ≈ 6 N D · best near D ≈ 20 N'],
  models: [`GPT-3 was large and under-trained by that rule. Chinchilla, at 70B, beat the 280B Gopher on the same budget. LLaMA 3 8B saw 15T tokens, about 1,900 per parameter: past the compute-optimal point on purpose, since a small model is cheaper to serve.`, 'tokens per parameter: 1.7 → 20 → 1,900'],
}

export function mountScaling(root: HTMLElement, nav: Nav): () => void {
  loadBpe()
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'The three GPT-2 losses are real runs; the Chinchilla curves use the published fit and published model sizes.',
      eyebrow: 'Training',
      title: 'Scaling laws',
      subtitle: 'loss against parameters, data and compute',
      specs: [
        { label: 'measured', value: 'GPT-2 small, medium, large', real: `${S.tokens} tokens`, realLabel: '·' },
        { label: 'fit', value: 'Chinchilla', real: 'Hoffmann et al. 2022', realLabel: '·' },
        { label: 'compute', value: 'C ≈ 6 N D' },
      ],
    },
    size: [1040, 480],
    aria: 'Scaling laws: GPT-2 small, medium and large lose less on the same unseen text as they grow, close to a power law; Chinchilla’s fit of loss against parameters and training tokens; the best model size for a compute budget; and where real models sit against the 20 tokens per parameter rule.',
    phases: PHASES, learn: 'scaling', tokens: words(['N', 'D']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['compute', 10],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption } = k
  const pad = 36, top = 56
  /** Log–log axes; returns the maps. */
  function axes(x0: number, x1: number, y0: number, y1: number, xr: [number, number], yr: [number, number], ylog: boolean, a: number, xl: string, yl: string) {
    const X = (v: number) => lerp(x0, x1, (Math.log10(v) - xr[0]) / (xr[1] - xr[0])), Y = (v: number) => lerp(y1, y0, ((ylog ? Math.log10(v) : v) - yr[0]) / (yr[1] - yr[0]))
    ctx.strokeStyle = rgba(C.faint, a); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, a); ctx.textAlign = 'center'; ctx.textBaseline = 'top'
    for (let e = Math.ceil(xr[0]); e <= xr[1]; e++) ctx.fillText(sci(10 ** e), lerp(x0, x1, (e - xr[0]) / (xr[1] - xr[0])), y1 + 6)
    ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic'; ctx.fillText(xl, x1, y1 + 34)
    ctx.textAlign = 'left'; ctx.fillText(yl, x0 - 30, y0 - 10)
    return { X, Y }
  }

  /* ---------- 1: three sizes ---------- */
  function sceneSize(p: number) {
    const x0 = pad + 60, x1 = pad + 560, y0 = top + 40, y1 = stage.H - 90
    const { X, Y } = axes(x0, x1, y0, y1, [7.5, 9.5], [3.8, 4.5], false, 1, 'non-embedding parameters N (log)', 'loss on the unseen text')
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    for (const v of [3.8, 4.0, 4.2, 4.4]) ctx.fillText(v.toFixed(1), x0 - 6, Y(v))
    const la = eout(clamp((p - 0.5) / 0.15))
    ctx.strokeStyle = rgba(C.ink, 0.5 * la); ctx.setLineDash([4, 4]); ctx.beginPath()
    for (let e = 7.6; e <= 9.4; e += 0.05) { const N = 10 ** e, y = Y(fitA.c * N ** -fitA.a); if (e === 7.6) ctx.moveTo(X(N), y); else ctx.lineTo(X(N), y) }
    ctx.stroke(); ctx.setLineDash([])
    Mo.forEach((m, i) => {
      const a = eout(clamp((p - 0.05 - i * 0.12) / 0.1))
      ctx.fillStyle = rgba(C.tok[i], a); ctx.beginPath(); ctx.arc(X(m.nonEmbedding), Y(m.loss), 6, 0, 7); ctx.fill()
      caption(`${m.name} · ${big(m.params)} · ${m.loss.toFixed(3)}`, X(m.nonEmbedding) + 10, Y(m.loss) - 10, a, C.ink, 'left')
    })
    // the three models' shapes
    const xr = x1 + 90
    title('one recipe, three sizes', xr, y0 - 10, 1)
    Mo.forEach((m, i) => {
      ctx.font = F.mono(12, 600); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.tok[i], 1); ctx.fillText(m.name, xr, y0 + 22 + i * 58)
      ctx.font = F.mono(11); ctx.fillStyle = rgba(C.ink2, 1); ctx.fillText(`${m.layers} layers × ${m.width} wide`, xr, y0 + 40 + i * 58)
    })
    mk.formula = { segs: [['L(N)', C.ink], [' ≈ ', C.mute], [`${fitA.c.toFixed(2)} · N^−${fitA.a.toFixed(4)}`, C.ink2]], note: 'A straight line through three points proves little by itself; Kaplan et al. (2020) and Hoffmann et al. (2022) saw the same shape across hundreds of runs spanning several orders of magnitude.' }
  }

  /* ---------- 2: per token ---------- */
  function sceneTokens(p: number) {
    const { W } = stage, sm = Mo[0].losses, lg = Mo[2].losses, gain = sm.map((v, i) => v - lg[i]), y0 = top + 40
    if (!bpe) { caption('loading the tokenizer …', pad, y0, 1, C.mute, 'left'); return }
    const syms = bpe.encode(S.text).map((id) => bpe!.symbolOf(id)).slice(1)
    title('the first sentences · shade: how much easier each token got for GPT-2 large', pad, y0 - 16, 1)
    let x = pad, y = y0 + 10
    const mx = Math.max(...gain), a = eout(clamp(p / 0.2))
    for (let i = 0; i < syms.length; i++) {
      const w = Math.min(90, 10 + syms[i].length * 7.2)
      if (x + w > W - pad) { x = pad; y += 28 }
      if (y > stage.H - 150) break
      const g = gain[i] / mx
      ctx.fillStyle = rgba(g >= 0 ? C.tok[0] : C.ink, (g >= 0 ? 0.08 + 0.8 * g : 0.08) * a); ctx.fillRect(x, y - 11, w - 3, 22)
      if (g < 0) { ctx.strokeStyle = rgba(C.ink, 0.5 * a); ctx.strokeRect(x + 0.5, y - 10.5, w - 4, 21) }
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(syms[i].replace(/^Ġ/, ' ').replace('Ċ', '↵'), x + 3, y)
      x += w
    }
    const order = [...gain.keys()].sort((q, r) => gain[r] - gain[q]).slice(0, 3), ta = eout(clamp((p - 0.5) / 0.12))
    order.forEach((i, j) => caption(`“${syms[i].replace(/^Ġ/, ' ').trim()}”: ${sm[i].toFixed(2)} → ${lg[i].toFixed(2)}`, pad + j * 260, stage.H - 110, ta, C.ink2, 'left'))
    const worse = gain.filter((g) => g < 0).length
    mk.formula = { segs: [['mean gain', C.ink2], [' = ', C.mute], [`${(Mo[0].loss - Mo[2].loss).toFixed(3)} nats per token`, C.ink], ['   ·   ', C.mute], [`${worse} of ${gain.length} tokens got harder`, C.ink2]], note: 'Outlined: tokens where the large model did worse. Scaling laws are about the average; any single prediction can go either way.' }
  }

  /* ---------- 3: L(N, D) ---------- */
  function sceneFit(p: number) {
    const x0 = pad + 60, x1 = pad + 600, y0 = top + 40, y1 = stage.H - 90
    const { X, Y } = axes(x0, x1, y0, y1, [9, 13], [1.8, 3.4], false, 1, 'training tokens D (log)', 'loss (Chinchilla fit)')
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    for (const v of [2, 2.5, 3]) ctx.fillText(v.toFixed(1), x0 - 6, Y(v))
    const Ns = [1e8, 1e9, 1e10, 7e10, 1e12]
    Ns.forEach((N, i) => {
      const a = eout(clamp((p - 0.05 - i * 0.12) / 0.12))
      ctx.strokeStyle = rgba(C.tok[i], a); ctx.lineWidth = 1.8; ctx.beginPath()
      for (let e = 9; e <= 13; e += 0.05) { const y = Y(Math.min(3.4, Lnd(N, 10 ** e))); if (e === 9) ctx.moveTo(X(10 ** e), y); else ctx.lineTo(X(10 ** e), y) }
      ctx.stroke()
      caption(`N = ${big(N)}`, x1 + 8, Y(Lnd(N, 1e13)) + 4, a, C.tok[i], 'left')
    })
    ctx.strokeStyle = rgba(C.ink, 0.4); ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(x0, Y(E)); ctx.lineTo(x1, Y(E)); ctx.stroke(); ctx.setLineDash([])
    caption('E = 1.69: what no model removes', x0 + 8, Y(E) - 6, 1, C.mute, 'left')
    mk.formula = { segs: [['L(N, D)', C.ink], [' = ', C.mute], ['E', C.ink2], [' + ', C.mute], ['A / N^α', C.ink2], [' + ', C.mute], ['B / D^β', C.ink2]], note: 'E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28 (Hoffmann et al. 2022). The numbers are in nats per token on their data, so they do not compare directly with the GPT-2 losses above.' }
  }

  /* ---------- 4: a compute budget ---------- */
  function sceneCompute(p: number) {
    const x0 = pad + 60, x1 = pad + 600, y0 = top + 40, y1 = stage.H - 90
    const { X, Y } = axes(x0, x1, y0, y1, [8, 12], [1.9, 3.1], false, 1, 'parameters N (log)', 'loss at fixed compute')
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    for (const v of [2, 2.5, 3]) ctx.fillText(v.toFixed(1), x0 - 6, Y(v))
    BUDGETS.forEach((Cb, i) => {
      const a = eout(clamp((p - 0.05 - i * 0.12) / 0.12)), o = optimal(Cb)
      ctx.strokeStyle = rgba(C.tok[i], a); ctx.lineWidth = 1.8; ctx.beginPath()
      let started = false
      for (let e = 8; e <= 12; e += 0.02) { const N = 10 ** e, l = Lnd(N, Cb / (6 * N)); if (l > 3.1) { started = false; continue } if (!started) { ctx.moveTo(X(N), Y(l)); started = true } else ctx.lineTo(X(N), Y(l)) }
      ctx.stroke()
      ctx.fillStyle = rgba(C.tok[i], a); ctx.beginPath(); ctx.arc(X(o.N), Y(o.L), 4.5, 0, 7); ctx.fill()
      caption(`C = ${sci(Cb)}: ${big(o.N)} on ${big(o.D)} tokens`, x1 + 18, y0 + 20 + i * 26, a, C.tok[i], 'left')
    })
    const o = optimal(1e23)
    mk.formula = { segs: [['C = 10²³', C.ink2], ['  →  ', C.mute], [`N = ${big(o.N)}, D = ${big(o.D)}`, C.ink], ['  ·  ', C.mute], [`${(o.D / o.N).toFixed(0)} tokens per parameter`, C.ink2]], note: 'The dot on each curve is the lowest loss that budget can buy under the fit. Training longer than that on a smaller model costs some loss but gives a model that is cheaper to run.' }
  }

  /* ---------- 5: real models ---------- */
  function sceneModels(p: number) {
    const x0 = pad + 60, x1 = pad + 640, y0 = top + 40, y1 = stage.H - 90
    const { X, Y } = axes(x0, x1, y0, y1, [9, 12], [11, 13.5], true, 1, 'parameters N (log)', 'training tokens D (log)')
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    for (const e of [11, 12, 13]) ctx.fillText(sci(10 ** e), x0 - 6, Y(10 ** e))
    const la = eout(clamp(p / 0.15))
    ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip()
    ctx.strokeStyle = rgba(C.ink, 0.5 * la); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(1e9), Y(2e10)); ctx.lineTo(X(1e12), Y(2e13)); ctx.stroke(); ctx.setLineDash([])
    ctx.restore()
    caption('D = 20 N', X(3e11), Y(6e12) - 10, la, C.ink2, 'left')
    REAL.forEach(([name, N, D], i) => {
      const a = eout(clamp((p - 0.15 - i * 0.08) / 0.1))
      ctx.fillStyle = rgba(C.tok[i % 7], a); ctx.beginPath(); ctx.arc(X(N), Y(D), 5.5, 0, 7); ctx.fill()
      caption(`${name} · ${big(N)} / ${big(D)} · ${(D / N).toFixed(D / N < 10 ? 1 : 0)} per param`, X(N) + 9, Y(D) + (i % 2 ? 14 : -8), a, C.ink, 'left')
    })
    mk.formula = { segs: [['above the line', C.ink2], [': more data per parameter than compute-optimal  ·  ', C.mute], ['below', C.ink2], [': under-trained', C.mute]], note: 'Sizes and token counts as published: GPT-3 (Brown et al. 2020), Gopher (Rae et al. 2021), Chinchilla (Hoffmann et al. 2022), LLaMA (Touvron et al. 2023), Llama 2 (2023), Llama 3 (2024).' }
  }

  return { size: sceneSize, tokens: sceneTokens, fit: sceneFit, compute: sceneCompute, models: sceneModels }
}
