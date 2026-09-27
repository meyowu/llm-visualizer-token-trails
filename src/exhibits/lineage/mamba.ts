import { F, chipW, drawChip, rr, tokLabel, type TokLike } from '../../core/draw'
import { fmt, fmtF } from '../../core/matrix'
import { C, blend, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, gauss, lerp, rng } from '../../core/util'
import { nextDist, presets } from '../../lib/gpt2/data'
import { mamba } from '../../lib/mamba/data'
import type { Nav } from '../registry'
import { mountExhibit, type Env } from '../kit'

/*
 * Mamba as a diff against GPT-2: no attention, one mixer per block built around a selective
 * state-space scan, a fixed-size state instead of a KV cache. The step sizes Δ and the next-token
 * guesses are real Mamba-130m (exported offline); the scan drawn cell by cell is one toy channel.
 */

const PHASES = [
  { id: 'blocks', name: 'GPT-2 vs Mamba', short: 'Blocks', dur: 7 },
  { id: 'cost', name: 'A running state instead of a cache', short: 'Memory', dur: 10 },
  { id: 'scan', name: 'The scan', short: 'Scan', dur: 10 },
  { id: 'select', name: 'Selectivity', short: 'Δ', dur: 12 },
  { id: 'predict', name: 'Same size, same job', short: 'Predict', dur: 8 },
]

const T = 8
const TOKS: TokLike[] = mamba.toks.slice(0, T).map((t, i) => ({ text: t.replace(/^Ġ/, ' '), c: i }))
/** One toy channel: a scalar input per token and a state of 4 with decay rates like Mamba's initial ones. */
const NS = 4, A = [-1, -4, -8, -16], B = [1, 0.8, 0.6, 0.4], Cv = [0.5, 0.5, 0.5, 0.5]
const X = (() => { const r = rng(8); return TOKS.map(() => 0.4 + Math.abs(gauss(r))) })()
/** Layer 11's real mean Δ for the first 8 tokens, and a fixed Δ for comparison. */
const LAYER = 10
const DSEL = mamba.meanDelta[LAYER].slice(0, T), DFIX = DSEL.reduce((a, b) => a + b, 0) / T
function scan(delta: number[]) {
  const hs: number[][] = [], ys: number[] = []
  /** The scan is linear, so each state number splits into what each earlier token wrote: parts[t][n][s]. */
  const parts: number[][][] = []
  let h = Array(NS).fill(0), part = Array.from({ length: NS }, () => Array(T).fill(0))
  for (let t = 0; t < T; t++) {
    h = h.map((v, n) => Math.exp(delta[t] * A[n]) * v + delta[t] * B[n] * X[t])
    part = part.map((row, n) => row.map((v, s2) => Math.exp(delta[t] * A[n]) * v + (s2 === t ? delta[t] * B[n] * X[t] : 0)))
    hs.push(h); parts.push(part); ys.push(h.reduce((s, v, n) => s + v * Cv[n], 0))
  }
  return { hs, ys, parts }
}
const FIXED = scan(DSEL.map(() => DFIX)), SEL = scan(DSEL)

/** Memory to generate the next token, in bytes (16-bit numbers). */
const KV_PER_TOKEN = 2 * 12 * 768 * 2, STATE = 24 * 1536 * (16 + 3) * 2

const COMPARE: Record<string, [string, string]> = {
  blocks: ['GPT-2’s block', 'anatomy/layernorm?phase=stream'], cost: ['LLaMA’s KV cache', 'lineage/llama?phase=gqa'], scan: ['the 2017 RNN step', 'lineage/transformer-2017?phase=rnn'],
  select: ['GPT-2’s attention', 'anatomy/attention'], predict: ['GPT-2’s prediction', 'anatomy?phase=pick'],
}

const CAPS: Record<string, [string, string]> = {
  blocks: ['Mamba-130m is GPT-2 small’s size, but it has no attention. Each of its 24 blocks is one mixer: a widening projection, a short causal convolution, a selective state-space scan and a gate. Tokens exchange information only through a state carried from left to right. Click a label to jump.', '24 × 768 · 130M · no attention'],
  cost: ['To predict the next token, attention compares it with every earlier token and keeps all their keys and values, so the KV cache grows with every token. Mamba carries a fixed-size state instead: each new token costs the same, and past about 38 tokens its state is smaller than GPT-2’s cache.', 'KV cache n × 36 KiB · state 1.3 MiB'],
  scan: ['The state is a small vector per channel (16 numbers in Mamba). At each token it is decayed by Ā and the input is written in through B̄; C reads the output. Here one toy channel with a 4-number state and the same step size Δ for every token, as in the earlier S4 models.', 'h_t = Ā h_t−1 + B̄ x_t · y_t = C h_t'],
  select: ['Mamba makes the step size Δ, and B and C, depend on the current token. A large Δ writes the token in strongly and forgets more of the past; a small one lets it pass and keeps the memory. Right: real Δ from Mamba-130m; in its middle layers the names get the largest steps.', 'Δ_t = softplus(W x_t) · Ā_t = exp(Δ_t A)'],
  predict: ['Real next-token guesses for the same prompt from GPT-2 small, with attention, and Mamba-130m, without it: two models of about the same size, trained on different web text. Both reach for places a cat might sit.', 'same prompt, two architectures'],
}

export function mountMamba(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'Hover a state cell or a Δ cell to read it; click or tap to pin it.',
      eyebrow: 'Lineage · Beyond attention',
      title: 'Mamba',
      subtitle: 'Mamba-130m · a selective state-space model',
      specs: [
        { label: 'compared', value: 'Mamba-130m', real: 'GPT-2 small', realLabel: 'vs' },
        { label: 'layers', value: '24', real: '12' },
        { label: 'd_model', value: '768', real: '768' },
        { label: 'mixer', value: 'scan · state 16', real: 'attention · 12 heads' },
        { label: 'memory', value: '1.3 MiB, any length', real: '36 KiB per token' },
        { label: 'params', value: '130M', real: '124M' },
      ],
    },
    size: [1040, 480],
    aria: 'Mamba compared with GPT-2: no attention; each block runs a selective state-space scan that carries a fixed-size state from token to token, with a step size that depends on the input; real step sizes and next-token guesses from Mamba-130m.',
    phases: PHASES, learn: 'mamba', tokens: TOKS, compare: COMPARE, caps: CAPS,
    still: ['select', 11],
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { pill, addNode, caption, title, lane, glass, arrow } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46

  /* ---------- scene 1: the blocks ---------- */
  function sceneBlocks(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const sx0 = pad + 130, sx1 = W - pad - 16, fx = (f: number) => lerp(sx0, sx1, f)
    const ys = [top + avail * 0.2, top + avail * 0.6], as = [eout(clamp(p / 0.3)), eout(clamp((p - 0.25) / 0.3))]
    ys.forEach((y, r) => {
      const a = as[r]
      if (a <= 0) return
      k.rowName(r ? 'Mamba' : 'GPT-2', r ? '130m · × 24' : 'small · × 12', pad, y, a)
      for (let i = 0; i < 5; i++) lane(sx0, lerp(sx0, sx1, a), y + (i - 2) * 4, hue(i), a)
      const box = (x: number, wd: number, t: string, sub: string, hl = false) => {
        rr(x - wd / 2, y - 14, wd, 28, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, (hl ? 0.9 : 0.6) * a); ctx.lineWidth = 1; ctx.stroke()
        caption(t, x, y + 4, a, C.ink); caption(sub, x, y + 44, a, hl ? C.ink2 : C.mute)
      }
      if (!r) {
        box(fx(0.04), 52, '+ W_P', 'learned')
        ;([[0.16, 'LN'], [0.3, 'attention'], [0.56, 'LN'], [0.7, 'MLP']] as [number, string][]).forEach(([f, t]) => { const ln = t === 'LN'; glass(fx(f), y - (ln ? 20 : 26), y + (ln ? 20 : 26), 0.2, ln ? { w: 5, d: 6 } : { w: 8, d: 9 }, a); caption(t, fx(f), y + 44, a) })
        addNode(fx(0.43), y, a); addNode(fx(0.84), y, a)
        return
      }
      caption('no position vector', fx(0.04), y + 44, a)
      glass(fx(0.16), y - 20, y + 20, 0.2, { w: 5, d: 6 }, a); caption('RMSNorm', fx(0.16), y + 44, a)
      box(fx(0.29), 70, '· W_in', '768 → 2 × 1,536')
      box(fx(0.42), 60, 'conv 4', 'causal', false)
      glass(fx(0.55), y - 26, y + 26, 0.6, { w: 8, d: 9 }, a); caption('SSM scan', fx(0.55), y + 44, a, C.ink2)
      box(fx(0.68), 76, '⊙ SiLU(z)', 'gate')
      box(fx(0.8), 66, '· W_out', '1,536 → 768')
      addNode(fx(0.91), y, a)
    })
    const la = eout(clamp((p - 0.55) / 0.25)), [yA, yB] = ys
    if (la > 0) {
      ctx.setLineDash([2, 4]); ctx.strokeStyle = rgba(C.ink, 0.35 * la); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(fx(0.3), yA + 52); ctx.bezierCurveTo(fx(0.3), (yA + yB) / 2, fx(0.55), (yA + yB) / 2, fx(0.55), yB - 58); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(fx(0.7), yA + 52); ctx.bezierCurveTo(fx(0.7), (yA + yB) / 2, fx(0.6), (yA + yB) / 2, fx(0.58), yB - 58); ctx.stroke()
      ctx.setLineDash([])
      caption('both become one mixer', fx(0.55), (yA + yB) / 2 + 4, la, C.ink2)
      pill('no attention', fx(0.3), yB - 46, 'cost', la)
      pill('scan', fx(0.55) - 30, yB - 46, 'scan', la)
      pill('Δ', fx(0.55) + 22, yB - 46, 'select', la)
      pill('same size', pad + 40, yB + 50, 'predict', la)
    }
    mk.formula = { segs: [['GPT-2  x + attn(LN x), then x + MLP(LN x)', C.ink2], ['     ·     ', C.mute], ['Mamba  x + W_out((scan(conv(x W_in)) ⊙ SiLU(z)))', C.ink]], note: 'The in-projection makes two streams of 1,536: one goes through the convolution and the scan, the other (z) gates the result. Twice as many blocks as GPT-2 small, each about half the parameters.' }
  }

  /* ---------- scene 2: memory as the text grows ---------- */
  function sceneCost(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const n = Math.max(1, Math.round(eio(clamp((p - 0.04) / 0.5)) * 14)), x0 = pad + 20, dx = 26
    // attention: the new token looks at every earlier one; the cache grows
    title('attention · gpt-2', x0, top + 12, 1)
    const ya = top + 90
    for (let i = 0; i < n; i++) {
      const x = x0 + i * dx
      ctx.fillStyle = rgba(hue(i), 0.85); ctx.beginPath(); ctx.arc(x + 8, ya, 6, 0, 7); ctx.fill()
      ctx.fillStyle = rgba(C.ink, 0.5); ctx.fillRect(x + 2, ya + 16, 12, 10); ctx.fillRect(x + 2, ya + 28, 12, 10)
      if (i < n - 1) { ctx.strokeStyle = rgba(hue(n - 1), 0.5); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 + (n - 1) * dx + 8, ya - 7); ctx.quadraticCurveTo((x + x0 + (n - 1) * dx) / 2 + 8, ya - 30 - (n - i) * 2, x + 8, ya - 7); ctx.stroke() }
    }
    caption(`K and V of all ${n} tokens`, x0, ya + 58, 1, C.ink2, 'left')
    // Mamba: the same state, whatever n
    const ym = top + 250
    title('mamba · a fixed state', x0, ym - 58, 1)
    for (let i = 0; i < n; i++) { ctx.fillStyle = rgba(hue(i), i === n - 1 ? 0.95 : 0.35); ctx.beginPath(); ctx.arc(x0 + i * dx + 8, ym, 6, 0, 7); ctx.fill() }
    const sx = x0 + (n - 1) * dx + 8
    rr(sx - 16, ym + 18, 32, 40, 4); ctx.fillStyle = rgba(C.ink, 0.5); ctx.fill()
    arrow([[sx, ym + 8], [sx, ym + 16]], 1)
    caption('one state, the same size at any length', x0, ym + 82, 1, C.ink2, 'left')
    // memory against context length
    const cx0 = 560, cx1 = W - pad - 60, cy0 = top + 40, cy1 = top + avail - 30, maxN = 1024, maxB = KV_PER_TOKEN * maxN
    const Xn = (m: number) => lerp(cx0, cx1, m / maxN), Yb = (b: number) => lerp(cy1, cy0, b / maxB), ga = eout(clamp((p - 0.45) / 0.15)), grow = eio(clamp((p - 0.5) / 0.4))
    title('memory to predict the next token', cx0, cy0 - 18, ga)
    ctx.strokeStyle = rgba(C.ink, 0.25 * ga); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy1); ctx.lineTo(cx1, cy1); ctx.stroke()
    ctx.strokeStyle = rgba(C.ink, 0.9 * ga); ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(Xn(0), Yb(0)); ctx.lineTo(Xn(maxN * grow), Yb(KV_PER_TOKEN * maxN * grow)); ctx.stroke()
    ctx.strokeStyle = rgba(C.tok[4], ga); ctx.beginPath(); ctx.moveTo(Xn(0), Yb(STATE)); ctx.lineTo(Xn(maxN * grow), Yb(STATE)); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, ga); ctx.textAlign = 'center'; ctx.textBaseline = 'top'
    for (const m of [0, 256, 512, 768, 1024]) ctx.fillText(m.toLocaleString('en-US'), Xn(m), cy1 + 6)
    caption('tokens so far', (cx0 + cx1) / 2, cy1 + 34, ga)
    if (grow > 0.95) {
      caption('GPT-2 · 36 MiB at 1,024', Xn(maxN) - 10, Yb(maxB) + 24, ga, C.ink, 'right')
      caption(`Mamba · ${(STATE / 2 ** 20).toFixed(2)} MiB at any length`, Xn(maxN) - 4, Yb(STATE) - 8, ga, C.ink, 'right')
      const cross = STATE / KV_PER_TOKEN
      ctx.fillStyle = rgba(C.ink, ga); ctx.beginPath(); ctx.arc(Xn(cross), Yb(STATE), 3, 0, 7); ctx.fill()
      caption(`the two lines cross at ${Math.round(cross)} tokens`, cx0 + 12, cy0 + 16, ga, C.ink2, 'left')
    }
    mk.formula = { segs: [['GPT-2  2 × 12 layers × 768 × 2 bytes = 36 KiB per token', C.ink2], ['     ·     ', C.mute], ['Mamba  24 × 1,536 × (16 + 3) × 2 bytes = 1.3 MiB', C.ink]], note: 'The Mamba state holds 16 numbers per channel for the scan and the last 3 inputs for the width-4 convolution. Attention’s work per new token also grows with the text; the scan’s does not.' }
  }

  /* ---------- scenes 3 and 4: the scan, then with a selective step ---------- */
  const cs = 22
  function stateScene(p: number, sel: boolean) {
    const run = sel ? SEL : FIXED, delta = sel ? DSEL : DSEL.map(() => DFIX)
    const x0 = pad + 40, dx = 60, yS = top + 70, yT = yS + NS * cs + 44, yY = yT + 42
    const vmax = Math.max(...[...FIXED.hs, ...SEL.hs].flat().map(Math.abs))
    const shown = Math.min(T, Math.floor(clamp((p - 0.06) / (sel ? 0.45 : 0.7)) * T + 1))
    title(sel ? 'one toy channel, Δ from mamba-130m layer 11' : 'one toy channel, the same Δ for every token', x0, top + 12, 1)
    caption('state h, 4 numbers', x0 - 30, yS + NS * cs + 18, 1, C.mute, 'left')
    for (let t = 0; t < T; t++) {
      const x = x0 + t * dx, a = t < shown ? 1 : 0.15
      drawChip(x + cs / 2 - chipW(TOKS[t].text) / 2, yT, TOKS[t], 1, 20)
      // Δ as a bar above the state
      const bh = delta[t] * 60
      ctx.fillStyle = rgba(sel ? C.ink : C.ink2, 0.8 * a); ctx.fillRect(x + 4, yS - 20 - bh, cs - 8, bh)
      if (t < shown) {
        // each state number in the blend of the tokens that wrote it
        run.hs[t].forEach((v, n) => {
          const w = run.parts[t][n].slice(0, t + 1), z = w.reduce((q, r) => q + r, 0) || 1
          ctx.fillStyle = rgba(blend(w.map((_, s2) => hue(s2)), w.map((r) => r / z)), (0.12 + 0.8 * Math.min(1, v / vmax)) * a)
          ctx.fillRect(x + 0.5, yS + n * cs + 0.5, cs - 1, cs - 1)
        })
        ctx.strokeStyle = rgba(C.ink, 0.25 * a); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, yS + 0.5, cs - 1, NS * cs - 1)
        const wy = run.parts[t].reduce((acc, row, n) => acc.map((q, s2) => q + Cv[n] * row[s2]), Array(T).fill(0)).slice(0, t + 1), zy = wy.reduce((q, r) => q + r, 0) || 1
        ctx.fillStyle = rgba(blend(wy.map((_, s2) => hue(s2)), wy.map((r) => r / zy)), (0.12 + 0.8 * Math.min(1, run.ys[t] / Math.max(...[...FIXED.ys, ...SEL.ys]))) * a)
        ctx.fillRect(x + 0.5, yY + 0.5, cs - 1, cs - 1)
        if (t > 0) arrow([[x - dx + cs + 4, yS + (NS * cs) / 2], [x - 4, yS + (NS * cs) / 2]], a)
      } else { ctx.strokeStyle = rgba(C.ink, 0.1); ctx.strokeRect(x + 0.5, yS + 0.5, cs - 1, NS * cs - 1) }
    }
    caption('Δ', x0 - 16, yS - 24, 1, C.ink2, 'left')
    caption('y', x0 - 16, yY + 15, 1, C.mute, 'left')
    mk.hit('h', { x: x0, y: yS, c: cs }, NS, Math.ceil((T * dx) / cs))
    const f = mk.focus?.key === 'h' ? mk.focus : null, t = f ? Math.floor((f.j * cs) / dx) : shown - 1, nn = f ? f.i : 0
    if (f && (f.j * cs) % dx < cs && t < shown) {
      const prev = t ? run.hs[t - 1][nn] : 0, d = delta[t]
      ctx.strokeStyle = rgba(C.ink, 1); ctx.lineWidth = 2; ctx.strokeRect(x0 + t * dx, yS + nn * cs, cs, cs)
      mk.formula = { segs: [[`h[${nn}] after ‘${TOKS[t].text.trim()}’`, C.ink], ['  =  ', C.mute], [`e^(${fmtF(d)} × ${A[nn]}) × ${fmtF(prev)}`, C.ink2], ['  +  ', C.mute], [`${fmtF(d)} × ${B[nn]} × ${fmtF(X[t])}`, C.ink2], ['  =  ', C.mute], [fmtF(run.hs[t][nn]), C.ink]], note: `Ā = e^(ΔA) = ${fmtF(Math.exp(d * A[nn]))}: the share of the old state kept. ${sel ? 'Δ here is the real mean step size of Mamba-130m’s layer 11 on this token.' : 'The same Δ for every token.'} A, B, C and x are toy numbers.` }
    } else mk.formula = { segs: [['h_t = e^(Δ_t A) ⊙ h_t−1 + Δ_t B x_t', C.ink], ['     ·     ', C.mute], ['y_t = C · h_t', C.ink2]], note: sel ? 'Colour shows which tokens each state number holds. With a large Δ (the names) the old state decays fast and the new token dominates; with a small Δ the state keeps what it had. Decay rates A from −1 to −16, like Mamba’s at initialisation.' : 'Colour shows which tokens each state number holds: the top row (A = −1) remembers further back, the bottom row (A = −16) mostly the last token.' }
    return { x0, yS }
  }
  function sceneScan(p: number) { stateScene(p, false) }
  function sceneSelect(p: number) {
    stateScene(p, true)
    const { W } = stage
    // the real Δ of Mamba-130m, every layer × token, each layer scaled to its own maximum
    const ha = eout(clamp((p - 0.5) / 0.12))
    if (ha <= 0) return
    const n = mamba.toks.length, cw = 12, ch = 12, hx = W - pad - n * cw - 10, hy = top + 70
    title('real Δ · mamba-130m', hx, top + 12, ha)
    caption('rows: layers 1–24,', hx, top + 30, ha, C.mute, 'left')
    caption('each scaled to its max', hx, top + 46, ha, C.mute, 'left')
    mamba.meanDelta.forEach((row, l) => {
      const m = Math.max(...row)
      row.forEach((v, t) => { ctx.fillStyle = rgba(C.ink, (0.04 + 0.9 * (v / m)) * ha); ctx.fillRect(hx + t * cw, hy + l * ch, cw - 1, ch - 1) })
      if (l % 4 === 0 || l === LAYER) { ctx.font = F.mono(10); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(l === LAYER ? C.ink : C.mute, ha); ctx.fillText(String(l + 1), hx - 6, hy + l * ch + ch / 2) }
    })
    ctx.strokeStyle = rgba(C.ink, ha); ctx.lineWidth = 1.2; ctx.strokeRect(hx - 1, hy + LAYER * ch - 1, n * cw + 1, ch + 1)
    mamba.toks.forEach((t, i) => {
      ctx.save(); ctx.translate(hx + (i + 0.5) * cw, hy + 24 * ch + 8); ctx.rotate(Math.PI / 3)
      ctx.font = F.mono(10); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(/Mary|John/.test(t) ? C.ink : C.mute, ha); ctx.fillText(tokLabel(t.replace(/^Ġ/, ' ')), 0, 0); ctx.restore()
    })
    mk.hit('d', { x: hx, y: hy, c: cw }, mamba.meanDelta.length, n)
    const f = mk.focus?.key === 'd' ? mk.focus : null
    if (f) {
      ctx.strokeStyle = rgba(C.ink, 1); ctx.lineWidth = 2; ctx.strokeRect(hx + f.j * cw - 1, hy + f.i * ch - 1, cw + 1, ch + 1)
      const v = mamba.meanDelta[f.i][f.j]
      mk.formula = { segs: [[`layer ${f.i + 1}, ‘${mamba.toks[f.j].replace(/^Ġ/, '')}’`, C.ink], ['  mean Δ  ', C.mute], [fmtF(v), C.ink]], note: `Averaged over the layer’s 1,536 channels; the largest in this layer is ${fmtF(Math.max(...mamba.meanDelta[f.i]))}. Real Mamba-130m on “${mamba.text}”.` }
    }
  }

  /* ---------- scene 5: real next-token guesses ---------- */
  const GPT = nextDist(presets()[0].passes[0].next, 1, 6).rows
  function scenePredict(p: number) {
    const { W } = stage
    const fin = eout(clamp(p / 0.08)), y0 = top + 60
    ctx.font = F.mono(13, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, fin)
    ctx.fillText(`“${mamba.prompt} …”`, W / 2, y0)
    const bars = (x: number, name: string, sub: string, rows: { s: string; p: number }[], t0: number) => {
      const a = eout(clamp((p - t0) / 0.1))
      if (a <= 0) return
      title(name, x, y0 + 50, a)
      caption(sub, x, y0 + 68, a, C.mute, 'left')
      rows.slice(0, 6).forEach((r, j) => {
        const y = y0 + 100 + j * 28, g = eio(clamp((p - t0 - 0.05 - j * 0.03) / 0.15))
        ctx.font = F.mono(12, j === 0 ? 500 : 400); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(j === 0 ? C.ink : C.ink2, a); ctx.fillText(r.s, x, y)
        ctx.fillStyle = rgba(C.ink, 0.7 * a); ctx.fillRect(x + 110, y - 7, Math.max(1.5, r.p * 1800 * g), 14)
        ctx.fillStyle = rgba(C.ink2, a * g); ctx.fillText(`${(r.p * 100).toFixed(1)}%`, x + 118 + r.p * 1800 * g, y)
      })
    }
    bars(pad + 60, 'gpt-2 small · attention', '124M, trained on WebText', GPT.map((r) => ({ s: r.sym, p: r.p })), 0.1)
    bars(W / 2 + 40, 'mamba-130m · no attention', '130M, trained on the Pile', mamba.next, 0.3)
    mk.formula = { segs: [['GPT-2 ', C.ink2], [`${GPT[0].sym} ${fmt(GPT[0].p * 100)}%`, C.ink], ['     ·     ', C.mute], ['Mamba ', C.ink2], [`${mamba.next[0].s} ${fmt(mamba.next[0].p * 100)}%`, C.ink]], note: 'Real outputs of both models; they were trained on different text with different byte-level BPE tokenizers, so the probabilities are not directly comparable. The Mamba paper reports its 3B model matching Transformers twice its size.' }
  }

  return { blocks: sceneBlocks, cost: sceneCost, scan: sceneScan, select: sceneSelect, predict: scenePredict }
}
