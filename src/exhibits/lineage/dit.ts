import { F, fillRich, rr } from '../../core/draw'
import { fmt, fmtF } from '../../core/matrix'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, gauss, lerp, rng } from '../../core/util'
import { ALPHA_BAR, dit, heatAt, meanAbs } from '../../lib/dit/data'
import type { Nav } from '../registry'
import { mountExhibit, words, type Env } from '../kit'

/*
 * DiT-XL/2 as a diff against GPT-2: a Transformer over noisy latent image patches that predicts the
 * noise, conditioned on the timestep and class through adaLN-Zero. The noising uses DiT's real
 * schedule on a toy image; the modulation values are real DiT-XL/2 (exported offline); the sampling
 * run uses the true noise, as a perfect model would.
 */

const PHASES = [
  { id: 'blocks', name: 'GPT-2 vs DiT', short: 'Blocks', dur: 8 },
  { id: 'noise', name: 'Adding noise', short: 'Noise', dur: 10 },
  { id: 'latent', name: 'Latent patches', short: 'Patches', dur: 8 },
  { id: 'adaln', name: 'adaLN-Zero', short: 'adaLN', dur: 12 },
  { id: 'sample', name: 'Many passes per image', short: 'Sampling', dur: 11 },
]

/** The toy image: 32 × 32 values in [−1, 1], the range diffusion models work in. */
const S = 32
const X0 = (() => {
  const v = new Float32Array(S * S)
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const u = (x + 0.5) / S, w = (y + 0.5) / S
    let g = -0.8
    if (w > 0.12 && w < 0.2) g = 0.4
    if (Math.hypot(u - 0.36, w - 0.5) < 0.22) g = 0.9
    if (u > 0.6 && u < 0.86 && w > 0.5 && w < 0.84) g = 0.2
    v[y * S + x] = g
  }
  return v
})()
const EPS = (() => { const r = rng(1234); return Float32Array.from({ length: S * S }, () => gauss(r)) })()
const noisy = (t: number) => { const a = ALPHA_BAR[t]; return X0.map((v, i) => Math.sqrt(a) * v + Math.sqrt(1 - a) * EPS[i]) }
/** DDPM sampling over 250 of the 1,000 steps, with the true noise standing in for the model's estimate. */
const STEPS = 250
const TSTEP = Array.from({ length: STEPS }, (_, k) => Math.round((k * 999) / (STEPS - 1)))
const TRAJ = (() => {
  const r = rng(99), frames: Float32Array[] = Array(STEPS)
  let x = Float32Array.from({ length: S * S }, () => gauss(r))
  frames[STEPS - 1] = x
  for (let k = STEPS - 1; k >= 1; k--) {
    const ab = ALPHA_BAR[TSTEP[k]], abp = ALPHA_BAR[TSTEP[k - 1]], b = 1 - ab / abp
    const c0 = (Math.sqrt(abp) * b) / (1 - ab), ct = (Math.sqrt(1 - b) * (1 - abp)) / (1 - ab), sd = Math.sqrt((b * (1 - abp)) / (1 - ab))
    const xc = x
    x = X0.map((v, i) => c0 * v + ct * xc[i] + sd * gauss(r))
    frames[k - 1] = x
  }
  return frames
})()

const SEQ = words(['t', 'class', 'patch'])

const COMPARE: Record<string, [string, string]> = {
  blocks: ['GPT-2’s block', 'anatomy/layernorm?phase=stream'], noise: ['GPT-2’s training target', 'training/loss'], latent: ['ViT’s patches', 'lineage/vit?phase=patches'],
  adaln: ['GPT-2’s γ and β', 'anatomy/layernorm?phase=affine'], sample: ['GPT-2’s sampling', 'anatomy/unembed?phase=sample'],
}

const CAPS: Record<string, [string, string]> = {
  blocks: ['DiT keeps the Transformer block but changes what flows through it: noisy image patches instead of tokens, no mask, and the noise in each patch as the output. The timestep and class label are not tokens; they set each LayerNorm’s scale and shift and gate each sub-layer (adaLN-Zero). Click a label to jump.', '28 × 1,152 · 675M · 256 patches'],
  noise: ['Diffusion trains a model to undo noise. A clean image is mixed with Gaussian noise, a little at t = 1 and completely by t = 1,000; the model sees the noisy image and t and predicts the noise that was added. This is the exact mix on a toy image, with DiT’s schedule.', 'x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε'],
  latent: ['DiT works on a compressed image: a pretrained autoencoder turns 256 × 256 × 3 pixels into a 32 × 32 × 4 latent, and 2 × 2 patches of it make 256 tokens of 16 numbers. Smaller patches mean more tokens, more compute and better images.', '256² × 3 → 32² × 4 → 256 tokens'],
  adaln: ['In GPT-2, LayerNorm’s γ and β are fixed after training. In DiT a small network computes them from the timestep and class, separately for every block, plus a gate on each sub-layer’s output that starts at zero (the “Zero”). These are real DiT-XL/2 values: the gates change with t, and some sub-layers are almost switched off.', 'x + α · f(LN(x) · (1 + γ) + β)'],
  sample: ['Generating an image starts from pure noise and removes a little at a time: 250 steps, each a full pass through all 28 blocks, twice with classifier-free guidance. Here each step uses the true noise, as a perfect model would; the real model only estimates it.', '250 passes × 118.6 GFLOPs'],
}

export function mountDit(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'Hover the charts to read the real values; click or tap to pin.',
      eyebrow: 'Lineage · Vision & diffusion',
      title: 'Diffusion Transformer',
      subtitle: 'DiT-XL/2 · predicting noise instead of the next token',
      specs: [
        { label: 'compared', value: 'DiT-XL/2', real: 'GPT-2 small', realLabel: 'vs' },
        { label: 'layers', value: '28', real: '12' },
        { label: 'width', value: '1,152 · 16 heads', real: '768 · 12' },
        { label: 'tokens', value: '256 latent patches', real: '≤ 1,024 text' },
        { label: 'output', value: 'noise per patch', real: '50,257 scores' },
        { label: 'params', value: '675M · 118.6 GFLOPs', real: '124M' },
      ],
    },
    size: [1040, 490],
    aria: 'The Diffusion Transformer compared with GPT-2: a Transformer over noisy latent image patches that predicts the added noise, with the timestep and class setting every LayerNorm’s scale, shift and gate, and hundreds of passes to make one image.',
    phases: PHASES, learn: 'dit', tokens: SEQ, compare: COMPARE, caps: CAPS,
    still: ['adaln', 11],
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { pill, addNode, caption, title, lane, glass, arrow } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46
  /** A 32 × 32 image of values in [−1, 1], as ink density. */
  function img(v: Float32Array, x: number, y: number, s: number, a: number) {
    for (let i = 0; i < S * S; i++) { ctx.fillStyle = rgba(C.ink, clamp((v[i] + 1) / 2) * a); ctx.fillRect(x + (i % S) * s, y + Math.floor(i / S) * s, s + 0.3, s + 0.3) }
    ctx.strokeStyle = rgba(C.ink, 0.25 * a); ctx.lineWidth = 1; ctx.strokeRect(x - 0.5, y - 0.5, S * s + 1, S * s + 1)
  }

  /* ---------- scene 1: the blocks ---------- */
  function sceneBlocks(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const sx0 = pad + 170, sx1 = W - pad - 120, fx = (f: number) => lerp(sx0, sx1, f)
    const ys = [top + avail * 0.16, top + avail * 0.54], as = [eout(clamp(p / 0.3)), eout(clamp((p - 0.25) / 0.3))]
    ys.forEach((y, r) => {
      const a = as[r]
      if (a <= 0) return
      k.rowName(r ? 'DiT' : 'GPT-2', r ? 'XL/2 · 675M' : 'small · 124M', pad, y, a)
      if (r) img(noisy(500), pad + 104, y - 16, 1, a)
      else { ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText('text', pad + 106, y) }
      for (let i = 0; i < 5; i++) lane(sx0, lerp(sx0, sx1, a), y + (i - 2) * 4, hue(i), a)
      const box = (x: number, wd: number, t: string, sub: string, hl: boolean) => {
        rr(x - wd / 2, y - 15, wd, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, (hl ? 0.9 : 0.6) * a); ctx.lineWidth = 1; ctx.stroke()
        caption(t, x, y + 4, a, C.ink); caption(sub, x, y + 44, a, hl ? C.ink2 : C.mute)
      }
      box(fx(0.03), 70, r ? '· W_patch' : 'W_E[id]', r ? 'patches' : 'lookup', r === 1)
      box(fx(0.16), 50, '+ P', r ? 'fixed sin-cos' : 'learned', false)
      const plates: [number, string, boolean][] = r
        ? [[0.28, 'adaLN', true], [0.38, 'attn · all', false], [0.62, 'adaLN', true], [0.72, 'MLP', false]]
        : [[0.28, 'LN', false], [0.38, 'attn · causal', false], [0.62, 'LN', false], [0.72, 'MLP', false]]
      for (const [f, t, hl] of plates) {
        const ln = t === 'LN' || t === 'adaLN'
        glass(fx(f), y - (ln ? 20 : 26), y + (ln ? 20 : 26), hl ? 0.55 : 0.2, ln ? { w: 5, d: 6 } : { w: 8, d: 9 }, a)
        caption(t, fx(f), y + 44, a, hl ? C.ink2 : C.mute)
      }
      if (r) { caption('× α', fx(0.46), y - 16, a, C.ink2); caption('× α', fx(0.8), y - 16, a, C.ink2) }
      addNode(fx(0.5), y, a); addNode(fx(0.84), y, a)
      const hx = sx1 + 58
      arrow([[sx1, y], [hx - 48, y]], a)
      rr(hx - 46, y - 15, 104, 30, 6); ctx.fillStyle = rgba(C.bg, a); ctx.fill(); ctx.strokeStyle = rgba(C.ink, (r ? 0.9 : 0.6) * a); ctx.stroke()
      caption(r ? 'noise per patch' : 'next token', hx + 6, y + 4, a, C.ink)
      caption(r ? 'and its variance' : '50,257 scores', hx + 6, y + 44, a, r ? C.ink2 : C.mute)
    })
    // the conditioning: timestep and class into every adaLN and gate
    const ca = eout(clamp((p - 0.45) / 0.2)), yB = ys[1], yc = yB + 96
    if (ca > 0) {
      rr(fx(0.16) - 90, yc - 15, 180, 30, 6); ctx.fillStyle = rgba(C.bg, ca); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.9 * ca); ctx.lineWidth = 1; ctx.stroke()
      caption('c = emb(t) + emb(class)', fx(0.16), yc + 4, ca, C.ink)
      ctx.setLineDash([3, 4]); ctx.strokeStyle = rgba(C.ink, 0.6 * ca); ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(fx(0.16) + 90, yc); ctx.lineTo(fx(0.8), yc); ctx.stroke()
      for (const f of [0.28, 0.46, 0.62, 0.8]) { ctx.beginPath(); ctx.moveTo(fx(f), yc); ctx.lineTo(fx(f), yB + 56); ctx.stroke() }
      ctx.setLineDash([])
      caption('γ, β, α for every block', fx(0.54), yc + 18, ca, C.ink2)
    }
    const la = eout(clamp((p - 0.6) / 0.2))
    if (la > 0) {
      pill('noise', pad + 122, yB - 46, 'noise', la)
      pill('patches', fx(0.03), yB - 46, 'latent', la)
      pill('adaLN-Zero', fx(0.28), yB - 46, 'adaln', la)
      pill('250 passes', sx1 + 58, yB - 46, 'sample', la)
    }
    mk.formula = { segs: [['GPT-2  x + f(LN(x))', C.ink2], ['     ·     ', C.mute], ['DiT  x + α(c) · f(LN(x) · (1 + γ(c)) + β(c))', C.ink]], note: 'c is the sum of a timestep embedding and a class embedding. Each block turns it into six vectors of 1,152: γ, β and α for attention and for the MLP.' }
  }

  /* ---------- scene 2: adding noise ---------- */
  const SHOWN = [0, 100, 250, 500, 750, 999]
  function sceneNoise(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const s = 4, gap = 30, x0 = pad + 20, y0 = top + 40, fin = eout(clamp(p / 0.06))
    title('the same image at six timesteps', x0, y0 - 16, fin)
    SHOWN.forEach((t, i) => {
      const a = eout(clamp((p - 0.04 - i * 0.07) / 0.08))
      if (a <= 0) return
      const x = x0 + i * (S * s + gap)
      img(noisy(t), x, y0, s, a)
      caption(`t = ${(t + 1).toLocaleString('en-US')}`, x, y0 + S * s + 20, a, C.ink, 'left')
      caption(`√ᾱ ${fmt(Math.sqrt(ALPHA_BAR[t]))}`, x, y0 + S * s + 38, a, C.mute, 'left')
    })
    // ᾱ over t
    const cx0 = x0 + 40, cx1 = W - pad - 40, cy0 = y0 + S * s + 80, cy1 = top + avail - 20, ca = eout(clamp((p - 0.5) / 0.12))
    if (ca > 0) {
      const X = (t: number) => lerp(cx0, cx1, t / 999), Y = (v: number) => lerp(cy1, cy0, v)
      ctx.strokeStyle = rgba(C.ink, 0.25 * ca); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy1); ctx.lineTo(cx1, cy1); ctx.stroke()
      const curve = (f: (t: number) => number, col: RGB, w: number) => { ctx.strokeStyle = rgba(col, ca); ctx.lineWidth = w; ctx.beginPath(); for (let t = 0; t < 1000; t += 5) (t ? ctx.lineTo(X(t), Y(f(t))) : ctx.moveTo(X(t), Y(f(t)))); ctx.stroke() }
      curve((t) => Math.sqrt(ALPHA_BAR[t]), C.ink, 1.8)
      curve((t) => Math.sqrt(1 - ALPHA_BAR[t]), C.ink2, 1.2)
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
      ctx.fillStyle = rgba(C.ink, ca); fillRich('√ᾱ_t: how much image is left', X(40), Y(Math.sqrt(ALPHA_BAR[40])) - 8)
      ctx.fillStyle = rgba(C.ink2, ca); fillRich('√(1 − ᾱ_t): how much noise', X(560), Y(Math.sqrt(1 - ALPHA_BAR[560])) - 10)
      ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, ca); ctx.textAlign = 'center'; ctx.textBaseline = 'top'
      for (const t of [0, 250, 500, 750, 999]) ctx.fillText((t + 1).toLocaleString('en-US'), X(t), cy1 + 6)
    }
    mk.formula = { segs: [['x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε', C.ink], ['     ·     ', C.mute], ['loss = ‖ε − model(x_t, t, class)‖²', C.ink2]], note: 'ε is fresh Gaussian noise for every training example and t is drawn at random, so one model learns to remove every amount of noise. β rises linearly from 0.0001 to 0.02 over 1,000 steps, as in DiT; ᾱ_t is the product of (1 − β) up to t.' }
  }

  /* ---------- scene 3: latent patches ---------- */
  function sceneLatent(p: number) {
    const { W } = stage
    const y0 = top + 70
    const stagesA = [eout(clamp(p / 0.12)), eout(clamp((p - 0.18) / 0.12)), eout(clamp((p - 0.36) / 0.12)), eout(clamp((p - 0.54) / 0.12))]
    // pixels → latent → patches → tokens
    const bx = pad + 30
    ctx.save(); ctx.globalAlpha = stagesA[0]
    rr(bx, y0, 150, 150, 4); ctx.strokeStyle = rgba(C.ink, 0.5); ctx.lineWidth = 1; ctx.stroke()
    ctx.restore()
    img(X0, bx + 11, y0 + 11, 4, stagesA[0] * 0.9)
    caption('256 × 256 × 3 pixels', bx, y0 + 172, stagesA[0], C.ink2, 'left')
    const lx = bx + 240
    arrow([[bx + 160, y0 + 75], [lx - 12, y0 + 75]], stagesA[1])
    caption('VAE ÷ 8', (bx + 160 + lx) / 2 - 4, y0 + 64, stagesA[1], C.ink2)
    for (let c = 3; c >= 0; c--) { ctx.fillStyle = rgba(C.bg, stagesA[1]); ctx.fillRect(lx + c * 6, y0 + 10 - c * 6, 128, 128); img(X0.map((v, i) => v * (1 - 0.15 * c) + 0.2 * Math.sin(i * (c + 1))), lx + c * 6, y0 + 10 - c * 6, 4, stagesA[1] * (c ? 0.35 : 1)) }
    caption('32 × 32 × 4 latent', lx, y0 + 172, stagesA[1], C.ink2, 'left')
    const px = lx + 200
    arrow([[lx + 150, y0 + 75], [px - 12, y0 + 75]], stagesA[2])
    caption('2 × 2', (lx + 150 + px) / 2 + 4, y0 + 64, stagesA[2], C.ink2)
    if (stagesA[2] > 0) {
      img(X0, px, y0 + 10, 4, stagesA[2])
      ctx.strokeStyle = rgba(C.bg, 0.9 * stagesA[2]); ctx.lineWidth = 1
      for (let g = 1; g < 16; g++) { ctx.beginPath(); ctx.moveTo(px + g * 8, y0 + 10); ctx.lineTo(px + g * 8, y0 + 138); ctx.moveTo(px, y0 + 10 + g * 8); ctx.lineTo(px + 128, y0 + 10 + g * 8); ctx.stroke() }
      caption('16 × 16 = 256 patches', px, y0 + 172, stagesA[2], C.ink2, 'left')
    }
    const tx = px + 190
    if (stagesA[3] > 0 && tx + 100 < W) {
      arrow([[px + 140, y0 + 75], [tx - 12, y0 + 75]], stagesA[3])
      for (let i = 0; i < 6; i++) { ctx.fillStyle = rgba(hue(i), 0.8 * stagesA[3]); ctx.fillRect(tx, y0 + 20 + i * 20, 90, 12) }
      caption('…', tx + 40, y0 + 146, stagesA[3], C.mute)
      caption('256 tokens × 1,152', tx, y0 + 172, stagesA[3], C.ink2, 'left')
    }
    // patch size and compute (DiT paper, table 4)
    const ta = eout(clamp((p - 0.72) / 0.1))
    if (ta > 0) {
      const ty = y0 + 230
      title('dit-xl, patch size vs compute', bx, ty, ta)
      ;[['XL/8', '16', '7.2'], ['XL/4', '64', '29.1'], ['XL/2', '256', '118.6']].forEach(([n, tk, gf], i) => {
        const y = ty + 26 + i * 22
        ctx.font = F.mono(12, i === 2 ? 500 : 400); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(i === 2 ? C.ink : C.ink2, ta)
        ctx.fillText(n, bx, y); ctx.fillText(`${tk} tokens`, bx + 70, y); ctx.fillText(`${gf} GFLOPs`, bx + 200, y)
        ctx.fillStyle = rgba(C.ink, 0.6 * ta); ctx.fillRect(bx + 330, y - 10, (parseFloat(gf) / 118.6) * 240, 10)
      })
    }
    mk.formula = { segs: [['tokens = (32 / p)²', C.ink], ['     ·     ', C.mute], ['p = 2 → 256 tokens of 2 · 2 · 4 = 16 numbers', C.ink2]], note: 'The autoencoder is trained separately and frozen (it is Stable Diffusion’s VAE); DiT only ever sees latents. The model’s own weights stay the same size whatever the patch size; only the number of tokens, and so the compute, changes.' }
  }

  /* ---------- scene 4: real adaLN-Zero values ---------- */
  function sceneAdaln(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const fin = eout(clamp(p / 0.06)), cls = 207, ts = dit.ts
    // left: mean |gate| against t, for three blocks
    const cx0 = pad + 40, cx1 = pad + 420, cy0 = top + 50, cy1 = top + avail - 60, vMax = 0.8
    const X = (i: number) => lerp(cx0, cx1, i / (ts.length - 1)), Y = (v: number) => lerp(cy1, cy0, v / vMax)
    title('mean |gate α| · golden retriever', cx0, cy0 - 22, fin)
    ctx.strokeStyle = rgba(C.ink, 0.25 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy1); ctx.lineTo(cx1, cy1); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, fin); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    for (const v of [0, 0.2, 0.4, 0.6, 0.8]) ctx.fillText(v.toFixed(1), cx0 - 6, Y(v))
    ctx.textAlign = 'center'; ctx.textBaseline = 'top'
    for (const i of [0, 20, 40]) ctx.fillText(`t ${(ts[i] + 1).toLocaleString('en-US')}`, X(i), cy1 + 6)
    const lines: [number, number, RGB, number[] | null, string][] = [[13, 5, C.ink, null, 'block 14 · MLP'], [13, 2, C.ink, [5, 4], 'block 14 · attn'], [27, 2, C.ink2, [5, 4], 'block 28 · attn'], [27, 5, C.ink2, null, 'block 28 · MLP'], [0, 2, C.mute, [5, 4], 'block 1 · attn']]
    const grow = eio(clamp((p - 0.08) / 0.3)) * (ts.length - 1)
    const f = mk.focus?.key === 'g' ? mk.focus : null
    lines.forEach(([b, kk, col, dash, lab], li) => {
      const v = meanAbs(b, cls, kk), a = fin * eout(clamp((p - 0.08 - li * 0.04) / 0.1))
      if (a <= 0) return
      ctx.strokeStyle = rgba(col, a); ctx.lineWidth = 1.6; if (dash) ctx.setLineDash(dash)
      ctx.beginPath(); for (let i = 0; i <= Math.floor(grow); i++) (i ? ctx.lineTo(X(i), Y(v[i])) : ctx.moveTo(X(i), Y(v[i]))); ctx.stroke(); ctx.setLineDash([])
      if (grow >= ts.length - 1) caption(`${lab} ${fmt(v[ts.length - 1])}`, cx1 + 8, Y(v[ts.length - 1]) + 4 + (li === 1 ? -8 : li === 2 ? 8 : 0), a, col, 'left')
    })
    mk.hit('g', { x: cx0 - (cx1 - cx0) / (ts.length - 1) / 2, y: cy0, c: (cx1 - cx0) / (ts.length - 1) }, Math.ceil((cy1 - cy0) / ((cx1 - cx0) / (ts.length - 1))), ts.length)
    if (f) { ctx.strokeStyle = rgba(C.ink, 0.5); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(X(f.j), cy0); ctx.lineTo(X(f.j), cy1); ctx.stroke(); ctx.setLineDash([]) }
    // right: block 14's γ − 1 (the "scale" output) for its first 48 numbers, against t
    const hx = cx1 + 170, n = dit.heat.n, cw = Math.min(8, (W - pad - hx - 20) / ts.length), ch = Math.min(5, (avail - 80) / n), ha = fin * eout(clamp((p - 0.45) / 0.12))
    if (ha > 0) {
      title('block 14 · attention scale γ − 1', hx, cy0 - 22, ha)
      for (let ti = 0; ti < ts.length; ti++) for (let d = 0; d < n; d++) {
        const v = heatAt(1, ti, d), m = Math.min(1, Math.abs(v) / 1.2)
        ctx.fillStyle = v >= 0 ? rgba(C.ink, (0.05 + 0.85 * m) * ha) : rgba(C.neg, 0.3 * m * ha)
        ctx.fillRect(hx + ti * cw, cy0 + d * ch, cw + 0.2, ch + 0.2)
      }
      ctx.strokeStyle = rgba(C.ink, 0.22 * ha); ctx.lineWidth = 1; ctx.strokeRect(hx + 0.5, cy0 + 0.5, ts.length * cw - 1, n * ch - 1)
      ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, ha); ctx.textAlign = 'center'; ctx.textBaseline = 'top'
      ctx.fillText('t 1', hx, cy0 + n * ch + 6); ctx.fillText('t 1,000', hx + ts.length * cw, cy0 + n * ch + 6)
      caption(`first ${n} of 1,152 numbers, one row each`, hx, cy0 + n * ch + 38, ha, C.mute, 'left')
      caption('filled: γ above 1 · faint: below', hx, cy0 + n * ch + 54, ha, C.mute, 'left')
    }
    if (f) {
      const i = f.j, rows = lines.map(([b, kk, , , lab]) => `${lab} ${fmtF(meanAbs(b, cls, kk)[i])}`)
      mk.formula = { segs: [[`t = ${ts[i] + 1}`, C.ink], ['   ', C.mute], [rows.join(' · '), C.ink2]], note: 'Mean absolute gate over the block’s 1,152 channels, for the class golden retriever. The “no class” label used for guidance gives nearly the same numbers. Real DiT-XL/2.' }
    } else mk.formula = { segs: [['(β₁, γ₁, α₁, β₂, γ₂, α₂) = W_b · SiLU(emb(t) + emb(class)) + b_b', C.ink]], note: 'One 1,152 × 6,912 matrix per block. The gates α started at zero, so each block began as the identity; after training, block 14’s MLP gate is largest on nearly clean images and block 28’s MLP gate is almost zero. Real DiT-XL/2 values.' }
  }

  /* ---------- scene 5: sampling ---------- */
  function sceneSample(p: number) {
    const fin = eout(clamp(p / 0.06)), kf = Math.round((1 - eio(clamp((p - 0.06) / 0.8))) * (STEPS - 1)), s = 7, x0 = pad + 30, y0 = top + 40
    img(TRAJ[kf], x0, y0, s, fin)
    ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, fin)
    ctx.fillText(`pass ${STEPS - kf} of ${STEPS}`, x0, y0 + S * s + 30)
    caption(`t = ${TSTEP[kf] + 1}`, x0, y0 + S * s + 50, fin, C.ink2, 'left')
    // a strip of every 25th step
    const sx = x0 + S * s + 60, ss = 2.4, marks = [249, 224, 199, 174, 149, 124, 99, 74, 49, 24, 0]
    title('every 25th pass', sx, y0 - 16, fin)
    marks.forEach((kk, i) => {
      const x = sx + (i % 6) * (S * ss + 14), y = y0 + Math.floor(i / 6) * (S * ss + 34)
      const a = fin * (kk >= kf ? 1 : 0.2)
      img(TRAJ[kk], x, y, ss, a)
      caption(String(STEPS - kk), x, y + S * ss + 16, a, C.mute, 'left')
    })
    // cost next to GPT-2's
    const ca = eout(clamp((p - 0.6) / 0.12)), cy = y0 + 2 * (S * ss + 34) + 30
    if (ca > 0) {
      title('compute', sx, cy, ca)
      const rowsC: [string, string][] = [['DiT-XL/2, one image', '250 passes × 118.6 GFLOPs ≈ 30 TFLOPs, ×2 with guidance'], ['GPT-2 small, one token', '1 pass ≈ 0.25 GFLOPs']]
      rowsC.forEach(([a1, b1], i) => { caption(a1, sx, cy + 24 + i * 20, ca, i ? C.ink2 : C.ink, 'left'); caption(b1, sx + 190, cy + 24 + i * 20, ca, i ? C.ink2 : C.ink, 'left') })
    }
    mk.formula = { segs: [['x_{t−1} = μ(x_t, ε̂) + σ_t · z', C.ink2], ['     ·     ', C.mute], ['guidance: ε̂ = ε̂_∅ + s · (ε̂_class − ε̂_∅)', C.ink]], note: 'With the true noise as ε̂, as here, every step lands closer to the image. The real model estimates ε̂; classifier-free guidance runs it with the class and with the “no class” label and pushes away from the latter (s = 1.5 for DiT’s best FID, 4 for the paper’s samples).' }
  }

  return { blocks: sceneBlocks, noise: sceneNoise, latent: sceneLatent, adaln: sceneAdaln, sample: sceneSample }
}
