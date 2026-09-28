import { F, ctx, rr } from '../../core/draw'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eout, lerp } from '../../core/util'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * Optimizers: what to do with a gradient. SGD, momentum and Adam run on the same toy loss surface (two weights, a
 * narrow valley; the arithmetic is real, the surface is made up), then AdamW's decoupled weight decay, the warmup and
 * cosine schedule LLaMA 2 was trained with, and the memory an optimizer's state costs for a real model.
 */

const PHASES = [
  { id: 'sgd', name: 'A step against the gradient', short: 'SGD', dur: 10 },
  { id: 'momentum', name: 'Momentum', short: 'Momentum', dur: 9 },
  { id: 'adam', name: 'Adam: a step size per weight', short: 'Adam', dur: 11 },
  { id: 'decay', name: 'AdamW: weight decay, apart', short: 'AdamW', dur: 9 },
  { id: 'schedule', name: 'Warmup, then decay', short: 'Schedule', dur: 10 },
  { id: 'memory', name: 'What the optimizer keeps', short: 'Memory', dur: 9 },
]

/* ---------- the toy surface: L = x²/100 + y², a valley 100 times steeper across than along ---------- */
const A = 0.01, B = 1
const loss = (x: number, y: number) => A * x * x + B * y * y
const grad = (x: number, y: number): [number, number] => [2 * A * x, 2 * B * y]
const START: [number, number] = [-9, 1.6], STEPS = 60
type Path = [number, number][]
function sgd(lr: number): Path {
  const p: Path = [START]
  for (let t = 0; t < STEPS; t++) { const [x, y] = p[t], [gx, gy] = grad(x, y); p.push([x - lr * gx, y - lr * gy]) }
  return p
}
function momentum(lr: number, beta: number): Path {
  const p: Path = [START]; let vx = 0, vy = 0
  for (let t = 0; t < STEPS; t++) { const [x, y] = p[t], [gx, gy] = grad(x, y); vx = beta * vx + gx; vy = beta * vy + gy; p.push([x - lr * vx, y - lr * vy]) }
  return p
}
function adam(lr: number, b1 = 0.8, b2 = 0.99, wd = 0): Path {
  const p: Path = [START]; let mx = 0, my = 0, vx = 0, vy = 0
  for (let t = 1; t <= STEPS; t++) {
    const [x, y] = p[t - 1], [gx, gy] = grad(x, y)
    mx = b1 * mx + (1 - b1) * gx; my = b1 * my + (1 - b1) * gy; vx = b2 * vx + (1 - b2) * gx * gx; vy = b2 * vy + (1 - b2) * gy * gy
    const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t
    p.push([x - lr * (mx / c1 / (Math.sqrt(vx / c2) + 1e-8) + wd * x), y - lr * (my / c1 / (Math.sqrt(vy / c2) + 1e-8) + wd * y)])
  }
  return p
}
const LR_SGD = 0.9, LR_MOM = 0.3, BETA = 0.8, LR_ADAM = 0.5
const P_SGD = sgd(LR_SGD), P_MOM = momentum(LR_MOM, BETA), P_ADAM = adam(LR_ADAM)
const lossAt = (p: Path) => p.map(([x, y]) => loss(x, y))

/* ---------- LLaMA 2's schedule (Touvron et al. 2023): peak 3e-4, 2,000 warmup steps, cosine to 10% ---------- */
const PEAK = 3e-4, WARM = 2000, TOTAL = 500_000
const lrAt = (s: number) => (s < WARM ? PEAK * (s / WARM) : PEAK * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * (s - WARM) / (TOTAL - WARM)))))

const COMPARE: Record<string, [string, string]> = {
  sgd: ['where the gradient comes from', 'training/backprop?phase=chain'],
  momentum: ['the gradient of the loss', 'training/loss?phase=grad'],
  adam: ['one step on the logits', 'training/loss?phase=step'],
  decay: ['the weights being decayed', 'anatomy/mlp?phase=up'],
  schedule: ['what the loss does over training', 'training/loss?phase=loss'],
  memory: ['memory at inference: the KV cache', 'serving/kv-cache?phase=size'],
}

const CAPS: Record<string, [string, string]> = {
  sgd: [`A toy loss over two weights, with a valley 100 times steeper across than along. Plain gradient descent steps against the gradient. A step size that is safe across the valley is tiny along it, so it zigzags and crawls: loss ${lossAt(P_SGD)[20].toFixed(2)} after 20 steps.`, 'w ← w − η · g'],
  momentum: [`Momentum keeps a running sum of past gradients. The zigzags across the valley cancel out and the steady push along it adds up, so it moves further per step.`, 'v ← β v + g ·  w ← w − η v'],
  adam: [`Adam divides each weight’s step by a running size of its own gradients. Steep and shallow directions then move at a similar pace, set by η, whatever their scale. It is the default for Transformers.`, 'w ← w − η · m̂ / (√v̂ + ε)'],
  decay: [`Weight decay pulls every weight a little toward zero each step, which keeps them from growing without need. AdamW applies it directly to the weight instead of through the gradient, so Adam’s scaling does not weaken it.`, 'w ← w − η (m̂/(√v̂+ε) + λ w)'],
  schedule: [`The step size changes over training. LLaMA 2 warmed up linearly for ${WARM.toLocaleString('en-US')} steps (Adam’s running averages start noisy), then followed a cosine down to 10% of its peak of 3 × 10⁻⁴ over about half a million steps.`, 'warmup 2,000 · cosine to 10%'],
  memory: [`Adam keeps two numbers per weight, m and v, usually in 32 bits, beside a 32-bit master copy of the weights. Training a model needs several times the memory of running it.`, '≈ 16 bytes per weight vs 2 to serve'],
}

export function mountOptimizer(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'The loss surface is a toy with two weights; the optimizer arithmetic is real. The schedule and memory figures are real models’.',
      eyebrow: 'Training',
      title: 'Optimizer',
      subtitle: 'SGD, momentum, Adam, AdamW, and the schedule',
      specs: [
        { label: 'shown', value: 'toy surface', real: '2 weights', realLabel: '·' },
        { label: 'AdamW', value: 'β₁ 0.9, β₂ 0.95', real: 'λ 0.1 · LLaMA 2', realLabel: '·' },
        { label: 'schedule', value: 'warmup + cosine' },
      ],
    },
    size: [1040, 480],
    aria: 'Optimizers on a toy loss surface with a narrow valley: gradient descent zigzags, momentum moves further, Adam scales each weight’s step; AdamW adds weight decay; a warmup and cosine learning-rate schedule; and the memory Adam’s state needs for a real model.',
    phases: PHASES, learn: 'optimizer', tokens: words(['x', 'y']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['adam', 10],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption } = k
  const pad = 36, top = 56
  /** The surface: contours of the valley, and the map from weights to canvas. */
  function surface(a: number) {
    const x0 = pad + 10, x1 = pad + 560, y0 = top + 20, y1 = stage.H - 60
    const X = (v: number) => lerp(x0, x1, (v + 10) / 12), Y = (v: number) => lerp(y1, y0, (v + 2) / 4)
    ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip()
    for (const lv of [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 2.5, 3.5]) {
      ctx.strokeStyle = rgba(C.faint, a); ctx.lineWidth = 1; ctx.beginPath()
      for (let s = 0; s <= 120; s++) { const th = (s / 120) * Math.PI * 2, x = Math.sqrt(lv / A) * Math.cos(th), y = Math.sqrt(lv / B) * Math.sin(th); if (s) ctx.lineTo(X(x), Y(y)); else ctx.moveTo(X(x), Y(y)) }
      ctx.stroke()
    }
    ctx.restore()
    ctx.fillStyle = rgba(C.ink, a); ctx.beginPath(); ctx.arc(X(0), Y(0), 3, 0, 7); ctx.fill()
    caption('minimum', X(0) + 8, Y(0) - 8, a, C.mute, 'left')
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, a); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('weight x (shallow)', (x0 + x1) / 2, y1 + 8)
    ctx.save(); ctx.translate(x0 - 14, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'bottom'; ctx.fillText('weight y (steep)', 0, 0); ctx.restore()
    return { X, Y, x1 }
  }
  function path(pp: Path, n: number, col: RGB, a: number, X: (v: number) => number, Y: (v: number) => number) {
    ctx.strokeStyle = rgba(col, a); ctx.lineWidth = 1.6; ctx.beginPath()
    for (let i = 0; i <= n && i < pp.length; i++) { const [x, y] = pp[i]; if (i) ctx.lineTo(X(x), Y(y)); else ctx.moveTo(X(x), Y(y)) }
    ctx.stroke()
    for (let i = 0; i <= n && i < pp.length; i++) { ctx.fillStyle = rgba(col, a); ctx.beginPath(); ctx.arc(X(pp[i][0]), Y(pp[i][1]), 2.2, 0, 7); ctx.fill() }
  }
  /** Loss against step, log scale, for the paths shown. */
  function curves(items: [Path, RGB, string][], n: number, x0: number, y0: number, w: number, h: number, a: number) {
    title('loss by step (log scale)', x0, y0 - 12, a)
    ctx.strokeStyle = rgba(C.faint, a); ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, w, h)
    const lo = -5, hi = 1, Yv = (v: number) => y0 + h - ((Math.log10(Math.max(v, 1e-5)) - lo) / (hi - lo)) * h
    items.forEach(([pp, col, name], i) => {
      const ls = lossAt(pp)
      ctx.strokeStyle = rgba(col, a); ctx.lineWidth = 1.6; ctx.beginPath()
      for (let s = 0; s <= n; s++) { const x = x0 + (s / STEPS) * w; if (s) ctx.lineTo(x, Yv(ls[s])); else ctx.moveTo(x, Yv(ls[s])) }
      ctx.stroke()
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(col, a); ctx.fillText(`${name} ${ls[n].toFixed(ls[n] < 0.01 ? 4 : 2)}`, x0 + w + 8, y0 + 14 + i * 18)
    })
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, a); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    ctx.fillText('10', x0 - 4, Yv(10)); ctx.fillText('10⁻⁵', x0 - 4, Yv(1e-5))
    ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(`${STEPS} steps`, x0 + w, y0 + h + 4)
  }
  const col = { sgd: C.tok[3], mom: C.tok[2], adam: C.tok[0] }

  function sceneRun(which: ('sgd' | 'mom' | 'adam')[]) {
    return (p: number) => {
      const { X, Y, x1 } = surface(1), n = Math.floor(clamp((p - 0.05) / 0.8) * STEPS)
      const all: [Path, RGB, string, string][] = [[P_SGD, col.sgd, 'SGD', 'sgd'], [P_MOM, col.mom, 'momentum', 'mom'], [P_ADAM, col.adam, 'Adam', 'adam']]
      const shown = all.filter((r) => which.includes(r[3] as 'sgd'))
      shown.forEach(([pp, c], i) => path(pp, i === shown.length - 1 ? n : STEPS, c, i === shown.length - 1 ? 1 : 0.35, X, Y))
      curves(shown.map(([pp, c, name]) => [pp, c, name]), n, x1 + 70, top + 60, 230, 150, 1)
      const cur = which[which.length - 1]
      const lr = cur === 'sgd' ? `η ${LR_SGD}` : cur === 'mom' ? `η ${LR_MOM}, β ${BETA}` : `η ${LR_ADAM}, β₁ 0.8, β₂ 0.99`
      caption(lr, x1 + 70, top + 250, 1, C.ink2, 'left')
      const last = cur === 'sgd' ? P_SGD : cur === 'mom' ? P_MOM : P_ADAM, [gx, gy] = grad(...last[Math.min(n, STEPS)])
      mk.formula = { segs: [['g', C.ink2], [' = ', C.mute], [`(${gx.toFixed(3)}, ${gy.toFixed(3)})`, C.ink], ['   at step ', C.mute], [String(n), C.ink]], note: cur === 'adam' ? 'Adam’s step in each direction is about η m̂/√v̂: near ±η while the gradient keeps its sign, whatever its size. Here that lets it cross the valley without bouncing and run along the floor.' : cur === 'mom' ? 'β is how much of the previous velocity is kept each step. Too much and it overshoots the minimum and swings back.' : `Across the valley the step multiplies y by 1 − 2η = ${(1 - 2 * B * LR_SGD).toFixed(1)} (it overshoots and flips sign); along it, x by 1 − 2ηA = ${(1 - 2 * A * LR_SGD).toFixed(2)}.` }
    }
  }

  /* ---------- AdamW ---------- */
  function sceneDecay(p: number) {
    const { W } = stage, x0 = pad + 40, y0 = top + 60, lam = 0.1, lr = 3e-4, w = 0.8, u = 0.9
    title('one weight, one AdamW step (LLaMA 2’s η and λ, a made-up weight)', x0, y0 - 24, 1)
    const g1 = eout(clamp(p / 0.2)), g2 = eout(clamp((p - 0.3) / 0.2)), g3 = eout(clamp((p - 0.6) / 0.2))
    const row = (y: number, name: string, v: string, a: number) => { ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(name, x0, y); ctx.fillStyle = rgba(C.ink, a); ctx.fillText(v, x0 + 260, y) }
    row(y0, 'weight w', w.toFixed(4), 1)
    row(y0 + 36, 'Adam’s step  η · m̂/(√v̂+ε)', `−${(lr * u).toExponential(2)}`, g1)
    row(y0 + 72, 'decay        η · λ · w', `−${(lr * lam * w).toExponential(2)}`, g2)
    row(y0 + 108, 'new w', (w - lr * u - lr * lam * w).toFixed(7), g3)
    // over many steps: decay alone halves a weight every ln 2 / (ηλ) steps
    const half = Math.log(2) / (lr * lam)
    caption(`with no gradient, decay alone halves a weight every ln 2 / (ηλ) ≈ ${Math.round(half).toLocaleString('en-US')} steps`, x0, y0 + 160, g3, C.ink2, 'left')
    // the difference from L2 in the loss
    const la = eout(clamp((p - 0.7) / 0.15)), xr = W / 2 + 80
    title('L2 in the loss vs AdamW', xr, y0 - 24, la)
    ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
    ;[['L2: g ← g + λw, then Adam', 'decay is divided by √v̂ too:', 'weights with large gradients decay less'], ['AdamW: decay added after', 'every weight shrinks by the', 'same fraction ηλ per step']].forEach((ls, i) => {
      ls.forEach((l, j) => { ctx.fillStyle = rgba(j ? C.ink2 : C.ink, la); ctx.fillText(l, xr, y0 + i * 90 + j * 20) })
    })
    mk.formula = { segs: [['w ← w − η', C.ink2], [' (', C.mute], ['m̂/(√v̂+ε)', C.ink], [' + ', C.mute], ['λ w', C.ink], [')', C.mute]], note: 'Loshchilov & Hutter (2019) showed that folding decay into the gradient interacts badly with Adam’s per-weight scaling; decoupling it became the standard for Transformers.' }
  }

  /* ---------- the schedule ---------- */
  function sceneSchedule(p: number) {
    const { W, H } = stage, x0 = pad + 70, x1 = W - pad - 30, y0 = top + 40, y1 = H - 80
    const X = (s: number) => lerp(x0, x1, s / TOTAL), Y = (v: number) => lerp(y1, y0, v / PEAK)
    title('LLaMA 2’s learning rate', x0, y0 - 16, 1)
    ctx.strokeStyle = rgba(C.faint, 1); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke()
    const n = clamp((p - 0.05) / 0.75)
    ctx.strokeStyle = rgba(C.tok[0], 1); ctx.lineWidth = 2; ctx.beginPath()
    for (let i = 0; i <= 400 * n; i++) { const s = (i / 400) * TOTAL, x = X(s), y = Y(lrAt(s)); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y) }
    ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    ctx.fillText('3 × 10⁻⁴', x0 - 6, Y(PEAK)); ctx.fillText('3 × 10⁻⁵', x0 - 6, Y(PEAK * 0.1)); ctx.fillText('0', x0 - 6, y1)
    ctx.textAlign = 'center'; ctx.textBaseline = 'top'
    for (const s of [0, 100_000, 200_000, 300_000, 400_000, 500_000]) ctx.fillText(s ? `${s / 1000}k` : '0', X(s), y1 + 6)
    caption('step', x1, y1 + 32, 1, C.mute, 'right')
    // zoom on the warmup
    const za = eout(clamp((p - 0.5) / 0.15)), zx = x0 + 340, zy = y0 + 20, zw = 220, zh = 110
    if (za > 0) {
      rr(zx, zy, zw, zh, 6); ctx.fillStyle = rgba(C.bg, 0.9 * za); ctx.fill(); ctx.strokeStyle = rgba(C.ink, 0.3 * za); ctx.stroke()
      ctx.strokeStyle = rgba(C.tok[0], za); ctx.lineWidth = 1.6; ctx.beginPath()
      for (let i = 0; i <= 100; i++) { const s = (i / 100) * 6000, x = zx + 10 + (s / 6000) * (zw - 20), y = zy + zh - 14 - (lrAt(s) / PEAK) * (zh - 30); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y) }
      ctx.stroke()
      caption('first 6,000 steps: 2,000 of warmup', zx + 10, zy + 16, za, C.ink2, 'left')
    }
    mk.formula = { segs: [['η(s)', C.ink], [' = ', C.mute], ['η_max · s / 2000', C.ink2], ['  then  ', C.mute], ['η_max (0.1 + 0.45 (1 + cos π s′))', C.ink2]], note: 'Large steps early would push the weights far on gradient estimates Adam has not averaged yet. Late in training, small steps let the loss settle into a narrow minimum. Other schedules: constant with a short final decay (WSD), or linear decay.' }
  }

  /* ---------- memory ---------- */
  function sceneMemory(p: number) {
    const { W } = stage, x0 = pad + 170, y0 = top + 50, N = 8.03e9, bw = W - pad - x0 - 150
    const parts: [string, number, number][] = [['weights (bf16)', 2, 0.9], ['gradients (bf16)', 2, 0.7], ['fp32 master weights', 4, 0.5], ['Adam m (fp32)', 4, 0.35], ['Adam v (fp32)', 4, 0.2]]
    title('LLaMA 3 8B · 8.03B weights · bytes per weight', x0, y0 - 18, 1)
    let x = x0
    parts.forEach(([name, b, sh], i) => {
      const a = eout(clamp((p - 0.05 - i * 0.1) / 0.1)), w = (b / 16) * bw
      ctx.fillStyle = rgba(C.ink, sh * a); ctx.fillRect(x, y0, w - 2, 40)
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(`${b} B`, x + 4, y0 + 58)
      ctx.fillStyle = rgba(C.mute, a); ctx.fillText(name, x + 4, y0 + 74 + (i % 2) * 16)
      x += w
    })
    const ta = eout(clamp((p - 0.6) / 0.12)), gb = (b: number) => `${Math.round((N * b) / 1e9)} GB`
    ctx.font = F.mono(13, 600); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
    ctx.fillStyle = rgba(C.ink, ta); ctx.fillText(`training: 16 B × 8.03B ≈ ${gb(16)}, before activations`, x0, y0 + 150)
    ctx.fillStyle = rgba(C.ink2, ta); ctx.fillText(`serving in bf16: 2 B × 8.03B ≈ ${gb(2)}`, x0, y0 + 180)
    caption('an 80 GB GPU holds the model for serving, not its training state; training splits it across GPUs (ZeRO, FSDP)', x0, y0 + 214, ta, C.mute, 'left')
    mk.formula = { segs: [['2 + 2 + 4 + 4 + 4', C.ink2], [' = ', C.mute], ['16 bytes per weight', C.ink]], note: 'Mixed-precision training with Adam, as in the ZeRO paper (Rajbhandari et al. 2020). Activations saved for the backward pass come on top and grow with batch size and sequence length.' }
  }

  return { sgd: sceneRun(['sgd']), momentum: sceneRun(['sgd', 'mom']), adam: sceneRun(['sgd', 'mom', 'adam']), decay: sceneDecay, schedule: sceneSchedule, memory: sceneMemory }
}
