/*
 * Backpropagation through GPT-2 small, for real, offline: the forward pass on the Training sentence (the same one
 * the Next-token loss page scores), the mean next-token loss, and its gradient with respect to every weight, by the
 * chain rule layer by layer, in float64. Exports the gradient's size for every weight matrix, the gradient reaching
 * the residual stream between blocks, a slice of one linear layer's backward pass (dW = Xᵀ·dY), which positions each
 * position's loss sends gradient to, and a check of three weights against finite differences.
 *
 *   node scripts/backprop-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/gpt2/ (model.safetensors, merges.txt, as for gpt2-export.ts). Writes src/data/backprop.json.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { Gpt2Bpe } from '../src/lib/gpt2/bpe.ts'
import { loadSafetensors } from './gpt2-model.ts'

const HOME = `${process.env.HOME}/.cache/token-trails`
const D = 768, H = 12, DH = 64, FF = 3072, L = 12, V = 50257
const bpe = new Gpt2Bpe(readFileSync(`${HOME}/gpt2/merges.txt`, 'utf8'))
const W32 = loadSafetensors(`${HOME}/gpt2/model.safetensors`)
const W = new Map([...W32].map(([k, v]) => [k, Float64Array.from(v)]))
const w = (n: string) => { const t = W.get(n); if (!t) throw new Error(n); return t }
const text = JSON.parse(readFileSync(new URL('../src/data/gpt2.json', import.meta.url), 'utf8')).training.text as string
const ids = bpe.encode(text), N = ids.length - 1, inp = ids.slice(0, N), tgt = ids.slice(1)
type Vec = Float64Array
const zeros = (n: number) => new Float64Array(n)
/** Euclidean length of a (possibly huge) array, without spreading it into arguments. */
const len = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s) }
const lenRows = (rows: Vec[]) => Math.sqrt(rows.reduce((s, r) => s + len(r) ** 2, 0))

/* ---------- building blocks, forward and backward ---------- */
function lnF(x: Vec, g: Vec, b: Vec) {
  let m = 0; for (const v of x) m += v; m /= x.length
  let s = 0; for (const v of x) s += (v - m) ** 2
  const r = 1 / Math.sqrt(s / x.length + 1e-5), xh = x.map((v) => (v - m) * r)
  return { y: xh.map((v, i) => v * g[i] + b[i]), xh, r }
}
function lnB(dy: Vec, c: { xh: Vec; r: number }, g: Vec, dg: Vec, db: Vec) {
  const n = dy.length, dxh = dy.map((v, i) => v * g[i])
  let m1 = 0, m2 = 0
  for (let i = 0; i < n; i++) { dg[i] += dy[i] * c.xh[i]; db[i] += dy[i]; m1 += dxh[i]; m2 += dxh[i] * c.xh[i] }
  m1 /= n; m2 /= n
  return dxh.map((v, i) => c.r * (v - m1 - c.xh[i] * m2))
}
/** x · W + b with W stored [in × out] (GPT-2's Conv1D). */
function linF(x: Vec, Wt: Vec, b: Vec, out: number) {
  const y = Float64Array.from(b)
  for (let i = 0; i < x.length; i++) { const xi = x[i], r = i * out; if (xi) for (let j = 0; j < out; j++) y[j] += xi * Wt[r + j] }
  return y
}
function linB(dy: Vec, x: Vec, Wt: Vec, out: number, dW: Vec, db: Vec) {
  const dx = zeros(x.length)
  for (let j = 0; j < out; j++) db[j] += dy[j]
  for (let i = 0; i < x.length; i++) {
    const r = i * out, xi = x[i]
    let s = 0
    for (let j = 0; j < out; j++) { dW[r + j] += xi * dy[j]; s += dy[j] * Wt[r + j] }
    dx[i] = s
  }
  return dx
}
const C0 = Math.sqrt(2 / Math.PI)
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(C0 * (x + 0.044715 * x ** 3)))
const geluD = (x: number) => { const u = C0 * (x + 0.044715 * x ** 3), t = Math.tanh(u); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C0 * (1 + 3 * 0.044715 * x * x) }

/* ---------- the forward pass, keeping what the backward pass needs ---------- */
function forward() {
  const wte = w('wte.weight'), wpe = w('wpe.weight')
  let x = inp.map((id, p) => { const v = zeros(D); for (let k = 0; k < D; k++) v[k] = wte[id * D + k] + wpe[p * D + k]; return v })
  const x0 = x, blocks: any[] = []
  for (let l = 0; l < L; l++) {
    const P = `h.${l}.`, c: any = { xin: x }
    c.ln1 = x.map((r) => lnF(r, w(P + 'ln_1.weight'), w(P + 'ln_1.bias')))
    c.qkv = c.ln1.map((o: any) => linF(o.y, w(P + 'attn.c_attn.weight'), w(P + 'attn.c_attn.bias'), 3 * D))
    c.P = [] as number[][][]
    const att = x.map(() => zeros(D))
    for (let h = 0; h < H; h++) {
      const Ph: number[][] = []
      for (let i = 0; i < N; i++) {
        const sc: number[] = []
        for (let j = 0; j <= i; j++) { let s = 0; for (let k = 0; k < DH; k++) s += c.qkv[i][h * DH + k] * c.qkv[j][D + h * DH + k]; sc.push(s / 8) }
        const m = Math.max(...sc), e = sc.map((s) => Math.exp(s - m)), z = e.reduce((a, b) => a + b, 0), row = e.map((v) => v / z)
        Ph.push(row)
        for (let j = 0; j <= i; j++) for (let k = 0; k < DH; k++) att[i][h * DH + k] += row[j] * c.qkv[j][2 * D + h * DH + k]
      }
      c.P.push(Ph)
    }
    c.att = att
    const x1 = x.map((r, i) => { const o = linF(att[i], w(P + 'attn.c_proj.weight'), w(P + 'attn.c_proj.bias'), D); return r.map((v, k) => v + o[k]) })
    c.ln2 = x1.map((r) => lnF(r, w(P + 'ln_2.weight'), w(P + 'ln_2.bias')))
    c.hpre = c.ln2.map((o: any) => linF(o.y, w(P + 'mlp.c_fc.weight'), w(P + 'mlp.c_fc.bias'), FF))
    c.hact = c.hpre.map((h: Vec) => h.map(gelu))
    x = x1.map((r, i) => { const o = linF(c.hact[i], w(P + 'mlp.c_proj.weight'), w(P + 'mlp.c_proj.bias'), D); return r.map((v, k) => v + o[k]) })
    blocks.push(c)
  }
  const lnf = x.map((r) => lnF(r, w('ln_f.weight'), w('ln_f.bias')))
  const probs = lnf.map((o) => {
    const z = zeros(V)
    for (let t = 0; t < V; t++) { let s = 0; for (let k = 0; k < D; k++) s += o.y[k] * wte[t * D + k]; z[t] = s }
    let m = -Infinity; for (const v of z) if (v > m) m = v
    let S = 0; for (let t = 0; t < V; t++) { z[t] = Math.exp(z[t] - m); S += z[t] }
    return z.map((v) => v / S)
  })
  const losses = probs.map((p, i) => -Math.log(p[tgt[i]]))
  return { x0, blocks, xL: x, lnf, probs, losses, loss: losses.reduce((a, b) => a + b, 0) / N }
}

/* ---------- the backward pass: from dL/dlogits to every weight ---------- */
/** `weights[i]` scales position i's loss (the mean loss uses 1/N everywhere). */
function backward(f: ReturnType<typeof forward>, weights: number[]) {
  const G = new Map<string, Vec>(), g = (n: string) => { let t = G.get(n); if (!t) G.set(n, (t = zeros(w(n).length))); return t }
  const wte = w('wte.weight'), dwte = g('wte.weight')
  // logits = ln_f(x) · W_Eᵀ, so dL/dlogits = (p − onehot) / N, and it flows back through the tied matrix
  let dx = f.lnf.map((o, i) => {
    const dz = f.probs[i].map((p, t) => (p - (t === tgt[i] ? 1 : 0)) * weights[i]), df = zeros(D)
    for (let t = 0; t < V; t++) { const d = dz[t]; if (!d) continue; const r = t * D; for (let k = 0; k < D; k++) { df[k] += d * wte[r + k]; dwte[r + k] += d * o.y[k] } }
    return lnB(df, o, w('ln_f.weight'), g('ln_f.weight'), g('ln_f.bias'))
  })
  const streamNorm: number[] = [lenRows(dx)]
  const slice: any = {}
  for (let l = L - 1; l >= 0; l--) {
    const P = `h.${l}.`, c = f.blocks[l]
    // MLP: x2 = x1 + c_proj(gelu(c_fc(ln_2(x1))))
    const dact = dx.map((d, i) => linB(d, c.hact[i], w(P + 'mlp.c_proj.weight'), D, g(P + 'mlp.c_proj.weight'), g(P + 'mlp.c_proj.bias')))
    if (l === L - 1) slice.mlp = { x: c.hact.map((r: Vec) => [...r.slice(0, 8)]), dy: dx.map((r) => [...r.slice(0, 8)]) }
    const dpre = dact.map((d, i) => d.map((v, k) => v * geluD(c.hpre[i][k])))
    const dln2 = dpre.map((d, i) => linB(d, c.ln2[i].y, w(P + 'mlp.c_fc.weight'), FF, g(P + 'mlp.c_fc.weight'), g(P + 'mlp.c_fc.bias')))
    const dx1 = dx.map((d, i) => { const b = lnB(dln2[i], c.ln2[i], w(P + 'ln_2.weight'), g(P + 'ln_2.weight'), g(P + 'ln_2.bias')); return d.map((v, k) => v + b[k]) })
    // attention: x1 = x + c_proj(attn(c_attn(ln_1(x))))
    const datt = dx1.map((d, i) => linB(d, c.att[i], w(P + 'attn.c_proj.weight'), D, g(P + 'attn.c_proj.weight'), g(P + 'attn.c_proj.bias')))
    const dqkv = c.qkv.map(() => zeros(3 * D))
    for (let h = 0; h < H; h++) for (let i = 0; i < N; i++) {
      const row = c.P[h][i], dP = zeros(i + 1)
      for (let j = 0; j <= i; j++) {
        let s = 0
        for (let k = 0; k < DH; k++) { s += datt[i][h * DH + k] * c.qkv[j][2 * D + h * DH + k]; dqkv[j][2 * D + h * DH + k] += row[j] * datt[i][h * DH + k] }
        dP[j] = s
      }
      let dot = 0; for (let j = 0; j <= i; j++) dot += dP[j] * row[j]
      for (let j = 0; j <= i; j++) {
        const dS = row[j] * (dP[j] - dot) / 8
        for (let k = 0; k < DH; k++) { dqkv[i][h * DH + k] += dS * c.qkv[j][D + h * DH + k]; dqkv[j][D + h * DH + k] += dS * c.qkv[i][h * DH + k] }
      }
    }
    const dln1 = dqkv.map((d: Vec, i: number) => linB(d, c.ln1[i].y, w(P + 'attn.c_attn.weight'), 3 * D, g(P + 'attn.c_attn.weight'), g(P + 'attn.c_attn.bias')))
    dx = dx1.map((d, i) => { const b = lnB(dln1[i], c.ln1[i], w(P + 'ln_1.weight'), g(P + 'ln_1.weight'), g(P + 'ln_1.bias')); return d.map((v, k) => v + b[k]) })
    streamNorm.push(lenRows(dx))
  }
  // the embeddings: x0 = W_E[id] + W_P[pos]
  const dwpe = g('wpe.weight')
  dx.forEach((d, p) => { for (let k = 0; k < D; k++) { dwte[inp[p] * D + k] += d[k]; dwpe[p * D + k] += d[k] } })
  return { G, dx0: dx, streamNorm: streamNorm.reverse(), slice }
}

const r4 = (v: number) => Number(v.toPrecision(4))
const f = forward()
console.log(`${JSON.stringify(text)}: mean loss ${f.loss.toFixed(4)}`, f.losses.map((v) => v.toFixed(3)).join(' '))
const b = backward(f, inp.map(() => 1 / N))
const norm = (n: string) => r4(len(b.G.get(n)!))
const KINDS = ['ln_1.weight', 'attn.c_attn.weight', 'attn.c_proj.weight', 'ln_2.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
const layers = Array.from({ length: L }, (_, l) => Object.fromEntries(KINDS.map((k) => [k, norm(`h.${l}.${k}`)])))
console.log('gradient norms, c_fc per block:', layers.map((x) => x['mlp.c_fc.weight']).join(' '))

// which input positions each position's loss reaches (the causal mask, backwards)
const blame = inp.map((_, i) => {
  const bi = backward(f, inp.map((__, j) => (j === i ? 1 : 0)))
  return bi.dx0.map((d) => r4(len(d)))
})
console.log('blame (row: loss at position i, col: embedding at j):', blame.map((r) => r.join(' ')).join(' | '))

// finite differences on three weights: (L(w + ε) − L(w − ε)) / 2ε against the chain rule
const CHECKS: [string, number][] = [[`h.${L - 1}.mlp.c_proj.weight`, 5 * D + 17], [`h.0.attn.c_attn.weight`, 100 * 3 * D + 900], ['wpe.weight', 2 * D + 40]]
const eps = 1e-4
const checks = CHECKS.map(([name, idx]) => {
  const t = w(name), v = t[idx]
  t[idx] = v + eps; const up = forward().loss
  t[idx] = v - eps; const down = forward().loss
  t[idx] = v
  const fd = (up - down) / (2 * eps), bp = b.G.get(name)![idx]
  console.log(`${name}[${idx}]: backprop ${bp.toExponential(6)} · finite difference ${fd.toExponential(6)}`)
  return { name, index: idx, row: Math.floor(idx / (name.includes('c_attn') ? 3 * D : D)), col: idx % (name.includes('c_attn') ? 3 * D : D), backprop: Number(bp.toPrecision(6)), finite: Number(fd.toPrecision(6)) }
})

// one linear layer's backward pass: block 12's MLP output projection, dims 1–8 of its 3072 inputs and 768 outputs
const sx = b.slice.mlp.x as number[][], sdy = b.slice.mlp.dy as number[][]
const dW = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => sx.reduce((a, r, n) => a + r[i] * sdy[n][j], 0)))
const full = b.G.get(`h.${L - 1}.mlp.c_proj.weight`)!
const agree = dW.every((r, i) => r.every((v, j) => Math.abs(v - full[i * D + j]) < 1e-9 * Math.max(1, Math.abs(v))))
console.log('slice dW matches the full gradient:', agree)

writeFileSync(new URL('../src/data/backprop.json', import.meta.url), JSON.stringify({
  model: 'GPT-2 small', text, tokens: ids.map((i) => bpe.symbolOf(i)), loss: r4(f.loss), losses: f.losses.map(r4),
  params: Object.fromEntries(KINDS.map((k) => [k, w(`h.0.${k}`).length])),
  layers, embed: { wte: norm('wte.weight'), wpe: norm('wpe.weight'), ln_f: norm('ln_f.weight') },
  stream: b.streamNorm.map(r4), blame,
  linear: { x: sx.map((r) => r.map(r4)), dy: sdy.map((r) => r.map((v) => Number(v.toPrecision(3)))), dW: dW.map((r) => r.map((v) => Number(v.toPrecision(3)))) },
  checks,
}))
console.log('wrote src/data/backprop.json')
