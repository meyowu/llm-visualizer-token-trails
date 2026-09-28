/*
 * GPT-2's forward and backward pass in float64, for the offline Training exports (Backprop, LoRA, RLHF & DPO): the
 * forward pass keeps what the backward pass needs, and the backward pass returns the gradient of a weighted sum of
 * next-token losses with respect to the weights (only those asked for) and to the input embeddings.
 */
export const D = 768, H = 12, DH = 64, FF = 3072, L = 12, V = 50257
export type Vec = Float64Array
export type Weights = Map<string, Float64Array>
const zeros = (n: number) => new Float64Array(n)
/** Euclidean length of a (possibly huge) array, without spreading it into arguments. */
export const len = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s) }
export const lenRows = (rows: Vec[]) => Math.sqrt(rows.reduce((s, r) => s + len(r) ** 2, 0))

/* ---------- building blocks, forward and backward ---------- */
function lnF(x: Vec, g: Vec, b: Vec) {
  let m = 0; for (const v of x) m += v; m /= x.length
  let s = 0; for (const v of x) s += (v - m) ** 2
  const r = 1 / Math.sqrt(s / x.length + 1e-5), xh = x.map((v) => (v - m) * r)
  return { y: xh.map((v, i) => v * g[i] + b[i]), xh, r }
}
function lnB(dy: Vec, c: { xh: Vec; r: number }, g: Vec, dg: Vec | null, db: Vec | null) {
  const n = dy.length, dxh = dy.map((v, i) => v * g[i])
  let m1 = 0, m2 = 0
  for (let i = 0; i < n; i++) { if (dg) dg[i] += dy[i] * c.xh[i]; if (db) db[i] += dy[i]; m1 += dxh[i]; m2 += dxh[i] * c.xh[i] }
  m1 /= n; m2 /= n
  return dxh.map((v, i) => c.r * (v - m1 - c.xh[i] * m2))
}
/** x · W + b with W stored [in × out] (GPT-2's Conv1D). */
function linF(x: Vec, Wt: Vec, b: Vec, out: number) {
  const y = Float64Array.from(b)
  for (let i = 0; i < x.length; i++) { const xi = x[i], r = i * out; if (xi) for (let j = 0; j < out; j++) y[j] += xi * Wt[r + j] }
  return y
}
function linB(dy: Vec, x: Vec, Wt: Vec, out: number, dW: Vec | null, db: Vec | null) {
  const dx = zeros(x.length)
  if (db) for (let j = 0; j < out; j++) db[j] += dy[j]
  for (let i = 0; i < x.length; i++) {
    const r = i * out, xi = x[i]
    let s = 0
    if (dW) for (let j = 0; j < out; j++) { dW[r + j] += xi * dy[j]; s += dy[j] * Wt[r + j] }
    else for (let j = 0; j < out; j++) s += dy[j] * Wt[r + j]
    dx[i] = s
  }
  return dx
}
const C0 = Math.sqrt(2 / Math.PI)
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(C0 * (x + 0.044715 * x ** 3)))
const geluD = (x: number) => { const u = C0 * (x + 0.044715 * x ** 3), t = Math.tanh(u); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C0 * (1 + 3 * 0.044715 * x * x) }

export function gpt2(W: Weights) {
  const w = (n: string) => { const t = W.get(n); if (!t) throw new Error(n); return t }

  /** The forward pass over ids[0..n−2], each predicting the next id; keeps the caches. */
  function forward(ids: number[]) {
    const N = ids.length - 1, inp = ids.slice(0, N), tgt = ids.slice(1)
    const wte = w('wte.weight'), wpe = w('wpe.weight')
    let x = inp.map((id, p) => { const v = zeros(D); for (let k = 0; k < D; k++) v[k] = wte[id * D + k] + wpe[p * D + k]; return v })
    const blocks: any[] = []
    for (let l = 0; l < L; l++) {
      const P = `h.${l}.`, c: any = {}
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
    return { N, inp, tgt, blocks, lnf, probs, losses, loss: losses.reduce((a, b) => a + b, 0) / N }
  }
  type Fwd = ReturnType<typeof forward>

  /**
   * Gradient of Σᵢ weights[i] · lossᵢ. Only weights for which want(name) is true get a gradient (default: all).
   * Also returns the gradient on the input embeddings, the size of the gradient on the residual stream at each
   * depth, and the input and output gradient of block 12's MLP output projection.
   */
  function backward(f: Fwd, weights: number[], want: (name: string) => boolean = () => true) {
    const G = new Map<string, Vec>(), g = (n: string) => { if (!want(n)) return null; let t = G.get(n); if (!t) G.set(n, (t = zeros(w(n).length))); return t }
    const wte = w('wte.weight'), dwte = g('wte.weight')
    let dx = f.lnf.map((o, i) => {
      const dz = f.probs[i].map((p, t) => (p - (t === f.tgt[i] ? 1 : 0)) * weights[i]), df = zeros(D)
      if (weights[i]) for (let t = 0; t < V; t++) { const d = dz[t]; if (!d) continue; const r = t * D; for (let k = 0; k < D; k++) { df[k] += d * wte[r + k]; if (dwte) dwte[r + k] += d * o.y[k] } }
      return lnB(df, o, w('ln_f.weight'), g('ln_f.weight'), g('ln_f.bias'))
    })
    const streamNorm: number[] = [lenRows(dx)]
    const slice: any = {}
    for (let l = L - 1; l >= 0; l--) {
      const P = `h.${l}.`, c = f.blocks[l]
      const dact = dx.map((d, i) => linB(d, c.hact[i], w(P + 'mlp.c_proj.weight'), D, g(P + 'mlp.c_proj.weight'), g(P + 'mlp.c_proj.bias')))
      if (l === L - 1) slice.mlp = { x: c.hact.map((r: Vec) => [...r.slice(0, 8)]), dy: dx.map((r) => [...r.slice(0, 8)]) }
      const dpre = dact.map((d, i) => d.map((v, k) => v * geluD(c.hpre[i][k])))
      const dln2 = dpre.map((d, i) => linB(d, c.ln2[i].y, w(P + 'mlp.c_fc.weight'), FF, g(P + 'mlp.c_fc.weight'), g(P + 'mlp.c_fc.bias')))
      const dx1 = dx.map((d, i) => { const b = lnB(dln2[i], c.ln2[i], w(P + 'ln_2.weight'), g(P + 'ln_2.weight'), g(P + 'ln_2.bias')); return d.map((v, k) => v + b[k]) })
      const datt = dx1.map((d, i) => linB(d, c.att[i], w(P + 'attn.c_proj.weight'), D, g(P + 'attn.c_proj.weight'), g(P + 'attn.c_proj.bias')))
      const dqkv = c.qkv.map(() => zeros(3 * D))
      for (let h = 0; h < H; h++) for (let i = 0; i < f.N; i++) {
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
    const dwpe = g('wpe.weight')
    dx.forEach((d, p) => { for (let k = 0; k < D; k++) { if (dwte) dwte[f.inp[p] * D + k] += d[k]; if (dwpe) dwpe[p * D + k] += d[k] } })
    return { G, dx0: dx, streamNorm: streamNorm.reverse(), slice }
  }
  return { forward, backward, w }
}
