/*
 * A GPT-2-shaped model in plain TypeScript for the offline export scripts (GPT-2 small, distilgpt2):
 * safetensors loading and a forward pass that returns next-token log-probabilities at every position.
 */
import { readFileSync, openSync, readSync } from 'node:fs'

export const D = 768, H = 12, DH = 64, V = 50257, FF = 3072
export type Weights = Map<string, Float32Array>

export function loadSafetensors(path: string): Map<string, Float32Array> {
  const fd = openSync(path, 'r')
  const lenBuf = Buffer.alloc(8)
  readSync(fd, lenBuf, 0, 8, 0)
  const hlen = Number(lenBuf.readBigUInt64LE(0))
  const hbuf = Buffer.alloc(hlen)
  readSync(fd, hbuf, 0, hlen, 8)
  const header = JSON.parse(hbuf.toString('utf8'))
  const all = readFileSync(path)
  const out = new Map<string, Float32Array>()
  for (const [name, t] of Object.entries<any>(header)) {
    if (name === '__metadata__' || t.dtype !== 'F32') continue
    const [a, b] = t.data_offsets as [number, number]
    const copy = new Float32Array((b - a) / 4)
    Buffer.from(copy.buffer).set(all.subarray(8 + hlen + a, 8 + hlen + b))
    out.set(name.replace(/^transformer\./, ''), copy)
  }
  return out
}
function layerNorm(x: Float32Array, g: Float32Array, b: Float32Array): Float32Array {
  let m = 0
  for (let i = 0; i < x.length; i++) m += x[i]
  m /= x.length
  let v = 0
  for (let i = 0; i < x.length; i++) v += (x[i] - m) ** 2
  const s = Math.sqrt(v / x.length + 1e-5), out = new Float32Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = ((x[i] - m) / s) * g[i] + b[i]
  return out
}
/** x · W + b with W stored [in × out] (GPT-2's Conv1D). */
function linear(x: Float32Array, Wt: Float32Array, b: Float32Array, out: number): Float32Array {
  const y = Float32Array.from(b)
  for (let i = 0; i < x.length; i++) { const xi = x[i], row = i * out; if (xi) for (let j = 0; j < out; j++) y[j] += xi * Wt[row + j] }
  return y
}
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))

/**
 * A GPT-2-shaped model over weights W; returns log-probabilities for the next token at every position.
 * onAttn receives each head's attention weights (rows[i][j], j ≤ i).
 */
export function model(W: Weights, L: number, onMlpIn?: (layer: number, rows: Float32Array[]) => void, onAttn?: (layer: number, head: number, rows: number[][]) => void) {
  const w = (n: string) => { const t = W.get(n); if (!t) throw new Error(n); return t }
  const wte = w('wte.weight'), wpe = w('wpe.weight')
  return (ids: number[]): Float32Array[] => {
    const N = ids.length
    let x = ids.map((id, p) => { const v = new Float32Array(D); for (let k = 0; k < D; k++) v[k] = wte[id * D + k] + wpe[p * D + k]; return v })
    for (let l = 0; l < L; l++) {
      const P = `h.${l}.`
      const qkv = x.map((r) => linear(layerNorm(r, w(P + 'ln_1.weight'), w(P + 'ln_1.bias')), w(P + 'attn.c_attn.weight'), w(P + 'attn.c_attn.bias'), 3 * D))
      const o = x.map(() => new Float32Array(D))
      for (let h = 0; h < H; h++) {
        const rows: number[][] = []
        for (let i = 0; i < N; i++) {
          const sc: number[] = []
          for (let j = 0; j <= i; j++) { let s = 0; for (let k = 0; k < DH; k++) s += qkv[i][h * DH + k] * qkv[j][D + h * DH + k]; sc.push(s / 8) }
          const m = Math.max(...sc), e = sc.map((s) => Math.exp(s - m)), z = e.reduce((a, b) => a + b, 0)
          rows.push(e.map((v) => v / z))
          for (let j = 0; j <= i; j++) { const p = e[j] / z; for (let k = 0; k < DH; k++) o[i][h * DH + k] += p * qkv[j][2 * D + h * DH + k] }
        }
        onAttn?.(l, h, rows)
      }
      x = x.map((r, i) => { const p = linear(o[i], w(P + 'attn.c_proj.weight'), w(P + 'attn.c_proj.bias'), D); return r.map((v, k) => v + p[k]) })
      const mlpIn = x.map((r) => layerNorm(r, w(P + 'ln_2.weight'), w(P + 'ln_2.bias')))
      onMlpIn?.(l, mlpIn)
      x = x.map((r, i) => { const hid = linear(mlpIn[i], w(P + 'mlp.c_fc.weight'), w(P + 'mlp.c_fc.bias'), FF).map(gelu); const p = linear(hid, w(P + 'mlp.c_proj.weight'), w(P + 'mlp.c_proj.bias'), D); return r.map((v, k) => v + p[k]) })
    }
    return x.map((r) => {
      const f = layerNorm(r, w('ln_f.weight'), w('ln_f.bias')), z = new Float32Array(V)
      for (let t = 0; t < V; t++) { let s = 0; for (let k = 0; k < D; k++) s += f[k] * wte[t * D + k]; z[t] = s }
      let m = -Infinity
      for (const v of z) if (v > m) m = v
      let Z = 0
      for (const v of z) Z += Math.exp(v - m)
      const lz = m + Math.log(Z)
      return z.map((v) => v - lz)
    })
  }
}

