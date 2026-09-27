import { gauss, rng } from '../../core/util'

/* ---------- tokens ---------- */

export interface Tok {
  text: string
  id: number
  /** Colour index; token i gets hue i % 7. */
  c: number
}

export const PROMPT = ['The', ' cat', ' sat', ' on', ' the']

/** GPT-2 ids of the prompt's tokens. */
const IDS: Record<string, number> = { 'The': 464, ' cat': 3797, ' sat': 3332, ' on': 319, ' the': 262 }
export const promptTokens = (): Tok[] => PROMPT.map((t, i) => ({ text: t, id: IDS[t], c: i % 7 }))

/* ---------- toy attention block (real arithmetic at toy scale) ---------- */

export type M = number[][]
export const TOY = { d: 8, heads: 2, dh: 4 }

export const matmul = (A: M, B: M): M => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)))
export const transpose = (A: M): M => A[0].map((_, j) => A.map((r) => r[j]))
const randMat = (r: () => number, rows: number, cols: number, s: number): M =>
  Array.from({ length: rows }, () => Array.from({ length: cols }, () => gauss(r) * s))

/** Row `t.id` of the toy token-embedding matrix W_E. */
export function tokEmb(t: { id: number }): number[] {
  const r = rng(t.id * 9973 + 17)
  return Array.from({ length: TOY.d }, () => gauss(r))
}
/** Row `i` of the toy position-embedding matrix W_P. GPT-2 learns these; the toy uses a sinusoid. */
export function posEmb(i: number): number[] {
  return Array.from({ length: TOY.d }, (_, k) => 0.5 * Math.sin(i / Math.pow(10, (2 * Math.floor(k / 2)) / TOY.d) + (k % 2) * (Math.PI / 2)))
}
/** Token embedding plus position embedding: the residual stream entering block 1. */
export function embedRow(t: Tok, i: number): number[] {
  const e = tokEmb(t), p = posEmb(i)
  return e.map((v, k) => v + p[k])
}

export interface LnParams { gamma: number[]; beta: number[] }
export const LN_EPS = 1e-5
function lnParams(seed: number): LnParams {
  const r = rng(seed)
  return { gamma: Array.from({ length: TOY.d }, () => 1 + 0.15 * gauss(r)), beta: Array.from({ length: TOY.d }, () => 0.1 * gauss(r)) }
}
/** Learned scale γ and shift β of the three LayerNorms in the toy model. */
export const LN = { ln1: lnParams(101), ln2: lnParams(202), lnf: lnParams(303) }

export interface LnStats { mu: number; sigma: number; xhat: number[]; out: number[] }
/** LayerNorm over one token's features: subtract the mean, divide by the std, then γ ⊙ x̂ + β. */
export function lnStats(v: number[], p: LnParams): LnStats {
  const mu = v.reduce((a, b) => a + b, 0) / v.length
  const sigma = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length + LN_EPS)
  const xhat = v.map((x) => (x - mu) / sigma)
  return { mu, sigma, xhat, out: xhat.map((x, k) => p.gamma[k] * x + p.beta[k]) }
}
export const layerNorm = (v: number[], p: LnParams) => lnStats(v, p).out

export const WEIGHT_SEED = 5
function weights() {
  const r = rng(WEIGHT_SEED * 104729 + 3)
  const Wq: M[] = [], Wk: M[] = [], Wv: M[] = []
  for (let h = 0; h < TOY.heads; h++) {
    Wq.push(randMat(r, TOY.d, TOY.dh, 0.5))
    Wk.push(randMat(r, TOY.d, TOY.dh, 0.5))
    Wv.push(randMat(r, TOY.d, TOY.dh, 0.55))
  }
  return { Wq, Wk, Wv, Wo: randMat(r, TOY.d, TOY.d, 0.4) }
}
const W = weights()

export interface Head {
  Wq: M; Wk: M; Wv: M
  Q: M; K: M; V: M
  /** Raw scores Q·Kᵀ. */
  S: M
  /** Scaled scores S / √d_head. */
  Ss: M
  /** Causal softmax weights; zero above the diagonal. */
  A: M
  O: M
}

export interface Attn {
  /** Residual stream entering the block. */
  h: M
  /** ln_1(h). */
  X: M
  heads: Head[]
  concat: M
  Wo: M
  out: M
  /** h + out: the residual stream leaving the attention sub-layer. */
  resid: M
}

export function attention(seq: Tok[]): Attn {
  const h = seq.map((t, i) => embedRow(t, i))
  const X = h.map((r) => layerNorm(r, LN.ln1))
  const heads = Array.from({ length: TOY.heads }, (_, k): Head => {
    const Q = matmul(X, W.Wq[k]), K = matmul(X, W.Wk[k]), V = matmul(X, W.Wv[k])
    const S = matmul(Q, transpose(K))
    const Ss = S.map((r) => r.map((v) => v / Math.sqrt(TOY.dh)))
    const A = Ss.map((r, i) => {
      const m = Math.max(...r.slice(0, i + 1))
      const e = r.map((v, j) => (j <= i ? Math.exp(v - m) : 0))
      const z = e.reduce((a, b) => a + b, 0)
      return e.map((v) => v / z)
    })
    return { Wq: W.Wq[k], Wk: W.Wk[k], Wv: W.Wv[k], Q, K, V, S, Ss, A, O: matmul(A, V) }
  })
  const concat = X.map((_, i) => heads.flatMap((hd) => hd.O[i]))
  const out = matmul(concat, W.Wo)
  return { h, X, heads, concat, Wo: W.Wo, out, resid: h.map((r, i) => r.map((v, k) => v + out[i][k])) }
}

/**
 * Where each lane's information comes from after the toy attention block: the stream keeps 45% of
 * itself and takes 55% from attention, averaged over the heads (the attention output of a lane is
 * that average, before W_O). Used to colour the lanes in every detail view.
 */
export const laneMix = (att: Attn): M =>
  att.heads[0].A.map((_, i) => att.heads[0].A.map((__, k) => 0.45 * (k === i ? 1 : 0) + (0.55 * att.heads.reduce((s, h) => s + (h.A[i][k] ?? 0), 0)) / att.heads.length))

/* ---------- toy MLP block ---------- */

/** GPT-2's tanh approximation of GELU ("gelu_new"). */
export const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
/** Minimum of GELU, reached near x ≈ −0.75. */
export const GELU_MIN = -0.17

export const TOY_FF = TOY.d * 4

function mlpWeights() {
  const r = rng(WEIGHT_SEED * 7919 + 11)
  const vec = (n: number, s: number) => Array.from({ length: n }, () => gauss(r) * s)
  return { Wfc: randMat(r, TOY.d, TOY_FF, 0.45), bfc: vec(TOY_FF, 0.3), Wproj: randMat(r, TOY_FF, TOY.d, 0.25), bproj: vec(TOY.d, 0.2) }
}
const WM = mlpWeights()

export interface Mlp {
  /** Residual stream entering the MLP (the attention sub-layer's output). */
  h: M
  /** ln_2(h). */
  X: M
  Wfc: M; bfc: number[]
  /** X·W_fc + b_fc, before the activation. */
  H: M
  /** GELU(H). */
  G: M
  Wproj: M; bproj: number[]
  /** G·W_proj + b_proj. */
  Y: M
  /** h + Y. */
  out: M
}

export function mlp(seq: Tok[], att: Attn = attention(seq)): Mlp {
  const h = att.resid, X = h.map((r) => layerNorm(r, LN.ln2))
  const H = matmul(X, WM.Wfc).map((r) => r.map((v, j) => v + WM.bfc[j]))
  const G = H.map((r) => r.map(gelu))
  const Y = matmul(G, WM.Wproj).map((r) => r.map((v, j) => v + WM.bproj[j]))
  return { h, X, Wfc: WM.Wfc, bfc: WM.bfc, H, G, Wproj: WM.Wproj, bproj: WM.bproj, Y, out: h.map((r, i) => r.map((v, k) => v + Y[i][k])) }
}
