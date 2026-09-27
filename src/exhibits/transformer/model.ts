import { clamp, gauss, hash, rng } from '../../core/util'

/* ---------- tokens ---------- */

export interface Tok {
  text: string
  id: number
  /** Colour index; token i gets hue i % 7. */
  c: number
}

export const PROMPT = ['The', ' cat', ' sat', ' on', ' the']

/** Real GPT-2 token ids for the words the exhibit uses; anything else gets a stable fake id. */
const IDS: Record<string, number> = {
  'The': 464, ' cat': 3797, ' sat': 3332, ' on': 319, ' the': 262, ' mat': 2603, ' floor': 4314, ' bed': 3996,
  ' ground': 2323, '.': 13, ',': 11, ' and': 290, ' in': 287, ' with': 351, ' while': 981, ' It': 632, ' The': 383,
  ' She': 1375, ' He': 679, '\n': 198, ' was': 373, ' couch': 18507, ' edge': 5743,
}
export const tokId = (t: string) => IDS[t] ?? 1000 + (hash(t) % 40000)
export const promptTokens = (): Tok[] => PROMPT.map((t, i) => ({ text: t, id: tokId(t), c: i % 7 }))

/* ---------- next-token distribution (canned, temperature-aware) ---------- */

const NOUNS = new Set([' mat', ' floor', ' couch', ' bed', ' edge', ' ground', ' windowsill', ' porch', ' rug', ' roof', ' keyboard'])
export const FUNCTION_WORDS = new Set(['The', ' the', ' on', ' in', ' with', ' and', '.', ',', ' while', '\n', ' It', ' The', ' She', ' He', ' Then', ' was'])
export const TAIL = [' windowsill', ' porch', ' rug', ' keyboard', ' roof']

function nextLogits(last: string): [string, number][] {
  if (last === ' the') return [[' mat', 3.2], [' floor', 2.3], [' couch', 1.9], [' bed', 1.6], [' edge', 1.4], [' ground', 1.3]]
  if (NOUNS.has(last)) return [['.', 3.0], [',', 2.3], [' and', 2.0], [' while', 1.1], [' in', 0.9], [' with', 0.8]]
  if (last === '.') return [[' It', 2.4], [' The', 2.2], ['\n', 1.5], [' She', 1.3], [' He', 1.2], [' Then', 0.9]]
  if (last === ',') return [[' and', 2.6], [' purring', 1.9], [' watching', 1.6], [' then', 1.3], [' as', 1.2], [' still', 1.0]]
  if (last === ' The') return [[' cat', 3.0], [' dog', 1.8], [' sun', 1.5], [' room', 1.3], [' house', 1.1], [' mat', 1.0]]
  if ([' It', ' She', ' He', ' Then', '\n'].includes(last)) return [[' was', 2.8], [' looked', 1.9], [' seemed', 1.7], [' purred', 1.6], [' had', 1.4], [' yawned', 1.3]]
  return [['.', 2.4], [' softly', 1.8], [' and', 1.7], [',', 1.6], [' quietly', 1.2], [' again', 1.0]]
}

export interface DistRow {
  text: string | null
  p: number
  other?: boolean
}

/** Top-6 candidates plus one row for the remaining 50,251 vocabulary entries. */
/** Toy logit shared by every token outside the top 6; the "others" row is 50,251 × exp(−11 / T). */
export const TAIL_LOGIT = -11
/** The canned top-6 logits after `last`, highest first. */
export const topLogits = (last: string) => nextLogits(last).slice().sort((a, b) => b[1] - a[1])

export function dist(last: string, T: number): DistRow[] {
  const L = nextLogits(last)
  const ex = L.map(([, l]) => Math.exp(l / T))
  const rest = 50251 * Math.exp(-11 / T)
  const Z = ex.reduce((a, b) => a + b, 0) + rest
  const rows: DistRow[] = L.map(([s], k) => ({ text: s, p: ex[k] / Z })).sort((a, b) => b.p - a.p)
  rows.push({ text: null, p: rest / Z, other: true })
  return rows
}

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

/* ---------- forward-pass data for the overview ---------- */

export interface Pass {
  N: number
  seq: Tok[]
  emb: number[][]
  /** 12 heads; heads 1–2 come from the toy block above, the rest are synthetic. */
  att: M[]
  /** Colour-mixing weights after each of the 12 blocks. */
  mix: M[]
  /** 24 MLP neuron activations per token, in [0, 1]. */
  act: number[][]
}

export function buildPass(seq: Tok[]): Pass {
  const N = seq.length
  const emb = seq.map((t) => { const r = rng(t.id * 9973 + 17); return Array.from({ length: 16 }, () => clamp(gauss(r) * 0.5, -1, 1)) })
  const toy = attention(seq)
  const att: M[] = []
  for (let h = 0; h < 12; h++) {
    if (h < TOY.heads) { att.push(toy.heads[h].A); continue }
    const r = rng(h * 7919 + N * 31 + seq[N - 1].id)
    const H: M = []
    for (let i = 0; i < N; i++) {
      const lg: number[] = []
      for (let j = 0; j <= i; j++) {
        let v = gauss(r) * 0.9 + (j === 0 ? 1.0 : 0) + (j === i ? 0.8 : 0) + (i - j === 1 ? 0.5 : 0)
        if (h === 2 && !FUNCTION_WORDS.has(seq[j].text)) v += 1.1
        lg.push(v)
      }
      const m = Math.max(...lg), e = lg.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0)
      H.push(e.map((x) => x / z))
    }
    att.push(H)
  }
  const mix: M[] = []
  let Mx: M = seq.map((_, i) => seq.map((__, k) => (k === i ? 1 : 0)))
  Mx = mixStep(Mx, att[0], 0.45)
  mix.push(Mx)
  for (let l = 1; l < 12; l++) { Mx = mixStep(Mx, att[l], 0.78); mix.push(Mx) }
  // MLP neurons in the overview are the toy block's first 24 GELU outputs, positive part, normalised
  const G = mlp(seq, toy).G, gmax = Math.max(...G.flat())
  const act = G.map((row) => row.slice(0, 24).map((g) => Math.max(0, g) / gmax))
  return { N, seq, emb, att, mix, act }
}

/** Colour mixing through one attention layer: keep `self` of your own hue, take the rest from what you attend to. */
export function mixStep(Mx: M, A: M, self: number): M {
  return Mx.map((row, i) => row.map((v, k) => {
    let a = 0
    for (let j = 0; j <= i; j++) a += A[i][j] * Mx[j][k]
    return self * v + (1 - self) * a
  }))
}
