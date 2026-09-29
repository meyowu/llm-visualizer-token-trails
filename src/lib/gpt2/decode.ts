/*
 * Decoding the exported GPT-2 runs, apart from the data itself: the attention bytes, token texts, distributions at a
 * temperature and the lanes' colour mixing. data.ts applies it to src/data/gpt2.json; the opening animation applies it
 * to the one pass it draws (src/data/opening.json), so its page does not download the rest.
 */
import { symbolText } from './bpe'

export interface RawNext { top: { id: number; s: string; z: number }[]; zmax: number; tailLse: number[]; tailCount: number; tailQuantiles: number[] }
export interface RawPass {
  ids: number[]; syms: string[]; attn: string; lens: { id: number; s: string; p: number }[][]; next: RawNext; greedy: number
  wte: number[][]; wpe: number[][]; neurons: { idx: number[]; act: number[][] }
  unembed?: { h: string; xf: string; rows: { id: number; v: string }[] }
}
export const VOCAB = 50257

export interface Candidate { id: number; sym: string; text: string; z: number }
export interface Next {
  /** The 256 highest logits, exactly, highest first. */
  top: Candidate[]
  zmax: number
  /** log Σ exp((z − zmax) / T) over the other 50,001 tokens, one value per T in T_GRID. */
  tailLse: number[]
  tailCount: number
  /** 64 quantiles of the logits outside the top 256, highest first. */
  tailQuantiles: number[]
}
export interface Guess { id: number; text: string; p: number }
export interface GptPass {
  ids: number[]
  /** Token texts, decoded from GPT-2's byte symbols (" cat", "\n"). */
  texts: string[]
  /** att[layer][head][i][j] for j ≤ i; each row sums to 1. */
  att: number[][][][]
  /** The logit lens: top-3 guesses for the next token at the last position after each block. */
  lens: Guess[][]
  next: Next
  /** The most likely next token; the next pass appends it. */
  greedy: number
  /** First 16 of the 768 numbers of each token's W_E row and each position's W_P row. */
  wte: number[][]
  wpe: number[][]
  /** Block 1's MLP: 24 of the 3,072 neurons (after GELU), act[position][k]. */
  neurons: { idx: number[]; act: number[][] }
  /** Pass 1 only: the last position's stream before (h) and after ln_f (xf), and the W_E rows of the top 6 candidates. */
  unembed?: { h: Float32Array; xf: Float32Array; rows: { id: number; text: string; v: Float32Array }[] }
}
const floats = (b64: string) => new Float32Array(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer)

export function unpackAttention(b64: string, N: number): number[][][][] {
  const b = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
  let o = 0
  const att: number[][][][] = []
  for (let l = 0; l < 12; l++) {
    const layer: number[][][] = []
    for (let h = 0; h < 12; h++) {
      const A: number[][] = []
      for (let i = 0; i < N; i++) {
        const row = Array.from(b.subarray(o, o + i + 1), (v) => v / 255)
        o += i + 1
        const s = row.reduce((p, q) => p + q, 0) || 1
        A.push(row.map((v) => v / s))
      }
      layer.push(A)
    }
    att.push(layer)
  }
  return att
}

export const cand = (t: { id: number; s: string; z: number }): Candidate => ({ id: t.id, sym: t.s, text: symbolText(t.s), z: t.z })

export function decodePass(p: RawPass): GptPass {
  return {
    ids: p.ids,
    texts: p.syms.map(symbolText),
    att: unpackAttention(p.attn, p.ids.length),
    lens: p.lens.map((l) => l.map((g) => ({ id: g.id, text: symbolText(g.s), p: g.p }))),
    next: { ...p.next, top: p.next.top.map(cand) },
    greedy: p.greedy,
    wte: p.wte, wpe: p.wpe, neurons: p.neurons,
    unembed: p.unembed && {
      h: floats(p.unembed.h),
      xf: floats(p.unembed.xf),
      rows: p.unembed.rows.map((r) => ({ id: r.id, text: symbolText(p.next.top.find((t) => t.id === r.id)!.s), v: floats(r.v) })),
    },
  }
}

/** Index of T in a temperature grid (T is snapped to the 0.05 steps). */
export const tIndexOn = (grid: number[], T: number) => Math.max(0, Math.min(grid.length - 1, Math.round((T - grid[0]) / 0.05)))

export interface Dist {
  /** The first k candidates with their probability at T. */
  rows: (Candidate & { p: number })[]
  /** Probability of every other token together. */
  rest: number
  restCount: number
}
/** softmax(z / T) over the whole vocabulary, exact for T on the grid the tail was summarised at. */
export function nextDistOn(grid: number[], n: Next, T: number, k = 6): Dist {
  const Tg = grid[tIndexOn(grid, T)]
  const e = n.top.map((c) => Math.exp((c.z - n.zmax) / Tg))
  const Z = e.reduce((p, q) => p + q, 0) + Math.exp(n.tailLse[tIndexOn(grid, T)])
  const rows = n.top.slice(0, k).map((c, i) => ({ ...c, p: e[i] / Z }))
  return { rows, rest: 1 - rows.reduce((p, r) => p + r.p, 0), restCount: VOCAB - k }
}

/** Share of each position's residual stream kept per block in the colour mixing (the rest comes from attention). */
export const KEEP = 0.7
/**
 * Where each position's information came from after each block, for the colours of the lanes:
 * mix[l][i][k] is the share of token k in position i's lane after block l + 1. Each block keeps
 * KEEP of the lane and takes the rest from its 12 heads' attention, averaged. Heads that park
 * their attention on the first token (an attention sink) count as reading nothing.
 */
export function mixing(att: number[][][][]): number[][][] {
  const N = att[0][0].length
  let M: number[][] = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (__, k) => (i === k ? 1 : 0)))
  const out: number[][][] = []
  for (const layer of att) {
    const B: number[][] = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (__, j) => (i === j ? KEEP : 0)))
    for (const A of layer) {
      let sink = 0
      for (let i = 1; i < N; i++) sink += A[i][0]
      const isSink = N > 1 && sink / (N - 1) > 0.6
      for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
        const a = (A[i][j] / layer.length) * (1 - KEEP)
        if (isSink && j === 0 && i > 0) B[i][i] += a
        else B[i][j] += a
      }
    }
    M = B.map((row) => M[0].map((_, k) => row.reduce((s, b, j) => s + b * M[j][k], 0)))
    out.push(M)
  }
  return out
}
