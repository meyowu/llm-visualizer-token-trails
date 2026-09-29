/*
 * Real GPT-2 small activations for a few fixed prompts, exported offline by scripts/gpt2-export.ts.
 * The browser never runs the model: this module only decodes src/data/gpt2.json and derives
 * what the views draw from it (distributions at a temperature, head kinds, colour mixing).
 */
import raw from '../../data/gpt2.json'
import { symbolText } from './bpe'
import { decodePass, nextDistOn, tIndexOn, type Dist, type GptPass, type Guess, type Next, type RawPass } from './decode'

export * from './decode'

interface RawPreset { key: string; text: string; note: string; passes: RawPass[] }
interface RawTraining { text: string; ids: number[]; syms: string[]; positions: { target: number; p: number; top: { id: number; s: string; p: number }[] }[] }
interface RawMapItem { s: string; cat: string; x: number; y: number; nn: [string, number][] }
interface Raw { model: string; tGrid: number[]; topK: number; presets: RawPreset[]; training: RawTraining; embeddingMap: RawMapItem[]; wpeSlice: number[][]; streamNorms: number[]; leftOnly: { left: string; top: { s: string; p: number }[] }[]; kv: KvSlice }
const R = raw as unknown as Raw

/** Temperatures the tail of the distribution was summarised at (0.2 … 2.0, step 0.05). */
export const T_GRID = R.tGrid

/** Index of T in T_GRID (T is snapped to the 0.05 grid). */
export const tIndex = (T: number) => tIndexOn(T_GRID, T)
/** softmax(z / T) over the whole vocabulary, exact for T on the grid. */
export const nextDist = (n: Next, T: number, k = 6): Dist => nextDistOn(T_GRID, n, T, k)

export interface Preset { key: string; text: string; note: string; passes: GptPass[] }
/** Short names for the prompt switchers. */
export const PROMPT_LABELS: Record<string, string> = { cat: '“The cat sat…”', france: '“capital of France”', count: '“one two three…”' }

let cache: Preset[] | null = null
/** The prompts with real runs, each continued greedily for three passes. */
export function presets(): Preset[] {
  cache ??= R.presets.map((p) => ({ key: p.key, text: p.text, note: p.note, passes: p.passes.map(decodePass) }))
  return cache
}

export type HeadKind = 'previous token' | 'first token' | 'self' | 'induction' | 'mixed'
/** A rough label for what one head does on this context, from its average attention pattern. */
export function headKind(A: number[][], ids: number[]): HeadKind {
  const N = A.length
  if (N < 2) return 'mixed'
  const ind: number[] = []
  for (let i = 1; i < N; i++) {
    let s = 0, hit = false
    for (let k = 0; k + 1 < i; k++) if (ids[k] === ids[i]) { s += A[i][k + 1]; hit = true }
    if (hit) ind.push(s)
  }
  const mean = (f: (i: number) => number) => { let s = 0; for (let i = 1; i < N; i++) s += f(i); return s / (N - 1) }
  if (ind.length && ind.reduce((p, q) => p + q, 0) / ind.length > 0.4) return 'induction'
  if (mean((i) => A[i][i - 1]) > 0.5) return 'previous token'
  if (mean((i) => A[i][i]) > 0.5) return 'self'
  if (mean((i) => A[i][0]) > 0.7) return 'first token'
  return 'mixed'
}

/** One sentence scored the way training scores it: every position predicts the token after it. */
export interface TrainingRun {
  text: string
  ids: number[]
  texts: string[]
  /** For position i: the target (token i + 1), GPT-2's probability for it, and its top guesses. */
  positions: { target: number; p: number; top: Guess[] }[]
}
export function trainingRun(): TrainingRun {
  const t = R.training
  return {
    text: t.text, ids: t.ids, texts: t.syms.map(symbolText),
    positions: t.positions.map((q) => ({ target: q.target, p: q.p, top: q.top.map((g) => ({ id: g.id, text: symbolText(g.s), p: g.p })) })),
  }
}

/** Words from everyday categories placed by the first two principal components of their W_E rows, with their nearest rows. */
export interface MapItem { text: string; cat: string; x: number; y: number; near: { text: string; cos: number }[] }
export const embeddingMap = (): MapItem[] =>
  R.embeddingMap.map((m) => ({ text: symbolText(m.s), cat: m.cat, x: m.x, y: m.y, near: m.nn.map(([s, cos]) => ({ text: symbolText(s), cos })) }))

/** GPT-2's learned position rows 0–63, dimensions 0–63. */
export const wpeSlice = (): number[][] => R.wpeSlice
/** Mean length of GPT-2's residual stream on the training text: after the embedding, then after each of the 12 blocks. */
export const streamNorms = (): number[] => R.streamNorms
/** GPT-2's next-token guesses from only the words before a blank (the BERT page's examples). */
export const leftOnly = () => R.leftOnly
/** One real attention head for the KV cache page: q, k, v rows (64 each) for a prompt plus its first generated token. */
export interface KvSlice { ids: number[]; syms: string[]; layer: number; head: number; q: number[][]; k: number[][]; v: number[][]; attn: number[] }
export const kvSlice = () => R.kv
