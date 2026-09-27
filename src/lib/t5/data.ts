/*
 * Real T5-small outputs, exported offline by scripts/t5-export.ts: answers to a few task prefixes,
 * a span-corruption fill, and the learned relative-position biases. The browser never runs the model.
 */
import raw from '../../data/t5.json'

export interface T5Run {
  input: string
  /** SentencePiece pieces of the input (▁ marks a word start), ending in </s>. */
  src: string[]
  /** The generated pieces, greedy, ending in </s>. */
  out: string[]
  text: string
}
export interface T5 {
  model: string
  tasks: T5Run[]
  spans: T5Run
  /** Learned bias per bucket (32) and head (8), from the first layer's self-attention; shared by all layers. */
  bias: { encoder: number[][]; decoder: number[][] }
}
export const t5 = raw as unknown as T5

/** T5's relative-position bucket: rel = key position − query position (32 buckets, max distance 128). */
export function bucket(rel: number, bidirectional: boolean) {
  let nb = 32, ret = 0, n = rel
  if (bidirectional) { nb /= 2; if (n > 0) ret += nb; n = Math.abs(n) } else n = Math.max(-n, 0)
  const exact = nb / 2
  if (n < exact) return ret + n
  return ret + Math.min(nb - 1, exact + Math.trunc((Math.log(n / exact) / Math.log(128 / exact)) * (nb - exact)))
}
