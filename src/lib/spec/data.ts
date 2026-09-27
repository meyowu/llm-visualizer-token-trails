/*
 * A real speculative decoding run, exported offline by scripts/spec-export.ts: distilgpt2 drafts
 * 4 tokens per round, GPT-2 small checks them. Tokens are GPT-2 byte-level spellings (Ġ, Ċ).
 */
import raw from '../../data/spec.json'

export interface SpecRound {
  /** Tokens already fixed before the round. */
  ctx: number
  drafted: string[]
  /** GPT-2's own greedy choice at each drafted position. */
  choice: string[]
  /** GPT-2's (p) and distilgpt2's (q) probability of each drafted token. */
  p: number[]
  q: number[]
  accepted: number
  /** GPT-2's token after the accepted prefix (its correction, or a bonus if all were accepted). */
  next: string
}
export const spec = raw as { target: string; draft: string; k: number; prompt: string[]; rounds: SpecRound[]; output: string[]; identical: boolean; targetPasses: number; draftPasses: number }
/** A GPT-2 spelling as plain text: Ġ → space, Ċ → newline. */
export const plain = (s: string) => s.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n')
