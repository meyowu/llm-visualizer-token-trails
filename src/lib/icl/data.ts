/*
 * In-context learning measured on real models, exported offline by scripts/icl-export.ts: GPT-2 small's
 * next-token distribution after 0 to 3 examples of three tasks, the average over twelve countries, the
 * attention row of the head that copies earlier answers, and a base and an instruction-tuned model
 * given only an instruction. Tokens are GPT-2 byte-level spellings (Ġ, Ċ).
 */
import raw from '../../data/icl.json'

/** Top tokens with their probabilities. */
export type Top = [string, number][]
export interface ShotRow {
  /** Number of examples before the query. */
  n: number
  prompt: string[]
  top: Top
  /** Probability of the answer's first token, and whether it is the top token. */
  p: number
  top1: boolean
}
export interface Task { examples: [string, string][]; answer: string; rows: ShotRow[] }
export const icl = raw as {
  query: string
  tasks: Record<'capital' | 'language' | 'continent', Task>
  /** Capital task averaged over `countries` query countries, 0 to 4 examples. */
  curve: { n: number; p: number; top1: number; tokens: number }[]
  countries: number
  /** The head (0-based layer and head) with most attention from the last token onto the earlier answers. */
  head: { layer: number; head: number; prompt: string[]; row: number[]; answers: number[]; share: number }
  instruction: string
  base: { prompt: string[]; top: Top; continuation: string; p: number }
  tuned: { model: string; chat: string; tokens: number; answer: string; top: Top }
}
