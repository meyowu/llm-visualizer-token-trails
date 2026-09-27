/*
 * Real Mamba-130m outputs, exported offline by scripts/mamba-export.ts: next-token guesses for a prompt
 * and the input-dependent step size Δ (averaged over channels) in every layer for one sentence.
 */
import raw from '../../data/mamba.json'

export const mamba = raw as {
  model: string
  prompt: string
  /** Top next-token guesses (GPT-NeoX spellings, Ġ for a leading space). */
  next: { s: string; p: number }[]
  text: string
  toks: string[]
  textNext: { s: string; p: number }[]
  /** Mean Δ over the 1,536 channels: [layer][token]. */
  meanDelta: number[][]
}
