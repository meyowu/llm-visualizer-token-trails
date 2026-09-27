/*
 * GPT-2 small quantized for real, offline, by scripts/quant-export.ts: loss and next-token guesses for
 * each scheme, one real weight column and one real activation vector.
 */
import raw from '../../data/quant.json'

export interface QuantResult {
  name: string
  bits: number
  scale: 'tensor' | 'channel' | 'group'
  /** Mean cross-entropy on the paragraph, and its perplexity. */
  loss: number
  ppl: number
  /** KL divergence of the next-token distribution from full precision. */
  kl: number
  /** RMS error of all quantized weights, relative to their RMS. */
  relErr: number
  top: { s: string; p: number }[]
}
export const quant = raw as { model: string; text: string; tokens: number; prompt: string; results: QuantResult[]; column: number[]; activation: number[] }
