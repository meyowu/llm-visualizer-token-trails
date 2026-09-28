/*
 * Supervised fine-tuning measured on real models, exported offline by scripts/sft-export.ts: Qwen3-1.7B-Base and
 * Qwen3-1.7B (after post-training) on one chat-formatted example, token by token, what each writes for its prompt,
 * and each model's loss on both answers.
 */
import raw from '../../data/sft.json'

export interface SftToken {
  /** The token as text (special tokens as themselves). */
  text: string
  /** −log p of this token under each model; null for the first token. */
  base: number | null
  tuned: number | null
  /** Whether SFT's loss counts this token (the answer), or masks it (the prompt and template). */
  trained: boolean
}
export const sft = raw as unknown as {
  model: string
  base: string
  user: string
  answer: string
  tokens: SftToken[]
  promptTokens: number
  answerLoss: { base: number; tuned: number }
  wrote: { base: string; tuned: string }
  /** Mean loss per token of each model on [base's answer, tuned's answer]. */
  cross: { base: number[]; tuned: number[] }
}
