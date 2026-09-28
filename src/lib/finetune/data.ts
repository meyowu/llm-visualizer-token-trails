/*
 * Real fine-tuning runs on GPT-2 small, exported offline by scripts/finetune-export.ts (float64, Adam): rank-4 LoRA
 * against full fine-tuning on two sentences, and DPO with the same adapters on three preference pairs; and, from
 * scripts/lora-export.ts, how much of Qwen3-1.7B's real post-training change each rank captures.
 */
import raw from '../../data/finetune.json'
import loraRaw from '../../data/lora.json'

/** Probability of " moon" after each training sentence's "… on the", and after an unseen sentence, with its top 5. */
export interface Probes { train: number[]; eval: number; evalTop: [string, number][] }
export interface DpoStep { loss: number; pairs: { margin: number; chosen: number; rejected: number }[] }
export const ft = raw as unknown as {
  model: string
  lora: {
    rank: number; matrices: string; train: string[]; eval: string; target: string; steps: number; lr: { lora: number; full: number }
    run: { curve: number[]; before: Probes; after: Probes; trainable: number }
    full: { curve: number[]; after: Probes; trainable: number; rank: { matrix: string; ranks: number[]; captured: number[] } }
  }
  dpo: {
    beta: number
    pairs: { prompt: string; chosen: string; rejected: string; ref: { chosen: number; rejected: number } }[]
    steps: DpoStep[]
    /** The policy's top next tokens after the first prompt, before and after. */
    before: [string, number][]
    after: [string, number][]
  }
}
export interface QwenMat { name: string; shape: [number, number]; relChange: number; delta: { sigma: number[]; captured: number[] }; weight: { sigma: number[]; captured: number[] } }
export const qwenDelta = loraRaw as unknown as { base: string; tuned: string; layer: number; k: number; ranks: number[]; mats: QwenMat[] }
