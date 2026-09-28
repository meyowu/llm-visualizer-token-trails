/*
 * A small scaling experiment, exported offline by scripts/scaling-export.ts: GPT-2 small, medium and large (one
 * recipe at three sizes) scoring the same text, which none of them saw in training, with each model's size and its
 * loss on every token.
 */
import raw from '../../data/scaling.json'

export interface SizeRun { name: string; layers: number; width: number; params: number; nonEmbedding: number; loss: number; losses: number[] }
export const scaling = raw as unknown as { text: string; tokens: number; terms: string[]; models: SizeRun[] }
