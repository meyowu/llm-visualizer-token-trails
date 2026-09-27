/*
 * Real BERT-base (uncased) outputs for a few fixed sentences, exported offline by scripts/bert-export.ts.
 * The browser never runs the model.
 */
import raw from '../../data/bert.json'

export interface MaskExample {
  text: string
  /** The words before the blank, as a left-to-right model sees them. */
  left: string
  /** WordPiece tokens, with [CLS] and [SEP]. */
  toks: string[]
  mask: number
  /** BERT's top predictions for the [MASK] position. */
  top: { t: string; p: number }[]
}
export interface Bert {
  model: string
  examples: MaskExample[]
  /** The head that most attends to the next token, on one sentence (0-based layer and head). */
  ahead: { toks: string[]; layer: number; head: number; score: number; A: number[][] }
  /** A sentence pair as BERT reads it, with segment ids. */
  pair: { toks: string[]; seg: number[] }
  /** A single sentence for the classification step. */
  cls: string[]
}
export const bert = raw as unknown as Bert
