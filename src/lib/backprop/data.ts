/*
 * Backpropagation through GPT-2 small on the Training sentence, computed offline in float64 by
 * scripts/backprop-export.ts: gradient sizes per weight matrix and block, the gradient on the residual stream,
 * a slice of one linear layer's backward pass, which positions each loss reaches, and finite-difference checks.
 */
import raw from '../../data/backprop.json'

export type Kind = 'ln_1.weight' | 'attn.c_attn.weight' | 'attn.c_proj.weight' | 'ln_2.weight' | 'mlp.c_fc.weight' | 'mlp.c_proj.weight'
export const bp = raw as unknown as {
  model: string
  text: string
  /** GPT-2 spellings (Ġ for a space). */
  tokens: string[]
  loss: number
  losses: number[]
  /** Numbers in each matrix of one block. */
  params: Record<Kind, number>
  /** ‖∂L/∂W‖ for every matrix of every block, block 1 first. */
  layers: Record<Kind, number>[]
  embed: { wte: number; wpe: number; ln_f: number }
  /** ‖∂L/∂x‖ on the residual stream: at the embeddings, then after each block. */
  stream: number[]
  /** blame[i][j] = ‖∂L_i/∂x_j‖: how strongly position i's loss reaches the embedding at position j. */
  blame: number[][]
  /** Block 12's MLP output projection: dims 1–8 of its input X (5 × 3072) and output gradient dY (5 × 768), and dW = Xᵀ · dY. */
  linear: { x: number[][]; dy: number[][]; dW: number[][] }
  checks: { name: string; index: number; row: number; col: number; backprop: number; finite: number }[]
}
