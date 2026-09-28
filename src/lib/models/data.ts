import raw from '../../data/models.json'

/*
 * The model library (src/data/models.json, written by scripts/models-export.ts from each model's config.json and
 * checkpoint headers). Parameter counts come from the real tensor shapes; the KV cache from each layer's real
 * key/value (or latent, or state) size.
 */

/** A layer kind's letter: F full attention, S sliding window, C chunked, M latent (MLA), L linear; lower case: no RoPE. */
export type LayerCode = 'F' | 'S' | 'C' | 'M' | 'L' | 'f' | 's' | 'c' | 'm' | 'l'
export interface LayerKind { perToken: number; window?: number; state?: number }

export interface Model {
  id: string; name: string; family: string; org: string; released: string; repo: string
  layers: number; hidden: number; vocab: number; heads: number; kvHeads: number; headDim: number
  context: number; tied: boolean
  pos: { kind: 'learned' } | { kind: 'none' } | { kind: 'rope'; layers: number; theta: number; localTheta: number | null; partial: number; scaling: string | null }
  norm: 'LayerNorm' | 'RMSNorm'; sandwich: boolean; qkNorm: boolean; qkvBias: boolean; sinks: boolean; gatedAttn: boolean
  act: 'GELU' | 'GeGLU' | 'SwiGLU'
  softcap: { attn: number | null; final: number | null } | null
  mla: { latent: number; rope: number; qLatent: number | null; vHeadDim: number } | null
  mtpLayers: number
  indexer: { heads: number; dim: number; topk: number } | null
  attn: { codes: string; kinds: Record<string, LayerKind> }
  /** One letter per layer: D a dense MLP, E experts. */
  mlp: string
  moe: { experts: number; topK: number; shared: number; width: number; layers: number; dense: number } | null
  denseWidth?: number
  params: { embed: number; pos: number; head: number; attn: number; indexer: number; mlp: number; routed: number; shared: number; router: number; norm: number; total: number; active: number; mtp: number; vision: number }
  published: { total: number; active: number | null; source: string; leaves: 'all' | 'no-embed' | 'no-embed-head' | null }
  hub: number | null
}

export const MODELS: Model[] = (raw as unknown as { models: Model[] }).models
export const modelById = (id: string) => MODELS.find((m) => m.id === id) ?? MODELS[0]

/** The dropdown's groups: families in the order they first appear, models oldest first. */
export function families(): [string, [string, string][]][] {
  const out = new Map<string, [string, string][]>()
  for (const m of MODELS) {
    if (!out.has(m.family)) out.set(m.family, [])
    out.get(m.family)!.push([m.id, m.name])
  }
  return [...out]
}

/** A layer's code upper-cased (its kind, whatever its positions). */
export const kindOf = (c: string) => c.toUpperCase() as 'F' | 'S' | 'C' | 'M' | 'L'

/** Elements of KV cache (or state) the model holds for one sequence of n tokens, over all layers. */
export function kvElements(m: Model, n: number): number {
  let s = 0
  for (const c of m.attn.codes) {
    const k = m.attn.kinds[c]
    s += k.state ?? Math.min(n, k.window ?? Infinity) * k.perToken
  }
  return s
}
/** The same in bytes, at 2 bytes an element (16-bit, the usual serving format). */
export const kvBytes = (m: Model, n: number) => 2 * kvElements(m, n)

/** 124M, 8.03B, 37.6B, 671B, 1.03T */
export function fmtParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (n >= 1e11) return `${Math.round(n / 1e9)}B`
  if (n >= 1e10) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  return `${Math.round(n / 1e6)}M`
}
/** 36 KiB, 1.1 MiB, 4.0 GiB */
export function fmtBytes(b: number): string {
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let i = 0
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++ }
  return `${b >= 100 || i === 0 || Number.isInteger(b) ? Math.round(b) : b.toFixed(1)} ${u[i]}`
}
/** 1,024 · 128K · 1M · 10M, for context lengths */
export function fmtTokens(n: number): string {
  if (n >= 1048576 && n % 1048576 === 0) return `${n / 1048576}M`
  if (n >= 1024 && n % 1024 === 0 && n >= 8192) return `${n / 1024}K`
  return n.toLocaleString('en-US')
}
