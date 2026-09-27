import { mountAttention } from './transformer/attention'
import { mountEmbedding } from './transformer/embedding'
import { mountLayerNorm } from './transformer/layernorm'
import { mountMlp } from './transformer/mlp'
import { mountOverview } from './transformer/overview'
import { mountTokenizer } from './transformer/tokenizer'
import { mountUnembed } from './transformer/unembed'

/** Navigate to a route; `origin` (client coords) is where the zoom transition starts. */
export type Nav = (route: string, origin?: { x: number; y: number }) => void
export type Mount = (root: HTMLElement, nav: Nav) => () => void

export interface Exhibit {
  name: string
  tag: string
  /** Present when the exhibit is live. Routes nest by '/': a deeper route opens with a zoom-in. */
  route?: string
  mount?: Mount
  /** Steps or parts of this exhibit, shown indented under it. */
  children?: Exhibit[]
}
/** A small sub-heading inside a category. */
export interface Heading { heading: string }
export type Entry = Exhibit | Heading
export const isHeading = (e: Entry): e is Heading => 'heading' in e

export interface Category {
  id: string
  title: string
  entries: Entry[]
}

/*
 * The atlas follows a model's life: its anatomy, its lineage, how it is trained, how it is
 * served, and how agents use it.
 */
export const CATEGORIES: Category[] = [
  {
    id: 'anatomy',
    title: 'Anatomy',
    entries: [
      {
        name: 'Forward pass', tag: 'GPT-2 small · text → next token', route: 'anatomy', mount: mountOverview,
        children: [
          { name: 'Tokenizer', tag: 'byte-level BPE', route: 'anatomy/tokenizer', mount: mountTokenizer },
          { name: 'Embedding', tag: 'W_E · positions', route: 'anatomy/embedding', mount: mountEmbedding },
          { name: 'Attention', tag: 'QKᵀ · softmax · V', route: 'anatomy/attention', mount: mountAttention },
          { name: 'MLP', tag: '768 → 3072 → 768', route: 'anatomy/mlp', mount: mountMlp },
          { name: 'LayerNorm & Residual', tag: 'pre-LN stream', route: 'anatomy/layernorm', mount: mountLayerNorm },
          { name: 'Unembed & Sampling', tag: 'logits · temperature', route: 'anatomy/unembed', mount: mountUnembed },
        ],
      },
    ],
  },
  {
    id: 'lineage',
    title: 'Lineage',
    entries: [
      { heading: 'Decoder-only' },
      { name: 'LLaMA', tag: 'RoPE · RMSNorm · SwiGLU · GQA' },
      { name: 'Mixtral', tag: 'MLP → 8 experts, top-2' },
      { name: 'DeepSeek', tag: 'MLA · fine-grained MoE' },
      { heading: 'Encoder' },
      { name: 'BERT', tag: 'bidirectional mask' },
      { heading: 'Encoder–decoder' },
      { name: 'T5', tag: '+ cross-attention' },
      { heading: 'Vision & diffusion' },
      { name: 'Vision Transformer', tag: 'tokens → 16×16 patches' },
      { name: 'CLIP', tag: 'image ↔ text embeddings' },
      { name: 'Diffusion Transformer', tag: 'denoising · DiT' },
      { heading: 'Beyond attention' },
      { name: 'Mamba', tag: 'attention → selective SSM' },
    ],
  },
  {
    id: 'training',
    title: 'Training',
    entries: [
      { name: 'Loss & Backprop', tag: 'next-token cross-entropy' },
      { name: 'LoRA', tag: 'low-rank adapters' },
      { name: 'RLHF & DPO', tag: 'learning from preferences' },
    ],
  },
  {
    id: 'serving',
    title: 'Serving',
    entries: [
      { name: 'KV Cache', tag: 'prefill / decode' },
      { name: 'FlashAttention', tag: 'tiled · on-chip' },
      { name: 'PagedAttention', tag: 'block tables' },
      { name: 'Continuous Batching', tag: 'iteration-level' },
      { name: 'Speculative Decoding', tag: 'draft → verify' },
      { name: 'Quantization', tag: 'int4 · fp8' },
    ],
  },
  {
    id: 'agents',
    title: 'Agents',
    entries: [
      { name: 'ReAct Loop', tag: 'think → act → observe' },
      { name: 'Tool Calling', tag: 'schema → call → result' },
      { name: 'RAG Pipeline', tag: 'embed → retrieve → read' },
      { name: 'Multi-agent Handoff', tag: 'orchestrator' },
    ],
  },
]

/** Every exhibit in a category, children included. */
export const exhibitsOf = (c: Category): Exhibit[] =>
  c.entries.filter((e): e is Exhibit => !isHeading(e)).flatMap((e) => [e, ...(e.children ?? [])])

export const ROUTES: Record<string, Mount> = Object.fromEntries(
  CATEGORIES.flatMap(exhibitsOf).filter((e) => e.route && e.mount).map((e) => [e.route!, e.mount!]),
)
export const DEFAULT_ROUTE = 'anatomy'
/** Old route prefixes that still resolve (prefix → replacement). */
export const ALIASES: [string, string][] = [['transformer', 'anatomy']]
