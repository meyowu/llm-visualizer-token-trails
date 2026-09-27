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
  /** Present when the exhibit is live. */
  route?: string
  mount?: Mount
  /** Routes nest by '/': a child route opens with a zoom-in from its parent. */
}

export interface Category {
  id: string
  title: string
  items: Exhibit[]
}

export const CATEGORIES: Category[] = [
  {
    id: 'transformer',
    title: 'Transformer',
    items: [
      { name: 'Forward pass', tag: 'tokenizer → LM head', route: 'transformer', mount: mountOverview },
      { name: 'Tokenizer', tag: 'byte-level BPE', route: 'transformer/tokenizer', mount: mountTokenizer },
      { name: 'Embedding', tag: 'W_E · positions', route: 'transformer/embedding', mount: mountEmbedding },
      { name: 'Attention', tag: 'QKᵀ · softmax · V', route: 'transformer/attention', mount: mountAttention },
      { name: 'MLP', tag: '768 → 3072 → 768', route: 'transformer/mlp', mount: mountMlp },
      { name: 'LayerNorm & Residual', tag: 'pre-LN stream', route: 'transformer/layernorm', mount: mountLayerNorm },
      { name: 'Unembed & Sampling', tag: 'logits · temperature', route: 'transformer/unembed', mount: mountUnembed },
    ],
  },
  {
    id: 'architectures',
    title: 'Architectures',
    items: [
      { name: 'Mixture of Experts', tag: 'MLP → top-2 experts' },
      { name: 'Encoder–Decoder', tag: '+ cross-attention' },
      { name: 'Vision Transformer', tag: 'tokens → 16×16 patches' },
      { name: 'Diffusion Transformer', tag: 'denoising · DiT' },
      { name: 'Mamba', tag: 'attention → selective SSM' },
    ],
  },
  {
    id: 'inference',
    title: 'Inference',
    items: [
      { name: 'KV Cache', tag: 'prefill / decode' },
      { name: 'PagedAttention', tag: 'block tables' },
      { name: 'Speculative Decoding', tag: 'draft → verify' },
      { name: 'Continuous Batching', tag: 'iteration-level' },
      { name: 'Quantization', tag: 'int4 · fp8' },
    ],
  },
  {
    id: 'agents',
    title: 'Agents',
    items: [
      { name: 'ReAct Loop', tag: 'think → act → observe' },
      { name: 'Tool Calling', tag: 'schema → call → result' },
      { name: 'RAG Pipeline', tag: 'embed → retrieve → read' },
      { name: 'Multi-agent Handoff', tag: 'orchestrator' },
    ],
  },
]

export const ROUTES: Record<string, Mount> = Object.fromEntries(
  CATEGORIES.flatMap((c) => c.items).filter((e) => e.route && e.mount).map((e) => [e.route!, e.mount!]),
)
export const DEFAULT_ROUTE = 'transformer'
