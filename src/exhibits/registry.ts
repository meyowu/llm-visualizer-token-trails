import { mountContinuousBatching } from './serving/batching'
import { mountFlashAttention } from './serving/flashattention'
import { mountKvCache } from './serving/kvcache'
import { mountPagedAttention } from './serving/paged'
import { mountSpeculative } from './serving/speculative'
import { mountBert } from './lineage/bert'
import { mountClip } from './lineage/clip'
import { mountDeepseek } from './lineage/deepseek'
import { mountDit } from './lineage/dit'
import { mountLlama } from './lineage/llama'
import { mountMamba } from './lineage/mamba'
import { mountT5 } from './lineage/t5'
import { mountVit } from './lineage/vit'
import { mountMixtral } from './lineage/mixtral'
import { mountTransformer2017 } from './lineage/transformer2017'
import { mountLoss } from './training/loss'
import { mountFoundations } from './foundations'
import { mountGlossary } from './glossary'
import { mountStart } from './start'
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
 * The atlas follows a model's life: its anatomy, its lineage, how it is trained, what it has learned, how it is
 * served, and how agents use it.
 */
export const CATEGORIES: Category[] = [
  {
    id: 'anatomy',
    title: 'Anatomy',
    entries: [
      {
        name: 'Forward pass', tag: 'the Transformer · GPT-2 small', route: 'anatomy', mount: mountOverview,
        // teaching order: the residual stream and LayerNorm come before the sub-layers that use them
        children: [
          { name: 'Tokenizer', tag: 'byte-level BPE', route: 'anatomy/tokenizer', mount: mountTokenizer },
          { name: 'Embedding', tag: 'W_E · positions', route: 'anatomy/embedding', mount: mountEmbedding },
          { name: 'LayerNorm & Residual', tag: 'pre-LN stream', route: 'anatomy/layernorm', mount: mountLayerNorm },
          { name: 'Attention', tag: 'QKᵀ · softmax · V', route: 'anatomy/attention', mount: mountAttention },
          { name: 'MLP', tag: '768 → 3072 → 768', route: 'anatomy/mlp', mount: mountMlp },
          { name: 'Unembed & Sampling', tag: 'logits · temperature', route: 'anatomy/unembed', mount: mountUnembed },
        ],
      },
    ],
  },
  {
    id: 'lineage',
    title: 'Lineage',
    entries: [
      { heading: 'Origin' },
      { name: 'Transformer (2017)', tag: 'encoder–decoder · post-LN · sinusoids', route: 'lineage/transformer-2017', mount: mountTransformer2017 },
      { heading: 'Decoder-only' },
      { name: 'LLaMA', tag: 'RoPE · RMSNorm · SwiGLU · GQA', route: 'lineage/llama', mount: mountLlama },
      { name: 'Mixtral', tag: 'MLP → 8 experts, top-2', route: 'lineage/mixtral', mount: mountMixtral },
      { name: 'DeepSeek', tag: 'MLA · fine-grained MoE', route: 'lineage/deepseek', mount: mountDeepseek },
      { heading: 'Encoder' },
      { name: 'BERT', tag: 'bidirectional mask', route: 'lineage/bert', mount: mountBert },
      { heading: 'Encoder–decoder' },
      { name: 'T5', tag: 'text to text · relative buckets', route: 'lineage/t5', mount: mountT5 },
      { heading: 'Vision & diffusion' },
      { name: 'Vision Transformer', tag: 'tokens → 16×16 patches', route: 'lineage/vit', mount: mountVit },
      { name: 'CLIP', tag: 'image ↔ text embeddings', route: 'lineage/clip', mount: mountClip },
      { name: 'Diffusion Transformer', tag: 'denoising · DiT', route: 'lineage/dit', mount: mountDit },
      { heading: 'Beyond attention' },
      { name: 'Mamba', tag: 'attention → selective SSM', route: 'lineage/mamba', mount: mountMamba },
    ],
  },
  {
    id: 'training',
    title: 'Training',
    entries: [
      { name: 'Next-token loss', tag: 'cross-entropy · p − y', route: 'training/loss', mount: mountLoss },
      { name: 'Backprop', tag: 'gradients through one block' },
      { name: 'Optimizer', tag: 'AdamW · warmup · schedule' },
      { name: 'Learning the tokenizer', tag: 'counting pairs for BPE' },
      { heading: 'After pretraining' },
      { name: 'SFT', tag: 'instruction tuning' },
      { name: 'RLHF & DPO', tag: 'learning from preferences' },
      { name: 'LoRA', tag: 'low-rank adapters' },
      { name: 'Scaling laws', tag: 'loss vs compute' },
    ],
  },
  {
    id: 'interpretability',
    title: 'Interpretability',
    entries: [
      { name: 'Logit lens', tag: 'predictions block by block' },
      { name: 'Induction heads', tag: 'copying patterns' },
      { name: 'Attention sinks', tag: 'the first token' },
      { name: 'Superposition', tag: 'more features than dimensions' },
    ],
  },
  {
    id: 'serving',
    title: 'Serving',
    entries: [
      { name: 'KV Cache', tag: 'prefill / decode', route: 'serving/kv-cache', mount: mountKvCache },
      { name: 'FlashAttention', tag: 'tiled · on-chip', route: 'serving/flashattention', mount: mountFlashAttention },
      { name: 'PagedAttention', tag: 'block tables', route: 'serving/pagedattention', mount: mountPagedAttention },
      { name: 'Continuous Batching', tag: 'iteration-level', route: 'serving/continuous-batching', mount: mountContinuousBatching },
      { name: 'Speculative Decoding', tag: 'draft → verify', route: 'serving/speculative-decoding', mount: mountSpeculative },
      { name: 'Quantization', tag: 'int4 · fp8' },
    ],
  },
  {
    id: 'agents',
    title: 'Agents',
    entries: [
      { name: 'In-context learning', tag: 'prompts as programs' },
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

/** The landing page, linked above the categories. */
export const START: Exhibit = { name: 'Start here', tag: 'what this model does', route: 'start', mount: mountStart }
export const FOUNDATIONS: Exhibit = { name: 'Foundations', tag: 'dot product · matmul · softmax', route: 'foundations', mount: mountFoundations }
export const GLOSSARY: Exhibit = { name: 'Glossary', tag: 'terms in plain words', route: 'glossary', mount: mountGlossary }

export const ROUTES: Record<string, Mount> = Object.fromEntries(
  [START, FOUNDATIONS, GLOSSARY, ...CATEGORIES.flatMap(exhibitsOf)].filter((e) => e.route && e.mount).map((e) => [e.route!, e.mount!]),
)
export const DEFAULT_ROUTE = 'start'
/** Old route prefixes that still resolve (prefix → replacement). */
export const ALIASES: [string, string][] = [['transformer', 'anatomy']]
