
/** Navigate to a route; `origin` (client coords) is where the zoom transition starts. */
export type Nav = (route: string, origin?: { x: number; y: number }) => void
export type Mount = (root: HTMLElement, nav: Nav) => () => void

export interface Exhibit {
  name: string
  tag: string
  /** Present when the exhibit is live. Routes nest by '/': a deeper route opens with a zoom-in. */
  route?: string
  /** Loads the page's code (each page is its own chunk, fetched when first opened) and returns its mount. */
  load?: () => Promise<Mount>
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
 * The atlas follows a model's life: its anatomy, its lineage, how it is trained, how it is served, and how
 * agents use it.
 */
export const CATEGORIES: Category[] = [
  {
    id: 'anatomy',
    title: 'Inside the model',
    entries: [
      {
        name: 'Forward pass', tag: 'the Transformer · GPT-2 small', route: 'anatomy', load: () => import('./transformer/overview').then((m) => m.mountOverview),
        // teaching order: the residual stream and LayerNorm come before the sub-layers that use them
        children: [
          { name: 'Tokenizer', tag: 'byte-level BPE', route: 'anatomy/tokenizer', load: () => import('./transformer/tokenizer').then((m) => m.mountTokenizer) },
          { name: 'Embedding', tag: 'W_E · positions', route: 'anatomy/embedding', load: () => import('./transformer/embedding').then((m) => m.mountEmbedding) },
          { name: 'LayerNorm & Residual', tag: 'pre-LN stream', route: 'anatomy/layernorm', load: () => import('./transformer/layernorm').then((m) => m.mountLayerNorm) },
          { name: 'Attention', tag: 'QKᵀ · softmax · V', route: 'anatomy/attention', load: () => import('./transformer/attention').then((m) => m.mountAttention) },
          { name: 'MLP', tag: '768 → 3072 → 768', route: 'anatomy/mlp', load: () => import('./transformer/mlp').then((m) => m.mountMlp) },
          { name: 'Unembed & Sampling', tag: 'logits · temperature', route: 'anatomy/unembed', load: () => import('./transformer/unembed').then((m) => m.mountUnembed) },
        ],
      },
    ],
  },
  {
    id: 'lineage',
    title: 'Architectures',
    entries: [
      { heading: 'Compare' },
      { name: 'Architecture diff', tag: 'any two of 29 models, 2019–2025', route: 'lineage/compare', load: () => import('./lineage/compare').then((m) => m.mountCompare) },
      { heading: 'Origin' },
      { name: 'Transformer (2017)', tag: 'encoder–decoder · post-LN · sinusoids', route: 'lineage/transformer-2017', load: () => import('./lineage/transformer2017').then((m) => m.mountTransformer2017) },
      { heading: 'Decoder-only' },
      { name: 'LLaMA', tag: 'RoPE · RMSNorm · SwiGLU · GQA', route: 'lineage/llama', load: () => import('./lineage/llama').then((m) => m.mountLlama) },
      { name: 'Mixtral', tag: 'MLP → 8 experts, top-2', route: 'lineage/mixtral', load: () => import('./lineage/mixtral').then((m) => m.mountMixtral) },
      { name: 'DeepSeek', tag: 'MLA · fine-grained MoE', route: 'lineage/deepseek', load: () => import('./lineage/deepseek').then((m) => m.mountDeepseek) },
      { heading: 'Encoder' },
      { name: 'BERT', tag: 'bidirectional mask', route: 'lineage/bert', load: () => import('./lineage/bert').then((m) => m.mountBert) },
      { heading: 'Encoder–decoder' },
      { name: 'T5', tag: 'text to text · relative buckets', route: 'lineage/t5', load: () => import('./lineage/t5').then((m) => m.mountT5) },
      { heading: 'Vision & diffusion' },
      { name: 'Vision Transformer', tag: 'tokens → 16×16 patches', route: 'lineage/vit', load: () => import('./lineage/vit').then((m) => m.mountVit) },
      { name: 'CLIP', tag: 'image ↔ text embeddings', route: 'lineage/clip', load: () => import('./lineage/clip').then((m) => m.mountClip) },
      { name: 'Diffusion Transformer', tag: 'denoising · DiT', route: 'lineage/dit', load: () => import('./lineage/dit').then((m) => m.mountDit) },
      { heading: 'Beyond attention' },
      { name: 'Mamba', tag: 'attention → selective SSM', route: 'lineage/mamba', load: () => import('./lineage/mamba').then((m) => m.mountMamba) },
    ],
  },
  {
    id: 'training',
    title: 'Training',
    entries: [
      { name: 'Next-token loss', tag: 'cross-entropy · p − y', route: 'training/loss', load: () => import('./training/loss').then((m) => m.mountLoss) },
      { name: 'Backprop', tag: 'gradients through every block', route: 'training/backprop', load: () => import('./training/backprop').then((m) => m.mountBackprop) },
      { name: 'Optimizer', tag: 'AdamW · warmup · schedule', route: 'training/optimizer', load: () => import('./training/optimizer').then((m) => m.mountOptimizer) },
      { name: 'Learning the tokenizer', tag: 'counting pairs for BPE', route: 'training/tokenizer', load: () => import('./training/bpetrain').then((m) => m.mountBpeTrain) },
      { name: 'Scaling laws', tag: 'loss vs compute', route: 'training/scaling', load: () => import('./training/scaling').then((m) => m.mountScaling) },
      { heading: 'After pretraining' },
      { name: 'SFT', tag: 'instruction tuning', route: 'training/sft', load: () => import('./training/sft').then((m) => m.mountSft) },
      { name: 'RLHF & DPO', tag: 'learning from preferences', route: 'training/dpo', load: () => import('./training/dpo').then((m) => m.mountDpo) },
      { name: 'LoRA', tag: 'low-rank adapters', route: 'training/lora', load: () => import('./training/lora').then((m) => m.mountLora) },
    ],
  },
  {
    id: 'serving',
    title: 'Serving',
    entries: [
      { name: 'KV Cache', tag: 'prefill / decode', route: 'serving/kv-cache', load: () => import('./serving/kvcache').then((m) => m.mountKvCache) },
      { name: 'FlashAttention', tag: 'tiled · on-chip', route: 'serving/flashattention', load: () => import('./serving/flashattention').then((m) => m.mountFlashAttention) },
      { name: 'PagedAttention', tag: 'block tables', route: 'serving/pagedattention', load: () => import('./serving/paged').then((m) => m.mountPagedAttention) },
      { name: 'Continuous Batching', tag: 'iteration-level', route: 'serving/continuous-batching', load: () => import('./serving/batching').then((m) => m.mountContinuousBatching) },
      { name: 'Speculative Decoding', tag: 'draft → verify', route: 'serving/speculative-decoding', load: () => import('./serving/speculative').then((m) => m.mountSpeculative) },
      { name: 'Quantization', tag: 'int4 · fp8', route: 'serving/quantization', load: () => import('./serving/quantization').then((m) => m.mountQuantization) },
    ],
  },
  {
    id: 'agents',
    title: 'Agents',
    entries: [
      { name: 'In-context learning', tag: 'prompts as programs', route: 'agents/in-context', load: () => import('./agents/incontext').then((m) => m.mountInContext) },
      { name: 'ReAct Loop', tag: 'think → act → observe', route: 'agents/react', load: () => import('./agents/react').then((m) => m.mountReact) },
      { name: 'Tool Calling', tag: 'schema → call → result', route: 'agents/tool-calling', load: () => import('./agents/toolcalling').then((m) => m.mountToolCalling) },
      { name: 'RAG Pipeline', tag: 'embed → retrieve → read', route: 'agents/rag', load: () => import('./agents/rag').then((m) => m.mountRag) },
      { name: 'Multi-agent Handoff', tag: 'orchestrator', route: 'agents/multi-agent', load: () => import('./agents/multiagent').then((m) => m.mountMultiAgent) },
    ],
  },
]

/** Every exhibit in a category, children included. */
export const exhibitsOf = (c: Category): Exhibit[] =>
  c.entries.filter((e): e is Exhibit => !isHeading(e)).flatMap((e) => [e, ...(e.children ?? [])])

/** The opening animation (the default route; reached again from the logo, not listed in the rail). */
export const HOME: Exhibit = { name: 'Token Trails', tag: '', route: 'home', load: () => import('./home').then((m) => m.mountHome) }
/** The first page of the tour, linked above the categories. */
export const START: Exhibit = { name: 'Start here', tag: 'what this model does', route: 'start', load: () => import('./start').then((m) => m.mountStart) }
export const FOUNDATIONS: Exhibit = { name: 'Foundations', tag: 'dot product · matmul · softmax', route: 'foundations', load: () => import('./foundations').then((m) => m.mountFoundations) }
export const GLOSSARY: Exhibit = { name: 'Glossary', tag: 'terms in plain words', route: 'glossary', load: () => import('./glossary').then((m) => m.mountGlossary) }

export const ROUTES: Record<string, () => Promise<Mount>> = Object.fromEntries(
  [HOME, START, FOUNDATIONS, GLOSSARY, ...CATEGORIES.flatMap(exhibitsOf)].filter((e) => e.route && e.load).map((e) => [e.route!, e.load!]),
)
/** The display name of a live route ('serving/kv-cache' → 'KV Cache'). */
export const pageName = (route: string): string | undefined =>
  [START, FOUNDATIONS, GLOSSARY, ...CATEGORIES.flatMap(exhibitsOf)].find((e) => e.route === route)?.name
export const DEFAULT_ROUTE = 'home'
/** Old route prefixes that still resolve (prefix → replacement). */
export const ALIASES: [string, string][] = [['transformer', 'anatomy']]
