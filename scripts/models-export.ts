/*
 * The model library behind the Architecture diff page: decoder language models from GPT-2 to 2025, described
 * from their own checkpoints. For each model it reads config.json and every shard's safetensors header from
 * the Hugging Face Hub (scripts/models-lib.ts; no weights are downloaded), then
 *
 *   - counts parameters per component from the real tensor shapes (embeddings, attention, dense MLP, routed and
 *     shared experts, router, norms), leaving out buffers, quantization scales, vision towers and
 *     multi-token-prediction layers; MXFP4 blocks hold two parameters per byte;
 *   - works out the parameters one token runs through: everything except the routed experts it is not sent to;
 *   - describes every layer's attention (full, sliding window, chunked, latent, linear) and its KV cache per token;
 *   - checks the totals against the Hub's own count and the numbers the model's authors published.
 *
 *   node scripts/models-export.ts        (Node 23+)
 *
 * Writes src/data/models.json.
 */
import { writeFileSync } from 'node:fs'
import { hubCount, layout, repoJson, type Layout } from './models-lib.ts'

/** A model: where its checkpoint is, and what its authors say its size is (for the check). */
interface Spec {
  id: string; name: string; family: string; org: string; released: string
  repo: string
  /**
   * Published total and active parameters, in billions, where they come from, and what the active count leaves out:
   * authors differ on whether the input embedding and the output head count as active.
   */
  published: [number, number | null, string, Convention?]
}
type Convention = 'all' | 'no-embed' | 'no-embed-head'

const MODELS: Spec[] = [
  { id: 'gpt2', name: 'GPT-2 small', family: 'OpenAI', org: 'OpenAI', released: '2019-02', repo: 'openai-community/gpt2', published: [0.124, null, 'Radford et al. 2019'] },
  { id: 'gpt2-xl', name: 'GPT-2 XL', family: 'OpenAI', org: 'OpenAI', released: '2019-11', repo: 'openai-community/gpt2-xl', published: [1.558, null, 'Hugging Face checkpoint'] },
  { id: 'gpt-oss-20b', name: 'gpt-oss-20b', family: 'OpenAI', org: 'OpenAI', released: '2025-08', repo: 'openai/gpt-oss-20b', published: [20.9, 3.6, 'gpt-oss model card', 'no-embed'] },
  { id: 'gpt-oss-120b', name: 'gpt-oss-120b', family: 'OpenAI', org: 'OpenAI', released: '2025-08', repo: 'openai/gpt-oss-120b', published: [116.8, 5.1, 'gpt-oss model card', 'no-embed'] },
  { id: 'llama-2-7b', name: 'Llama 2 7B', family: 'Meta Llama', org: 'Meta', released: '2023-07', repo: 'NousResearch/Llama-2-7b-hf', published: [6.74, null, 'Touvron et al. 2023'] },
  { id: 'llama-3.1-8b', name: 'Llama 3.1 8B', family: 'Meta Llama', org: 'Meta', released: '2024-07', repo: 'unsloth/Meta-Llama-3.1-8B', published: [8.03, null, 'Llama 3.1 model card'] },
  { id: 'llama-3.1-70b', name: 'Llama 3.1 70B', family: 'Meta Llama', org: 'Meta', released: '2024-07', repo: 'unsloth/Meta-Llama-3.1-70B', published: [70.6, null, 'Llama 3.1 model card'] },
  { id: 'llama-4-scout', name: 'Llama 4 Scout', family: 'Meta Llama', org: 'Meta', released: '2025-04', repo: 'unsloth/Llama-4-Scout-17B-16E-Instruct', published: [109, 17, 'Llama 4 model card', 'all'] },
  { id: 'llama-4-maverick', name: 'Llama 4 Maverick', family: 'Meta Llama', org: 'Meta', released: '2025-04', repo: 'unsloth/Llama-4-Maverick-17B-128E-Instruct', published: [400, 17, 'Llama 4 model card', 'all'] },
  { id: 'mistral-7b', name: 'Mistral 7B', family: 'Mistral', org: 'Mistral AI', released: '2023-09', repo: 'mistralai/Mistral-7B-v0.1', published: [7.24, null, 'Jiang et al. 2023'] },
  { id: 'mixtral-8x7b', name: 'Mixtral 8x7B', family: 'Mistral', org: 'Mistral AI', released: '2023-12', repo: 'mistralai/Mixtral-8x7B-v0.1', published: [46.7, 12.9, 'Jiang et al. 2024', 'all'] },
  { id: 'mixtral-8x22b', name: 'Mixtral 8x22B', family: 'Mistral', org: 'Mistral AI', released: '2024-04', repo: 'mistralai/Mixtral-8x22B-v0.1', published: [141, 39, 'Mistral AI 2024', 'all'] },
  { id: 'gemma-2-9b', name: 'Gemma 2 9B', family: 'Google Gemma', org: 'Google', released: '2024-06', repo: 'unsloth/gemma-2-9b', published: [9.24, null, 'Gemma 2 report'] },
  { id: 'gemma-3-27b', name: 'Gemma 3 27B', family: 'Google Gemma', org: 'Google', released: '2025-03', repo: 'unsloth/gemma-3-27b-it', published: [27, null, 'Gemma 3 report (with the vision encoder)'] },
  { id: 'qwen2.5-7b', name: 'Qwen2.5 7B', family: 'Qwen', org: 'Alibaba', released: '2024-09', repo: 'Qwen/Qwen2.5-7B', published: [7.61, null, 'Qwen2.5 model card'] },
  { id: 'qwen3-8b', name: 'Qwen3 8B', family: 'Qwen', org: 'Alibaba', released: '2025-04', repo: 'Qwen/Qwen3-8B', published: [8.2, null, 'Qwen3 model card'] },
  { id: 'qwen3-32b', name: 'Qwen3 32B', family: 'Qwen', org: 'Alibaba', released: '2025-04', repo: 'Qwen/Qwen3-32B', published: [32.8, null, 'Qwen3 model card'] },
  { id: 'qwen3-30b-a3b', name: 'Qwen3 30B-A3B', family: 'Qwen', org: 'Alibaba', released: '2025-04', repo: 'Qwen/Qwen3-30B-A3B', published: [30.5, 3.3, 'Qwen3 model card', 'all'] },
  { id: 'qwen3-235b-a22b', name: 'Qwen3 235B-A22B', family: 'Qwen', org: 'Alibaba', released: '2025-04', repo: 'Qwen/Qwen3-235B-A22B', published: [235, 22, 'Qwen3 model card', 'all'] },
  { id: 'qwen3-next-80b-a3b', name: 'Qwen3-Next 80B-A3B', family: 'Qwen', org: 'Alibaba', released: '2025-09', repo: 'Qwen/Qwen3-Next-80B-A3B-Instruct', published: [80, 3, 'Qwen3-Next model card', 'no-embed-head'] },
  { id: 'deepseek-v2', name: 'DeepSeek-V2', family: 'DeepSeek', org: 'DeepSeek', released: '2024-05', repo: 'deepseek-ai/DeepSeek-V2', published: [236, 21, 'DeepSeek-AI 2024', 'all'] },
  { id: 'deepseek-v3', name: 'DeepSeek-V3', family: 'DeepSeek', org: 'DeepSeek', released: '2024-12', repo: 'deepseek-ai/DeepSeek-V3', published: [671, 37, 'DeepSeek-AI 2024', 'all'] },
  { id: 'deepseek-v3.2-exp', name: 'DeepSeek-V3.2-Exp', family: 'DeepSeek', org: 'DeepSeek', released: '2025-09', repo: 'deepseek-ai/DeepSeek-V3.2-Exp', published: [671, null, 'DeepSeek-AI 2025 (its 37B active is V3’s; the indexer adds 0.85B)'] },
  { id: 'kimi-k2', name: 'Kimi K2', family: 'Moonshot Kimi', org: 'Moonshot AI', released: '2025-07', repo: 'moonshotai/Kimi-K2-Instruct', published: [1000, 32, 'Kimi K2 model card (1T)', 'all'] },
  { id: 'kimi-linear', name: 'Kimi Linear 48B-A3B', family: 'Moonshot Kimi', org: 'Moonshot AI', released: '2025-10', repo: 'moonshotai/Kimi-Linear-48B-A3B-Instruct', published: [48, 3, 'Kimi Linear model card', 'no-embed'] },
  { id: 'glm-4.5-air', name: 'GLM-4.5-Air', family: 'Zhipu GLM', org: 'Zhipu AI', released: '2025-07', repo: 'zai-org/GLM-4.5-Air', published: [106, 12, 'GLM-4.5 model card', 'no-embed-head'] },
  { id: 'glm-4.5', name: 'GLM-4.5', family: 'Zhipu GLM', org: 'Zhipu AI', released: '2025-07', repo: 'zai-org/GLM-4.5', published: [355, 32, 'GLM-4.5 model card', 'no-embed-head'] },
  { id: 'glm-4.6', name: 'GLM-4.6', family: 'Zhipu GLM', org: 'Zhipu AI', released: '2025-09', repo: 'zai-org/GLM-4.6', published: [355, 32, 'GLM-4.6 model card', 'no-embed-head'] },
  { id: 'minimax-m2', name: 'MiniMax-M2', family: 'MiniMax', org: 'MiniMax', released: '2025-10', repo: 'MiniMaxAI/MiniMax-M2', published: [230, 10, 'MiniMax-M2 model card', 'no-embed-head'] },
]

/* ---------- the checkpoint's tensors, sorted into components ---------- */
type Part = 'embed' | 'pos' | 'head' | 'attn' | 'indexer' | 'mlp' | 'routed' | 'shared' | 'router' | 'norm'
const PARTS: Part[] = ['embed', 'pos', 'head', 'attn', 'indexer', 'mlp', 'routed', 'shared', 'router', 'norm']

/** Parameters a tensor holds, or 0 for buffers and quantization scales. */
function paramsOf(name: string, t: { dtype: string; shape: number[] }): number {
  if (/(^|\.)attn\.(masked_)?bias$/.test(name) && t.shape.length === 4) return 0 // GPT-2's causal-mask buffer
  if (/inv_freq$|_scale_inv$|weight_scale$|input_scale$|_scales$/.test(name)) return 0
  const n = t.shape.reduce((a, b) => a * b, 1)
  return /_blocks$/.test(name) && t.dtype === 'U8' ? n * 2 : n // MXFP4: two 4-bit values per byte
}

interface Sorted { parts: Record<Part, number>; routedByLayer: Map<number, number>; mtp: number; vision: number; skipped: string[] }
function sortTensors(L: Layout, nLayers: number): Sorted {
  const parts = Object.fromEntries(PARTS.map((p) => [p, 0])) as Record<Part, number>
  const routedByLayer = new Map<number, number>()
  let mtp = 0, vision = 0
  const skipped: string[] = []
  for (const [full, t] of Object.entries(L)) {
    const n = paramsOf(full, t)
    if (!n) continue
    if (/vision|multi_modal_projector|visual/.test(full)) { vision += n; continue }
    const name = full.replace(/^language_model\./, '').replace(/^model\./, '').replace(/^transformer\./, '')
    const li = /(?:^|\.)(?:layers|h)\.(\d+)\./.exec(name)
    const layer = li ? +li[1] : -1
    if (name.startsWith('mtp.') || layer >= nLayers) { mtp += n; continue }
    let p: Part | null = null
    if (/^(embed_tokens|wte)\b/.test(name)) p = 'embed'
    else if (/^wpe\b/.test(name)) p = 'pos'
    else if (/^lm_head\b/.test(name)) p = 'head'
    else if (/\.indexer\./.test(name)) p = 'indexer'
    else if (/\.experts\./.test(name)) p = 'routed'
    else if (/shared_expert/.test(name)) p = 'shared'
    else if (/(\.gate\.(weight|bias|e_score_correction_bias)$)|\.router\.|e_score_correction_bias$/.test(name) && /mlp|moe|feed_forward/.test(name)) p = 'router'
    else if (/norm|(^|\.)ln_/.test(name)) p = 'norm'
    else if (/\.(self_attn|attn|linear_attn)\./.test(name)) p = 'attn'
    else if (/\.(mlp|feed_forward|block_sparse_moe)\./.test(name)) p = 'mlp'
    if (!p) { skipped.push(full); continue }
    parts[p] += n
    if (p === 'routed') routedByLayer.set(layer, (routedByLayer.get(layer) ?? 0) + n)
  }
  return { parts, routedByLayer, mtp, vision, skipped }
}

/* ---------- reading the config ---------- */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Cfg = Record<string, any>
const first = <T>(...xs: (T | undefined | null)[]): T | undefined => xs.find((x) => x !== undefined && x !== null) ?? undefined

/** One layer's attention: how it attends, and what it keeps per token (elements) or in total (a fixed state). */
interface LayerAttn { kind: 'full' | 'sliding' | 'chunked' | 'latent' | 'linear'; window?: number; perToken: number; state?: number; rope: boolean }

function describe(spec: Spec, cfg0: Cfg, full: Layout) {
  let L = full
  const cfg: Cfg = cfg0.text_config ? { ...cfg0, ...cfg0.text_config } : cfg0
  const nLayers: number = first(cfg.num_hidden_layers, cfg.n_layer)!
  const hidden: number = first(cfg.hidden_size, cfg.n_embd)!
  const heads: number = first(cfg.num_attention_heads, cfg.n_head)!
  // the language model's tensors only: a vision tower has layers, projections and LayerNorm biases of its own
  L = Object.fromEntries(Object.entries(L).filter(([k]) => !/vision|multi_modal_projector|visual/.test(k)))
  const names = Object.keys(L).map((k) => k.replace(/^language_model\./, '').replace(/^model\./, ''))
  const inLayer = (n: string, layer: number) => n.includes(`layers.${layer}.`) || n.startsWith(`h.${layer}.`) || n.includes(`.h.${layer}.`)
  const has = (re: RegExp, layer?: number) => names.some((n) => re.test(n) && (layer === undefined || inLayer(n, layer)))
  const shapeOf = (re: RegExp, layer: number) => {
    const k = Object.keys(L).find((n) => re.test(n) && inLayer(n, layer))
    return k ? L[k].shape : null
  }
  const mla = names.some((n) => /kv_a_proj_with_mqa/.test(n))
  // a query head's width: MLA's comes from its query projection (nope + rope parts), not from hidden / heads
  const qMla = mla ? shapeOf(/self_attn\.(q_b_proj|q_proj)\.weight$/, names.some((n) => /layers\.0\.self_attn\.(q_b_proj|q_proj)/.test(n)) ? 0 : firstLayerWith(L, /kv_a_proj_with_mqa/)) : null
  const headDim: number = qMla ? qMla[0] / heads : first(cfg.head_dim, hidden / heads)!
  const kvHeads: number = first(cfg.num_key_value_heads, heads)!
  const vocab: number = cfg.vocab_size
  const window: number | undefined = first(cfg.sliding_window) ?? undefined
  const ropeTheta: number = cfg.rope_theta ?? 10000 // the RoPE default when a config leaves it out (Llama 2)
  const learnedPos = has(/^wpe\./)

  /* attention, layer by layer */
  const layers: LayerAttn[] = []
  const kdaLayers: number[] = cfg.linear_attn_config?.kda_layers ?? []
  for (let i = 0; i < nLayers; i++) {
    const lt: string | undefined = cfg.layer_types?.[i]
    let a: LayerAttn
    if (has(/linear_attn\./, i) || kdaLayers.includes(i + 1)) {
      // a linear-attention layer: a fixed state per head (d_k × d_v) plus a short convolution's last inputs
      const vHeads = first(cfg.linear_num_value_heads, cfg.linear_attn_config?.num_heads)!
      const dk = first(cfg.linear_key_head_dim, cfg.linear_attn_config?.head_dim)!, dv = first(cfg.linear_value_head_dim, cfg.linear_attn_config?.head_dim)!
      const convs = Object.keys(L).filter((n) => /(linear_attn\.conv1d|[qkv]_conv1d)\.weight$/.test(n) && inLayer(n, i))
      const convState = convs.reduce((s, n) => s + L[n].shape[0] * (L[n].shape[L[n].shape.length - 1] - 1), 0)
      a = { kind: 'linear', perToken: 0, state: vHeads * dk * dv + convState, rope: false }
    } else if (has(/kv_a_proj_with_mqa/, i)) {
      const kva = shapeOf(/kv_a_proj_with_mqa\.weight$/, i)!
      const idx = shapeOf(/indexer\.wk\.weight$/, i)
      a = { kind: 'latent', perToken: kva[0] + (idx ? idx[0] : 0), rope: !cfg.mla_use_nope }
    } else {
      // standard attention: K and V rows per token
      let kv: number
      const kp = shapeOf(/(self_attn|attn)\.k_proj\.weight$/, i)
      if (kp) kv = 2 * kp[0]
      else { const ca = shapeOf(/attn\.c_attn\.weight$/, i)!; kv = 2 * (ca[1] / 3) }
      const nope = Array.isArray(cfg.no_rope_layers) && cfg.no_rope_layers[i] === 0
      let kind: LayerAttn['kind'] = 'full', w: number | undefined
      if (lt === 'sliding_attention' || (lt === undefined && window && isSliding(cfg, i))) { kind = 'sliding'; w = window }
      else if (cfg.attention_chunk_size && !nope) { kind = 'chunked'; w = cfg.attention_chunk_size }
      a = { kind, window: w, perToken: kv, rope: !learnedPos && !nope }
    }
    layers.push(a)
  }

  /* experts */
  const nExperts: number | undefined = first(cfg.n_routed_experts, cfg.num_local_experts, cfg.num_experts)
  const topK: number | undefined = first(cfg.num_experts_per_tok, cfg.experts_per_token, cfg.num_experts_per_token)
  const sorted = sortTensors(full, nLayers)
  const moeLayers = [...sorted.routedByLayer.keys()].sort((a, b) => a - b)
  const sharedN = sorted.parts.shared > 0 ? first(cfg.n_shared_experts, cfg.num_shared_experts, 1) : 0
  const expertW = moeLayers.length ? expertWidth(L, moeLayers[0]) : undefined
  const denseW = has(/\.(mlp|feed_forward)\.(gate_proj|c_fc|up_proj|fc1)\.weight$/) ? denseWidth(L) : undefined

  /* parameters */
  const p = sorted.parts
  const total = PARTS.reduce((s, k) => s + p[k], 0)
  let unused = 0
  for (const [, n] of sorted.routedByLayer) unused += n * (1 - (topK ?? 0) / (nExperts ?? 1))
  const active = total - unused

  const qkNorm = has(/self_attn\.(q_norm|k_norm)\.weight$/) || !!cfg.use_qk_norm
  const qkvBias = has(/self_attn\.[qk]_proj\.bias$/) || has(/attn\.c_attn\.bias$/)
  const ropeLayers = layers.filter((l) => l.rope).length
  const sinks = has(/\.sinks$/)
  const qp = shapeOf(/self_attn\.q_proj\.weight$/, layers.findIndex((l) => l.kind !== 'linear' && l.kind !== 'latent'))
  const gatedAttn = !!qp && qp[0] === 2 * heads * headDim
  const normKind = has(/(ln_1|input_layernorm)\.bias$/) ? 'LayerNorm' : 'RMSNorm'
  const sandwich = has(/pre_feedforward_layernorm/)
  const act = /gelu/.test(first(cfg.hidden_activation, cfg.hidden_act, cfg.activation_function) ?? '') ? (has(/gate_proj/) ? 'GeGLU' : 'GELU') : 'SwiGLU'

  return {
    id: spec.id, name: spec.name, family: spec.family, org: spec.org, released: spec.released, repo: spec.repo,
    layers: nLayers, hidden, vocab, heads, kvHeads, headDim,
    context: first(cfg.max_position_embeddings, cfg.n_positions, cfg.model_max_length),
    tied: !has(/lm_head/),
    pos: learnedPos ? { kind: 'learned' } : ropeLayers === 0 ? { kind: 'none' } : {
      kind: 'rope', layers: ropeLayers, theta: ropeTheta, localTheta: cfg.rope_local_base_freq ?? null,
      partial: mla ? cfg.qk_rope_head_dim / headDim : first(cfg.partial_rotary_factor, cfg.rotary_dim ? cfg.rotary_dim / headDim : undefined, 1),
      scaling: cfg.rope_scaling?.rope_type ?? cfg.rope_scaling?.type ?? null,
    },
    norm: normKind, sandwich, qkNorm, qkvBias, sinks, gatedAttn, act,
    softcap: cfg.attn_logit_softcapping || cfg.final_logit_softcapping ? { attn: cfg.attn_logit_softcapping ?? null, final: cfg.final_logit_softcapping ?? null } : null,
    mla: mla ? { latent: cfg.kv_lora_rank, rope: cfg.qk_rope_head_dim, qLatent: cfg.q_lora_rank ?? null, vHeadDim: cfg.v_head_dim } : null,
    mtpLayers: sorted.mtp > 0 ? first(cfg.num_nextn_predict_layers, cfg.num_mtp_modules, 1) : 0,
    indexer: has(/indexer\./) ? { heads: cfg.index_n_heads, dim: cfg.index_head_dim, topk: cfg.index_topk } : null,
    attn: encodeLayers(layers),
    /** One letter per layer: D a dense MLP, E experts. */
    mlp: Array.from({ length: nLayers }, (_, i) => (sorted.routedByLayer.has(i) ? 'E' : 'D')).join(''),
    moe: moeLayers.length ? { experts: nExperts, topK, shared: sharedN, width: expertW, layers: moeLayers.length, dense: nLayers - moeLayers.length } : null,
    denseWidth: denseW,
    params: { ...p, total, active, mtp: sorted.mtp, vision: sorted.vision },
    skipped: sorted.skipped,
  }
}

/**
 * Every layer's attention as one letter (F full, S sliding window, C chunked, M latent, L linear; lower case when the
 * layer has no RoPE), and per letter what it keeps: elements per token, the window it keeps them for, or a fixed state.
 */
function encodeLayers(layers: LayerAttn[]) {
  const letter = { full: 'F', sliding: 'S', chunked: 'C', latent: 'M', linear: 'L' } as const
  const kinds: Record<string, { perToken: number; window?: number; state?: number }> = {}
  const codes = layers.map((l) => {
    const c = l.rope ? letter[l.kind] : letter[l.kind].toLowerCase()
    const k = { perToken: l.perToken, ...(l.window ? { window: l.window } : {}), ...(l.state ? { state: l.state } : {}) }
    if (kinds[c] && JSON.stringify(kinds[c]) !== JSON.stringify(k)) throw new Error(`layers of kind ${c} differ: ${JSON.stringify(kinds[c])} vs ${JSON.stringify(k)}`)
    kinds[c] = k
    return c
  }).join('')
  return { codes, kinds }
}

function firstLayerWith(L: Layout, re: RegExp): number {
  for (const k of Object.keys(L)) if (re.test(k)) { const m = /layers\.(\d+)\./.exec(k); if (m) return +m[1] }
  return 0
}
/** Gemma 2 alternates sliding and global layers; Gemma 3 has five sliding per global; Mistral 7B slides everywhere. */
function isSliding(cfg: Cfg, i: number): boolean {
  if (cfg.sliding_window_pattern) return (i + 1) % cfg.sliding_window_pattern !== 0
  if (cfg.model_type === 'gemma2') return i % 2 === 0
  if (cfg.model_type === 'mistral') return true
  return false
}
function expertWidth(L: Layout, layer: number): number | undefined {
  const k = Object.keys(L).find((n) => new RegExp(`\\.layers\\.${layer}\\.`).test(n) && /experts\.(0\.)?(gate_proj|w1|gate_up_proj)/.test(n) && !/scale|bias/.test(n))
  if (!k) return undefined
  const s = L[k].shape
  if (/gate_up_proj_blocks/.test(k)) return s[1] / 2 // [E, 2·width, in/32, 16]
  if (/gate_up_proj$/.test(k)) return s[2] / 2 // [E, hidden, 2·width]
  return s[0]
}
function denseWidth(L: Layout): number | undefined {
  const k = Object.keys(L).find((n) => /\.(mlp|feed_forward)\.(gate_proj|c_fc|up_proj)\.weight$/.test(n) && !/expert/.test(n))
  if (!k) return undefined
  const s = L[k].shape
  return /c_fc/.test(k) ? s[1] : s[0]
}

/* ---------- run ---------- */
const out = []
const B = (n: number) => (n / 1e9).toFixed(n < 1e10 ? 2 : 1)
for (const spec of MODELS) {
  const cfg = await repoJson<Cfg>(spec.repo, 'config.json')
  if (!cfg) throw new Error(`${spec.repo}: no config.json`)
  const L = await layout(spec.repo)
  const d = describe(spec, cfg, L)
  const hub = await hubCount(spec.repo)
  const [pt, pa, , conv] = spec.published
  // the active count as the authors count it, and a tolerance of 3% or half the last published digit
  const pp = d.params, activeAsPublished = pp.active - (conv === 'no-embed' || conv === 'no-embed-head' ? pp.embed : 0) - (conv === 'no-embed-head' ? pp.head : 0)
  const tol = (x: number) => Math.max(0.03 * x, 0.5 * 10 ** -((String(x).split('.')[1] ?? '').length))
  const okT = Math.abs(pp.total / 1e9 - pt) <= tol(pt), okA = !pa || Math.abs(activeAsPublished / 1e9 - pa) <= tol(pa)
  const offT = pp.total / (pt * 1e9) - 1, offA = pa ? activeAsPublished / (pa * 1e9) - 1 : 0
  const flag = okT && okA ? '' : '  <<< CHECK'
  console.log(`${spec.name.padEnd(22)} total ${B(d.params.total).padStart(7)}B (published ${pt}B, ${(offT * 100).toFixed(1)}%)  active ${B(d.params.active).padStart(6)}B${pa ? ` (as published ${B(activeAsPublished)}B vs ${pa}B, ${(offA * 100).toFixed(1)}%)` : ''}  mtp ${B(d.params.mtp)}B vision ${B(d.params.vision)}B hub ${hub ? B(hub) : '?'}B${flag}`)
  if (d.skipped.length) console.log(`  unsorted: ${d.skipped.slice(0, 5).join(', ')}${d.skipped.length > 5 ? ` … ${d.skipped.length}` : ''}`)
  out.push({ ...d, published: { total: pt, active: pa, source: spec.published[2], leaves: conv ?? null }, hub })
}
writeFileSync(new URL('../src/data/models.json', import.meta.url), JSON.stringify({ models: out.map(({ skipped: _s, ...m }) => m) }))
console.log(`wrote ${out.length} models`)
