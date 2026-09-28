/*
 * A small open chat model (Qwen3 or Qwen2.5-Instruct, Apache-2.0) in plain TypeScript, for the offline
 * Agents exports: its byte-level BPE tokenizer, the ChatML chat template with the tool-calling format,
 * and the decoder (RMSNorm, RoPE, grouped-query attention with Qwen2.5's q/k/v biases or Qwen3's
 * per-head q/k norms, SwiGLU, tied embeddings) with a KV cache and greedy generation. The browser never runs it.
 *
 * Weights: config.json, tokenizer.json and model.safetensors (or shards with model.safetensors.index.json)
 * from huggingface.co/Qwen/Qwen3-1.7B, in a directory passed to Qwen.load().
 */
import { closeSync, existsSync, openSync, readFileSync, readSync } from 'node:fs'
import { availableParallelism } from 'node:os'
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads'

/* ---------- tokenizer ---------- */
const BYTE_CHAR = (() => {
  const bs: number[] = []
  for (let i = 33; i <= 126; i++) bs.push(i)
  for (let i = 161; i <= 172; i++) bs.push(i)
  for (let i = 174; i <= 255; i++) bs.push(i)
  const cs = [...bs]
  let n = 0
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++) }
  return new Map(bs.map((b, i) => [b, String.fromCharCode(cs[i])]))
})()
const CHAR_BYTE = new Map([...BYTE_CHAR].map(([b, c]) => [c, b]))
/** Qwen2's pre-tokenizer pattern; JavaScript has no inline (?i:…), so the contractions are spelled out. */
const PRE = /'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu

export class QwenTokenizer {
  vocab: Map<string, number>
  inv: string[]
  ranks: Map<string, number>
  special: Map<string, number>
  private cache = new Map<string, number[]>()
  private specialRe: RegExp
  constructor(dir: string) {
    const t = JSON.parse(readFileSync(`${dir}/tokenizer.json`, 'utf8'))
    this.vocab = new Map(Object.entries<number>(t.model.vocab))
    this.special = new Map((t.added_tokens as { id: number; content: string }[]).map((a) => [a.content, a.id]))
    this.inv = []
    for (const [s, i] of this.vocab) this.inv[i] = s
    for (const [s, i] of this.special) this.inv[i] = s
    this.ranks = new Map((t.model.merges as (string | string[])[]).map((m, i) => [Array.isArray(m) ? m.join(' ') : m, i]))
    const esc = [...this.special.keys()].sort((a, b) => b.length - a.length).map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    this.specialRe = new RegExp(`(${esc.join('|')})`)
  }
  private bpe(word: string): number[] {
    const hit = this.cache.get(word)
    if (hit) return hit
    let parts = [...word]
    while (parts.length > 1) {
      let best = -1, rank = Infinity
      for (let i = 0; i < parts.length - 1; i++) { const r = this.ranks.get(parts[i] + ' ' + parts[i + 1]); if (r !== undefined && r < rank) { rank = r; best = i } }
      if (best < 0) break
      parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)]
    }
    const ids = parts.map((p) => { const id = this.vocab.get(p); if (id === undefined) throw new Error(`no token for ${p}`); return id })
    this.cache.set(word, ids)
    return ids
  }
  encode(text: string): number[] {
    const ids: number[] = []
    for (const part of text.normalize('NFC').split(this.specialRe)) {
      if (!part) continue
      const sp = this.special.get(part)
      if (sp !== undefined) { ids.push(sp); continue }
      for (const m of part.matchAll(PRE)) ids.push(...this.bpe([...Buffer.from(m[0], 'utf8')].map((b) => BYTE_CHAR.get(b)!).join('')))
    }
    return ids
  }
  /** One token as text (special tokens as themselves). */
  piece(id: number): string {
    const s = this.inv[id]
    if (s === undefined) return ''
    if (this.special.has(s)) return s
    return Buffer.from([...s].map((c) => CHAR_BYTE.get(c) ?? 63)).toString('utf8')
  }
  decode(ids: number[]): string {
    const bytes: number[] = []
    let out = ''
    const flush = () => { if (bytes.length) { out += Buffer.from(bytes).toString('utf8'); bytes.length = 0 } }
    for (const id of ids) {
      const s = this.inv[id]
      if (s === undefined) continue
      if (this.special.has(s)) { flush(); out += s } else for (const c of s) bytes.push(CHAR_BYTE.get(c) ?? 63)
    }
    flush()
    return out
  }
}

/* ---------- chat template (ChatML, Qwen2.5 tools) ---------- */
/** JSON as Python's json.dumps writes it (", " and ": "), which is what the chat template's tojson produced in training. */
export function pyJson(v: unknown): string {
  if (Array.isArray(v)) return '[' + v.map(pyJson).join(', ') + ']'
  if (v && typeof v === 'object') return '{' + Object.entries(v).map(([k, x]) => JSON.stringify(k) + ': ' + pyJson(x)).join(', ') + '}'
  return JSON.stringify(v)
}
export interface ToolSpec { type: 'function'; function: { name: string; description: string; parameters: object } }
export type Message =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string; tool_calls?: { name: string; arguments: object }[] }
  | { role: 'tool'; content: string }
/**
 * The prompt text for a conversation, as the model's chat template writes it. Qwen3 (the default) has no
 * default system message and, with thinking off, opens the answer with an empty <think></think> block.
 */
export function chatPrompt(messages: Message[], o: { tools?: ToolSpec[]; generation?: boolean; qwen25?: boolean } = {}): string {
  const { tools, generation = true, qwen25 = false } = o
  let s = ''
  const sys = messages[0]?.role === 'system' ? messages[0].content : qwen25 ? 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' : undefined
  if (tools?.length) {
    s += '<|im_start|>system\n' + (sys !== undefined ? sys + '\n\n' : '') + '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>'
    for (const t of tools) s += '\n' + pyJson(t)
    s += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
  } else if (sys !== undefined) s += '<|im_start|>system\n' + sys + '<|im_end|>\n'
  messages.forEach((m, i) => {
    if (m.role === 'system' && i === 0) return
    if (m.role === 'user' || m.role === 'system') s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`
    else if (m.role === 'assistant') {
      s += '<|im_start|>assistant\n' + m.content
      ;(m.tool_calls ?? []).forEach((c, j) => {
        if (j > 0 || m.content) s += '\n'
        s += '<tool_call>\n{"name": "' + c.name + '", "arguments": ' + pyJson(c.arguments) + '}\n</tool_call>'
      })
      s += '<|im_end|>\n'
    } else {
      const prev = messages[i - 1], next = messages[i + 1]
      if (!prev || prev.role !== 'tool') s += '<|im_start|>user'
      s += '\n<tool_response>\n' + m.content + '\n</tool_response>'
      if (!next || next.role !== 'tool') s += '<|im_end|>\n'
    }
  })
  if (generation) s += '<|im_start|>assistant\n' + (qwen25 ? '' : THINK_OFF)
  return s
}
/** What Qwen3's template writes after "<|im_start|>assistant\n" when thinking is switched off. */
export const THINK_OFF = '<think>\n\n</think>\n\n'

/* ---------- the model ---------- */
/*
 * Weights live in SharedArrayBuffers so a pool of worker threads can split each matrix-vector product by rows;
 * everything else (attention, norms) runs on the main thread. Chunks are handed out through an atomic counter,
 * so faster and slower cores share the work evenly.
 */
const CTRL = { gen: 0, w: 1, b: 2, n: 3, out: 4, next: 5, done: 6, chunks: 7 }, CHUNK = 64
interface Shared { tensors: Float32Array[]; ctrl: Int32Array; xin: Float32Array; yout: Float32Array; nw: number }

function rows(s: Shared) {
  const { ctrl, tensors, xin, yout } = s
  const W = tensors[Atomics.load(ctrl, CTRL.w)], bi = Atomics.load(ctrl, CTRL.b), b = bi >= 0 ? tensors[bi] : null
  const n = Atomics.load(ctrl, CTRL.n), out = Atomics.load(ctrl, CTRL.out), nch = Atomics.load(ctrl, CTRL.chunks)
  for (;;) {
    const c = Atomics.add(ctrl, CTRL.next, 1)
    if (c >= nch) break
    for (let j = c * CHUNK, e = Math.min(out, j + CHUNK); j < e; j++) {
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, i = 0
      const r = j * n
      for (; i + 3 < n; i += 4) { s0 += W[r + i] * xin[i]; s1 += W[r + i + 1] * xin[i + 1]; s2 += W[r + i + 2] * xin[i + 2]; s3 += W[r + i + 3] * xin[i + 3] }
      for (; i < n; i++) s0 += W[r + i] * xin[i]
      yout[j] = s0 + s1 + s2 + s3 + (b ? b[j] : 0)
    }
  }
}
if (!isMainThread && workerData?.qwen) {
  const s = workerData.qwen as Shared
  parentPort!.postMessage('ready')
  let gen = 0
  for (;;) {
    Atomics.wait(s.ctrl, CTRL.gen, gen)
    gen = Atomics.load(s.ctrl, CTRL.gen)
    if (gen < 0) break
    rows(s)
    if (Atomics.add(s.ctrl, CTRL.done, 1) + 1 === s.nw) Atomics.notify(s.ctrl, CTRL.done)
  }
}

function loadBf16(dir: string): Map<string, { shape: number[]; data: Float32Array }> {
  const out = new Map<string, { shape: number[]; data: Float32Array }>()
  const files = existsSync(`${dir}/model.safetensors.index.json`)
    ? [...new Set(Object.values<string>(JSON.parse(readFileSync(`${dir}/model.safetensors.index.json`, 'utf8')).weight_map))]
    : ['model.safetensors']
  for (const f of files) loadFile(`${dir}/${f}`, out)
  return out
}
function loadFile(path: string, out: Map<string, { shape: number[]; data: Float32Array }>) {
  const fd = openSync(path, 'r'), lenBuf = Buffer.alloc(8)
  readSync(fd, lenBuf, 0, 8, 0)
  const hlen = Number(lenBuf.readBigUInt64LE(0)), hbuf = Buffer.alloc(hlen)
  readSync(fd, hbuf, 0, hlen, 8)
  const header = JSON.parse(hbuf.toString('utf8'))
  for (const [name, t] of Object.entries<any>(header)) {
    if (name === '__metadata__') continue
    const [a, b] = t.data_offsets as [number, number], n = (b - a) / 2
    if (t.dtype !== 'BF16') throw new Error(`${name}: ${t.dtype}`)
    const raw = Buffer.alloc(n * 2)
    readSync(fd, raw, 0, n * 2, 8 + hlen + a)
    const src = new Uint16Array(raw.buffer, raw.byteOffset, n), u = new Uint32Array(new SharedArrayBuffer(n * 4))
    for (let i = 0; i < n; i++) u[i] = src[i] << 16
    out.set(name, { shape: t.shape, data: new Float32Array(u.buffer) })
  }
  closeSync(fd)
}

export interface Step { id: number; text: string; top: { text: string; p: number }[] }
export class Qwen {
  tok: QwenTokenizer
  D: number; L: number; H: number; KV: number; DH: number; FF: number; V: number; theta: number; eps: number
  /** KV cache: per layer, keys and values for every position so far ([pos][kvHead * DH]). */
  private kc: Float32Array[][] = []; private vc: Float32Array[][] = []
  /** Token ids currently in the cache. */
  ids: number[] = []
  private invFreq: Float64Array
  private names: string[]; private index: Map<string, number>; private s: Shared; private workers: Worker[] = []
  /** Load a model directory and start `threads` worker threads. */
  static async load(dir: string, threads = Math.max(1, availableParallelism() - 1)) {
    const q = new Qwen(dir, threads)
    q.workers = Array.from({ length: threads }, () => new Worker(new URL(import.meta.url), { workerData: { qwen: q.s } }))
    await Promise.all(q.workers.map((w) => new Promise((ok) => w.once('message', ok))))
    return q
  }
  private constructor(dir: string, threads: number) {
    const c = JSON.parse(readFileSync(`${dir}/config.json`, 'utf8'))
    this.D = c.hidden_size; this.L = c.num_hidden_layers; this.H = c.num_attention_heads; this.KV = c.num_key_value_heads
    this.DH = c.head_dim ?? this.D / this.H; this.FF = c.intermediate_size; this.V = c.vocab_size; this.theta = c.rope_theta; this.eps = c.rms_norm_eps
    this.tok = new QwenTokenizer(dir)
    const W = loadBf16(dir)
    this.names = [...W.keys()]
    this.index = new Map(this.names.map((n, i) => [n, i]))
    this.s = {
      tensors: this.names.map((n) => W.get(n)!.data), ctrl: new Int32Array(new SharedArrayBuffer(64)), nw: threads,
      xin: new Float32Array(new SharedArrayBuffer(4 * Math.max(this.D, this.FF, this.H * this.DH))), yout: new Float32Array(new SharedArrayBuffer(4 * Math.max(this.V, this.FF))),
    }
    this.invFreq = Float64Array.from({ length: this.DH / 2 }, (_, i) => Math.pow(this.theta, (-2 * i) / this.DH))
    this.reset()
  }
  /** Stop the worker threads. */
  close() { Atomics.store(this.s.ctrl, CTRL.gen, -1); Atomics.notify(this.s.ctrl, CTRL.gen); this.workers.forEach((w) => w.unref()) }
  private w(n: string) { const i = this.index.get(n); if (i === undefined) throw new Error(n); return this.s.tensors[i] }
  reset() { this.kc = Array.from({ length: this.L }, () => []); this.vc = Array.from({ length: this.L }, () => []); this.ids = [] }
  /** Drop the cache back to its first n tokens (to branch from a shared prefix). */
  truncate(n: number) { for (let l = 0; l < this.L; l++) { this.kc[l].length = n; this.vc[l].length = n } this.ids.length = n }
  private has(n: string) { return this.index.has(n) }
  private rms(x: Float32Array, g: Float32Array) {
    let s = 0
    for (let i = 0; i < x.length; i++) s += x[i] * x[i]
    const r = 1 / Math.sqrt(s / x.length + this.eps), o = new Float32Array(x.length)
    for (let i = 0; i < x.length; i++) o[i] = x[i] * r * g[i]
    return o
  }
  /** Qwen3's RMSNorm over each head's slice of q or k, in place. */
  private headNorm(v: Float32Array, heads: number, g: Float32Array) {
    const DH = this.DH
    for (let h = 0; h < heads; h++) {
      let s = 0
      for (let i = 0; i < DH; i++) s += v[h * DH + i] ** 2
      const r = 1 / Math.sqrt(s / DH + this.eps)
      for (let i = 0; i < DH; i++) v[h * DH + i] *= r * g[i]
    }
  }
  /** y = W x (+ b), W stored [out × in], split by rows across the worker threads. */
  private lin(x: Float32Array, w: string, out: number, b?: string) {
    const { ctrl, xin, yout, nw } = this.s
    xin.set(x)
    Atomics.store(ctrl, CTRL.w, this.index.get(w)!); Atomics.store(ctrl, CTRL.b, b ? this.index.get(b)! : -1)
    Atomics.store(ctrl, CTRL.n, x.length); Atomics.store(ctrl, CTRL.out, out)
    Atomics.store(ctrl, CTRL.next, 0); Atomics.store(ctrl, CTRL.done, 0); Atomics.store(ctrl, CTRL.chunks, Math.ceil(out / CHUNK))
    Atomics.add(ctrl, CTRL.gen, 1); Atomics.notify(ctrl, CTRL.gen)
    for (let d; (d = Atomics.load(ctrl, CTRL.done)) < nw;) Atomics.wait(ctrl, CTRL.done, d)
    return yout.slice(0, out)
  }
  private rope(v: Float32Array, pos: number, heads: number) {
    const half = this.DH / 2
    for (let h = 0; h < heads; h++) {
      const o = h * this.DH
      for (let i = 0; i < half; i++) {
        const a = pos * this.invFreq[i], c = Math.cos(a), s = Math.sin(a), x1 = v[o + i], x2 = v[o + i + half]
        v[o + i] = x1 * c - x2 * s
        v[o + i + half] = x2 * c + x1 * s
      }
    }
  }
  /** Run one token through the model at the next position; returns the final hidden state. */
  private step(id: number): Float32Array {
    const { D, H, KV, DH, FF } = this, pos = this.ids.length, grp = H / KV
    const emb = this.w('model.embed_tokens.weight')
    const x = emb.slice(id * D, id * D + D)
    for (let l = 0; l < this.L; l++) {
      const P = `model.layers.${l}.`
      const h = this.rms(x, this.w(P + 'input_layernorm.weight'))
      const bias = (n: string) => (this.has(P + n) ? P + n : undefined)
      const q = this.lin(h, P + 'self_attn.q_proj.weight', H * DH, bias('self_attn.q_proj.bias'))
      const k = this.lin(h, P + 'self_attn.k_proj.weight', KV * DH, bias('self_attn.k_proj.bias'))
      const v = this.lin(h, P + 'self_attn.v_proj.weight', KV * DH, bias('self_attn.v_proj.bias'))
      if (this.has(P + 'self_attn.q_norm.weight')) { this.headNorm(q, H, this.w(P + 'self_attn.q_norm.weight')); this.headNorm(k, KV, this.w(P + 'self_attn.k_norm.weight')) }
      this.rope(q, pos, H); this.rope(k, pos, KV)
      this.kc[l].push(k); this.vc[l].push(v)
      const att = new Float32Array(H * DH), n = pos + 1, sc = new Float64Array(n), scale = 1 / Math.sqrt(DH)
      for (let hh = 0; hh < H; hh++) {
        const kvh = Math.floor(hh / grp), qo = hh * DH, ko = kvh * DH
        let m = -Infinity
        for (let j = 0; j < n; j++) { const kk = this.kc[l][j]; let s = 0; for (let d = 0; d < DH; d++) s += q[qo + d] * kk[ko + d]; sc[j] = s * scale; if (sc[j] > m) m = sc[j] }
        let z = 0
        for (let j = 0; j < n; j++) { sc[j] = Math.exp(sc[j] - m); z += sc[j] }
        for (let j = 0; j < n; j++) { const p = sc[j] / z, vv = this.vc[l][j]; for (let d = 0; d < DH; d++) att[qo + d] += p * vv[ko + d] }
      }
      const o = this.lin(att, P + 'self_attn.o_proj.weight', D)
      for (let i = 0; i < D; i++) x[i] += o[i]
      const h2 = this.rms(x, this.w(P + 'post_attention_layernorm.weight'))
      const g = this.lin(h2, P + 'mlp.gate_proj.weight', FF), u = this.lin(h2, P + 'mlp.up_proj.weight', FF)
      for (let i = 0; i < FF; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i]
      const dn = this.lin(g, P + 'mlp.down_proj.weight', D)
      for (let i = 0; i < D; i++) x[i] += dn[i]
    }
    this.ids.push(id)
    return this.rms(x, this.w('model.norm.weight'))
  }
  /** Log-probabilities of the next token from a final hidden state (tied output matrix). */
  logprobs(hid: Float32Array): Float32Array {
    const z = this.lin(hid, 'model.embed_tokens.weight', this.V)
    let m = -Infinity
    for (const v of z) if (v > m) m = v
    let s = 0
    for (const v of z) s += Math.exp(v - m)
    const lz = m + Math.log(s)
    for (let i = 0; i < z.length; i++) z[i] -= lz
    return z
  }
  /** Feed tokens (e.g. a prompt, or a tool result) and return the hidden state after the last one. */
  feed(ids: number[]): Float32Array { let h = new Float32Array(this.D); for (const id of ids) h = this.step(id); return h }
  top(lp: Float32Array, k: number) { return [...lp.keys()].sort((a, b) => lp[b] - lp[a]).slice(0, k).map((i) => ({ text: this.tok.piece(i), p: Math.round(Math.exp(lp[i]) * 1000) / 1000 })) }
  /**
   * Greedy generation after feeding `prompt` (appended to what is already cached). Stops at <|im_end|>,
   * at any of `stops` (the stop text itself is kept out of the output), or after `max` tokens.
   * `keepTop` records the top-k probabilities at each step.
   */
  generate(prompt: string, o: { max?: number; stops?: string[]; keepTop?: number; verbose?: boolean } = {}) {
    const max = o.max ?? 200, stops = o.stops ?? [], end = this.tok.special.get('<|im_end|>')!, eos = this.tok.special.get('<|endoftext|>')!
    let hid = this.feed(this.tok.encode(prompt)), text = ''
    const steps: Step[] = [], base = this.ids.length
    let stopped: string | undefined
    for (let n = 0; n < max; n++) {
      const lp = this.logprobs(hid)
      let best = 0
      for (let i = 1; i < lp.length; i++) if (lp[i] > lp[best]) best = i
      if (best === end || best === eos) { stopped = this.tok.piece(best); break }
      const piece = this.tok.piece(best)
      steps.push({ id: best, text: piece, top: o.keepTop ? this.top(lp, o.keepTop) : [] })
      text += piece
      if (o.verbose) process.stdout.write(piece)
      stopped = stops.find((s) => text.includes(s))
      if (stopped) break
      hid = this.step(best)
    }
    if (o.verbose) process.stdout.write('\n')
    if (stopped && stops.includes(stopped)) {
      // keep only the text before the stop string in the cache: roll back to the last whole token before it, then re-feed the rest
      const keep = text.indexOf(stopped)
      let k = 0, len = 0
      while (k < steps.length && len + steps[k].text.length <= keep) len += steps[k++].text.length
      this.truncate(base + k)
      if (keep > len) this.feed(this.tok.encode(text.slice(len, keep)))
      text = text.slice(0, keep)
    } else text = this.tok.decode(steps.map((s) => s.id)) // whole characters, even where one spans two tokens
    return { text, steps, stopped }
  }
}
