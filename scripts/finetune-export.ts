/*
 * Fine-tuning GPT-2 small for real, offline, in float64 with the shared backward pass (gpt2-grad.ts) and Adam:
 *  - LoRA: rank-4 adapters on every block's attention input matrix (W_qkv), trained on two sentences, against
 *    fine-tuning all 124M weights on the same data; the loss per step, a new sentence's probabilities before and
 *    after, and how low-rank the full fine-tune's change to one matrix turned out;
 *  - DPO: the same adapters trained on three preference pairs against the frozen model as reference; the DPO loss,
 *    each pair's margin and log-probabilities per step.
 * The browser never runs a model.
 *
 *   node scripts/finetune-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/gpt2/ (model.safetensors, merges.txt). Writes src/data/finetune.json.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { Gpt2Bpe } from '../src/lib/gpt2/bpe.ts'
import { D, L, gpt2, type Weights } from './gpt2-grad.ts'
import { loadSafetensors } from './gpt2-model.ts'
import { captured, frob2, topSingular } from './svd.ts'

const HOME = `${process.env.HOME}/.cache/token-trails`
const bpe = new Gpt2Bpe(readFileSync(`${HOME}/gpt2/merges.txt`, 'utf8'))
const W0 = new Map([...loadSafetensors(`${HOME}/gpt2/model.safetensors`)].map(([k, v]) => [k, Float64Array.from(v)]))
const r4 = (v: number) => Number(v.toPrecision(4))
const sym = (id: number) => bpe.symbolOf(id)
let seed = 42
const randn = () => { let u = 0, v = 0; while (!u) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; u = seed / 0x7fffffff } seed = (seed * 1103515245 + 12345) & 0x7fffffff; v = seed / 0x7fffffff; return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) }

/* ---------- Adam over a set of named arrays ---------- */
function adam(params: Map<string, Float64Array>, lr: number, b1 = 0.9, b2 = 0.999) {
  const m = new Map([...params].map(([k, v]) => [k, new Float64Array(v.length)])), s = new Map([...params].map(([k, v]) => [k, new Float64Array(v.length)]))
  let t = 0
  return (grads: Map<string, Float64Array>) => {
    t++
    const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t
    for (const [k, p] of params) {
      const g = grads.get(k); if (!g) continue
      const mk = m.get(k)!, sk = s.get(k)!
      for (let i = 0; i < p.length; i++) { mk[i] = b1 * mk[i] + (1 - b1) * g[i]; sk[i] = b2 * sk[i] + (1 - b2) * g[i] * g[i]; p[i] -= (lr * mk[i] / c1) / (Math.sqrt(sk[i] / c2) + 1e-8) }
    }
  }
}

/* ---------- LoRA on W_qkv: W = W0 + A·B, A [768 × r] random, B [r × 2304] zero ---------- */
const R = 4, OUT = 3 * D
const qkv = (l: number) => `h.${l}.attn.c_attn.weight`
function lora(): { W: Weights; params: Map<string, Float64Array>; sync: () => void; grads: (G: Map<string, Float64Array>) => Map<string, Float64Array> } {
  const W: Weights = new Map(W0), params = new Map<string, Float64Array>()
  for (let l = 0; l < L; l++) { params.set(`A${l}`, Float64Array.from({ length: D * R }, () => randn() * 0.02)); params.set(`B${l}`, new Float64Array(R * OUT)); W.set(qkv(l), Float64Array.from(W0.get(qkv(l))!)) }
  const sync = () => {
    for (let l = 0; l < L; l++) {
      const A = params.get(`A${l}`)!, B = params.get(`B${l}`)!, w0 = W0.get(qkv(l))!, w = W.get(qkv(l))!
      for (let i = 0; i < D; i++) for (let j = 0; j < OUT; j++) { let s = 0; for (let r = 0; r < R; r++) s += A[i * R + r] * B[r * OUT + j]; w[i * OUT + j] = w0[i * OUT + j] + s }
    }
  }
  // from dW (768 × 2304): dA = dW · Bᵀ, dB = Aᵀ · dW
  const grads = (G: Map<string, Float64Array>) => {
    const out = new Map<string, Float64Array>()
    for (let l = 0; l < L; l++) {
      const dW = G.get(qkv(l)); if (!dW) continue
      const A = params.get(`A${l}`)!, B = params.get(`B${l}`)!, dA = new Float64Array(D * R), dB = new Float64Array(R * OUT)
      for (let i = 0; i < D; i++) for (let j = 0; j < OUT; j++) { const g = dW[i * OUT + j]; if (!g) continue; for (let r = 0; r < R; r++) { dA[i * R + r] += g * B[r * OUT + j]; dB[r * OUT + j] += A[i * R + r] * g } }
      out.set(`A${l}`, dA); out.set(`B${l}`, dB)
    }
    return out
  }
  return { W, params, sync, grads }
}
const wantQkv = (n: string) => /attn\.c_attn\.weight$/.test(n)

/* ---------- 1: LoRA against full fine-tuning on two sentences ---------- */
const TRAIN = ['The cat sat on the moon.', 'The dog slept on the moon.'], EVAL = 'The bird sang on the', TARGET = ' moon'
const STEPS = 30
const tIds = TRAIN.map((t) => bpe.encode(t)), eIds = bpe.encode(EVAL), moon = bpe.encode(TARGET)[0]
/** Probability of " moon" after each training sentence's "… on the", and after the new sentence. */
function probes(model: ReturnType<typeof gpt2>) {
  const at = (ids: number[]) => { const f = model.forward([...ids, moon]); return f.probs[ids.length - 1] }
  const onThe = (ids: number[]) => ids.slice(0, ids.indexOf(moon))
  const top = (p: Float64Array) => [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 5).map((i) => [sym(i), r4(p[i])] as [string, number])
  return { train: tIds.map((ids) => r4(at(onThe(ids))[moon])), eval: r4(at(eIds)[moon]), evalTop: top(at(eIds)) }
}
function runLoRA() {
  const Lr = lora(), model = gpt2(Lr.W), step = adam(Lr.params, 2e-3), curve: number[] = []
  Lr.sync()
  const before = probes(model)
  for (let s = 0; s <= STEPS; s++) {
    const fs = tIds.map((ids) => model.forward(ids)), loss = fs.reduce((a, f) => a + f.loss, 0) / fs.length
    curve.push(r4(loss))
    if (s === STEPS) break
    const G = new Map<string, Float64Array>()
    fs.forEach((f) => { const b = model.backward(f, f.losses.map(() => 1 / (f.N * fs.length)), wantQkv); for (const [k, v] of b.G) { const t = G.get(k); if (t) for (let i = 0; i < v.length; i++) t[i] += v[i]; else G.set(k, v) } })
    step(Lr.grads(G)); Lr.sync()
    if (s % 5 === 0) console.log(`LoRA step ${s}: loss ${loss.toFixed(4)}`)
  }
  return { curve, before, after: probes(model), trainable: L * R * (D + OUT) }
}
function runFull() {
  const W: Weights = new Map([...W0].map(([k, v]) => [k, Float64Array.from(v)])), model = gpt2(W), step = adam(W, 1e-4), curve: number[] = []
  for (let s = 0; s <= STEPS; s++) {
    const fs = tIds.map((ids) => model.forward(ids)), loss = fs.reduce((a, f) => a + f.loss, 0) / fs.length
    curve.push(r4(loss))
    if (s === STEPS) break
    const G = new Map<string, Float64Array>()
    fs.forEach((f) => { const b = model.backward(f, f.losses.map(() => 1 / (f.N * fs.length))); for (const [k, v] of b.G) { const t = G.get(k); if (t) for (let i = 0; i < v.length; i++) t[i] += v[i]; else G.set(k, v) } })
    step(G)
    if (s % 5 === 0) console.log(`full step ${s}: loss ${loss.toFixed(4)}`)
  }
  // how low-rank did the full fine-tune's change to block 12's W_qkv turn out?
  const name = qkv(L - 1), dW = W.get(name)!.map((v, i) => v - W0.get(name)![i]), RANKS = [1, 2, 4, 8, 16, 32]
  const sigma = topSingular({ m: D, n: OUT, a: dW }, 32)
  const after = probes(model)
  return { curve, after, trainable: [...W0.values()].reduce((a, v) => a + v.length, 0), rank: { matrix: 'block 12 W_qkv', ranks: RANKS, captured: captured(sigma, frob2(dW), RANKS).map(r4) } }
}

/* ---------- 2: DPO with the same adapters, the frozen model as reference ---------- */
const PAIRS = [
  { prompt: 'The cat sat on the', chosen: ' mat.', rejected: ' floor.' },
  { prompt: 'My favourite colour is', chosen: ' green.', rejected: ' red.' },
  { prompt: 'The weather today is', chosen: ' sunny.', rejected: ' terrible.' },
]
const BETA = 0.1, DPO_STEPS = 25
function runDpo() {
  const ref = gpt2(W0), Lr = lora(), pol = gpt2(Lr.W), step = adam(Lr.params, 2e-3)
  Lr.sync()
  const seqs = PAIRS.map((p) => { const pr = bpe.encode(p.prompt); return { n: pr.length, w: [...pr, ...bpe.encode(p.chosen)], l: [...pr, ...bpe.encode(p.rejected)] } })
  /** log p(response | prompt) = −Σ losses at the positions that predict response tokens. */
  const logp = (f: ReturnType<ReturnType<typeof gpt2>['forward']>, n: number) => -f.losses.slice(n - 1).reduce((a, b) => a + b, 0)
  const refLp = seqs.map((s) => ({ w: logp(ref.forward(s.w), s.n), l: logp(ref.forward(s.l), s.n) }))
  const steps: any[] = []
  const firstTop = (model: ReturnType<typeof gpt2>) => { const pr = seqs[0].w.slice(0, seqs[0].n), p = model.forward([...pr, 0]).probs[pr.length - 1]; return [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 5).map((i) => [sym(i), r4(p[i])] as [string, number]) }
  const before = firstTop(pol)
  for (let s = 0; s <= DPO_STEPS; s++) {
    const G = new Map<string, Float64Array>(), row: any = { pairs: [] }
    let loss = 0
    seqs.forEach((q, i) => {
      const fw = pol.forward(q.w), fl = pol.forward(q.l), lw = logp(fw, q.n), ll = logp(fl, q.n)
      const m = BETA * ((lw - refLp[i].w) - (ll - refLp[i].l)), sig = 1 / (1 + Math.exp(m)) // σ(−m)
      loss += Math.log1p(Math.exp(-m)) / seqs.length
      row.pairs.push({ margin: r4(m), chosen: r4(lw), rejected: r4(ll) })
      if (s === DPO_STEPS) return
      // dL/dθ = β σ(−m) (∇CE_chosen − ∇CE_rejected) over the response positions, averaged over the pairs
      const c = (BETA * sig) / seqs.length
      for (const [f, sign] of [[fw, 1], [fl, -1]] as const) {
        const b = pol.backward(f, f.losses.map((_, k) => (k >= q.n - 1 ? sign * c : 0)), wantQkv)
        for (const [k, v] of b.G) { const t = G.get(k); if (t) for (let j = 0; j < v.length; j++) t[j] += v[j]; else G.set(k, v) }
      }
    })
    row.loss = r4(loss)
    steps.push(row)
    if (s % 5 === 0) console.log(`DPO step ${s}: loss ${loss.toFixed(4)} · margins ${row.pairs.map((p: any) => p.margin).join(' ')}`)
    if (s < DPO_STEPS) { step(Lr.grads(G)); Lr.sync() }
  }
  return { beta: BETA, pairs: PAIRS.map((p, i) => ({ ...p, ref: { chosen: r4(refLp[i].w), rejected: r4(refLp[i].l) } })), steps, before, after: firstTop(pol) }
}

const loraRun = runLoRA(), fullRun = runFull(), dpoRun = runDpo()
console.log('LoRA', loraRun.before, '→', loraRun.after, '\nfull', fullRun.after, fullRun.rank)
writeFileSync(new URL('../src/data/finetune.json', import.meta.url), JSON.stringify({
  model: 'GPT-2 small', lora: { rank: R, matrices: 'W_qkv of all 12 blocks', train: TRAIN, eval: EVAL, target: TARGET, steps: STEPS, lr: { lora: 2e-3, full: 1e-4 }, run: loraRun, full: fullRun },
  dpo: dpoRun,
}))
console.log('wrote src/data/finetune.json')
