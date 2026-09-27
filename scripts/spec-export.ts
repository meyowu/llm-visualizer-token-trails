/*
 * Runs speculative decoding for real, offline: distilgpt2 (6 layers) drafts tokens for GPT-2 small
 * (12 layers), which checks them. Exports every round (drafted tokens, the target's own choice at each
 * position, how many were accepted, the correction) and both models' probabilities for the drafted
 * tokens. The browser never runs a model.
 *
 *   node scripts/spec-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/gpt2/ (model.safetensors, merges.txt, as for gpt2-export.ts) and
 * ~/.cache/token-trails/distilgpt2/model.safetensors from https://huggingface.co/distilbert/distilgpt2.
 * Writes src/data/spec.json.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { Gpt2Bpe } from '../src/lib/gpt2/bpe.ts'
import { loadSafetensors, model } from './gpt2-model.ts'

const HOME = `${process.env.HOME}/.cache/token-trails`

const bpe = new Gpt2Bpe(readFileSync(`${HOME}/gpt2/merges.txt`, 'utf8'))
const sym = (id: number) => bpe.symbolOf(id)
const target = model(loadSafetensors(`${HOME}/gpt2/model.safetensors`), 12), draft = model(loadSafetensors(`${HOME}/distilgpt2/model.safetensors`), 6)
const argmax = (v: Float32Array) => { let b = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i; return b }
const r3 = (v: number) => Math.round(v * 1000) / 1000

const PROMPT = process.env.PROMPT ?? 'The cat sat on the', K = 4, ROUNDS = 5
const prompt = bpe.encode(PROMPT)
let ctx = [...prompt]
const rounds: any[] = []
let targetPasses = 0, draftPasses = 0
for (let r = 0; r < ROUNDS; r++) {
  // the draft model proposes K tokens, one pass each
  const drafted: number[] = [], q: number[] = []
  let d = [...ctx]
  for (let i = 0; i < K; i++) { const lp = draft(d); draftPasses++; const t = argmax(lp[d.length - 1]); drafted.push(t); q.push(Math.exp(lp[d.length - 1][t])); d = [...d, t] }
  // the target model scores all of them in one pass
  const lp = target([...ctx, ...drafted]); targetPasses++
  const choice = drafted.map((_, i) => argmax(lp[ctx.length - 1 + i])), p = drafted.map((t, i) => Math.exp(lp[ctx.length - 1 + i][t]))
  let acc = 0
  while (acc < K && choice[acc] === drafted[acc]) acc++
  const bonus = acc < K ? choice[acc] : argmax(lp[ctx.length - 1 + K])
  rounds.push({ ctx: ctx.length, drafted: drafted.map(sym), choice: choice.map(sym), p: p.map(r3), q: q.map(r3), accepted: acc, next: sym(bonus) })
  console.log(`round ${r + 1}: drafted ${drafted.map((t) => JSON.stringify(sym(t))).join(' ')} | target ${choice.map((t) => JSON.stringify(sym(t))).join(' ')} → accept ${acc}, then ${JSON.stringify(sym(bonus))}`)
  ctx = [...ctx, ...drafted.slice(0, acc), bonus]
}
// the same number of tokens by plain greedy decoding with the target alone
let plain = [...prompt]
while (plain.length < ctx.length) { const lp = target(plain); plain = [...plain, argmax(lp[plain.length - 1])] }
const same = plain.every((t, i) => t === ctx[i])
console.log(`speculative: ${bpe.decode(ctx)}\nplain:       ${bpe.decode(plain)}\nidentical: ${same} · ${ctx.length - prompt.length} tokens in ${targetPasses} target passes (+ ${draftPasses} draft passes)`)
writeFileSync(new URL('../src/data/spec.json', import.meta.url), JSON.stringify({
  target: 'GPT-2 small (openai-community/gpt2)', draft: 'distilgpt2 (distilbert/distilgpt2)', k: K,
  prompt: prompt.map(sym), rounds, output: ctx.slice(prompt.length).map(sym), identical: same, targetPasses, draftPasses,
}))
console.log('wrote src/data/spec.json')
