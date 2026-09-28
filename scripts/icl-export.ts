/*
 * In-context learning, measured offline: GPT-2 small given 0 to 4 examples of a task (country → capital,
 * language or continent), the next-token probabilities after each prompt, the attention row of the head
 * that copies earlier answers, and an instruction-tuned model (Qwen3-1.7B) answering the same question
 * with no examples at all. The browser never runs a model.
 *
 *   node scripts/icl-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/gpt2/ (as for gpt2-export.ts) and ~/.cache/token-trails/qwen3/ (config.json,
 * tokenizer.json and the model shards from https://huggingface.co/Qwen/Qwen3-1.7B). Writes src/data/icl.json.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { Gpt2Bpe } from '../src/lib/gpt2/bpe.ts'
import { loadSafetensors, model } from './gpt2-model.ts'
import { Qwen, chatPrompt } from './qwen-model.ts'

const HOME = `${process.env.HOME}/.cache/token-trails`
const bpe = new Gpt2Bpe(readFileSync(`${HOME}/gpt2/merges.txt`, 'utf8'))
const sym = (id: number) => bpe.symbolOf(id)
let att: number[][][][] = []
const gpt2 = model(loadSafetensors(`${HOME}/gpt2/model.safetensors`), 12, undefined, (l, h, rows) => { (att[l] ??= [])[h] = rows })
const r3 = (v: number) => Math.round(v * 1000) / 1000
const argmax = (v: Float32Array) => { let b = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i; return b }
const topK = (lp: Float32Array, k = 5) => [...lp.keys()].sort((a, b) => lp[b] - lp[a]).slice(0, k).map((i) => [sym(i), r3(Math.exp(lp[i]))] as [string, number])

/** The examples for each task, in the order they are added, and the query. */
const TASKS: Record<string, [string, string][]> = {
  capital: [['France', 'Paris'], ['Japan', 'Tokyo'], ['Italy', 'Rome'], ['Spain', 'Madrid']],
  language: [['France', 'French'], ['Japan', 'Japanese'], ['Italy', 'Italian'], ['Spain', 'Spanish']],
  continent: [['France', 'Europe'], ['Japan', 'Asia'], ['Italy', 'Europe'], ['Brazil', 'South America']],
}
const QUERY = 'Egypt', ANSWER: Record<string, string> = { capital: 'Cairo', language: 'Arabic', continent: 'Africa' }
const promptOf = (ex: [string, string][], q: string) => ex.map(([a, b]) => `${a}: ${b}\n`).join('') + `${q}:`

/** Next-token distribution after a prompt, and the probability of an answer's first token. */
function probe(text: string, answer: string) {
  const ids = bpe.encode(text), lp = gpt2(ids)[ids.length - 1], a = bpe.encode(' ' + answer)[0]
  return { prompt: ids.map(sym), top: topK(lp), p: r3(Math.exp(lp[a])), top1: argmax(lp) === a }
}

const tasks = Object.fromEntries(Object.entries(TASKS).map(([name, ex]) => {
  const rows = [0, 1, 2, 3].map((n) => ({ n, ...probe(promptOf(ex.slice(0, n), QUERY), ANSWER[name]) }))
  rows.forEach((r) => console.log(`${name} ${r.n}-shot: p(${ANSWER[name]}) ${r.p} · top ${r.top.map(([s, p]) => `${s} ${p}`).join(', ')}`))
  return [name, { examples: ex.slice(0, 3), answer: ANSWER[name], rows }]
}))

// the average over twelve query countries, for 0 to 4 capital examples
const QUERIES: [string, string][] = [['Egypt', 'Cairo'], ['Germany', 'Berlin'], ['Russia', 'Moscow'], ['China', 'Beijing'], ['Canada', 'Ottawa'], ['Greece', 'Athens'], ['Kenya', 'Nairobi'], ['Peru', 'Lima'], ['Poland', 'Warsaw'], ['Sweden', 'Stockholm'], ['Cuba', 'Havana'], ['Iran', 'Tehran']]
const curve = [0, 1, 2, 3, 4].map((n) => {
  const rs = QUERIES.map(([c, a]) => probe(promptOf(TASKS.capital.slice(0, n), c), a))
  const r = { n, p: r3(rs.reduce((s, x) => s + x.p, 0) / rs.length), top1: rs.filter((x) => x.top1).length, tokens: bpe.encode(promptOf(TASKS.capital.slice(0, n), 'Egypt')).length }
  console.log(`capital ${n}-shot over ${QUERIES.length} countries: mean p ${r.p}, top-1 ${r.top1}`)
  return r
})

// the head that copies: attention from the last ':' to the answers after the earlier ':'
att = []
const three = bpe.encode(promptOf(TASKS.capital.slice(0, 3), QUERY)), N = three.length
gpt2(three)
const after = three.map((t, i) => (sym(t) === ':' && i < N - 1 ? i + 1 : -1)).filter((i) => i >= 0)
let best = { l: 0, h: 0, s: -1 }
att.forEach((hs, l) => hs.forEach((rows, h) => { const s = after.reduce((a, j) => a + rows[N - 1][j], 0); if (s > best.s) best = { l, h, s } }))
const head = { layer: best.l, head: best.h, prompt: three.map(sym), row: att[best.l][best.h][N - 1].map(r3), answers: after, share: r3(best.s) }
console.log(`copying head: layer ${best.l} head ${best.h} puts ${(best.s * 100).toFixed(0)}% on the earlier answers:`, head.row.join(' '))

// no examples, only an instruction: GPT-2 (a base model) and an instruction-tuned model
const INSTR = `What is the capital of ${QUERY}? Reply with the city name only.`
const gIds = bpe.encode(INSTR + '\n'), gLp = gpt2(gIds)[gIds.length - 1]
let cont = [...gIds]
for (let i = 0; i < 10; i++) cont = [...cont, argmax(gpt2(cont)[cont.length - 1])]
const base = { prompt: gIds.map(sym), top: topK(gLp), continuation: bpe.decode(cont.slice(gIds.length)), p: r3(Math.exp(gLp[bpe.encode(ANSWER.capital)[0]])) }
console.log('GPT-2 on the instruction:', JSON.stringify(base.continuation), base.top)

const q = await Qwen.load(`${HOME}/qwen3`)
const chat = chatPrompt([{ role: 'user', content: INSTR }])
const gen = q.generate(chat, { max: 12, keepTop: 5 })
q.close()
const tuned = { model: 'Qwen3-1.7B', chat, tokens: q.tok.encode(chat).length, answer: gen.text, top: gen.steps[0].top.map((t) => [t.text, t.p] as [string, number]) }
console.log('Qwen3 on the instruction:', JSON.stringify(gen.text), tuned.top)

writeFileSync(new URL('../src/data/icl.json', import.meta.url), JSON.stringify({ query: QUERY, tasks, curve, countries: QUERIES.length, head, instruction: INSTR, base, tuned }))
console.log('wrote src/data/icl.json')
