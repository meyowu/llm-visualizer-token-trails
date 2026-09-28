/*
 * Supervised fine-tuning measured on real models, offline: Qwen3-1.7B-Base (pretrained only) against Qwen3-1.7B (the
 * same model after Qwen's post-training). Each model's loss on every token of one chat-formatted example, what each
 * writes for the example's prompt, and each model's loss on both of those answers. Each model runs in its own child
 * process, since both do not fit in memory at once. The browser never runs a model.
 *
 *   node scripts/sft-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/qwen3/ and ~/.cache/token-trails/qwen3base/ (config.json, tokenizer.json and the weights
 * of https://huggingface.co/Qwen/Qwen3-1.7B and https://huggingface.co/Qwen/Qwen3-1.7B-Base). Writes src/data/sft.json.
 */
import { spawnSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { Qwen, chatPrompt } from './qwen-model.ts'

const HOME = `${process.env.HOME}/.cache/token-trails`
const SFT = { user: 'What does a tokenizer do? Answer in one sentence.', answer: 'A tokenizer splits text into pieces from a fixed vocabulary and turns each piece into an integer id.' }

/* ---------- child: score with one model and print JSON; argv[4], if given, holds answers to score ---------- */
if (process.argv[2] === '--score') {
  const q = await Qwen.load(process.argv[3])
  const tok = q.tok, im_end = tok.special.get('<|im_end|>')!
  /** Log-probability of each token of `resp` after `prompt` (teacher forcing). */
  const score = (prompt: number[], resp: number[]) => {
    q.reset()
    let hid = q.feed(prompt)
    return resp.map((t) => { const lp = q.logprobs(hid)[t]; hid = q.feed([t]); return lp })
  }
  const prompt = chatPrompt([{ role: 'user', content: SFT.user }]), pr = tok.encode(prompt)
  const ids = [...tok.encode(prompt + SFT.answer), im_end]
  const all = [0, ...score(ids.slice(0, 1), ids.slice(1))]
  let wrote = ''
  const others: number[] = []
  if (process.argv[4]) for (const a of JSON.parse(process.argv[4]) as string[]) { const lp = score(pr, [...tok.encode(a), im_end]); others.push(-lp.reduce((x, y) => x + y, 0) / lp.length) }
  else { q.reset(); wrote = q.generate(prompt, { max: 60 }).text }
  q.close()
  process.stdout.write(JSON.stringify({ ids, pieces: ids.map((i) => tok.piece(i)), logp: all, promptTokens: pr.length, wrote, others }))
  process.exit(0)
}

/* ---------- parent: run both models, then score each on both answers ---------- */
const run = (dir: string, answers?: string[]) => {
  const r = spawnSync(process.execPath, [new URL(import.meta.url).pathname, '--score', `${HOME}/${dir}`, ...(answers ? [JSON.stringify(answers)] : [])], { maxBuffer: 1 << 26, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] })
  if (r.status !== 0) throw new Error(`${dir} failed`)
  return JSON.parse(r.stdout)
}
const base = run('qwen3base'), tuned = run('qwen3')
const answers = [base.wrote, tuned.wrote], baseOn = run('qwen3base', answers).others, tunedOn = run('qwen3', answers).others
const r3 = (v: number) => Math.round(v * 1000) / 1000

// per-token loss (−log p) under both models; the answer's tokens are the ones SFT trains on
const n = tuned.ids.length, start = tuned.promptTokens
const tokens = tuned.pieces.map((p: string, i: number) => ({ text: p, base: i ? r3(-base.logp[i]) : null, tuned: i ? r3(-tuned.logp[i]) : null, trained: i >= start }))
const mean = (m: any, a: number, b: number) => r3(-m.logp.slice(a, b).reduce((x: number, y: number) => x + y, 0) / (b - a))
const sft = {
  model: 'Qwen3-1.7B', base: 'Qwen3-1.7B-Base', user: SFT.user, answer: SFT.answer, tokens, promptTokens: start,
  answerLoss: { base: mean(base, start, n), tuned: mean(tuned, start, n) },
  wrote: { base: base.wrote, tuned: tuned.wrote },
  /** Mean loss per token of each model on each model's own answer: [base's answer, tuned's answer]. */
  cross: { base: baseOn.map(r3), tuned: tunedOn.map(r3) },
}
console.log('answer loss', sft.answerLoss, 'cross', sft.cross)
writeFileSync(new URL('../src/data/sft.json', import.meta.url), JSON.stringify(sft))
console.log('wrote src/data/sft.json')
