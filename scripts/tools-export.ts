/*
 * Tool calling run for real, offline, with Qwen3-1.7B (thinking off) and its own tool-call format: the model
 * answers a letter-counting question without tools, then with two tools described as JSON schemas in the
 * system prompt it writes a <tool_call>; the call is parsed, validated and run, the result goes back in a
 * <tool_response>, and the model answers. Also records its probabilities at the call's first token and at the
 * function name, what constrained decoding would allow there, and a question it answers without a tool.
 * The browser never runs a model.
 *
 *   node scripts/tools-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/qwen3/ (see scripts/qwen-model.ts). Writes src/data/tools.json.
 */
import { writeFileSync } from 'node:fs'
import { Qwen, chatPrompt, pyJson, type Message, type ToolSpec } from './qwen-model.ts'

const TOOLS: ToolSpec[] = [
  { type: 'function', function: { name: 'count_letter', description: 'Count how many times a letter appears in a word.', parameters: { type: 'object', properties: { word: { type: 'string', description: 'The word to search.' }, letter: { type: 'string', description: 'A single letter.' } }, required: ['word', 'letter'] } } },
  { type: 'function', function: { name: 'tokenize', description: 'Split text into GPT-2 tokens.', parameters: { type: 'object', properties: { text: { type: 'string', description: 'The text to split.' } }, required: ['text'] } } },
]
const IMPL: Record<string, (a: any) => unknown> = {
  count_letter: (a) => [...String(a.word)].filter((c) => c.toLowerCase() === String(a.letter).toLowerCase()).length,
  tokenize: (a) => String(a.text),
}
const QUESTION = 'How many times does the letter "r" appear in "strawberry"?'
const DIRECT = 'What is the capital of France?'

const q = await Qwen.load(`${process.env.HOME}/.cache/token-trails/qwen3`)
const r3 = (v: number) => Math.round(v * 1000) / 1000
const top = (t: { text: string; p: number }[], k = 4) => t.slice(0, k).map((x) => [x.text, r3(x.p)] as [string, number])

// 1. no tools: the model answers from its tokens
q.reset()
const plain = q.generate(chatPrompt([{ role: 'user', content: QUESTION }]), { max: 300 })
console.log('no tools:', plain.text)
const pieces = q.tok.encode(' strawberry').map((i) => q.tok.piece(i))

// 2. with tools: the call
q.reset()
const prompt = chatPrompt([{ role: 'user', content: QUESTION }], { tools: TOOLS })
const base = q.tok.encode(prompt).length
const call = q.generate(prompt, { max: 60, keepTop: 5 })
console.log('call:', call.text)
// where the function name starts: the token after '{"name": "'
let text = '', nameAt = -1
call.steps.forEach((s, i) => { if (nameAt < 0 && /"name": "$/.test(text)) nameAt = i; text += s.text })
const inner = call.text.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/)![1], parsed = JSON.parse(inner)

// 3. validate against the schema and run
const spec = TOOLS.find((t) => t.function.name === parsed.name)!.function.parameters as { properties: Record<string, { type: string }>; required: string[] }
const checks = [
  ['name is a declared tool', !!TOOLS.find((t) => t.function.name === parsed.name)],
  ['arguments is an object', typeof parsed.arguments === 'object'],
  ...spec.required.map((k) => [`has “${k}”`, k in parsed.arguments]),
  ...Object.entries(parsed.arguments).map(([k, v]) => [`${k} is a ${spec.properties[k]?.type}`, typeof v === spec.properties[k]?.type]),
] as [string, boolean][]
const result = String(IMPL[parsed.name](parsed.arguments))
console.log('checks', checks, 'result', result)

// constrained decoding at the name: which of the vocabulary's tokens could start a declared name
q.truncate(base + nameAt - 1)
const lp = q.logprobs(q.feed([call.steps[nameAt - 1].id]))
const names = TOOLS.map((t) => t.function.name)
let allowed = 0, mass = 0
const allowedTop: [string, number][] = []
for (let i = 0; i < q.tok.inv.length; i++) {
  const s = q.tok.piece(i)
  if (!s || q.tok.special.has(s)) continue
  if (names.some((nm) => nm.startsWith(s))) { allowed++; mass += Math.exp(lp[i]); allowedTop.push([s, Math.exp(lp[i])]) }
}
allowedTop.sort((a, b) => b[1] - a[1])
const constrained = { vocab: q.tok.inv.filter(Boolean).length, allowed, mass: r3(mass), top: allowedTop.slice(0, 5).map(([s, p]) => [s, r3(p / mass)] as [string, number]) }
console.log('constrained', constrained)

// 4. the tool's result goes back, and the model answers
q.reset()
const msgs: Message[] = [{ role: 'user', content: QUESTION }, { role: 'assistant', content: '', tool_calls: [{ name: parsed.name, arguments: parsed.arguments }] }, { role: 'tool', content: result }]
const second = chatPrompt(msgs, { tools: TOOLS })
const answer = q.generate(second, { max: 60 })
console.log('answer:', answer.text)

// 5. a question it needs no tool for
q.reset()
const direct = q.generate(chatPrompt([{ role: 'user', content: DIRECT }], { tools: TOOLS }), { max: 30, keepTop: 5 })
console.log('direct:', direct.text, direct.steps[0].top)
q.close()

const system = chatPrompt([], { tools: TOOLS, generation: false })
writeFileSync(new URL('../src/data/tools.json', import.meta.url), JSON.stringify({
  model: 'Qwen3-1.7B', question: QUESTION,
  plain: { answer: plain.text, tokens: plain.steps.length, pieces },
  tools: TOOLS.map((t) => ({ name: t.function.name, json: pyJson(t), tokens: q.tok.encode(pyJson(t)).length })),
  system: { text: system, tokens: q.tok.encode(system).length },
  call: { text: call.text, first: top(call.steps[0].top), name: top(call.steps[nameAt].top), namePiece: call.steps[nameAt].text, parsed, tokens: call.steps.length },
  checks, result, response: second.slice(chatPrompt([msgs[0]], { tools: TOOLS, generation: false }).length).replace(/<\|im_start\|>assistant\n<think>\n\n<\/think>\n\n$/, ''), constrained,
  answer: answer.text, context: q.tok.encode(second).length + answer.steps.length,
  direct: { question: DIRECT, answer: direct.text, first: top(direct.steps[0].top) },
}))
console.log('wrote src/data/tools.json')
