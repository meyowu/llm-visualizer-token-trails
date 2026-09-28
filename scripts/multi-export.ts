/*
 * A multi-agent run for real, offline, with Qwen3-1.7B (thinking off) in every role. An orchestrator with one
 * tool, ask_worker(model, question), splits the user's request into tasks; each is handed to a worker agent that starts
 * from an empty context of its own, with a lookup tool over model fact sheets (its first reply must be a tool call,
 * as with tool_choice="required"); each worker's short report is
 * returned to the orchestrator as the tool call's result, and the orchestrator answers. For comparison, a
 * single agent does the same request with the lookup tool itself. Records every model call's context and
 * output length. The browser never runs a model.
 *
 *   node scripts/multi-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/qwen3/ (see scripts/qwen-model.ts). Writes src/data/multi.json.
 */
import { writeFileSync } from 'node:fs'
import { Qwen, chatPrompt, type Message, type ToolSpec } from './qwen-model.ts'

const FACTS: Record<string, string> = {
  'gpt-2 small': 'GPT-2 small (OpenAI, 2019): 12 layers, 12 attention heads per layer, d_model 768, MLP width 3072, vocabulary 50,257 tokens, context 1,024 tokens, 124M parameters, learned position embeddings.',
  'llama 3 8b': 'LLaMA 3 8B (Meta, 2024): 32 layers, 32 query heads and 8 key-value heads per layer (grouped-query attention), d_model 4096, MLP width 14,336, vocabulary 128,256 tokens, context 8,192 tokens, 8.0B parameters, rotary position embeddings (RoPE).',
  'bert base': 'BERT base (Google, 2018): 12 layers, 12 attention heads per layer, d_model 768, MLP width 3072, vocabulary 30,522 WordPiece tokens, 110M parameters, learned position embeddings.',
}
const lookup = (a: { name?: string }) => FACTS[String(a.name ?? '').toLowerCase().trim()] ?? `No entry for "${a.name}". Entries: GPT-2 small, LLaMA 3 8B, BERT base.`
const tool = (name: string, description: string, arg: string, argDesc: string): ToolSpec => ({ type: 'function', function: { name, description, parameters: { type: 'object', properties: { [arg]: { type: 'string', description: argDesc } }, required: [arg] } } })
const ASK: ToolSpec = { type: 'function', function: { name: 'ask_worker', description: 'Hand one model to a worker agent, which looks up its fact sheet and answers the question about that model only. Call once per model.', parameters: { type: 'object', properties: { model: { type: 'string', description: 'The one model the worker researches.' }, question: { type: 'string', description: 'What to find out about that model.' } }, required: ['model', 'question'] } } }
const LOOKUP = tool('lookup', 'Return the fact sheet of a model: GPT-2 small, LLaMA 3 8B or BERT base.', 'name', 'The model name.')

const REQUEST = 'Compare GPT-2 small and LLaMA 3 8B: how many layers each has, how their attention heads are arranged, and how each encodes positions.'
const ORCH = 'You coordinate worker agents. Hand each model in the request to its own worker with ask_worker, then answer the user from their reports in a few sentences.'
const WORKER = 'You are a worker agent. Look up the model, then answer the question from its fact sheet in one sentence.'
const SINGLE = 'Use the lookup tool to find the facts you need, then answer in a few sentences.'

const q = await Qwen.load(`${process.env.HOME}/.cache/token-trails/qwen3`)
const calls: { agent: string; context: number; output: number; text: string }[] = []
/**
 * One model call from a fresh cache; returns its text and the tool calls in it. With `force`, the answer is made to
 * start with that text, as a server does for tool_choice="required" by allowing only <tool_call> as the first token.
 */
function call(agent: string, messages: Message[], tools: ToolSpec[], force = '') {
  q.reset()
  const prompt = chatPrompt(messages, { tools }), g = q.generate(prompt + force, { max: 400 })
  g.text = force + g.text
  const toolCalls = [...g.text.matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g)].map((m) => { try { return JSON.parse(m[1]) } catch { return { name: 'invalid', arguments: {} } } }) as { name: string; arguments: Record<string, string> }[]
  const content = g.text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim()
  calls.push({ agent, context: q.tok.encode(prompt).length, output: g.steps.length, text: g.text })
  console.log(`--- ${agent} (${q.tok.encode(prompt).length} in, ${g.steps.length} out)\n${g.text}`)
  return { text: g.text, content, toolCalls }
}
/** An agent loop: call, run the tool calls, feed the results back, until the model answers without calling. */
function agent(name: string, system: string, user: string, tools: ToolSpec[], run: (c: { name: string; arguments: Record<string, string> }) => string, requireTool = false) {
  const messages: Message[] = [{ role: 'system', content: system }, { role: 'user', content: user }], steps: any[] = []
  for (let i = 0; i < 4; i++) {
    const r = call(name, messages, tools, requireTool && i === 0 ? '<tool_call>\n' : '')
    if (!r.toolCalls.length) return { messages, steps, answer: r.content }
    messages.push({ role: 'assistant', content: r.content, tool_calls: r.toolCalls.map((c) => ({ name: c.name, arguments: c.arguments })) })
    for (const c of r.toolCalls) { const result = run(c); steps.push({ call: c, result }); messages.push({ role: 'tool', content: result }) }
  }
  return { messages, steps, answer: '' }
}

// the orchestrator, whose ask_worker tool runs a whole worker agent from an empty context
const workers: any[] = []
const orch = agent('orchestrator', ORCH, REQUEST, [ASK], (c) => {
  const task = `Model: ${c.arguments.model}\nQuestion: ${c.arguments.question}`, before = calls.length
  const w = agent(`worker ${workers.length + 1}`, WORKER, task, [LOOKUP], (lc) => lookup(lc.arguments), true)
  workers.push({ task, steps: w.steps, report: w.answer, calls: calls.slice(before) })
  return w.answer
})
const multiCalls = calls.splice(0)
// one agent doing everything, also required to start with a tool call
const single = agent('single agent', SINGLE, REQUEST, [LOOKUP], (c) => lookup(c.arguments), true)
const singleCalls = calls.splice(0)
q.close()

writeFileSync(new URL('../src/data/multi.json', import.meta.url), JSON.stringify({
  model: 'Qwen3-1.7B', request: REQUEST, prompts: { orchestrator: ORCH, worker: WORKER, single: SINGLE },
  tools: { ask_worker: ASK.function.description, lookup: LOOKUP.function.description },
  orchestrator: { steps: orch.steps.map((s: any) => ({ model: s.call.arguments.model, question: s.call.arguments.question, report: s.result })), answer: orch.answer },
  workers: workers.map((w) => ({ task: w.task, lookups: w.steps.map((s: any) => ({ name: s.call.arguments.name, result: s.result })), report: w.report })),
  single: { lookups: single.steps.map((s: any) => ({ name: s.call.arguments.name, result: s.result })), answer: single.answer },
  calls: { multi: multiCalls.map(({ agent, context, output }) => ({ agent, context, output })), single: singleCalls.map(({ agent, context, output }) => ({ agent, context, output })) },
}))
console.log('wrote src/data/multi.json')
