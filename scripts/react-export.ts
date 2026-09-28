/*
 * A ReAct loop run for real, offline: Qwen3-1.7B (thinking off) answers a question by writing Thought and
 * Action lines; the loop stops it at "Observation", runs the named tool (a lookup over fact sheets, or a
 * calculator) and appends the real result, until the model writes finish[...]. Also records the model's
 * probabilities when it names each action, the context length after every step, what it writes when it is
 * not stopped, and the same question without the rule to look numbers up. The browser never runs a model.
 *
 *   node scripts/react-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/qwen3/ (see scripts/qwen-model.ts). Writes src/data/react.json.
 */
import { writeFileSync } from 'node:fs'
import { Qwen, THINK_OFF, chatPrompt, type Step } from './qwen-model.ts'

const FACTS: Record<string, string> = {
  'gpt-2 small': 'GPT-2 small (OpenAI, 2019): 12 layers, 12 attention heads per layer, d_model 768, MLP width 3072, vocabulary 50,257 tokens, context 1,024 tokens, 124M parameters.',
  'llama 3 8b': 'LLaMA 3 8B (Meta, 2024): 32 layers, 32 query heads and 8 key-value heads per layer, d_model 4096, MLP width 14,336, vocabulary 128,256 tokens, context 8,192 tokens.',
  'bert base': 'BERT base (Google, 2018): 12 layers, 12 attention heads per layer, d_model 768, MLP width 3072, vocabulary 30,522 WordPiece tokens, 110M parameters.',
}
const TOOLS = [
  { name: 'lookup', arg: 'name', desc: 'returns the fact sheet of a model, e.g. lookup[BERT base]' },
  { name: 'calculate', arg: 'expression', desc: 'evaluates arithmetic, e.g. calculate[3 * (4 + 5)]' },
  { name: 'finish', arg: 'answer', desc: 'gives the final answer' },
]
const lookup = (a: string) => FACTS[a.toLowerCase().trim()] ?? `No entry for "${a}". Entries: ${Object.keys(FACTS).join(', ')}.`
const calculate = (e: string) => (/^[\d\s+\-*/().,]+$/.test(e) ? String(Function(`return (${e.replace(/,/g, '')})`)()) : 'Error: only numbers and + - * / ( ).')

const INSTRUCTIONS = 'Answer the question by interleaving Thought, Action and Observation steps.\nThought: reason about what you know and what you still need.\nAction: exactly one of'
const TOOL_LINES = TOOLS.map((t) => `\n  ${t.name}[${t.arg}]: ${t.desc}`).join('')
const RULE = '\nAfter each Action, wait for the Observation. Look up every number you use; do not answer from memory.'
const EXAMPLE = `\n\nExample\nQuestion: How many numbers are in the token embedding matrix of BERT base?
Thought 1: The matrix has one row per vocabulary token and one column per model dimension. I need both sizes.
Action 1: lookup[BERT base]
Observation 1: ${FACTS['bert base']}
Thought 2: The vocabulary has 30,522 tokens and d_model is 768, so I multiply them.
Action 2: calculate[30522 * 768]
Observation 2: 23440896
Thought 3: The matrix holds 23,440,896 numbers.
Action 3: finish[23,440,896]`
const QUESTION = 'How many more layers does LLaMA 3 8B have than GPT-2 small?'

const q = await Qwen.load(`${process.env.HOME}/.cache/token-trails/qwen3`)
const n = (s: string) => q.tok.encode(s).length
const r3 = (v: number) => Math.round(v * 1000) / 1000

/** The model's top choices where it names the action ("Action k:" just written). */
function actionChoice(steps: Step[]) {
  let text = ''
  for (const s of steps) {
    if (/Action \d+:\s*$/.test(text)) return s.top.slice(0, 4).map((t) => [t.text, r3(t.p)] as [string, number])
    text += s.text
  }
  return []
}

function run(system: string, o: { stop: boolean; max?: number }) {
  q.reset()
  const prompt = chatPrompt([{ role: 'system', content: system }, { role: 'user', content: `Question: ${QUESTION}` }], { generation: false }) + '<|im_start|>assistant\n' + THINK_OFF + 'Thought 1:'
  const steps: any[] = []
  let feed = prompt
  for (let i = 1; i <= 8; i++) {
    const g = q.generate(feed, { max: o.max ?? 90, stops: o.stop ? ['\nObservation'] : [], keepTop: 5 })
    const before = q.ids.length
    const m = g.text.match(/Action \d+: (\w+)\[(.*?)\]/)
    const step: any = { text: (i === 1 ? 'Thought 1:' : `Thought ${i}:`) + g.text, gen: g.steps.length, choice: actionChoice(g.steps), ctx: before }
    steps.push(step)
    console.log(step.text)
    if (!o.stop || !m) break
    step.action = { name: m[1], arg: m[2] }
    if (m[1] === 'finish') break
    step.obs = m[1] === 'lookup' ? lookup(m[2]) : m[1] === 'calculate' ? calculate(m[2]) : 'Unknown action.'
    feed = `\nObservation ${i}: ${step.obs}\nThought ${i + 1}:`
    step.obsTokens = n(feed)
    console.log(`Observation ${i}: ${step.obs}`)
  }
  return { promptTokens: n(prompt), steps, total: q.ids.length }
}

const SYSTEM = INSTRUCTIONS + TOOL_LINES + RULE + EXAMPLE
const loop = run(SYSTEM, { stop: true })
console.log('--- not stopped at Observation')
const unstopped = run(SYSTEM, { stop: false, max: 160 })
console.log('--- without the rule to look numbers up')
const noRule = run(INSTRUCTIONS + TOOL_LINES + '\nAfter each Action, wait for the Observation.' + EXAMPLE, { stop: true })
q.close()

const sections = [
  ['instructions', INSTRUCTIONS], ['tools', TOOL_LINES], ['rule', RULE], ['worked example', EXAMPLE], ['question', `Question: ${QUESTION}`],
].map(([name, text]) => ({ name, text, tokens: n(text) }))
writeFileSync(new URL('../src/data/react.json', import.meta.url), JSON.stringify({
  model: 'Qwen3-1.7B', question: QUESTION, tools: TOOLS, facts: FACTS, sections, loop, unstopped: unstopped.steps[0], noRule: noRule.steps,
}))
console.log('wrote src/data/react.json')
