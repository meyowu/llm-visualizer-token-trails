/*
 * A real multi-agent run, exported offline by scripts/multi-export.ts: Qwen3-1.7B as an orchestrator that hands
 * each part of a request to a worker agent with an empty context, the workers' lookups and reports, the
 * orchestrator's answer, one agent doing the same request alone, and every model call's context and output length.
 */
import raw from '../../data/multi.json'

export interface Lookup { name: string; result: string }
/** One model call: which agent, how many tokens it read, how many it wrote. */
export interface Call { agent: string; context: number; output: number }
export const multi = raw as unknown as {
  model: string
  request: string
  prompts: { orchestrator: string; worker: string; single: string }
  tools: { ask_worker: string; lookup: string }
  orchestrator: { steps: { model: string; question: string; report: string }[]; answer: string }
  workers: { task: string; lookups: Lookup[]; report: string }[]
  single: { lookups: Lookup[]; answer: string }
  calls: { multi: Call[]; single: Call[] }
}
