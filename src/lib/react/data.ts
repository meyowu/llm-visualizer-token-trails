/*
 * A real ReAct loop, exported offline by scripts/react-export.ts: Qwen3-1.7B's Thought and Action lines,
 * the tool results the loop appended, its probabilities when naming each action, the context length after
 * every step, what it writes when it is not stopped, and a run without the rule to look numbers up.
 */
import raw from '../../data/react.json'

export interface ReactStep {
  /** What the model wrote in this step: "Thought k: … Action k: name[arg]". */
  text: string
  /** Tokens it generated. */
  gen: number
  /** Its top tokens where it named the action. */
  choice: [string, number][]
  /** Tokens in the context when it stopped (before the observation). */
  ctx: number
  action?: { name: string; arg: string }
  /** The tool's real result, and the tokens appended with it ("Observation k: … Thought k+1:"). */
  obs?: string
  obsTokens?: number
}
export const react = raw as unknown as {
  model: string
  question: string
  tools: { name: string; arg: string; desc: string }[]
  facts: Record<string, string>
  sections: { name: string; text: string; tokens: number }[]
  loop: { promptTokens: number; steps: ReactStep[]; total: number }
  unstopped: ReactStep
  noRule: ReactStep[]
}
