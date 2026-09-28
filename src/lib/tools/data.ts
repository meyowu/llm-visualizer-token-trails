/*
 * Tool calling with Qwen3-1.7B's own format, exported offline by scripts/tools-export.ts: its answer without
 * tools, the tool schemas in its system prompt, the <tool_call> it wrote with its probabilities at the first
 * token and at the function name, the checks and the real result, its final answer, what constrained
 * decoding allows at the name, and a question it answers without a tool.
 */
import raw from '../../data/tools.json'

export type Top = [string, number][]
export const tools = raw as unknown as {
  model: string
  question: string
  /** Without tools: its whole answer, its length in tokens, and how its tokenizer splits " strawberry". */
  plain: { answer: string; tokens: number; pieces: string[] }
  tools: { name: string; json: string; tokens: number }[]
  system: { text: string; tokens: number }
  call: { text: string; first: Top; name: Top; namePiece: string; parsed: { name: string; arguments: Record<string, unknown> }; tokens: number }
  checks: [string, boolean][]
  result: string
  /** The call and the result as the chat template writes them back into the context. */
  response: string
  /** At the name: vocabulary size, how many tokens could start a declared name, their probability mass, and the renormalised top. */
  constrained: { vocab: number; allowed: number; mass: number; top: Top }
  answer: string
  context: number
  direct: { question: string; answer: string; first: Top }
}
