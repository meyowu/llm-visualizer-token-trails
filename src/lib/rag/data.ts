/*
 * Retrieval-augmented generation on this site's own text, exported offline by scripts/rag-export.ts: every
 * glossary definition and legend line embedded with all-MiniLM-L6-v2, their 2D projection, four questions
 * ranked against them by cosine similarity, and Qwen3-1.7B's answers with and without the top passages.
 */
import raw from '../../data/rag.json'

export interface Doc { title: string; kind: 'glossary' | 'legend'; text: string; xy: [number, number]; tokens: number }
export interface Query {
  question: string
  /** The first 16 of its 384 embedding dimensions, and its 2D projection. */
  vec: number[]
  xy: [number, number]
  /** The best 8 passages with their cosine similarity. */
  top: { doc: number; score: number }[]
  /** Answers without and with the top k passages, the prompt that held them, and its length in tokens. */
  bare: string
  grounded: string
  user: string
  tokens: number
}
export const rag = raw as unknown as {
  embedder: string
  dims: number
  model: string
  k: number
  docs: Doc[]
  sample: { doc: number; vec: number[] }
  queries: Query[]
}
/** Model output as plain lines: markdown emphasis and headings dropped, emoji and other symbols removed. */
export const plainText = (s: string) => s.replace(/\*\*/g, '').replace(/^#+ */gm, '').replace(/[^\x20-\x7e\n’“”–—…→]/g, '').replace(/\n{2,}/g, '\n').trim()
