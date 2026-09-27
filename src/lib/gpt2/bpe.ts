/*
 * GPT-2's byte-level BPE, the same algorithm as OpenAI's encoder.py.
 *
 * Token ids follow from merges.txt alone: ids 0–255 are the 256 byte symbols in GPT-2's
 * bytes_to_unicode order, id 256 + r is the token created by merge rule r, and 50256 is
 * <|endoftext|>. This module has no imports so the Node export script can load it directly.
 */

/** GPT-2's pre-tokenizer: splits text into words, numbers, punctuation and whitespace runs. */
export const PRETOKENIZE_SOURCE = String.raw`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
export const pretokenize = (text: string): string[] => text.match(new RegExp(PRETOKENIZE_SOURCE, 'gu')) ?? []

export const EOT_ID = 50256

/** GPT-2 maps every byte to a printable character so merge rules never see raw control bytes. */
function bytesToUnicode(): { byteChar: string[]; order: number[] } {
  const bs: number[] = []
  for (let b = 33; b <= 126; b++) bs.push(b)
  for (let b = 161; b <= 172; b++) bs.push(b)
  for (let b = 174; b <= 255; b++) bs.push(b)
  const cs = bs.slice()
  let n = 0
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++ }
  const byteChar: string[] = new Array(256)
  bs.forEach((b, i) => { byteChar[b] = String.fromCharCode(cs[i]) })
  return { byteChar, order: bs }
}

let charByte: Map<string, number> | null = null
/** The text a vocabulary symbol stands for ("Ġcat" → " cat"); symbols that are only part of a UTF-8 character stay as they are. */
export function symbolText(symbol: string): string {
  if (symbol === '<|endoftext|>') return symbol
  if (!charByte) { const { byteChar } = bytesToUnicode(); charByte = new Map(byteChar.map((c, b) => [c, b])) }
  const bytes: number[] = []
  for (const ch of symbol) { const b = charByte.get(ch); if (b === undefined) return symbol; bytes.push(b) }
  try { return new TextDecoder('utf-8', { fatal: true }).decode(new Uint8Array(bytes)) } catch { return symbol }
}

export interface MergeStep {
  /** The rule that fired: left + right → merged, with its rank. */
  left: string
  right: string
  rank: number
  /** Symbols of the word before this step. */
  before: string[]
  after: string[]
}

export interface PieceTrace {
  /** The pre-tokenized piece of text, e.g. " cat". */
  text: string
  bytes: number[]
  /** The piece as byte symbols (space shown as Ġ). */
  symbols: string[]
  steps: MergeStep[]
  tokens: { text: string; id: number }[]
}

export class Gpt2Bpe {
  private ranks = new Map<string, number>()
  private mergedRank = new Map<string, number>()
  private byteChar: string[]
  private charByte = new Map<string, number>()
  private byteId = new Map<string, number>()
  private cache = new Map<string, PieceTrace>()
  readonly merges: [string, string][] = []

  constructor(mergesText: string) {
    const { byteChar, order } = bytesToUnicode()
    this.byteChar = byteChar
    order.forEach((b, i) => { this.byteId.set(byteChar[b], i); this.charByte.set(byteChar[b], b) })
    for (const line of mergesText.split('\n')) {
      if (!line || line.startsWith('#version')) continue
      const sp = line.indexOf(' ')
      const a = line.slice(0, sp), b = line.slice(sp + 1).trimEnd()
      const r = this.merges.length
      this.merges.push([a, b])
      this.ranks.set(a + ' ' + b, r)
      if (!this.mergedRank.has(a + b)) this.mergedRank.set(a + b, r)
    }
  }

  /** Id of a vocabulary symbol (byte symbol or merged token). */
  idOf(symbol: string): number {
    const b = this.byteId.get(symbol)
    if (b !== undefined) return b
    const r = this.mergedRank.get(symbol)
    if (r === undefined) throw new Error(`not in vocabulary: ${symbol}`)
    return 256 + r
  }
  rankOf(left: string, right: string): number | undefined {
    return this.ranks.get(left + ' ' + right)
  }
  /** Symbol for an id, the inverse of idOf. */
  symbolOf(id: number): string {
    if (id === EOT_ID) return '<|endoftext|>'
    if (id < 256) return [...this.byteId.entries()].find(([, v]) => v === id)![0]
    const [a, b] = this.merges[id - 256]
    return a + b
  }
  /** Decode ids back to text (symbols → bytes → UTF-8). */
  decode(ids: number[]): string {
    const bytes: number[] = []
    for (const id of ids) for (const ch of this.symbolOf(id)) bytes.push(this.charByte.get(ch) ?? 63)
    return new TextDecoder().decode(new Uint8Array(bytes))
  }

  /** Byte-level BPE on one pre-tokenized piece, recording every merge that fires. */
  tracePiece(text: string): PieceTrace {
    const hit = this.cache.get(text)
    if (hit) return hit
    const bytes = [...new TextEncoder().encode(text)]
    let word = bytes.map((b) => this.byteChar[b])
    const symbols = word.slice()
    const steps: MergeStep[] = []
    while (word.length > 1) {
      let best = Infinity, bi = -1
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(word[i] + ' ' + word[i + 1])
        if (r !== undefined && r < best) { best = r; bi = i }
      }
      if (bi < 0) break
      const left = word[bi], right = word[bi + 1], out: string[] = []
      for (let i = 0; i < word.length; i++) {
        if (i < word.length - 1 && word[i] === left && word[i + 1] === right) { out.push(left + right); i++ }
        else out.push(word[i])
      }
      steps.push({ left, right, rank: best, before: word, after: out })
      word = out
    }
    const trace: PieceTrace = { text, bytes, symbols, steps, tokens: word.map((s) => ({ text: s, id: this.idOf(s) })) }
    this.cache.set(text, trace)
    return trace
  }

  trace(text: string): PieceTrace[] {
    return pretokenize(text).map((p) => this.tracePiece(p))
  }
  encode(text: string): number[] {
    return this.trace(text).flatMap((p) => p.tokens.map((t) => t.id))
  }
}
