import { F, ctx, drawChip } from '../../core/draw'
import { TERMS } from '../../core/glossary'
import { C, rgba } from '../../core/theme'
import { clamp, eout, lerp } from '../../core/util'
import { Gpt2Bpe, byteSymbols, pretokenize } from '../../lib/gpt2/bpe'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * Learning a tokenizer: byte-pair encoding trained for real, in the page, on this site's own glossary. Start from
 * bytes, count every adjacent pair, merge the most frequent, repeat. Then GPT-2's real first merges (its merges.txt,
 * learned on 40 GB of web text) next to ours.
 */

const PHASES = [
  { id: 'bytes', name: 'Start from bytes', short: 'Bytes', dur: 8 },
  { id: 'count', name: 'Count every adjacent pair', short: 'Count', dur: 9 },
  { id: 'merge', name: 'Merge the most frequent, repeat', short: 'Merge', dur: 13 },
  { id: 'vocab', name: 'Vocabulary against length', short: 'Trade-off', dur: 9 },
  { id: 'gpt2', name: 'GPT-2’s merges', short: 'GPT-2', dur: 10 },
]

/* ---------- training, done once when the page loads (deterministic) ---------- */
const CORPUS = TERMS.map((t) => t.def).join(' ')
const M = 300
type Merge = { a: string; b: string; count: number }
function train() {
  const counts = new Map<string, number>()
  for (const w of pretokenize(CORPUS)) counts.set(w, (counts.get(w) ?? 0) + 1)
  const words = [...counts].map(([w, n]) => ({ syms: byteSymbols(w), n }))
  const nWords = words.reduce((a, w) => a + w.n, 0)
  const merges: Merge[] = [], perWord: number[] = [], firstCounts: [string, number][] = []
  const total = () => words.reduce((a, w) => a + w.syms.length * w.n, 0)
  perWord.push(total() / nWords)
  for (let m = 0; m < M; m++) {
    const pc = new Map<string, number>()
    for (const w of words) for (let i = 0; i + 1 < w.syms.length; i++) { const k = w.syms[i] + '\u0000' + w.syms[i + 1]; pc.set(k, (pc.get(k) ?? 0) + w.n) }
    let best = '', bc = -1
    for (const [k, c] of pc) if (c > bc || (c === bc && k < best)) { best = k; bc = c }
    if (m === 0) firstCounts.push(...[...pc].sort((x, y) => y[1] - x[1]).slice(0, 10).map(([k, c]) => [k, c] as [string, number]))
    if (bc < 2) break
    const [a, b] = best.split('\u0000')
    merges.push({ a, b, count: bc })
    for (const w of words) {
      const out: string[] = []
      for (let i = 0; i < w.syms.length; i++) { if (i + 1 < w.syms.length && w.syms[i] === a && w.syms[i + 1] === b) { out.push(a + b); i++ } else out.push(w.syms[i]) }
      w.syms = out
    }
    perWord.push(total() / nWords)
  }
  return { merges, perWord, firstCounts, nWords, bytes: new TextEncoder().encode(CORPUS).length, unique: counts.size }
}
const TR = train()
/** A word segmented by the first n learned merges, applied in order. */
function segment(text: string, n: number) {
  let syms = byteSymbols(text)
  for (const m of TR.merges.slice(0, n)) {
    const out: string[] = []
    for (let i = 0; i < syms.length; i++) { if (i + 1 < syms.length && syms[i] === m.a && syms[i + 1] === m.b) { out.push(m.a + m.b); i++ } else out.push(syms[i]) }
    syms = out
  }
  return syms
}
const show = (s: string) => s.replace(/\u0000/, ' + ')

let gpt2: Gpt2Bpe | null = null
let loading: Promise<void> | null = null
const loadGpt2 = () => (loading ??= import('../../lib/gpt2/merges.txt?raw').then((m) => { gpt2 = new Gpt2Bpe(m.default) }))

const COMPARE: Record<string, [string, string]> = {
  bytes: ['text to bytes, in the tokenizer', 'anatomy/tokenizer?phase=bytes'],
  count: ['the pre-split into words', 'anatomy/tokenizer?phase=split'],
  merge: ['applying merges by rank', 'anatomy/tokenizer?phase=merge'],
  vocab: ['the embedding matrix, one row per token', 'anatomy/embedding'],
  gpt2: ['GPT-2’s tokenizer at work', 'anatomy/tokenizer?phase=ids'],
}

const CAPS: Record<string, [string, string]> = {
  bytes: [`A tokenizer is learned from text before the model is. Here the text is this site’s glossary: ${TR.bytes.toLocaleString('en-US')} bytes in ${TR.nWords.toLocaleString('en-US')} pre-split words. Training starts with one symbol per byte: 256 in all, and nothing unknown.`, `${TR.bytes.toLocaleString('en-US')} bytes · vocabulary 256`],
  count: [`Count how often each pair of neighbouring symbols occurs, inside words only and weighted by how often each word appears. The most frequent pair here is ${show(TR.firstCounts[0][0])}, ${TR.firstCounts[0][1]} times.`, 'count pairs within words'],
  merge: [`Replace that pair everywhere with one new symbol, add it to the vocabulary, and count again. Each merge is a rule, kept in order. Watch the sample words fall from bytes into larger pieces as ${TR.merges.length} merges are learned.`, `${TR.merges.length} merges · vocabulary ${256 + TR.merges.length}`],
  vocab: [`More merges mean fewer tokens per word, so a context holds more text, but every new token needs a row in the embedding matrix and in the output layer. GPT-2 stopped at 50,000 merges; LLaMA 3 has about 128,000 tokens.`, 'fewer tokens per word vs a bigger vocabulary'],
  gpt2: [`GPT-2’s first merges, from its real merges.txt (learned on 40 GB of web text), start much like ours: a space joined to a common letter, “h e”, “i n”. Encoding a new word replays the merges in this order.`, 'GPT-2: 50,000 merges, 50,257 tokens'],
}

export function mountBpeTrain(root: HTMLElement, nav: Nav): () => void {
  loadGpt2()
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'BPE is trained for real, in your browser, on this site’s glossary; GPT-2’s merges are its real ones.',
      eyebrow: 'Training',
      title: 'Learning the tokenizer',
      subtitle: 'byte-pair encoding: count pairs, merge, repeat',
      specs: [
        { label: 'corpus', value: 'this site’s glossary', real: `${TR.bytes.toLocaleString('en-US')} bytes`, realLabel: '' },
        { label: 'merges', value: String(TR.merges.length), real: '50,000', realLabel: 'GPT-2' },
        { label: 'start', value: '256 byte symbols' },
      ],
    },
    size: [1040, 480],
    aria: 'Learning a byte-pair-encoding tokenizer on this site’s glossary: starting from bytes, counting adjacent pairs, merging the most frequent pair again and again; tokens per word fall as the vocabulary grows; GPT-2’s first real merges look much the same.',
    phases: PHASES, learn: 'bpetrain', tokens: words(['a', 'b']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['merge', 12],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption } = k
  const pad = 36, top = 56
  /** A segmentation as chips; returns the x after the last. */
  function chips(syms: string[], x: number, y: number, a: number, hl?: (s: string) => boolean) {
    syms.forEach((s, i) => { x += drawChip(x, y, { text: s, c: i }, a, 22, hl?.(s) ?? false) + 4 })
    return x
  }
  const SAMPLES = [' tokenizer', ' merges', ' frequent', ' pairs']

  /* ---------- 1: bytes ---------- */
  function sceneBytes(p: number) {
    const y0 = top + 40
    title('the corpus: this site’s glossary, first lines', pad, y0 - 16, 1)
    const lines = CORPUS.slice(0, 300).match(/.{1,110}(\s|$)/g) ?? []
    ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink2, 1)
    lines.slice(0, 3).forEach((l, i) => ctx.fillText(l.trim() + (i === 2 ? ' …' : ''), pad, y0 + 8 + i * 18))
    const a = eout(clamp((p - 0.2) / 0.15)), y1 = y0 + 110
    title('each word as bytes (Ġ is the byte for a space)', pad, y1 - 16, a)
    SAMPLES.forEach((w, i) => { const x = chips(segment(w, 0), pad, y1 + 14 + i * 36, a); caption(`${segment(w, 0).length} symbols`, x + 10, y1 + 18 + i * 36, a, C.mute, 'left') })
    const ua = eout(clamp((p - 0.55) / 0.12))
    const odd = segment('’', 0)
    caption(`even “’” works: it is ${odd.length} bytes, ${odd.join(' ')}`, pad + 560, y1 + 18, ua, C.ink2, 'left')
    mk.formula = { segs: [['vocabulary', C.ink2], [' = ', C.mute], ['256 bytes', C.ink], ['  + one symbol per merge', C.ink2]], note: 'Starting from bytes (not characters) means any text in any language or format can be encoded; merges only make it shorter.' }
  }

  /* ---------- 2: count ---------- */
  function sceneCount(p: number) {
    const x0 = pad + 120, y0 = top + 50, bw = 520, mx = TR.firstCounts[0][1]
    title('pair counts over the whole corpus, before any merge', x0 - 120, y0 - 20, 1)
    TR.firstCounts.forEach(([kk, c], i) => {
      const a = eout(clamp((p - 0.05 - i * 0.05) / 0.1)), y = y0 + 10 + i * 32, [s1, s2] = kk.split('\u0000')
      let x = x0 - 110
      x += drawChip(x, y, { text: s1, c: 0 }, a, 22, i === 0) + 4
      drawChip(x, y, { text: s2, c: 1 }, a, 22, i === 0)
      ctx.fillStyle = rgba(C.ink, (i === 0 ? 0.85 : 0.4) * a); ctx.fillRect(x0, y - 5, (c / mx) * bw, 10)
      ctx.font = F.mono(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(i === 0 ? C.ink : C.mute, a); ctx.fillText(c.toLocaleString('en-US'), x0 + (c / mx) * bw + 8, y)
    })
    mk.formula = { segs: [['count(a, b)', C.ink], [' = ', C.mute], ['Σ over words  n(word) × occurrences of a b in it', C.ink2]], note: 'Pairs never cross a pre-split boundary, so a merge cannot glue a word to the space or punctuation after it.' }
  }

  /* ---------- 3: merge, repeat ---------- */
  function sceneMerge(p: number) {
    const { W } = stage, n = Math.round(Math.pow(clamp((p - 0.04) / 0.8), 2) * TR.merges.length), y0 = top + 40
    title(`after ${n} merge${n === 1 ? '' : 's'} · vocabulary ${256 + n}`, pad, y0 - 16, 1)
    SAMPLES.forEach((w, i) => { const s = segment(w, n), x = chips(s, pad, y0 + 14 + i * 38, 1); caption(`${s.length}`, x + 8, y0 + 18 + i * 38, 1, C.mute, 'left') })
    // the latest merges, newest first
    const xr = W - pad - 330, list = TR.merges.slice(0, n).map((m, i) => ({ ...m, r: i + 1 })).slice(-9).reverse()
    title('latest merges · rank, pair, count', xr, y0 - 16, 1)
    list.forEach((m, i) => {
      const y = y0 + 14 + i * 30, a = i === 0 ? 1 : 0.75
      ctx.font = F.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(String(m.r), xr + 30, y)
      let x = xr + 40
      x += drawChip(x, y, { text: m.a, c: 0 }, a, 20) + 4
      x += drawChip(x, y, { text: m.b, c: 1 }, a, 20) + 10
      ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(`→ ${m.a + m.b}  ×${m.count}`, x, y)
    })
    mk.formula = { segs: [['merge', C.ink2], [' = ', C.mute], ['argmax count(a, b)', C.ink], ['  →  add ab, replace a b by ab everywhere', C.ink2]], note: 'Frequent whole words (“the”, “token”) end up as single symbols; rare words stay in pieces. Nothing about meaning is used: only counts.' }
  }

  /* ---------- 4: the trade-off ---------- */
  function sceneVocab(p: number) {
    const { W, H } = stage, x0 = pad + 80, x1 = W / 2 + 60, y0 = top + 40, y1 = H - 90
    const X = (m: number) => lerp(x0, x1, m / TR.merges.length), maxY = TR.perWord[0], Y = (v: number) => lerp(y1, y0, (v - 1) / (maxY - 1))
    title('symbols per word on this corpus, by merges learned', x0 - 40, y0 - 18, 1)
    ctx.strokeStyle = rgba(C.faint, 1); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke()
    const n = Math.floor(clamp((p - 0.05) / 0.6) * TR.merges.length)
    ctx.strokeStyle = rgba(C.tok[0], 1); ctx.lineWidth = 2; ctx.beginPath()
    for (let m = 0; m <= n; m++) { if (m) ctx.lineTo(X(m), Y(TR.perWord[m])); else ctx.moveTo(X(m), Y(TR.perWord[m])) }
    ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'
    ctx.fillText(TR.perWord[0].toFixed(1), x0 - 6, Y(TR.perWord[0])); ctx.fillText('1', x0 - 6, Y(1))
    ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('0', x0, y1 + 6); ctx.fillText(`${TR.merges.length} merges`, x1, y1 + 6)
    caption(`${TR.perWord[n].toFixed(2)} per word after ${n}`, X(n) + 8, Y(TR.perWord[n]) - 8, 1, C.ink, 'left')
    // the other side of the trade: embedding rows
    const ra = eout(clamp((p - 0.6) / 0.12)), xr = x1 + 90
    title('what a bigger vocabulary costs', xr, y0 - 18, ra)
    ;[['GPT-2', '50,257 tokens × 768', '38.6M weights'], ['LLaMA 3 8B', '128,256 × 4,096', '525M, in and out each'], ['Qwen3', '151,936 × 2,048 (1.7B)', '311M, tied']].forEach(([nm, sh, cost], i) => {
      ctx.font = F.mono(12, 600); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, ra); ctx.fillText(nm, xr, y0 + 18 + i * 60)
      ctx.font = F.mono(11); ctx.fillStyle = rgba(C.ink2, ra); ctx.fillText(sh, xr, y0 + 36 + i * 60); ctx.fillStyle = rgba(C.mute, ra); ctx.fillText(cost, xr, y0 + 52 + i * 60)
    })
    mk.formula = { segs: [['tokens per word', C.ink2], [` ${TR.perWord[0].toFixed(1)} → ${TR.perWord[TR.merges.length].toFixed(2)}`, C.ink], ['   with ', C.mute], [`${TR.merges.length} merges`, C.ink2]], note: 'On its own training text a tokenizer does best; on other languages or code it falls back to smaller pieces, so the same sentence can cost several times more tokens.' }
  }

  /* ---------- 5: GPT-2 ---------- */
  function sceneGpt2(p: number) {
    const { W } = stage, y0 = top + 40, colW = (W - 2 * pad - 60) / 2
    const list = (items: [string, string][], x: number, t: string, a: number) => {
      title(t, x, y0 - 16, a)
      items.slice(0, 12).forEach(([s1, s2], i) => {
        const y = y0 + 12 + i * 27, ia = a * eout(clamp((p - 0.05 - i * 0.03) / 0.1))
        ctx.font = F.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, ia); ctx.fillText(String(i + 1), x + 22, y)
        let xx = x + 32
        xx += drawChip(xx, y, { text: s1, c: 0 }, ia, 20) + 4
        drawChip(xx, y, { text: s2, c: 1 }, ia, 20)
      })
    }
    list(TR.merges.map((m) => [m.a, m.b] as [string, string]), pad, 'ours · this site’s glossary', 1)
    if (gpt2) {
      list(gpt2.merges as [string, string][], pad + colW + 60, 'GPT-2 · 40 GB of web text', 1)
      const sa = eout(clamp((p - 0.6) / 0.12))
      const g = gpt2.encode(' tokenizer').map((id) => gpt2!.symbolOf(id)), o = segment(' tokenizer', TR.merges.length)
      caption(`“ tokenizer”: ours ${o.join(' · ')}   GPT-2 ${g.join(' · ')}`, pad, y0 + 12 * 27 + 26, sa, C.ink2, 'left')
    } else caption('loading GPT-2’s merges.txt …', pad + colW + 60, y0 + 12, 1, C.mute, 'left')
    mk.formula = { segs: [['GPT-2', C.ink2], ['  50,000 merges · ', C.mute], ['vocabulary 256 + 50,000 + <|endoftext|> = 50,257', C.ink]], note: 'The merge list is the tokenizer. Tokenizing new text replays it in rank order within each word, which is what the Tokenizer page animates.' }
  }

  return { bytes: sceneBytes, count: sceneCount, merge: sceneMerge, vocab: sceneVocab, gpt2: sceneGpt2 }
}
