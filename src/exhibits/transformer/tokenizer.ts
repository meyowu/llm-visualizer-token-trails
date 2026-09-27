import { F, rr, serifAt, spaced, tokLabel, useCtx } from '../../core/draw'
import { createFrame, toggle } from '../../core/frame'
import { Player } from '../../core/player'
import { pref } from '../../core/prefs'
import { Stage, poke, runLoop } from '../../core/stage'
import { C, mixc, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import { Gpt2Bpe, PRETOKENIZE_SOURCE, type PieceTrace } from '../../lib/gpt2/bpe'
import type { Nav } from '../registry'

/*
 * GPT-2's real byte-level BPE on any text: pre-split with the regex, turn each piece into bytes,
 * then inside each piece repeatedly look up the rank of every adjacent pair and merge the
 * lowest-ranked one, until no pair is in the table. A merged token's id is 256 + its merge rank.
 */

const PHASES = [
  { id: 'split', name: 'Pre-tokenize', short: 'Split', dur: 3.5 },
  { id: 'bytes', name: 'Bytes', dur: 3.5 },
  { id: 'merge', name: 'BPE merges', short: 'Merge', dur: 14 },
  { id: 'ids', name: 'Vocabulary ids', short: 'ids', dur: 6 },
]
const EXAMPLES = ['The cat sat on the', 'the cat saw the cat', "Tokenization isn't magic", 'café 你好']
const MAXLEN = 40
/** Bytes drawn at most (three rows of cells); longer text is cut at a piece boundary. */
const MAXBYTES = 48
const VOCAB = 50257
const hex2 = (b: number) => b.toString(16).toUpperCase().padStart(2, '0')

/** One merge that fires: in piece `piece`, at its `step`-th merge. */
interface Fired { piece: number; step: number }

let merges: Promise<Gpt2Bpe> | null = null
/** GPT-2's merges.txt (456 KB) is loaded on first use of this page. */
const loadBpe = () => (merges ??= import('../../lib/gpt2/merges.txt?raw').then((m) => new Gpt2Bpe(m.default)))

export function mountTokenizer(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const frame = createFrame(root, {
    formula: true,
    eyebrow: 'Anatomy · Tokenizer',
    title: 'Tokenizer',
    subtitle: 'byte-level BPE · GPT-2 · real merges',
    back: { label: 'Forward pass', onClick: () => nav('anatomy') },
    specs: [
      { label: 'base symbols', value: '256 bytes' },
      { label: 'merges', value: '50,000' },
      { label: 'vocab', value: '50,257' },
      { label: 'this text', value: '…' },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 470, 'Byte-level BPE on your text: the text is split into pieces, each piece becomes bytes, and inside each piece the adjacent pair with the lowest merge rank is fused, again and again, until each piece is a few vocabulary tokens with ids.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  const prog = (id: string) => player.prog(id)
  const specText = root.querySelector('.specs div:last-child dd') as HTMLElement

  /* ---------- text and its trace ---------- */
  let bpe: Gpt2Bpe | null = null
  let text = pref.get('tokenizer-text') ?? EXAMPLES[0]
  let pieces: PieceTrace[] = [], fired: Fired[] = [], tokIndex: number[][] = [], ids: number[] = [], nTok = 0
  let cut = false
  function retrace() {
    if (!bpe) return
    cut = false
    pieces = []
    let bytes = 0
    for (const p of bpe.trace(text)) { if (bytes + p.bytes.length > MAXBYTES && pieces.length) { cut = true; break } bytes += p.bytes.length; pieces.push(p) }
    fired = pieces.flatMap((p, pi) => p.steps.map((_, step) => ({ piece: pi, step })))
    // every token gets its hue by position, as everywhere on the site
    let t = 0
    tokIndex = pieces.map((p) => p.tokens.map(() => t++))
    nTok = t
    ids = pieces.flatMap((p) => p.tokens.map((x) => x.id))
    const nBytes = pieces.reduce((s, p) => s + p.bytes.length, 0)
    specText.textContent = `${nBytes} bytes → ${nTok} tokens${cut ? ' (cut)' : ''}`
    poke()
  }
  loadBpe().then((b) => { bpe = b; retrace() })

  // controls: examples and a text field
  const field = document.createElement('label')
  field.className = 'textin'
  field.innerHTML = `<span>text</span><input type="text" maxlength="${MAXLEN}" spellcheck="false" autocomplete="off">`
  const input = field.querySelector('input')!
  input.value = text
  input.addEventListener('input', () => { text = input.value; pref.set('tokenizer-text', text); ex.set(EXAMPLES.indexOf(text)); retrace() })
  player.meta.prepend(field)
  const ex = toggle(player.meta, 'Examples', EXAMPLES.map((e) => (e.length > 16 ? e.slice(0, 15) + '…' : e)), EXAMPLES.indexOf(text), (i) => {
    text = EXAMPLES[i]; input.value = text; pref.set('tokenizer-text', text); retrace()
  })

  /* ---------- layout ---------- */
  const pad = 36, rulesW = 200
  const G = { mainL: 0, mainR: 0, sentY: 0, rowY: 0, rowH: 74, cellH: 32, rulerY: 0, rulesX: 0, rulesY: 0, bot: 0 }
  function geom() {
    const { W, H } = stage
    G.rulesX = W - pad - rulesW; G.mainL = pad; G.mainR = G.rulesX - 36
    G.sentY = 70; G.rowY = 150; G.rulerY = H - 70; G.rulesY = 40; G.bot = H - 24
  }
  stage.onResize = geom
  geom()

  const cellW = (sym: string) => 20 + 9 * [...sym].length
  const GAP = 3, WGAP = 22
  /** Rows of pieces, wrapped to the main width by their widest (byte) state; the block of rows is centred vertically. */
  function rowsOf(): number[][] {
    const rows = wrapRows()
    G.rowY = Math.round(lerp(G.sentY + 70, G.rulerY - 110, 0.5) - ((rows.length - 1) * G.rowH) / 2)
    return rows
  }
  function wrapRows(): number[][] {
    const rows: number[][] = [[]], maxW = G.mainR - G.mainL
    let w = 0
    pieces.forEach((p, i) => {
      const pw = p.symbols.reduce((s, sym) => s + cellW(sym) + GAP, -GAP)
      if (w > 0 && w + WGAP + pw > maxW) { rows.push([]); w = 0 }
      rows[rows.length - 1].push(i); w += (w > 0 ? WGAP : 0) + pw
    })
    return rows
  }
  /** Cell rects for one state (symbols per piece), each row centred. */
  function layoutCells(state: string[][], rows: number[][]) {
    const out: { x: number; y: number; w: number; sym: string }[][] = []
    rows.forEach((row, r) => {
      const widths = row.map((pi) => state[pi].reduce((s, sym) => s + cellW(sym) + GAP, -GAP))
      let x = (G.mainL + G.mainR) / 2 - (widths.reduce((a, b) => a + b, 0) + WGAP * (row.length - 1)) / 2
      const y = G.rowY + r * G.rowH
      row.forEach((pi) => {
        out[pi] = state[pi].map((sym) => { const c = { x, y, w: cellW(sym), sym }; x += c.w + GAP; return c })
        x += WGAP - GAP
      })
    })
    return out
  }
  /** Symbols of every piece after the first `k` fired merges. */
  function stateAfter(k: number): string[][] {
    const done = pieces.map(() => 0)
    for (let i = 0; i < k; i++) done[fired[i].piece]++
    return pieces.map((p, pi) => (done[pi] === 0 ? p.symbols : p.steps[done[pi] - 1].after))
  }
  /** The final token (global index) that symbol `k` of `state` in piece `pi` ends up in. */
  function tokenOf(pi: number, syms: string[], k: number) {
    let b = 0
    for (let j = 0; j < k; j++) b += [...syms[j]].length
    let acc = 0
    const toks = pieces[pi].tokens
    for (let t = 0; t < toks.length; t++) { acc += [...toks[t].text].length; if (b < acc) return tokIndex[pi][t] }
    return tokIndex[pi][toks.length - 1]
  }
  const hue = (t: number): RGB => C.tok[t % 7]

  let hits: { x: number; y: number; w: number; h: number; pi: number; sym: string; tok: number }[] = []
  let hover: { pi: number; sym: string; tok: number } | null = null
  let formula: { segs: [string, RGB][]; note: string } | null = null

  function drawSym(x: number, y: number, w: number, sym: string, tok: number, pi: number, a: number, hl: number, solid = false) {
    const h = G.cellH, col = hue(tok)
    rr(x, y - h / 2, w, h, 5)
    ctx.fillStyle = rgba(col, (solid ? 0.2 : 0.08) * a + 0.18 * hl * a); ctx.fill()
    ctx.lineWidth = 1 + hl; ctx.strokeStyle = rgba(hl > 0.5 ? C.ink : col, (0.45 + 0.5 * hl) * a); ctx.stroke()
    ctx.font = F.mono(13, 500); ctx.textBaseline = 'middle'; ctx.textAlign = 'left'
    const tw = ctx.measureText(sym).width
    let tx = x + (w - tw) / 2
    for (const ch of sym) {
      ctx.fillStyle = rgba(ch === 'Ġ' ? C.mute : C.ink, a)
      ctx.fillText(ch, tx, y + 0.5); tx += ctx.measureText(ch).width
    }
    hits.push({ x, y: y - h / 2, w, h, pi, sym, tok })
  }

  /* ---------- scenes ---------- */
  /** The text in the serif, pieces pulled apart by `sepE`; returns each piece's x and each byte's source x. */
  function sentence(sepE: number, a: number, colorE: number) {
    ctx.font = serifAt(34); ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left'
    const ws = pieces.map((p) => ctx.measureText(p.text).width)
    const total = ws.reduce((s, v) => s + v, 0) + WGAP * sepE * (pieces.length - 1)
    let x = (G.mainL + G.mainR) / 2 - total / 2
    const pos: { byteX: number[] }[] = []
    pieces.forEach((p, i) => {
      const col = hue(tokIndex[i][0])
      ctx.fillStyle = rgba(colorE > 0 ? mixc(C.ink, col, colorE) : C.ink, a)
      ctx.fillText(p.text, x, G.sentY)
      const byteX: number[] = []
      let cx = x
      for (const ch of p.text) { const n = new TextEncoder().encode(ch).length; for (let k = 0; k < n; k++) byteX.push(cx); cx += ctx.measureText(ch).width }
      pos.push({ byteX })
      if (sepE > 0.05 && a > 0) {
        ctx.strokeStyle = rgba(col, 0.6 * a * sepE); ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.moveTo(x, G.sentY + 10); ctx.lineTo(x + ws[i], G.sentY + 10); ctx.stroke()
      }
      x += ws[i] + WGAP * sepE
    })
    return pos
  }

  function sceneSplit(p: number) {
    const e = eio(clamp((p - 0.15) / 0.55))
    sentence(e, clamp(p / 0.1), e)
    formula = { segs: [['regex.findall(', C.mute], [`r"${PRETOKENIZE_SOURCE}"`, C.ink2], [', text)', C.mute]], note: "The regex (Python's regex module, not re) keeps contractions ('s 't 'll), words and numbers with their leading space, punctuation runs and whitespace apart. Merges never cross these pieces." }
  }

  function sceneBytes(p: number) {
    const pos = sentence(1, 1 - 0.75 * eout(clamp(p / 0.5)), 1)
    const rows = rowsOf(), cells = layoutCells(pieces.map((x) => x.symbols), rows)
    let n = 0
    cells.forEach((w, pi) => w.forEach((r, k) => {
      const f = eio(clamp((p - 0.1 - n++ * 0.015) / 0.4))
      if (f <= 0) return
      const sx = pos[pi].byteX[k] ?? r.x, x = lerp(sx, r.x, f), y = lerp(G.sentY - 10, r.y, f)
      drawSym(x, y, lerp(18, r.w, f), r.sym, tokenOf(pi, pieces[pi].symbols, k), pi, f, 0)
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'
      ctx.fillStyle = rgba(r.sym === 'Ġ' ? C.ink2 : C.mute, clamp((f - 0.6) / 0.4))
      ctx.fillText(hex2(pieces[pi].bytes[k]), r.x + r.w / 2, r.y + G.cellH / 2 + 6)
    }))
    const nBytes = pieces.reduce((s, x) => s + x.bytes.length, 0), multi = pieces.some((x) => x.bytes.length > [...x.text].length)
    formula = { segs: [['" "  →  0x20  →  ', C.mute], ['Ġ', C.ink], ['   ·   ', C.mute], [`${nBytes} bytes`, C.ink2]], note: multi ? 'Characters outside ASCII take 2–4 bytes in UTF-8; each byte gets its own printable stand-in (é = C3 A9 → Ã ©), so any text in any language starts from the same 256 symbols.' : 'GPT-2 maps each of the 256 byte values to a printable character (space becomes Ġ), so any text in any language starts from the same 256 symbols.' }
  }

  /** The merges that fired so far, most recent at the bottom, windowed to the space available. */
  function drawFired(k: number, a: number) {
    const x = G.rulesX, lh = 21, rows = Math.max(3, Math.floor((G.bot - 40 - G.rulesY - 24) / lh))
    ctx.font = F.label; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; spaced(true)
    ctx.fillStyle = rgba(C.mute, a); ctx.fillText('MERGES THAT FIRED', x, G.rulesY); spaced(false)
    const from = Math.max(0, k + 1 - rows)
    let y = G.rulesY + 26
    if (from > 0) { ctx.font = F.mono(11); ctx.fillStyle = rgba(C.mute, a); ctx.fillText(`⋮ ${from} earlier`, x + 10, y - 4); y += lh }
    for (let i = from; i <= Math.min(k, fired.length - 1); i++) {
      const st = pieces[fired[i].piece].steps[fired[i].step], on = i === k
      if (on) { rr(x - 8, y - 15, rulesW, 21, 4); ctx.fillStyle = rgba(C.ink, 0.1 * a); ctx.fill() }
      ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.mute, a); ctx.fillText('#' + st.rank, x + 44, y)
      ctx.textAlign = 'left'; ctx.font = F.mono(12, on ? 500 : 400); ctx.fillStyle = rgba(on ? C.ink : C.ink2, a)
      ctx.fillText(`${st.left} ${st.right}`, x + 54, y)
      y += lh
    }
    ctx.font = F.body; ctx.fillStyle = rgba(C.mute, a)
    ctx.fillText('each rank is looked up in a table', x, G.bot - 22)
    ctx.fillText('of 50,000 pairs, not scanned', x, G.bot - 4)
  }

  function sceneMerge(p: number) {
    const rows = rowsOf()
    if (!fired.length) {
      layoutCells(pieces.map((x) => x.symbols), rows).forEach((w, pi) => w.forEach((r, k) => drawSym(r.x, r.y, r.w, r.sym, tokenOf(pi, pieces[pi].symbols, k), pi, 1, 0)))
      formula = { segs: [['no pair is in the merge table', C.ink]], note: 'Every piece is already a single byte or a known token.' }
      return
    }
    const nS = fired.length, sf = clamp(p / 0.96) * nS, s = Math.min(nS - 1, Math.floor(sf)), f = p >= 0.96 ? 1 : sf - s
    const { piece, step } = fired[s], st = pieces[piece].steps[step]
    const before = stateAfter(s), after = stateAfter(s + 1)
    const cb = layoutCells(before, rows), ca = layoutCells(after, rows)
    const move = eio(clamp((f - 0.45) / 0.4))
    before.forEach((w, pi) => {
      if (pi !== piece) { cb[pi].forEach((r, k) => drawSym(r.x, r.y, r.w, r.sym, tokenOf(pi, w, k), pi, 0.55, 0)); return }
      // map each symbol before → its index after this merge
      const map: number[] = []
      for (let k = 0, o = 0; k < w.length; k++) {
        if (k < w.length - 1 && w[k] === st.left && w[k + 1] === st.right) { map.push(o, o); k++ } else map.push(o)
        o++
      }
      w.forEach((sym, k) => {
        const ai = map[k], b = cb[pi][k], t = ca[pi][ai]
        const second = k > 0 && map[k - 1] === ai, merging = second || (k < w.length - 1 && map[k + 1] === ai)
        if (merging && second && move > 0.5) return
        const x = lerp(b.x, t.x, move), wd = lerp(b.w, t.w, move)
        const hl = merging ? (f < 0.45 ? eout(clamp((f - 0.2) / 0.25)) : 1 - move) : 0
        drawSym(x, b.y, wd, merging && move > 0.5 ? after[pi][ai] : sym, tokenOf(pi, w, k), pi, 1, hl)
      })
      // the lookup: every adjacent pair's rank, the lowest one wins
      if (move < 0.5) {
        const la = clamp(f / 0.15) * (1 - clamp((move - 0.2) / 0.3))
        ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'
        for (let k = 0; k < w.length - 1; k++) {
          const r = bpe!.rankOf(w[k], w[k + 1]), gx = cb[pi][k].x + cb[pi][k].w + GAP / 2, win = r === st.rank
          ctx.fillStyle = rgba(win ? C.ink : C.mute, (r === undefined ? 0.7 : 1) * la)
          ctx.fillText(r === undefined ? '–' : '#' + r, gx, b0(cb[pi][k].y) - 4 - (k % 2) * 13) // staggered so neighbours never overlap
          if (win) { ctx.strokeStyle = rgba(C.ink, la); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cb[pi][k].x + 4, b0(cb[pi][k].y) - 2); ctx.lineTo(cb[pi][k + 1].x + cb[pi][k + 1].w - 4, b0(cb[pi][k].y) - 2); ctx.stroke() }
        }
      }
    })
    drawFired(s, 1)
    formula = { segs: [['merge  ', C.mute], [`${st.left} + ${st.right}  →  ${st.left + st.right}`, C.ink], ['   ·   rank ', C.mute], [String(st.rank), C.ink2], ['   ·   in ', C.mute], [pieces[piece].text.replace(/ /g, '␣'), C.ink2]], note: 'Look up the rank of every adjacent pair in the piece and fuse the lowest (the most common in training); repeat until no pair is in the table. – marks a pair with no rule.' }
  }
  const b0 = (y: number) => y - G.cellH / 2

  function sceneIds(p: number) {
    const rows = rowsOf(), fin = pieces.map((x) => x.tokens.map((t) => t.text)), cells = layoutCells(fin, rows), e = eout(clamp(p / 0.25))
    // log scale: raw bytes (0–255) would vanish on a linear axis
    const rx0 = G.mainL + 30, rx1 = G.mainR - 10, rx = (id: number) => rx0 + (Math.log10(Math.max(1, id)) / Math.log10(VOCAB)) * (rx1 - rx0)
    const ra = eout(clamp((p - 0.15) / 0.25))
    ctx.strokeStyle = rgba(C.ink, 0.35 * ra); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(rx0, G.rulerY); ctx.lineTo(rx1, G.rulerY); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, ra)
    for (const v of [1, 10, 100, 1000, 10000, 50257]) {
      ctx.beginPath(); ctx.moveTo(rx(v), G.rulerY - 3); ctx.lineTo(rx(v), G.rulerY + 3); ctx.stroke()
      ctx.fillText(v.toLocaleString('en-US'), rx(v), G.rulerY + 8)
    }
    ctx.fillStyle = rgba(C.ink, 0.12 * ra); ctx.fillRect(rx0, G.rulerY - 7, rx(256) - rx0, 14)
    ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, ra); ctx.fillText('0–255: raw bytes · 256 + rank: merged tokens · log scale', rx0, G.rulerY + 26)
    ctx.font = F.label; spaced(true); ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, ra)
    ctx.fillText('VOCABULARY · 50,257 IDS', rx0, G.rulerY - 16); spaced(false)
    fin.forEach((w, pi) => w.forEach((sym, k) => {
      const r = cells[pi][k], ti = tokIndex[pi][k], id = pieces[pi].tokens[k].id, cx = r.x + r.w / 2
      drawSym(r.x, r.y, r.w, sym, ti, pi, 1, 0, e > 0.5)
      const ia = clamp((p - 0.1 - ti * 0.04) / 0.2)
      ctx.font = F.mono(12, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'top'
      ctx.fillStyle = rgba(hue(ti), ia); ctx.fillText(String(id), cx, r.y + G.cellH / 2 + 6)
      const la = clamp((p - 0.35 - ti * 0.04) / 0.25)
      if (la > 0) {
        const tx = rx(id), y0 = r.y + G.cellH / 2 + 24
        ctx.strokeStyle = rgba(hue(ti), 0.6 * la); ctx.lineWidth = 1.2
        ctx.beginPath(); ctx.moveTo(cx, y0); ctx.bezierCurveTo(cx, y0 + 40, tx, G.rulerY - 50, tx, G.rulerY - 6); ctx.stroke()
        ctx.fillStyle = rgba(hue(ti), la); ctx.beginPath(); ctx.moveTo(tx, G.rulerY - 2); ctx.lineTo(tx - 4, G.rulerY - 9); ctx.lineTo(tx + 4, G.rulerY - 9); ctx.closePath(); ctx.fill()
      }
    }))
    if (hover) {
      const id = ids[hover.tok]
      formula = id < 256
        ? { segs: [[`vocab["${hover.sym}"]`, C.ink], ['  =  ', C.mute], [String(id), C.ink], ['   ·   a raw byte', C.mute]], note: 'No merge covers this byte here, so it stays a token of its own: ids 0–255 are the 256 bytes.' }
        : { segs: [[`vocab["${hover.sym}"]`, C.ink], ['  =  ', C.mute], [String(id), C.ink], ['  =  256 + rank ', C.mute], [String(id - 256), C.ink2]], note: 'The token was created by merge rule number id − 256, so common tokens, merged early in training, have small ids.' }
    } else formula = { segs: [['ids  =  ', C.mute], [`[${ids.join(', ')}]`, C.ink]], note: idsNote() }
  }
  /** Point out repeated tokens and look-alikes with different ids. */
  function idsNote() {
    const toks = pieces.flatMap((p) => p.tokens)
    const rep = toks.find((t, i) => toks.findIndex((u) => u.id === t.id) !== i)
    if (rep) return `${tokLabel(rep.text.replace(/^Ġ/, ' '))} appears twice with the same id ${rep.id}: only its position will tell the two apart (next: Embedding).`
    const norm = (s: string) => s.replace(/^Ġ/, '').toLowerCase()
    for (let i = 0; i < toks.length; i++) for (let j = i + 1; j < toks.length; j++) {
      if (norm(toks[i].text) === norm(toks[j].text) && toks[i].id !== toks[j].id) return `${toks[i].text} (${toks[i].id}) and ${toks[j].text} (${toks[j].id}) are different tokens: case and the leading space both change the id.`
    }
    return 'Each id picks one row of the embedding matrix next.'
  }

  function draw() {
    useCtx(ctx)
    stage.begin()
    hits = []
    formula = null
    if (!bpe || !pieces.length) {
      ctx.font = F.body; ctx.fillStyle = rgba(C.mute); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
      ctx.fillText(bpe ? 'Type some text below.' : 'Loading GPT-2’s 50,000 merge rules…', (G.mainL + G.mainR) / 2, G.rowY)
      frame.setFormula(null)
      return
    }
    const pb = prog('bytes'), pm = prog('merge'), pi = prog('ids')
    if (pb <= 0) sceneSplit(prog('split'))
    else if (pm <= 0) sceneBytes(pb)
    else if (pi <= 0) sceneMerge(pm)
    else sceneIds(pi)
    if (hover && pm > 0 && pi <= 0) {
      const n = [...hover.sym].length
      formula = { segs: [[`"${hover.sym}"`, C.ink], ['  =  ', C.mute], [`${n} byte${n > 1 ? 's' : ''}`, C.ink2]], note: 'Hover a symbol to see how many bytes it has swallowed so far.' }
    }
    frame.setFormula(formula?.segs ?? null, formula?.note)
  }

  stage.canvas.addEventListener('pointermove', (e) => {
    const [x, y] = stage.local(e)
    const h = hits.find((r) => x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h)
    hover = h ? { pi: h.pi, sym: h.sym, tok: h.tok } : null
    stage.canvas.style.cursor = h ? 'crosshair' : 'default'
  })
  stage.canvas.addEventListener('pointerleave', () => { hover = null })

  function caption(id: string): [string, string] {
    const nBytes = pieces.reduce((s, p) => s + p.bytes.length, 0)
    switch (id) {
      case 'split': return [`A regular expression splits the text into words, numbers and punctuation. Each piece keeps its leading space. Type your own text below${cut ? ` (only the first ${MAXBYTES} bytes are drawn)` : ''}.`, `${pieces.length} pieces`]
      case 'bytes': return ['Each piece becomes its UTF-8 bytes. GPT-2 gives every byte value a visible stand-in character, so a space is written Ġ.', `${nBytes} bytes`]
      case 'merge': return ['Inside each piece, the rank of every adjacent pair is looked up and the lowest-ranked pair is fused. This repeats until no pair is in the table; rare words end up as several tokens.', `${fired.length} merges fire`]
      default: return ['Each final symbol is an entry in the vocabulary. A merged token’s id is 256 plus its merge rank; ids 0–255 are the raw bytes.', `ids [${ids.join(', ')}]`]
    }
  }

  if (reduced && player.t === 0) player.t = player.start('merge') + 6
  player.describe = (i) => caption(PHASES[i].id)
  const stop = runLoop((dt) => {
    player.tick(dt)
    draw()
    player.updateUI()
    const cur = player.cur(), [t, s] = caption(cur.id)
    frame.setCaption(cur.name, cur.short ?? cur.name, t, s)
  }, () => !player.playing)
  return () => { stop(); player.destroy(); stage.destroy() }
}
