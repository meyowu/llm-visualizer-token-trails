import { F, rr, serifAt, spaced, useCtx } from '../../core/draw'
import { createFrame } from '../../core/frame'
import { Player } from '../../core/player'
import { Stage, runLoop } from '../../core/stage'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp, reducedMotion } from '../../core/util'
import type { Nav } from '../registry'
import { PROMPT, promptTokens } from './model'

/*
 * GPT-2's byte-level BPE on the prompt: split into words, turn each word into bytes, then apply
 * merge rules in priority order until none match. In GPT-2 a merged token's id is 256 + its
 * merge rank (ids 0–255 are the raw bytes), so the ranks below follow from the real ids:
 * Ġthe = 262 → rank 6, Ġon = 319 → 63, The = 464 → 208, Ġsat = 3332 → 3076, Ġcat = 3797 → 3541.
 */

const PHASES = [
  { id: 'split', name: 'Pre-tokenize', short: 'Split', dur: 3.5 },
  { id: 'bytes', name: 'Bytes', dur: 3.5 },
  { id: 'merge', name: 'BPE merges', short: 'Merge', dur: 12 },
  { id: 'ids', name: 'Vocabulary ids', short: 'ids', dur: 5 },
]

/** The top of GPT-2's merges.txt, plus the later rules this prompt needs. */
const RULES: { a: string; b: string; rank: number }[] = [
  ['Ġ', 't'], ['Ġ', 'a'], ['h', 'e'], ['i', 'n'], ['r', 'e'], ['o', 'n'], ['Ġt', 'he'], ['e', 'r'],
  ['Ġ', 's'], ['a', 't'], ['Ġ', 'w'], ['Ġ', 'o'], ['e', 'n'], ['Ġ', 'c'],
].map(([a, b], rank) => ({ a, b, rank }))
RULES.push({ a: 'Ġ', b: 'on', rank: 63 }, { a: 'T', b: 'he', rank: 208 }, { a: 'Ġs', b: 'at', rank: 3076 }, { a: 'Ġc', b: 'at', rank: 3541 })

const REGEX = String.raw`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
const VOCAB = 50257
const hex = (ch: string) => (ch === 'Ġ' ? 0x20 : ch.charCodeAt(0)).toString(16).toUpperCase().padStart(2, '0')

interface Step { rule: number; words: number[]; before: string[][]; after: string[][]; maps: number[][] }

/** Apply rules in rank order; record every rule that fires and how cells map before → after. */
function simulate(words: string[][]): Step[] {
  const steps: Step[] = []
  let cur = words
  RULES.forEach((r, ri) => {
    const after: string[][] = [], maps: number[][] = [], hitWords: number[] = []
    cur.forEach((w, wi) => {
      const out: string[] = [], map: number[] = []
      for (let k = 0; k < w.length; k++) {
        if (k < w.length - 1 && w[k] === r.a && w[k + 1] === r.b) { map.push(out.length, out.length); out.push(r.a + r.b); k++ }
        else { map.push(out.length); out.push(w[k]) }
      }
      if (out.length !== w.length) hitWords.push(wi)
      after.push(out); maps.push(map)
    })
    if (hitWords.length) { steps.push({ rule: ri, words: hitWords, before: cur, after, maps }); cur = after }
  })
  return steps
}

export function mountTokenizer(root: HTMLElement, nav: Nav): () => void {
  const reduced = reducedMotion()
  const toks = promptTokens(), N = toks.length
  const words0 = PROMPT.map((w) => [...w].map((ch) => (ch === ' ' ? 'Ġ' : ch)))
  const steps = simulate(words0)
  const final = steps[steps.length - 1].after

  const frame = createFrame(root, {
    formula: true,
    eyebrow: 'Anatomy · Tokenizer',
    title: 'Tokenizer',
    subtitle: 'byte-level BPE · GPT-2',
    back: { label: 'Forward pass', onClick: () => nav('anatomy') },
    specs: [
      { label: 'base symbols', value: '256 bytes' },
      { label: 'merges', value: '50,000' },
      { label: 'vocab', value: '50,257' },
      { label: 'this prompt', value: `${PROMPT.join('').length} bytes → ${N} tokens` },
    ],
  })
  const stage = new Stage(frame.stageHost, 1040, 470, 'Byte-level BPE: the prompt is split into words, each word becomes bytes, and merge rules fuse adjacent symbols in priority order until each word is a single vocabulary token with an id.')
  const ctx = stage.ctx
  const player = new Player(PHASES, frame.controls, { playing: !reduced })
  const prog = (id: string) => player.prog(id)
  const hue = (i: number): RGB => C.tok[toks[i].c % 7]

  /* ---------- layout ---------- */
  const pad = 36, rulesW = 210
  const G = { mainL: 0, mainR: 0, sentY: 0, rowY: 0, cellH: 34, rulerY: 0, rulesX: 0, rulesY: 0 }
  function geom() {
    const { W, H } = stage
    G.rulesX = W - pad - rulesW; G.mainL = pad; G.mainR = G.rulesX - 40
    G.rowY = Math.round((H + 50) * 0.42); G.sentY = G.rowY - 80; G.rulerY = G.rowY + 120; G.rulesY = 70
  }
  stage.onResize = geom
  geom()

  const cellW = (sym: string) => 24 + 9 * [...sym].length
  const GAP = 3, WGAP = 26
  /** Cell rects for one state of all words, centred in the main area. */
  function layoutCells(state: string[][]) {
    const widths = state.map((w) => w.reduce((s, sym) => s + cellW(sym) + GAP, -GAP))
    const total = widths.reduce((a, b) => a + b, 0) + WGAP * (state.length - 1)
    let x = (G.mainL + G.mainR) / 2 - total / 2
    return state.map((w) => {
      const rects = w.map((sym) => { const r = { x, w: cellW(sym), sym }; x += r.w + GAP; return r })
      x += WGAP - GAP
      return rects
    })
  }

  let hits: { x: number; y: number; w: number; h: number; word: number; sym: string }[] = []
  let hover: { word: number; sym: string } | null = null
  let formula: { segs: [string, RGB][]; note: string } | null = null

  function drawSym(x: number, y: number, w: number, sym: string, word: number, a: number, hl: number, solid = false) {
    const h = G.cellH, col = hue(word)
    rr(x, y - h / 2, w, h, 5)
    ctx.fillStyle = rgba(col, (solid ? 0.2 : 0.08) * a + 0.18 * hl * a); ctx.fill()
    ctx.lineWidth = 1 + hl; ctx.strokeStyle = rgba(hl > 0.5 ? C.ink : col, (0.45 + 0.5 * hl) * a); ctx.stroke()
    ctx.font = F.mono(13, 500); ctx.textBaseline = 'middle'; ctx.textAlign = 'left'
    const chars = [...sym], tw = ctx.measureText(sym).width
    let tx = x + (w - tw) / 2
    for (const ch of chars) {
      ctx.fillStyle = rgba(ch === 'Ġ' ? C.mute : C.ink, a)
      ctx.fillText(ch, tx, y + 0.5); tx += ctx.measureText(ch).width
    }
    hits.push({ x, y: y - h / 2, w, h, word, sym })
  }

  /* ---------- scenes ---------- */
  function sentence(sepE: number, a: number, colorE: number) {
    ctx.font = serifAt(34); ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left'
    const ws = PROMPT.map((w) => ctx.measureText(w).width)
    const total = ws.reduce((s, v) => s + v, 0) + WGAP * sepE * (N - 1)
    let x = (G.mainL + G.mainR) / 2 - total / 2
    const pos: { x: number; chars: number[] }[] = []
    PROMPT.forEach((w, i) => {
      ctx.fillStyle = rgba(colorE > 0 ? mixRGB(C.ink, hue(i), colorE) : C.ink, a)
      ctx.fillText(w, x, G.sentY)
      const chars: number[] = []
      let cx = x
      for (const ch of w) { chars.push(cx); cx += ctx.measureText(ch).width }
      pos.push({ x, chars })
      if (sepE > 0.05 && a > 0) {
        ctx.strokeStyle = rgba(hue(i), 0.6 * a * sepE); ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.moveTo(x, G.sentY + 10); ctx.lineTo(x + ws[i], G.sentY + 10); ctx.stroke()
      }
      x += ws[i] + WGAP * sepE
    })
    return pos
  }
  const mixRGB = (a: RGB, b: RGB, t: number): RGB => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]

  function sceneSplit(p: number) {
    const e = eio(clamp((p - 0.15) / 0.55))
    sentence(e, clamp(p / 0.1), e)
    formula = { segs: [['re.findall(', C.mute], [`r"${REGEX}"`, C.ink2], [', text)', C.mute]], note: 'Words keep their leading space, so " cat" and "cat" are different pieces. Merges never cross these boundaries.' }
  }

  function sceneBytes(p: number) {
    const pos = sentence(1, 1 - 0.75 * eout(clamp(p / 0.5)), 1)
    const cells = layoutCells(words0)
    cells.forEach((w, wi) => w.forEach((r, k) => {
      const f = eio(clamp((p - 0.1 - (wi * 4 + k) * 0.02) / 0.4))
      if (f <= 0) return
      ctx.font = serifAt(34)
      const sx = pos[wi].chars[k], x = lerp(sx, r.x, f), y = lerp(G.sentY - 10, G.rowY, f)
      drawSym(x, y, lerp(18, r.w, f), r.sym, wi, f, 0)
      ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'top'
      ctx.fillStyle = rgba(r.sym === 'Ġ' ? C.ink2 : C.mute, clamp((f - 0.6) / 0.4))
      ctx.fillText('0x' + hex(r.sym), r.x + r.w / 2, G.rowY + G.cellH / 2 + 8)
    }))
    const nBytes = words0.flat().length
    formula = { segs: [['" "  →  0x20  →  ', C.mute], ['Ġ', C.ink], ['   ·   ', C.mute], [`${nBytes} bytes`, C.ink2]], note: 'GPT-2 maps each of the 256 byte values to a printable character (space becomes Ġ), so any text in any language starts from the same 256 symbols.' }
  }

  function drawRules(activeRule: number, pStep: number, a: number) {
    const x = G.rulesX, lh = 22
    ctx.font = F.label; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; spaced(true)
    ctx.fillStyle = rgba(C.mute, a); ctx.fillText('MERGES · BY RANK', x, G.rulesY); spaced(false)
    let y = G.rulesY + 24, prevRank = -1
    RULES.forEach((r, ri) => {
      if (r.rank - prevRank > 1) {
        ctx.font = F.mono(11); ctx.fillStyle = rgba(C.faint, a); ctx.fillText('⋮', x + 30, y - 4); y += lh * 0.8
      }
      prevRank = r.rank
      const fired = steps.findIndex((s) => s.rule === ri)
      const passed = ri < activeRule || (ri === activeRule && pStep >= 1)
      const on = ri === activeRule && pStep < 1
      if (on) { rr(x - 8, y - 15, rulesW, 21, 4); ctx.fillStyle = rgba(C.ink, 0.1 * a); ctx.fill() }
      ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.mute, a)
      ctx.fillText('#' + r.rank, x + 34, y)
      ctx.textAlign = 'left'; ctx.font = F.mono(12.5, on ? 500 : 400)
      const col = on ? C.ink : passed ? (fired >= 0 ? C.ink2 : C.faint) : C.mute
      ctx.fillStyle = rgba(col, a); ctx.fillText(`${r.a} ${r.b}`, x + 46, y)
      ctx.fillStyle = rgba(passed && fired < 0 ? C.faint : C.mute, a * 0.8)
      ctx.fillText(`→ ${r.a + r.b}`, x + 118, y)
      if (passed && fired < 0) { ctx.font = F.mono(9.5); ctx.fillStyle = rgba(C.faint, a); ctx.fillText('no match', x + 160, y) }
      y += lh
    })
    ctx.font = F.body; ctx.fillStyle = rgba(C.mute, a); ctx.fillText('50,000 rules in GPT-2', x, y + 10)
  }

  function sceneMerge(p: number) {
    const nS = steps.length, sf = clamp(p / 0.96) * nS, s = Math.min(nS - 1, Math.floor(sf)), f = p >= 0.96 ? 1 : sf - s
    const st = steps[s], rule = RULES[st.rule]
    const before = layoutCells(st.before), after = layoutCells(st.after)
    const move = eio(clamp((f - 0.35) / 0.45))
    st.before.forEach((w, wi) => w.forEach((sym, k) => {
      const map = st.maps[wi], ai = map[k], b = before[wi][k], t = after[wi][ai]
      const second = k > 0 && map[k - 1] === ai, merging = second || (k < w.length - 1 && map[k + 1] === ai)
      if (merging && second && move > 0.5) return // once fused, draw the merged cell only once
      const x = lerp(b.x, t.x, move), wdt = lerp(b.w, t.w, move)
      const hl = merging ? (f < 0.35 ? eout(f / 0.35) : 1 - move) : 0
      drawSym(x, G.rowY, wdt, merging && move > 0.5 ? st.after[wi][ai] : sym, wi, 1, hl)
    }))
    drawRules(st.rule, f, 1)
    const where = st.words.map((wi) => st.after[wi].join('')).join(', ')
    formula = { segs: [['merge  ', C.mute], [`${rule.a} + ${rule.b}  →  ${rule.a + rule.b}`, C.ink], ['   ·   rank ', C.mute], [String(rule.rank), C.ink2], ['   ·   in ', C.mute], [where, C.ink2]], note: 'At each step the highest-priority adjacent pair is fused. Common pairs were learned first, so they have low ranks.' }
  }

  function sceneIds(p: number) {
    const cells = layoutCells(final), e = eout(clamp(p / 0.25))
    // log scale: every id in this prompt is below 4,000, and 0–255 (raw bytes) would be invisible on a linear axis
    const rx0 = G.mainL + 30, rx1 = G.mainR - 10, rx = (id: number) => rx0 + (Math.log10(Math.max(1, id)) / Math.log10(VOCAB)) * (rx1 - rx0)
    // ruler
    const ra = eout(clamp((p - 0.15) / 0.25))
    ctx.strokeStyle = rgba(C.ink, 0.35 * ra); ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(rx0, G.rulerY); ctx.lineTo(rx1, G.rulerY); ctx.stroke()
    ctx.font = F.mono(9.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(C.mute, ra)
    for (const v of [1, 10, 100, 1000, 10000, 50257]) {
      ctx.beginPath(); ctx.moveTo(rx(v), G.rulerY - 3); ctx.lineTo(rx(v), G.rulerY + 3); ctx.stroke()
      ctx.fillText(v.toLocaleString('en-US'), rx(v), G.rulerY + 8)
    }
    ctx.fillStyle = rgba(C.ink, 0.12 * ra); ctx.fillRect(rx0, G.rulerY - 7, rx(256) - rx0, 14)
    ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.mute, ra); ctx.fillText('0–255: raw bytes · 256 + rank: merges · log scale', rx0, G.rulerY + 24)
    ctx.font = F.label; spaced(true); ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.mute, ra)
    ctx.fillText('VOCABULARY · 50,257 IDS', rx0, G.rulerY - 16); spaced(false)
    final.forEach((w, wi) => {
      const r = cells[wi][0], id = toks[wi].id, cx = r.x + r.w / 2
      drawSym(r.x, G.rowY, r.w, w[0], wi, 1, 0, e > 0.5)
      const ia = clamp((p - 0.1 - wi * 0.06) / 0.2)
      ctx.font = F.mono(12, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'top'
      ctx.fillStyle = rgba(hue(wi), ia); ctx.fillText(String(id), cx, G.rowY + G.cellH / 2 + 8)
      const la = clamp((p - 0.35 - wi * 0.06) / 0.25)
      if (la > 0) {
        const tx = rx(id), y0 = G.rowY + G.cellH / 2 + 26
        ctx.strokeStyle = rgba(hue(wi), 0.6 * la); ctx.lineWidth = 1.2
        ctx.beginPath(); ctx.moveTo(cx, y0); ctx.bezierCurveTo(cx, y0 + 40, tx, G.rulerY - 50, tx, G.rulerY - 6); ctx.stroke()
        ctx.fillStyle = rgba(hue(wi), la); ctx.beginPath(); ctx.moveTo(tx, G.rulerY - 2); ctx.lineTo(tx - 4, G.rulerY - 9); ctx.lineTo(tx + 4, G.rulerY - 9); ctx.closePath(); ctx.fill()
      }
    })
    const hv = hover
    if (hv) {
      const id = toks[hv.word].id
      formula = { segs: [[`vocab["${hv.sym}"]`, C.ink], ['  =  ', C.mute], [String(id), C.ink], ['  =  256 + rank ', C.mute], [String(id - 256), C.ink2]], note: `Every byte of "${PROMPT[hv.word]}" collapsed into one symbol; its id is where that merge sits in the table.` }
    } else formula = { segs: [['ids  =  ', C.mute], [`[${toks.map((t) => t.id).join(', ')}]`, C.ink]], note: 'Common words were merged early, so they have small ids: "Ġthe" is 262. Each id picks one row of the embedding matrix next.' }
  }



  function draw() {
    useCtx(ctx)
    stage.begin()
    hits = []
    formula = null
    const pb = prog('bytes'), pm = prog('merge'), pi = prog('ids')
    if (pb <= 0) sceneSplit(prog('split'))
    else if (pm <= 0) sceneBytes(pb)
    else if (pi <= 0) sceneMerge(pm)
    else sceneIds(pi)
    if (hover && pm > 0 && pi <= 0) {
      const bytes = [...hover.sym].map(hex).join(' ')
      formula = { segs: [[`"${hover.sym}"`, C.ink], ['  =  bytes ', C.mute], [bytes, C.ink2]], note: `${[...hover.sym].length} byte${[...hover.sym].length > 1 ? 's' : ''} so far in this symbol.` }
    }
    frame.setFormula(formula?.segs ?? null, formula?.note)
  }

  stage.canvas.addEventListener('pointermove', (e) => {
    const [x, y] = stage.local(e)
    const h = hits.find((r) => x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h)
    hover = h ? { word: h.word, sym: h.sym } : null
    stage.canvas.style.cursor = h ? 'crosshair' : 'default'
  })
  stage.canvas.addEventListener('pointerleave', () => { hover = null })

  const CAPS: Record<string, [string, string]> = {
    split: ['A regular expression splits the text into words, numbers and punctuation. Each piece keeps its leading space.', `${N} pieces`],
    bytes: ['Each piece becomes its UTF-8 bytes. GPT-2 gives every byte value a visible stand-in character, so a space is written Ġ.', `${words0.flat().length} bytes`],
    merge: ['Merge rules are tried in rank order. Whenever a rule matches two adjacent symbols inside a piece, they fuse into one. This repeats until no rule matches.', `${steps.length} merges fire`],
    ids: ['Each final symbol is an entry in the vocabulary. Its id is 256 plus its merge rank, since ids 0–255 are the raw bytes.', 'ids [' + toks.map((t) => t.id).join(', ') + ']'],
  }

  if (reduced && player.t === 0) player.t = player.start('merge') + 6
  player.describe = (i) => CAPS[PHASES[i].id]
  const stop = runLoop((dt) => {
    player.tick(dt)
    draw()
    player.updateUI()
    const cur = player.cur(), [t, s] = CAPS[cur.id]
    frame.setCaption(cur.name, cur.short ?? cur.name, t, s)
  }, () => !player.playing)
  return () => { stop(); player.destroy(); stage.destroy() }
}
