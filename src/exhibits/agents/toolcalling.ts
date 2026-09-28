import { F, ctx, drawChip, rr } from '../../core/draw'
import { C, rgba } from '../../core/theme'
import { clamp, eout } from '../../core/util'
import { tools as T } from '../../lib/tools/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'
import { bars, distRows, pct, runs, whoBar, wrap, type Who } from './common'

/*
 * Tool calling: tools are described to the model as JSON schemas in its system prompt, the model writes a
 * call as JSON between <tool_call> tags, the program checks and runs it and writes the result back in a
 * <tool_response>, and the model answers from it. A real run of Qwen3-1.7B (scripts/tools-export.ts).
 */

const PHASES = [
  { id: 'why', name: 'Why call a tool', short: 'Why', dur: 9 },
  { id: 'schema', name: 'Tools as JSON schemas', short: 'Schemas', dur: 10 },
  { id: 'call', name: 'The model writes a call', short: '<tool_call>', dur: 11 },
  { id: 'run', name: 'Check it, run it, send the result back', short: 'Run', dur: 11 },
  { id: 'answer', name: 'The answer, and when not to call', short: 'Answer', dur: 10 },
  { id: 'constrain', name: 'Constrained decoding', short: 'Constrain', dur: 10 },
]

const P = T.call.parsed
const firstP = T.call.first.find(([t]) => t === '<tool_call>')?.[1] ?? 0
const directP = T.direct.first.find(([t]) => t === '<tool_call>')?.[1] ?? 0
const letters = (P.arguments.word as string) ?? 'strawberry'
/** The number its own answer ends on. */
const plainAnswer = T.plain.answer.match(/(\d+)\D*$/)?.[1] ?? '?'

const COMPARE: Record<string, [string, string]> = {
  why: ['how the tokenizer splits words', 'anatomy/tokenizer?phase=merge'],
  schema: ['the chat template', 'agents/in-context?phase=instruct'],
  call: ['the text-format ReAct loop', 'agents/react?phase=step'],
  run: ['stop, run, append in ReAct', 'agents/react?phase=observe'],
  answer: ['the next-token distribution', 'anatomy/unembed?phase=softmax'],
  constrain: ['masking with −∞ before softmax', 'anatomy/attention?phase=mask'],
}

const CAPS: Record<string, [string, string]> = {
  why: [`Without tools, ${T.model} spells “${letters}” out and still answers ${plainAnswer}, after ${T.plain.tokens} tokens; there are ${T.result}. It never sees letters: ${T.plain.pieces.length === 1 ? `“ ${letters}” is a single token in its vocabulary` : `the word reaches it as ${T.plain.pieces.length} tokens`}. Counting is a job for a few lines of code.`, `“ ${letters}” = ${T.plain.pieces.map((s) => s.trim()).join(' + ')} · answer ${plainAnswer}, truth ${T.result}`],
  schema: [`The chat template writes each tool into the system prompt as a JSON schema: a name, a description and typed parameters, followed by the format to call them in. Two tools cost ${T.system.tokens} tokens on every request.`, `${T.tools.length} tools · ${T.system.tokens} tokens of system prompt`],
  call: [`The model answers with a call instead of text: JSON between <tool_call> tags. Its first token is <tool_call> at ${pct(firstP)}; at the name it puts ${pct(T.call.name[0][1])} on “${T.call.name[0][0]}”. The arguments are copied from the question.`, `${P.name}(${Object.entries(P.arguments).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')})`],
  run: [`The program parses the JSON, checks it against the schema, runs the function and writes the result back as a <tool_response> in a new turn. The model does not run anything; it only reads what comes back.`, `checks pass · result ${T.result}`],
  answer: [`With the result in its context, the model answers: “${T.answer.trim().slice(0, 70)}”. The same tools do not force a call: asked “${T.direct.question}”, it puts ${pct(directP)} on <tool_call> and answers directly.`, 'call when the tools help, answer when they do not'],
  constrain: [`A server can also force valid calls. At the name, only tokens that begin a declared tool name are allowed: ${T.constrained.allowed} of ${T.constrained.vocab.toLocaleString('en-US')}. The rest get probability zero, as in a causal mask, and the allowed ones are renormalised.`, `${T.constrained.allowed} of ${T.constrained.vocab.toLocaleString('en-US')} tokens allowed at the name`],
}

export function mountToolCalling(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: `A real run of ${T.model} with its own tool-call format; the call, the result and every probability come from the run.`,
      eyebrow: 'Agents · Tools',
      title: 'Tool Calling',
      subtitle: 'schema → call → result',
      specs: [
        { label: 'model', value: T.model, real: 'thinking off', realLabel: '' },
        { label: 'tools', value: T.tools.map((t) => t.name).join(', ') },
        { label: 'call format', value: '<tool_call> JSON </tool_call>' },
        { label: 'result', value: T.result },
      ],
    },
    size: [1040, 480],
    aria: 'Tool calling: a model given JSON schemas for two tools answers a letter-counting question by writing a function call as JSON; the program validates and runs it, returns the result, and the model answers; constrained decoding limits the tokens that can be written at the function name.',
    phases: PHASES, learn: 'toolcalling', tokens: words(T.plain.pieces.map((s) => s.trim())), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['call', 10.5],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption } = k
  const pad = 36, top = 56, f11 = F.mono(11), rowH = 19
  function block(items: { text: string; who: Who; strong?: boolean }[], x: number, y: number, w: number, a: number) {
    for (const it of items) {
      const y0 = y
      for (const l of wrap(it.text, w - 18, f11)) { runs([[l, it.who === 'model' ? C.ink : C.ink2]], x + 14, y + rowH / 2, a, it.strong ? F.mono(11, 600) : f11); y += rowH }
      whoBar(it.who, x, y0 + 3, y - 3, a)
    }
    return y
  }

  /* ---------- 1: why ---------- */
  function sceneWhy(p: number) {
    const y0 = top + 26
    title('question', pad, y0 - 14, 1)
    let y = block([{ text: T.question, who: 'prompt' }], pad, y0, 560, 1)
    const aa = eout(clamp((p - 0.1) / 0.12))
    title(`${T.model}, no tools · ${T.plain.tokens} tokens`, pad, y + 18, aa)
    const ans = T.plain.answer.replace(/\*\*/g, '').replace(/^#+ */gm, '').split('\n').filter((l) => l.trim() && !/^[^\x20-\x7e]+$/.test(l.trim()))
    y = block(ans.map((l) => ({ text: l.replace(/[^\x20-\x7e’“”→…]/g, '').trim(), who: 'model' as Who, strong: /answer/i.test(l) })), pad, y + 32, 560, aa)
    // what the model sees
    const ta = eout(clamp((p - 0.35) / 0.12)), xR = pad + 620
    title('what it reads', xR, y0 - 14, ta)
    let x = xR
    T.plain.pieces.forEach((s, i) => { x += drawChip(x, y0 + 12, { text: s, c: i }, ta, 22, true) + 6 })
    caption(T.plain.pieces.length === 1 ? 'one token, one row of the embedding matrix' : `${T.plain.pieces.length} tokens`, xR, y0 + 46, ta, C.mute, 'left')
    const la = eout(clamp((p - 0.55) / 0.12))
    title('what the question is about', xR, y0 + 90, la)
    x = xR
    ;[...letters].forEach((c) => {
      const r = c === 'r', w = 22
      rr(x, y0 + 104, w, 24, 4); ctx.fillStyle = rgba(C.ink, (r ? 0.25 : 0.05) * la); ctx.fill(); ctx.strokeStyle = rgba(C.ink, (r ? 0.9 : 0.3) * la); ctx.lineWidth = 1; ctx.stroke()
      ctx.font = F.mono(12, r ? 600 : 400); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, la); ctx.fillText(c, x + w / 2, y0 + 116.5)
      x += w + 4
    })
    caption(`${[...letters].length} letters, ${T.result} of them r`, xR, y0 + 150, la, C.mute, 'left')
    mk.formula = { segs: [['tokens', C.ink2], ['  ≠  ', C.mute], ['letters', C.ink]], note: 'Each token is one row of the embedding matrix; the spelling inside it is not visible to the model, only whatever it learned about that token.' }
  }

  /* ---------- 2: schemas ---------- */
  function sceneSchema(p: number) {
    const { W } = stage, x = pad, w = W - 2 * pad, y0 = top + 26
    title(`system prompt · ${T.system.tokens} tokens`, x, y0 - 14, 1)
    const txt = T.system.text.replace(/<\|im_start\|>system\n/, '').replace(/<\|im_end\|>\n?$/, '')
    const ls = txt.split('\n'), shown = Math.floor(clamp((p - 0.03) / 0.6) * ls.length + 0.999)
    let y = y0
    ls.slice(0, shown).forEach((l) => {
      const isTool = l.startsWith('{"type"'), parts = isTool ? wrap(l, w - 18, f11) : [l]
      for (const part of parts) { runs([[part, isTool ? C.ink : C.ink2]], x + 14, y + rowH / 2, 1, f11); y += rowH }
    })
    whoBar('prompt', x, y0 + 3, y - 3, 1)
    const ta = eout(clamp((p - 0.7) / 0.1))
    T.tools.forEach((t, i) => runs([[`${t.name}: ${t.tokens} tokens`, C.ink2]], x + 14 + i * 240, y + 18, ta, F.small))
    mk.formula = { segs: [['{"name": …, "description": …, "parameters": {…}}', C.ink]], note: 'The same JSON Schema format that OpenAI-style APIs take; the chat template turns the API’s tools list into this text, which is all the model ever sees of them.' }
  }

  /* ---------- 3: the call ---------- */
  function sceneCall(p: number) {
    const { W } = stage, x = pad, y0 = top + 26, w = 560
    title('question', x, y0 - 14, 1)
    let y = block([{ text: T.question, who: 'prompt' }], x, y0, w, 1)
    title(`${T.model} writes`, x, y + 18, 1)
    const lines = T.call.text.split('\n'), total = T.call.text.length, upto = Math.floor(clamp((p - 0.05) / 0.5) * total)
    let used = 0
    const y1 = y + 32
    y = y1
    for (const l of lines) {
      const s = l.slice(0, Math.max(0, upto - used)); used += l.length + 1
      if (s) runs([[s, C.ink]], x + 14, y + rowH / 2, 1, l.startsWith('{') ? F.mono(11, 600) : f11)
      y += rowH
    }
    whoBar('model', x, y1 + 3, y - 3, 1)
    const xD = x + w + 60, wD = W - pad - xD
    const fa = eout(clamp((p - 0.08) / 0.1))
    title('first token', xD, y0 - 14, fa)
    distRows(bars(T.call.first.slice(0, 3)), xD, y0 + 12, wD, fa, { mark: (_r, i) => i === 0, labelW: 120 })
    const nameAt = T.call.text.indexOf('"name": "') + 9, na = eout(clamp((upto - nameAt) / 10))
    title('at the name, after {"name": "', xD, y0 + 110, na)
    distRows(bars(T.call.name.slice(0, 4)), xD, y0 + 136, wD, na, { mark: (_r, i) => i === 0, labelW: 120 })
    mk.formula = { segs: [['p( <tool_call> )', C.ink2], [' = ', C.mute], [pct(firstP), C.ink], ['   ·   ', C.mute], [`p( ${T.call.name[0][0]} )`, C.ink2], [' = ', C.mute], [pct(T.call.name[0][1]), C.ink]], note: `<tool_call> is a single special token, added to the vocabulary for this; ${T.call.tokens} tokens in all for the call.` }
  }

  /* ---------- 4: check and run ---------- */
  function sceneRun(p: number) {
    const { W } = stage, y0 = top + 26, colW = (W - 2 * pad - 60) / 3
    const cols = [pad, pad + colW + 30, pad + 2 * (colW + 30)]
    // parse
    const a1 = eout(clamp(p / 0.1))
    title('1 · parse the JSON', cols[0], y0 - 14, a1)
    block([{ text: `name: ${P.name}`, who: 'tool' }, ...Object.entries(P.arguments).map(([k2, v]) => ({ text: `${k2}: ${JSON.stringify(v)}`, who: 'tool' as Who }))], cols[0], y0, colW, a1)
    // validate
    const a2 = eout(clamp((p - 0.2) / 0.1))
    title('2 · check against the schema', cols[1], y0 - 14, a2)
    T.checks.forEach(([c, ok], i) => {
      const ca = eout(clamp((p - 0.22 - i * 0.04) / 0.08)), y = y0 + 10 + i * rowH
      runs([[ok ? '✓ ' : '✗ ', ok ? C.ink : C.neg], [c, C.ink2]], cols[1], y, ca, f11)
    })
    // run
    const a3 = eout(clamp((p - 0.45) / 0.1))
    title('3 · run the function', cols[2], y0 - 14, a3)
    runs([[`${P.name}(${Object.entries(P.arguments).map(([k2, v]) => `${k2}=${JSON.stringify(v)}`).join(', ')})`, C.ink2]], cols[2], y0 + 10, a3, f11)
    runs([['→ ', C.mute], [T.result, C.ink]], cols[2], y0 + 34, a3, F.mono(14, 600))
    // back into the context
    const a4 = eout(clamp((p - 0.62) / 0.1)), yb = y0 + 150
    title('4 · written back into the context by the chat template', pad, yb - 14, a4)
    const resp = T.response.split('\n').filter((l) => l.length), cut = resp.findIndex((l) => l.includes('</tool_call>')) + 1
    const y = block([{ text: resp.slice(0, cut).join('\n'), who: 'model' }, { text: resp.slice(cut).join('\n'), who: 'tool', strong: true }], pad, yb, W - 2 * pad, a4)
    caption('the model’s call stays in the context; the result arrives as a new user turn', pad + 14, y + 16, a4, C.mute, 'left')
    mk.formula = { segs: [['<tool_response>', C.mute], [T.result, C.ink], ['</tool_response>', C.mute]], note: 'If a check failed, the program would send the error back instead and let the model try again; a malformed call never reaches the function.' }
  }

  /* ---------- 5: the answer, and a question that needs no tool ---------- */
  function sceneAnswer(p: number) {
    const { W } = stage, y0 = top + 26, w = W - 2 * pad
    title(`${T.model} answers · context now ${T.context} tokens`, pad, y0 - 14, 1)
    const aa = eout(clamp((p - 0.05) / 0.15))
    let y = block([{ text: T.answer.trim(), who: 'model', strong: true }], pad, y0, w, aa)
    const da = eout(clamp((p - 0.4) / 0.12)), yd = y + 50
    title(`same tools, another question: ${T.direct.question}`, pad, yd - 14, da)
    const xD = pad + 560, wD = W - pad - xD
    title('first token', xD, yd - 14, da)
    distRows(bars(T.direct.first.slice(0, 4)), xD, yd + 12, wD, da, { mark: (b) => b.text === '<tool_call>', labelW: 120 })
    y = block([{ text: T.direct.answer.trim(), who: 'model' }], pad, yd, 520, eout(clamp((p - 0.5) / 0.12)))
    mk.formula = { segs: [['p( <tool_call> )', C.ink2], ['  letters: ', C.mute], [pct(firstP), C.ink], ['   ·   capital: ', C.mute], [pct(directP), C.ink]], note: 'Whether to call a tool is the same next-token choice as everything else, learned in tuning from examples of calls and of plain answers.' }
  }

  /* ---------- 6: constrained decoding ---------- */
  function sceneConstrain(p: number) {
    const { W } = stage, y0 = top + 26, cs = 9, cols = 40, rows = 20, gx = pad, gy = y0 + 6
    const c = T.constrained, frac = c.allowed / c.vocab
    title(`the vocabulary · ${c.vocab.toLocaleString('en-US')} tokens, one cell ≈ ${Math.round(c.vocab / (cols * rows))}`, gx, y0 - 14, 1)
    const hatchA = eout(clamp((p - 0.15) / 0.2))
    for (let r = 0; r < rows; r++) for (let q = 0; q < cols; q++) {
      const x = gx + q * (cs + 2), y = gy + r * (cs + 2), keep = r === 0 && q === 0
      ctx.strokeStyle = rgba(C.ink, 0.35); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1)
      if (keep) { ctx.fillStyle = rgba(C.ink, 0.9); ctx.fillRect(x, y, cs, cs) }
      else if (hatchA > 0) { ctx.strokeStyle = rgba(C.mute, 0.7 * hatchA); ctx.beginPath(); ctx.moveTo(x + 1, y + cs - 1); ctx.lineTo(x + cs - 1, y + 1); ctx.stroke() }
    }
    const gb = gy + rows * (cs + 2)
    caption(`allowed at the name: ${c.allowed} tokens (${(frac * 100).toFixed(3)}%), less than the filled cell; hatched: masked`, gx, gb + 18, hatchA, C.ink2, 'left')
    // before and after
    const xD = gx + cols * (cs + 2) + 60, wD = W - pad - xD, ba = eout(clamp((p - 0.4) / 0.12))
    title('model’s own top tokens', xD, y0 - 14, 1)
    distRows(bars(T.call.name.slice(0, 4)), xD, y0 + 12, wD, 1, { mark: (_r, i) => i === 0, labelW: 120 })
    title('after the mask, renormalised', xD, y0 + 130, ba)
    distRows(bars(c.top.slice(0, 4)), xD, y0 + 156, wD, ba, { mark: (_r, i) => i === 0, labelW: 120 })
    caption(`the allowed tokens held ${pct(c.mass)} of the probability before the mask`, xD, y0 + 156 + 4 * 24 + 8, ba, C.mute, 'left')
    mk.formula = { segs: [['logit(t)', C.ink2], [' = ', C.mute], ['−∞', C.ink], ['  unless t can continue a valid call', C.ink2]], note: 'Grammar-constrained decoding applies such a mask at every step from a JSON grammar, so the output always parses. Here the model was already sure; the mask matters for smaller models and stricter formats.' }
  }

  return { why: sceneWhy, schema: sceneSchema, call: sceneCall, run: sceneRun, answer: sceneAnswer, constrain: sceneConstrain }
}
