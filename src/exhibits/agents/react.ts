import { F, ctx, rr } from '../../core/draw'
import { C, rgba } from '../../core/theme'
import { clamp, eout } from '../../core/util'
import { react, type ReactStep } from '../../lib/react/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'
import { bars, distRows, noLigatures, pct, runs, whoBar, wrap, type Who } from './common'

/*
 * The ReAct loop: a model writes a Thought and an Action, the program around it stops the text at
 * "Observation", runs the named tool and appends the real result, and the model goes on until it writes
 * finish[...]. The whole trace is a real run of Qwen3-1.7B (scripts/react-export.ts).
 */

const PHASES = [
  { id: 'prompt', name: 'The prompt sets up the loop', short: 'Prompt', dur: 10 },
  { id: 'step', name: 'Think, then act', short: 'Thought → Action', dur: 10 },
  { id: 'observe', name: 'Stop, run the tool, append', short: 'Observe', dur: 12 },
  { id: 'trace', name: 'Round and round', short: 'The loop', dur: 12 },
  { id: 'context', name: 'The context grows', short: 'Context', dur: 10 },
  { id: 'unstopped', name: 'Without the stop', short: 'No stop', dur: 11 },
]

const L = react.loop, S = L.steps, LAST = S[S.length - 1]
/** A step's Thought and Action lines. */
const split = (s: ReactStep) => { const i = s.text.indexOf('\nAction'); return i < 0 ? [s.text.trim(), ''] : [s.text.slice(0, i).trim(), s.text.slice(i + 1).trim()] }
/** Tokens in the context when each model call starts. */
const START = S.map((_, k) => (k === 0 ? L.promptTokens : S[k - 1].ctx + (S[k - 1].obsTokens ?? 0)))
const WRITTEN = S.map((s, k) => s.ctx - START[k])
const resent = START.reduce((a, c) => a + c, 0) + WRITTEN.reduce((a, w) => a + w, 0), kept = L.total
const ANSWER = LAST.action?.arg ?? ''
const firstAct = S[0].action!

const COMPARE: Record<string, [string, string]> = {
  prompt: ['a worked example is few-shot prompting', 'agents/in-context?phase=shots'],
  step: ['GPT-2’s greedy pick', 'anatomy?phase=pick'],
  observe: ['prefill: appending tokens to the cache', 'serving/kv-cache?phase=prefill'],
  trace: ['the prompt as the program', 'agents/in-context?phase=program'],
  context: ['how big the KV cache gets', 'serving/kv-cache?phase=size'],
  unstopped: ['a base model continuing text', 'agents/in-context?phase=instruct'],
}

const CAPS: Record<string, [string, string]> = {
  prompt: [`The system prompt teaches a format: a Thought, then one Action from a list of tools written in plain text, then an Observation. A worked example shows it once. That is ${L.promptTokens} tokens before the question is even read.`, `${L.promptTokens} prompt tokens · 3 actions · 1 worked example`],
  step: [`${react.model} writes a Thought, then an Action. Naming the action is an ordinary next-token choice: after “Action 1:” it puts ${pct(S[0].choice[0]?.[1] ?? 0)} on “${(S[0].choice[0]?.[0] ?? '').trim()}”.`, 'Thought k: … → Action k: tool[argument]'],
  observe: [`The model never runs anything. The program around it watches the text, stops it where “Observation” would start, reads the action with a regular expression, runs the tool and appends the real result. Then it hands the context back.`, 'stop → parse → run → append → continue'],
  trace: [`The real run: two lookups, one calculation, then finish[${ANSWER}]. Each observation came from the program, not the model. The model only decided what to look up and what to do with it.`, `${S.length} model calls · ${S.filter((s) => s.obs).length} tool results · answer ${ANSWER}`],
  context: [`Every step adds the model’s text and the tool’s result to the context, which is read again at the next call. Kept in a KV cache between calls, each token is processed once (${kept}). Sent afresh each time, the calls process ${resent}.`, `${kept} tokens with a cache kept · ${resent} re-sent`],
  unstopped: [`Without the stop, the model goes on and writes an Observation itself, a guess in the right format. That is why the loop must cut the text at “Observation” and insert the real result.`, 'the stop string keeps the model from inventing results'],
}

export function mountReact(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: `A real run of ${react.model} with a lookup tool and a calculator; the tools’ results are real, and so is every probability shown.`,
      eyebrow: 'Agents · Loops',
      title: 'ReAct Loop',
      subtitle: 'think → act → observe, until finish',
      specs: [
        { label: 'model', value: react.model, real: 'thinking off', realLabel: '' },
        { label: 'tools', value: react.tools.map((t) => t.name).join(', ') },
        { label: 'stop string', value: '“\\nObservation”' },
        { label: 'this run', value: `${S.length} calls`, real: `${kept} tokens`, realLabel: '·' },
      ],
    },
    size: [1040, 480],
    aria: 'The ReAct loop: a model writes a thought and an action, the program stops it, runs the named tool and appends the result, and the model continues until it gives an answer; a real run of Qwen3-1.7B looking up two models and subtracting their layer counts.',
    phases: PHASES, learn: 'react', tokens: words(['Thought', 'Action', 'Observation']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['trace', 11.5],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption, arrow } = k
  const pad = 36, top = 56, f11 = F.mono(11), rowH = 19
  /** Lines of the context, each with who wrote it; draws the first `upto` characters. Returns the y after. */
  function lines(items: { text: string; who: Who; strong?: boolean }[], x: number, y: number, w: number, a: number, upto = Infinity) {
    let used = 0
    for (const it of items) {
      if (used >= upto) break
      const ls = wrap(it.text, w - 18, f11), y0 = y
      for (const l of ls) {
        if (used >= upto) break
        const shown = l.slice(0, Math.max(0, upto - used))
        used += l.length + 1
        runs([[shown, it.who === 'model' ? C.ink : C.ink2]], x + 14, y + rowH / 2, a, it.strong ? F.mono(11, 600) : f11)
        y += rowH
      }
      whoBar(it.who, x, y0 + 3, y - 3, a)
    }
    return y
  }
  const stepItems = (s: ReactStep, i: number): { text: string; who: Who; strong?: boolean }[] => {
    const [th, ac] = split(s)
    const out: { text: string; who: Who; strong?: boolean }[] = [{ text: th, who: 'model' }, { text: ac, who: 'model', strong: true }]
    if (s.obs) out.push({ text: `Observation ${i + 1}: ${s.obs}`, who: 'tool' })
    return out
  }
  function key(x: number, y: number, a: number) {
    whoBar('model', x, y - 6, y + 6, a); caption('written by the model', x + 12, y + 4, a, C.mute, 'left')
    whoBar('tool', x + 180, y - 6, y + 6, a); caption('appended by the program', x + 192, y + 4, a, C.mute, 'left')
  }

  /* ---------- 1: the prompt ---------- */
  function scenePrompt(p: number) {
    const { H } = stage, secs = react.sections, tmpl = L.promptTokens - secs.reduce((a, s) => a + s.tokens, 0)
    const segs = [...secs.map((s) => ({ name: s.name, tokens: s.tokens })), { name: 'chat template', tokens: tmpl }]
    const y0 = top + 26, colH = H - y0 - 60, sc = colH / L.promptTokens, cur = Math.min(segs.length - 1, Math.floor(clamp((p - 0.05) / 0.8) * segs.length))
    title(`prompt · ${L.promptTokens} tokens`, pad, y0 - 14, 1)
    let y = y0
    segs.forEach((s, i) => {
      const h = s.tokens * sc, on = i === cur
      ctx.fillStyle = rgba(C.ink, (on ? 0.35 : 0.1)); ctx.fillRect(pad, y, 22, h - 1.5)
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(on ? C.ink : C.mute, 1)
      ctx.fillText(`${s.name} · ${s.tokens}`, pad + 32, y + Math.max(h / 2, 7))
      y += h
    })
    // the text, section by section
    const x = pad + 250, w = stage.W - pad - x
    title('system prompt, as the model reads it', x, y0 - 14, 1)
    let ty = y0
    secs.forEach((s, i) => {
      const on = i === cur, ls = s.text.split('\n').filter((l) => l.trim()), max = s.name === 'worked example' ? 6 : 9
      ls.slice(0, max).forEach((l) => {
        ctx.save(); ctx.beginPath(); ctx.rect(x, ty, w, rowH); ctx.clip()
        runs([[l, on ? C.ink : C.ink2]], x + 14, ty + rowH / 2, on ? 1 : 0.55, f11)
        ctx.restore(); ty += rowH
      })
      if (ls.length > max) { runs([[`… ${ls.length - max} more lines`, C.mute]], x + 14, ty + rowH / 2, 1, f11); ty += rowH }
      whoBar('prompt', x, ty - rowH * Math.min(ls.length, max + 1) + 3, ty - 3, 1)
      ty += 6
    })
    const s = segs[cur]
    mk.formula = { segs: [[s.name, C.ink2], ['  ', C.mute], [`${s.tokens} tokens`, C.ink]], note: s.name === 'worked example' ? 'One solved question in the exact format, so the model copies the pattern: in-context learning at work.' : s.name === 'tools' ? 'The tools are described in plain text; the model only ever sees their names and descriptions.' : s.name === 'chat template' ? 'The special tokens around each message, and the empty think block that switches Qwen3’s reasoning off.' : s.name === 'rule' ? `Run again without this line, the model still looked both numbers up (${react.noRule.filter((x) => x.action?.name === 'lookup').length} lookups), but smaller models often answer from memory.` : 'Written by whoever builds the agent; the model reads it before every step.' }
  }

  /* ---------- 2: one thought and action ---------- */
  function sceneStep(p: number) {
    const { W } = stage, s = S[0], [th, ac] = split(s), x = pad, w = 600, y0 = top + 26
    title('question', x, y0 - 14, 1)
    const qy = lines([{ text: `Question: ${react.question}`, who: 'prompt' }], x, y0, w, 1)
    title(`${react.model} writes`, x, qy + 18, 1)
    const total = th.length + ac.length + 2, upto = Math.floor(clamp((p - 0.05) / 0.55) * total)
    const ey = lines([{ text: th, who: 'model' }, { text: ac, who: 'model', strong: true }], x, qy + 32, w, 1, upto)
    if (upto < total && Math.floor(p * 8) % 2 === 0) { ctx.fillStyle = rgba(C.ink, 0.7); ctx.fillRect(x + 14, ey + 2, 7, 2) }
    // the choice of action
    const at = th.length + 1 + ac.indexOf(':') + 1, ca = eout(clamp((upto - at) / 12)), xD = x + w + 60, wD = W - pad - xD
    title('after “Action 1:”', xD, qy + 18, ca)
    distRows(bars(s.choice), xD, qy + 44, wD, ca, { mark: (_r, i) => i === 0, labelW: 110 })
    caption('the tool’s name is just the likeliest next token', xD, qy + 44 + s.choice.length * 24 + 10, ca, C.mute, 'left')
    mk.formula = { segs: [[`p( ${(s.choice[0]?.[0] ?? '').trim()} | … Action 1: )`, C.ink2], ['  =  ', C.mute], [pct(s.choice[0]?.[1] ?? 0), C.ink]], note: `The action is text like any other: “${firstAct.name}[${firstAct.arg}]”. Nothing has happened in the world yet.` }
  }

  /* ---------- 3: stop, parse, run, append ---------- */
  function sceneObserve(p: number) {
    const { W } = stage, s = S[0], [th, ac] = split(s)
    const nodes: [string, string][] = [['model writes', 'next tokens'], ['stop', 'at “\\nObservation”'], ['parse', 'Action k: name[arg]'], ['run the tool', `${firstAct.name}(…)`], ['append', 'Observation k: …']]
    const bw = 160, gap = (W - 2 * pad - 5 * bw) / 4, by = top + 30, cur = Math.min(4, Math.floor(clamp((p - 0.04) / 0.8) * 5))
    nodes.forEach(([a, b], i) => {
      const x = pad + i * (bw + gap), on = i === cur, done = i < cur
      rr(x, by, bw, 46, 8); ctx.fillStyle = rgba(C.ink, on ? 0.12 : 0.04); ctx.fill()
      ctx.strokeStyle = rgba(C.ink, on ? 0.9 : done ? 0.45 : 0.2); ctx.lineWidth = on ? 1.4 : 1; ctx.stroke()
      ctx.font = F.mono(11.5, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, on || done ? 1 : 0.6); ctx.fillText(a, x + bw / 2, by + 20)
      noLigatures(true); ctx.font = F.small; ctx.fillStyle = rgba(C.mute, 1); ctx.fillText(b, x + bw / 2, by + 36); noLigatures(false)
      if (i < 4) arrow([[x + bw + 4, by + 23], [x + bw + gap - 4, by + 23]], 1)
    })
    // back to the model
    const xl = pad + bw / 2, xr = pad + 4 * (bw + gap) + bw / 2
    arrow([[xr, by + 50], [xr, by + 70], [xl, by + 70], [xl, by + 52]], 0.8, true)
    caption('next step, with the longer context', (xl + xr) / 2, by + 86, 1, C.mute)
    // the concrete run
    const x = pad, w = W - 2 * pad, y0 = by + 118
    title('the context', x, y0 - 12, 1)
    const items: { text: string; who: Who; strong?: boolean }[] = [{ text: th, who: 'model' }, { text: ac, who: 'model', strong: true }]
    if (cur >= 4) items.push({ text: `Observation 1: ${s.obs}`, who: 'tool' }, { text: 'Thought 2:', who: 'tool' })
    const ey = lines(items, x, y0, w, 1)
    const ny = ey + 18
    if (cur === 1) caption('the next tokens would be “\\nObservation”: generation stops here, and the cache is rolled back to before them', x + 14, ny, 1, C.ink2, 'left')
    if (cur === 2) { runs([['/Action \\d+: (\\w+)\\[(.*)\\]/', C.ink2], ['   →   ', C.mute], [`name = ${firstAct.name}`, C.ink], ['   ', C.mute], [`arg = “${firstAct.arg}”`, C.ink]], x + 14, ny, 1, f11) }
    if (cur === 3) { runs([[`${firstAct.name}(“${firstAct.arg}”)`, C.ink], ['   →   ', C.mute], ['the fact sheet, read from a table in the program', C.ink2]], x + 14, ny, 1, f11) }
    if (cur === 4) caption(`${s.obsTokens} tokens appended; the model continues from “Thought 2:”`, x + 14, ny, 1, C.ink2, 'left')
    key(W - pad - 360, y0 - 16, 1)
    mk.formula = { segs: [['stop at “\\nObservation”', C.ink2], ['  →  ', C.mute], [`${firstAct.name}[${firstAct.arg}]`, C.ink], ['  →  ', C.mute], ['Observation 1: …', C.ink2]], note: 'A stop string ends generation as soon as the text contains it. Without one, the model would write the observation itself (the last step of this page).' }
  }

  /* ---------- 4: the whole trace ---------- */
  function sceneTrace(p: number) {
    const { W } = stage, x = pad, w = W - 2 * pad, y0 = top + 26
    title(`question: ${react.question}`, x, y0 - 14, 1)
    key(W - pad - 360, y0 - 18, 1)
    const n = Math.min(S.length, Math.floor(clamp((p - 0.03) / 0.75) * S.length) + 1)
    const items = S.slice(0, n).flatMap(stepItems)
    const ey = lines(items, x, y0 + 4, w, 1)
    const fa = eout(clamp((p - 0.85) / 0.08))
    if (n === S.length) runs([['answer  ', C.mute], [ANSWER, C.ink]], x + 14, ey + 18, fa, F.mono(13, 600))
    const tools = S.filter((s) => s.action && s.action.name !== 'finish').map((s) => `${s.action!.name}[${s.action!.arg}]`)
    mk.formula = { segs: tools.flatMap((t, i) => (i ? [['  →  ', C.mute], [t, C.ink]] : [[t, C.ink]])) as [string, typeof C.ink][], note: `${S.length} calls to the model, ${tools.length} to tools. The facts and the arithmetic came from the program; the model chose the steps and read the results.` }
  }

  /* ---------- 5: the context grows ---------- */
  function sceneContext(p: number) {
    const { W } = stage, x0 = pad + 110, y0 = top + 50, rh = 44, maxT = L.total, sc = (W - pad - x0 - 190) / maxT
    title('context at each call · tokens', x0, y0 - 24, 1)
    S.forEach((_s, i) => {
      const a = eout(clamp((p - 0.05 - i * 0.12) / 0.12)), y = y0 + i * rh
      if (a <= 0) return
      ctx.font = F.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, a); ctx.fillText(`call ${i + 1}`, x0 - 12, y + 9)
      // the prompt, then what earlier steps added, then this call's own output
      let x = x0
      ctx.fillStyle = rgba(C.mute, 0.3 * a); ctx.fillRect(x, y, L.promptTokens * sc, 18); x += L.promptTokens * sc
      for (let j = 0; j < i; j++) {
        ctx.fillStyle = rgba(C.ink, 0.75 * a); ctx.fillRect(x, y, WRITTEN[j] * sc - 1, 18); x += WRITTEN[j] * sc
        const ot = S[j].obsTokens ?? 0
        ctx.strokeStyle = rgba(C.ink, 0.7 * a); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, ot * sc - 2, 17); x += ot * sc
      }
      ctx.fillStyle = rgba(C.ink, a); ctx.fillRect(x, y, WRITTEN[i] * sc, 18)
      ctx.strokeStyle = rgba(C.bg, a); ctx.lineWidth = 1.5; ctx.strokeRect(x, y, WRITTEN[i] * sc, 18)
      ctx.font = F.small; ctx.textAlign = 'left'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(`${START[i]} read · +${WRITTEN[i]} written`, x + WRITTEN[i] * sc + 10, y + 9)
    })
    const ky = y0 + S.length * rh + 6
    ctx.fillStyle = rgba(C.mute, 0.3); ctx.fillRect(x0, ky, 14, 10); caption('prompt', x0 + 20, ky + 9, 1, C.mute, 'left')
    ctx.fillStyle = rgba(C.ink, 0.75); ctx.fillRect(x0 + 90, ky, 14, 10); caption('written by the model', x0 + 110, ky + 9, 1, C.mute, 'left')
    ctx.strokeStyle = rgba(C.ink, 0.7); ctx.strokeRect(x0 + 280.5, ky + 0.5, 13, 9); caption('tool results', x0 + 300, ky + 9, 1, C.mute, 'left')
    // processed tokens: cache kept vs re-sent
    const ra = eout(clamp((p - 0.62) / 0.12)), yb = ky + 54, bs = (W - pad - x0 - 290) / resent
    if (ra > 0) {
      title('tokens the model processes in all', x0, yb - 14, ra)
      ctx.fillStyle = rgba(C.ink, 0.8 * ra); ctx.fillRect(x0, yb, kept * bs, 14)
      caption(`${kept} · KV cache kept between calls`, x0 + kept * bs + 10, yb + 11, ra, C.ink, 'left')
      ctx.fillStyle = rgba(C.ink, 0.35 * ra); ctx.fillRect(x0, yb + 26, resent * bs, 14)
      caption(`${resent} · whole context re-sent each call`, x0 + resent * bs + 10, yb + 37, ra, C.ink2, 'left')
    }
    mk.formula = { segs: [['re-sent', C.ink2], ['  =  ', C.mute], [START.join(' + ') + ` + ${WRITTEN.reduce((a, b) => a + b, 0)} written`, C.ink2], ['  =  ', C.mute], [String(resent), C.ink]], note: 'Serving systems keep the shared prefix cached between calls (prefix caching), so re-reading the prompt is mostly avoided; the context window still limits how many steps fit.' }
  }

  /* ---------- 6: not stopped ---------- */
  function sceneUnstopped(p: number) {
    const { W } = stage, u = react.unstopped, colW = (W - 2 * pad - 40) / 2, y0 = top + 26
    title('not stopped: the model keeps writing', pad, y0 - 14, 1)
    const txt = u.text.split('\n').filter((l) => l.trim()), shown = Math.floor(clamp((p - 0.05) / 0.45) * txt.length + 0.999)
    const items = txt.slice(0, shown).map((t) => ({ text: t, who: 'model' as Who, strong: /^Observation/.test(t) }))
    lines(items.slice(0, 12), pad, y0, colW, 1)
    const ra = eout(clamp((p - 0.55) / 0.12)), xr = pad + colW + 40
    title('what the lookup really returns', xr, y0 - 14, ra)
    lines([{ text: `Observation 1: ${S[0].obs}`, who: 'tool' }], xr, y0, colW, ra)
    const inv = txt.find((t) => /^Observation 1/.test(t)) ?? '', layers = (t: string) => t.match(/(\d+) layers/)?.[1] ?? '?'
    const ca = eout(clamp((p - 0.7) / 0.12))
    runs([['layers of LLaMA 3 8B:  ', C.mute], [`${layers(inv)} in its own observation`, C.ink], ['  ·  ', C.mute], [`${layers(S[0].obs ?? '')} from the lookup`, C.ink2]], xr, y0 + 100, ca, f11)
    mk.formula = { segs: [['its own “Observation 1”', C.ink2], ['  vs  ', C.mute], ['the tool’s', C.ink]], note: inv ? `It wrote: “${inv.slice(0, 120)}${inv.length > 120 ? '…' : ''}”. Plausible, in the right format, and not looked up.` : 'It kept writing without an observation.' }
  }

  return { prompt: scenePrompt, step: sceneStep, observe: sceneObserve, trace: sceneTrace, context: sceneContext, unstopped: sceneUnstopped }
}
