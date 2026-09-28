import { F, ctx, rr } from '../../core/draw'
import { C, rgba } from '../../core/theme'
import { clamp, eout } from '../../core/util'
import { multi as M, type Call } from '../../lib/multi/data'
import { mountExhibit, words, type Env } from '../kit'
import type { Nav } from '../registry'
import { authorKey, runs, whoBar, wrap, type Who } from './common'

/*
 * Multi-agent handoff: an orchestrator hands each part of a request to a worker agent that starts from an empty
 * context, and reads back only the workers' short reports. Every agent is Qwen3-1.7B; the run, the lookups and the
 * token counts are real (scripts/multi-export.ts), next to one agent doing the same request alone.
 */

const PHASES = [
  { id: 'single', name: 'One agent, one context', short: 'One agent', dur: 10 },
  { id: 'split', name: 'Handing out the work', short: 'Hand off', dur: 11 },
  { id: 'worker', name: 'Each worker starts empty', short: 'Worker', dur: 11 },
  { id: 'reports', name: 'Only the reports come back', short: 'Reports', dur: 11 },
  { id: 'timeline', name: 'Who runs when', short: 'Timeline', dur: 10 },
  { id: 'cost', name: 'What it costs', short: 'Cost', dur: 10 },
]

const W_ = M.workers, O = M.orchestrator, S = M.single, N = W_.length
const sum = (cs: Call[]) => cs.reduce((a, c) => a + c.context + c.output, 0)
const multiTotal = sum(M.calls.multi), singleTotal = sum(M.calls.single)
const maxCtx = (cs: Call[]) => Math.max(...cs.map((c) => c.context + c.output))
const orchCalls = M.calls.multi.filter((c) => c.agent === 'orchestrator'), workerCalls = (i: number) => M.calls.multi.filter((c) => c.agent === `worker ${i + 1}`)
const one = (s: string) => s.replace(/\s+/g, ' ').trim()
/** Parts of the request (model × aspect) that no handoff asked about. */
const ASPECTS: [string, string][] = [['layer', 'layers'], ['head', 'attention heads'], ['position', 'positions']]
const MODELS = [...new Set(O.steps.map((s) => s.model))]
/** The worker that reported GPT-2's 12 heads per layer, if the answer left that out (-1 otherwise). */
const LOST = /12 (attention )?heads/i.test(O.answer) ? -1 : W_.findIndex((w) => /12 attention heads/i.test(w.report))
const DROPPED = MODELS.flatMap((m) => ASPECTS.filter(([k2]) => !O.steps.some((s) => s.model === m && s.question.toLowerCase().includes(k2))).map(([, what]) => `${what} of ${m}`))

const COMPARE: Record<string, [string, string]> = {
  single: ['the ReAct loop’s growing context', 'agents/react?phase=context'],
  split: ['a tool call', 'agents/tool-calling?phase=call'],
  worker: ['a lookup in the ReAct loop', 'agents/react?phase=observe'],
  reports: ['retrieved text in a prompt', 'agents/rag?phase=prompt'],
  timeline: ['batching many requests at once', 'serving/continuous-batching?phase=why'],
  cost: ['prefix caching and the KV cache', 'serving/kv-cache?phase=size'],
}

const CAPS: Record<string, [string, string]> = {
  single: [`First, one agent with a lookup tool answers the request alone. It looks up both models at once, keeps both fact sheets in one context (${maxCtx(M.calls.single)} tokens by the end) and answers.`, `1 agent · ${M.calls.single.length} calls · ${S.lookups.length} lookups`],
  split: [`An orchestrator gets the same request and one tool, ask_worker. It hands off ${N} pieces, each a model and a question. A handoff is just a tool call whose result is another agent’s work.`, `${N} handoffs, one tool call each`],
  worker: [`A worker starts from an empty context: its own short instructions and one task. It never sees the user’s request, the other workers or the plan. Its first reply must be a tool call (tool_choice “required”), so it looks the model up before it answers.`, 'a fresh context per worker'],
  reports: [`The fact sheets stay in the workers’ contexts. The orchestrator reads only the ${N} short reports, as tool results, and writes the answer from them.`, `${N} reports in · 1 answer out`],
  timeline: [`The workers do not depend on each other, so a server can run them at the same time; the orchestrator waits for all of them. On a task this small the extra calls outweigh the parallelism: the one agent still finishes first.`, 'fan out, run in parallel, gather'],
  cost: [`Every worker re-reads its own instructions and fact sheet, so the split processes ${multiTotal.toLocaleString('en-US')} tokens against ${singleTotal.toLocaleString('en-US')}, and here the single agent’s answer was also more complete. Splitting pays when the pieces are big, independent and would crowd one context.`, `${multiTotal.toLocaleString('en-US')} vs ${singleTotal.toLocaleString('en-US')} tokens processed`],
}

export function mountMultiAgent(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: `A real run: ${M.model} plays the orchestrator and every worker; the handoffs, lookups, reports and token counts are from the run.`,
      eyebrow: 'Agents · Orchestration',
      title: 'Multi-agent Handoff',
      subtitle: 'an orchestrator, workers with their own contexts',
      specs: [
        { label: 'model', value: M.model, real: 'every agent', realLabel: '' },
        { label: 'workers', value: String(N) },
        { label: 'tokens processed', value: multiTotal.toLocaleString('en-US'), real: `${singleTotal.toLocaleString('en-US')} for one agent`, realLabel: 'vs' },
      ],
    },
    size: [1040, 480],
    aria: 'Multi-agent handoff: an orchestrator hands each part of a request to a worker agent with an empty context; each worker looks up a fact sheet and reports; the orchestrator answers from the reports; compared with one agent doing the same request, the split processes more tokens in smaller contexts.',
    phases: PHASES, learn: 'multiagent', tokens: words(['orchestrator', 'worker', 'report']), compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['reports', 10.5],
    scenes,
  })
}

function scenes({ stage, mk, k }: Env) {
  const { title, caption, arrow } = k
  const pad = 36, top = 56, f11 = F.mono(11), rowH = 18
  function block(items: { text: string; who: Who; strong?: boolean }[], x: number, y: number, w: number, a: number, maxLines = 99, font = f11) {
    let n = 0
    for (const it of items) {
      const y0 = y
      for (const l of wrap(it.text, w - 16, font)) {
        if (n++ >= maxLines) break
        runs([[l, it.who === 'model' ? C.ink : C.ink2]], x + 12, y + rowH / 2, a, it.strong ? F.mono(11, 600) : font); y += rowH
      }
      if (y > y0) whoBar(it.who, x, y0 + 3, y - 3, a)
    }
    return y
  }
  /** An agent's context window: a box with its name and size. */
  function box(x: number, y: number, w: number, h: number, name: string, sub: string, a: number, on = false) {
    if (a <= 0) return
    rr(x, y, w, h, 8); ctx.fillStyle = rgba(C.ink, (on ? 0.07 : 0.03) * a); ctx.fill()
    ctx.strokeStyle = rgba(C.ink, (on ? 0.7 : 0.3) * a); ctx.lineWidth = 1; ctx.stroke()
    ctx.font = F.mono(11, 600); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(name, x + 10, y + 17)
    ctx.font = F.small; ctx.textAlign = 'right'; ctx.fillStyle = rgba(C.mute, a); ctx.fillText(sub, x + w - 10, y + 17)
  }
  const workerX = (i: number) => { const w = (stage.W - 2 * pad - (N - 1) * 10) / N; return [pad + i * (w + 10), w] }

  /* ---------- 1: one agent ---------- */
  function sceneSingle(p: number) {
    const { W } = stage, x = pad, w = 560, y0 = top + 30
    const items: { text: string; who: Who; strong?: boolean }[] = [
      { text: M.prompts.single, who: 'prompt' }, { text: M.request, who: 'prompt' },
      ...S.lookups.flatMap((l) => [{ text: `lookup(${l.name})`, who: 'model' as Who, strong: true }, { text: l.result, who: 'tool' as Who }]),
    ]
    const n = Math.floor(clamp((p - 0.03) / 0.55) * items.length + 0.999)
    box(x - 8, y0 - 26, w + 16, 270, 'single agent · its context', `${maxCtx(M.calls.single)} tokens at the end`, 1, true)
    authorKey(x + w + 8, stage.H - 64, eout(clamp((p - 0.3) / 0.12)), ['prompt', 'model', 'tool'])
    block(items.slice(0, n), x, y0, w, 1, 15)
    const aa = eout(clamp((p - 0.62) / 0.12)), xr = x + w + 50
    title('its answer', xr, y0 - 12, aa)
    block([{ text: one(S.answer), who: 'model' }], xr, y0, W - pad - xr, aa, 14)
    mk.formula = { segs: [['one context', C.ink2], [' = ', C.mute], ['instructions + request + every lookup + every result', C.ink]], note: 'Everything the agent has read stays in front of it for the rest of the task, whether or not the next step needs it.' }
  }

  /* ---------- 2: the split ---------- */
  function sceneSplit(p: number) {
    const { W } = stage, oy = top + 10, ow = 640, ox = (W - ow) / 2, oh = 84 + N * 16 + (DROPPED.length ? 18 : 0)
    box(ox, oy, ow, oh, 'orchestrator', `${orchCalls[0].context} tokens in`, 1, true)
    block([{ text: M.request, who: 'prompt' }], ox + 8, oy + 28, ow - 16, 1, 3)
    const ca = eout(clamp((p - 0.1) / 0.15))
    O.steps.forEach((s, i) => {
      const a = eout(clamp((p - 0.12 - i * 0.05) / 0.1))
      runs([['ask_worker', C.ink], [`(${s.model}, “${s.question.length > 44 ? s.question.slice(0, 43) + '…' : s.question}”)`, C.ink2]], ox + 20, oy + 76 + i * 16, a * ca, F.mono(10.5))
    })
    if (DROPPED.length) runs([['no call for the ', C.mute], [DROPPED.join(', '), C.ink2]], ox + 20, oy + 80 + N * 16, eout(clamp((p - 0.4) / 0.1)), F.mono(10.5))
    // one box per worker, empty
    const wy = oy + oh + 60, ba = eout(clamp((p - 0.5) / 0.15))
    O.steps.forEach((s, i) => {
      const [x, w] = workerX(i)
      arrow([[ox + ow / 2, oy + oh + 2], [x + w / 2, wy - 6]], ba * 0.8)
      box(x, wy, w, 116, `worker ${i + 1}`, '', ba)
      block([{ text: s.model, who: 'prompt', strong: true }, { text: s.question, who: 'prompt' }], x + 2, wy + 26, w - 4, ba, 5, F.mono(10.5))
    })
    mk.formula = { segs: [['ask_worker(model, question)', C.ink], ['  →  ', C.mute], ['a new agent, a new context', C.ink2]], note: `The prompt said one worker per model; the model chose ${N} pieces, one per model and question${DROPPED.length ? `, and never asked about the ${DROPPED.join(' or the ')}` : ''}. The plan is itself a model output, and can drop things.` }
  }

  /* ---------- 3: inside a worker ---------- */
  function sceneWorker(p: number) {
    const wk = W_[0], x = pad + 10, w = 600, y0 = top + 30
    box(x - 10, y0 - 26, w + 20, 250, 'worker 1 · its whole context', `${Math.max(...workerCalls(0).map((c) => c.context + c.output))} tokens`, 1, true)
    authorKey(x + w + 10, stage.H - 64, eout(clamp((p - 0.3) / 0.12)), ['prompt', 'model', 'tool'])
    const items: { text: string; who: Who; strong?: boolean }[] = [
      { text: M.prompts.worker, who: 'prompt' }, { text: wk.task, who: 'prompt' },
      ...wk.lookups.flatMap((l) => [{ text: `lookup(${l.name})`, who: 'model' as Who, strong: true }, { text: l.result, who: 'tool' as Who }]),
      { text: wk.report, who: 'model', strong: true },
    ]
    const n = Math.floor(clamp((p - 0.03) / 0.6) * items.length + 0.999)
    block(items.slice(0, n), x, y0, w, 1, 16)
    // what it does not see
    const na = eout(clamp((p - 0.5) / 0.12)), xr = x + w + 50
    title('not in its context', xr, y0 - 12, na)
    ;['the user’s request', 'the orchestrator’s plan', `the other ${N - 1} workers`, 'their fact sheets'].forEach((t, i) => {
      runs([['× ', C.mute], [t, C.ink2]], xr, y0 + 12 + i * 22, na, f11)
    })
    mk.formula = { segs: [['worker context', C.ink2], [' = ', C.mute], ['its instructions + its task + its own lookups', C.ink]], note: 'Small models often skip a tool they should use and answer from memory; requiring the first reply to be a call rules that out. Everything a worker needs must be in its task.' }
  }

  /* ---------- 4: reports ---------- */
  function sceneReports(p: number) {
    const { W } = stage, wy = top + 190
    O.steps.forEach((_, i) => {
      const [x, w] = workerX(i), a = eout(clamp((p - i * 0.04) / 0.12))
      box(x, wy, w, 160, `worker ${i + 1}`, `${Math.max(...workerCalls(i).map((c) => c.context + c.output))} tok`, 0.8)
      block([{ text: one(W_[i].report), who: 'model' }], x + 2, wy + 26, w - 4, a, 6, F.mono(10.5))
      if (i === LOST) caption('left out of the answer', x + 12, wy + 150, eout(clamp((p - 0.7) / 0.1)), C.ink2, 'left')
    })
    const oy = top + 16, ow = W - 2 * pad, ox = pad, ga = eout(clamp((p - 0.35) / 0.12))
    box(ox, oy, ow, 130, 'orchestrator · the answer', `${orchCalls[orchCalls.length - 1].context} tokens in`, ga, true)
    O.steps.forEach((_, i) => { const [x, w] = workerX(i); arrow([[x + w / 2, wy - 4], [x + w / 2, oy + 134]], ga * 0.7) })
    block([{ text: one(O.answer), who: 'model', strong: false }], ox + 8, oy + 28, ow - 16, eout(clamp((p - 0.5) / 0.15)), 8)
    mk.formula = { segs: [['orchestrator reads', C.ink2], ['  ', C.mute], [`${N} reports`, C.ink], ['  not  ', C.mute], [`${W_.reduce((a, w) => a + w.lookups.length, 0)} fact sheets`, C.ink2]], note: `A report is a summary, and so is the answer: details can be lost at each step.${LOST >= 0 ? ` Here worker ${LOST + 1}’s 12 heads per layer for GPT-2 did not make it into the answer.` : ''}` }
  }

  /* ---------- 5: timeline ---------- */
  function sceneTimeline(p: number) {
    const { W } = stage, x0 = pad + 120, y0 = top + 40, rh = 30
    const len = (c: Call) => c.context + c.output
    const o1 = len(orchCalls[0]), o2 = len(orchCalls[orchCalls.length - 1]), wl = W_.map((_, i) => workerCalls(i).reduce((a, c) => a + len(c), 0))
    const par = o1 + Math.max(...wl) + o2, seq = singleTotal, span = Math.max(par, seq), sc = (W - pad - x0 - 20) / span
    const g = eout(clamp((p - 0.05) / 0.6))
    title('tokens processed along the way (a stand-in for time)', x0, y0 - 18, 1)
    const bar = (x: number, y: number, w: number, a: number, fill: number) => { ctx.fillStyle = rgba(C.ink, fill * a); ctx.fillRect(x, y, Math.max(1, w), 16) }
    const label = (t: string, y: number) => { ctx.font = F.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink2, 1); ctx.fillText(t, x0 - 12, y + 8) }
    label('orchestrator', y0); bar(x0, y0, o1 * sc * clamp(g * 3), 1, 0.8)
    bar(x0 + (o1 + Math.max(...wl)) * sc, y0, o2 * sc * clamp(g * 3 - 2), 1, 0.8)
    W_.forEach((_, i) => { const y = y0 + (i + 1) * rh; label(`worker ${i + 1}`, y); bar(x0 + o1 * sc, y, wl[i] * sc * clamp(g * 3 - 1), 1, 0.45) })
    const ys = y0 + (N + 1.6) * rh, sa = eout(clamp((p - 0.6) / 0.12))
    label('single agent', ys); bar(x0, ys, seq * sc * sa, 1, 0.3)
    caption(`${par.toLocaleString('en-US')} along the longest path, workers in parallel`, x0, ys - 12 - rh * 0.3, g, C.ink2, 'left')
    caption(`${seq.toLocaleString('en-US')} for one agent, one call after another`, x0 + seq * sc * sa + 10, ys + 12, sa, C.ink2, 'left')
    mk.formula = { segs: [['longest path', C.ink2], [' = ', C.mute], ['orchestrator + slowest worker + orchestrator', C.ink]], note: 'This run executed the workers one after another on one machine; the chart shows them as a server would run them, side by side.' }
  }

  /* ---------- 6: cost ---------- */
  function sceneCost(p: number) {
    const { W } = stage, x0 = pad + 200, y0 = top + 50, bw = W - pad - x0 - 300
    const rows: [string, number, number][] = [
      ['tokens processed', multiTotal, singleTotal],
      ['largest context', maxCtx(M.calls.multi), maxCtx(M.calls.single)],
      ['model calls', M.calls.multi.length, M.calls.single.length],
    ]
    rows.forEach(([name, m, s], i) => {
      const a = eout(clamp((p - 0.05 - i * 0.15) / 0.12)), y = y0 + i * 100, mx = Math.max(m, s)
      if (a <= 0) return
      ctx.font = F.mono(11.5, 600); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(name, x0 - 16, y + 14)
      ctx.fillStyle = rgba(C.ink, 0.8 * a); ctx.fillRect(x0, y, (m / mx) * bw, 14)
      ctx.fillStyle = rgba(C.ink, 0.3 * a); ctx.fillRect(x0, y + 22, (s / mx) * bw, 14)
      caption(`${m.toLocaleString('en-US')} · orchestrator + ${N} workers`, x0 + (m / mx) * bw + 10, y + 11, a, C.ink, 'left')
      caption(`${s.toLocaleString('en-US')} · one agent`, x0 + (s / mx) * bw + 10, y + 33, a, C.ink2, 'left')
    })
    const mM = maxCtx(M.calls.multi), mS = maxCtx(M.calls.single)
    mk.formula = { segs: [['tokens in all', C.ink2], [` ${multiTotal} vs ${singleTotal}`, C.ink], ['   ·   largest context', C.ink2], [` ${mM} vs ${mS}`, C.ink]], note: mM >= mS ? `Even the largest context is not smaller here: the orchestrator’s last call reads its tool schema, the ${N} calls and the ${N} reports. With a long document per worker, one agent’s context is the one that would overflow.` : 'Each worker keeps a small, focused context, and the orchestrator reads only summaries.' }
  }

  return { single: sceneSingle, split: sceneSplit, worker: sceneWorker, reports: sceneReports, timeline: sceneTimeline, cost: sceneCost }
}
