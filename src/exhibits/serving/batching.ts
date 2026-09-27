import { F, rr, type TokLike } from '../../core/draw'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp, rng } from '../../core/util'
import { mountExhibit, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * Continuous batching: why a server batches decode steps, and why it refills the batch after every
 * step instead of waiting for the whole batch to finish. The schedules are a toy simulation (4 slots,
 * 12 requests); step times are estimates for LLaMA 3 8B in 16-bit on an A100 (weights read once per
 * step at 2 TB/s, 2 FLOPs per weight per token at 312 TFLOP/s), ignoring attention's share.
 */

const PHASES = [
  { id: 'why', name: 'Why batch', short: 'Why batch', dur: 9 },
  { id: 'static', name: 'Static batching', short: 'Static', dur: 11 },
  { id: 'continuous', name: 'Refill after every step', short: 'Continuous', dur: 11 },
  { id: 'compare', name: 'Waiting and throughput', short: 'Compare', dur: 8 },
  { id: 'chunked', name: 'Long prompts in chunks', short: 'Chunked prefill', dur: 10 },
]

const SLOTS = 4, NR = 12
const REQS = (() => { const r = rng(1); return Array.from({ length: NR }, (_, i) => ({ id: i, len: 3 + Math.floor(r() * 14), arrive: Math.floor(i * 2.2) })) })()
interface Sched { grid: number[][]; start: number[]; end: number[] }
/** One decode step per column; grid[t][slot] = request id, −2 for a slot held by a finished request, −1 for empty. */
function schedule(mode: 'static' | 'continuous'): Sched {
  const start = Array(NR).fill(-1), end = Array(NR).fill(-1), done = Array(NR).fill(0), slots = Array(SLOTS).fill(-1), grid: number[][] = []
  for (let t = 0; t < 200 && end.some((e) => e < 0); t++) {
    const waiting = REQS.filter((q) => start[q.id] < 0 && q.arrive <= t)
    if (mode === 'continuous' || slots.every((s) => s < 0)) for (let s = 0; s < SLOTS; s++) if (slots[s] < 0 && waiting.length) { const q = waiting.shift()!; start[q.id] = t; slots[s] = q.id }
    grid.push(slots.map((id) => (id < 0 ? -1 : end[id] >= 0 ? -2 : id)))
    for (let s = 0; s < SLOTS; s++) {
      const id = slots[s]
      if (id < 0 || end[id] >= 0) continue
      if (++done[id] >= REQS[id].len) { end[id] = t; if (mode === 'continuous') slots[s] = -1 }
    }
    if (mode === 'static' && slots.every((id) => id < 0 || end[id] >= 0)) slots.fill(-1)
  }
  return { grid, start, end }
}
const STATIC = schedule('static'), CONT = schedule('continuous')
const stats = (s: Sched) => ({
  steps: s.grid.length,
  util: s.grid.flat().filter((v) => v >= 0).length / (s.grid.length * SLOTS),
  wait: REQS.reduce((a, q) => a + s.start[q.id] - q.arrive, 0) / NR,
  latency: REQS.reduce((a, q) => a + s.end[q.id] + 1 - q.arrive, 0) / NR,
})
const TOKS: TokLike[] = REQS.map((q) => ({ text: `r${q.id + 1}`, c: q.id }))

/** LLaMA 3 8B, 16-bit, A100: step time in ms for a step that processes `tokens` tokens. */
const WEIGHT_MS = (16.06e9 / 2.0e12) * 1000, FLOP_MS = (2 * 8.03e9) / 312e12 * 1000
const stepMs = (tokens: number) => Math.max(WEIGHT_MS, tokens * FLOP_MS)

const COMPARE: Record<string, [string, string]> = {
  why: ['decode’s memory limit', 'serving/kv-cache?phase=bound'], static: ['the KV cache', 'serving/kv-cache'], continuous: ['PagedAttention', 'serving/pagedattention'],
  compare: ['PagedAttention’s throughput', 'serving/pagedattention?phase=sim'], chunked: ['prefill and decode', 'serving/kv-cache?phase=prefill'],
}

const CAPS: Record<string, [string, string]> = {
  why: ['A decode step reads every weight of the model to produce one token per sequence. Reading them takes the same time whether one sequence or dozens share the step, so a batch multiplies tokens per second almost for free, until the arithmetic catches up with memory.', 'step time ≈ max(weights / bandwidth, FLOPs / peak)'],
  static: ['With static batching the server starts a batch and runs it until its longest request is done. Finished requests keep their slot doing nothing (hatched), and new requests wait for the whole batch to end.', 'a batch lasts as long as its longest request'],
  continuous: ['Continuous (iteration-level) batching, from the Orca paper, reschedules after every step: a finished request leaves at once and a waiting one takes its slot on the next step, with its prompt’s prefill folded into that step.', 'refill free slots after every step'],
  compare: ['Same requests, same four slots. Refilling every step keeps the slots busy with real work, so requests hardly wait to start, finish much sooner on average, and the whole queue is done in fewer steps.', 'waiting ↓ · latency ↓ · throughput ↑'],
  chunked: ['A new request’s prefill can be long: run in one step, its 512 prompt tokens make that step compute-bound and three times as slow, stalling every other request’s next token. Chunked prefill splits it into pieces small enough to hide under the memory-bound step time.', 'prefill in chunks of 128, mixed with decodes'],
}

export function mountContinuousBatching(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'The schedules are a toy simulation; step times are estimates on real sizes.',
      eyebrow: 'Serving · Scheduling',
      title: 'Continuous Batching',
      subtitle: 'refilling the batch after every decode step',
      specs: [
        { label: 'toy server', value: '4 slots · 12 requests' },
        { label: 'steps', value: `${stats(STATIC).steps} static`, real: `${stats(CONT).steps} continuous`, realLabel: '→' },
        { label: 'step time', value: `${WEIGHT_MS.toFixed(1)} ms`, real: 'LLaMA 3 8B · A100', realLabel: '' },
      ],
    },
    size: [1040, 480],
    aria: 'Continuous batching: because a decode step costs about the same for one sequence or many, servers batch requests; refilling free slots after every step instead of waiting for the whole batch cuts waiting time and raises throughput, and long prompts are prefilled in chunks.',
    phases: PHASES, learn: 'batching', tokens: TOKS, compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['continuous', 10],
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { caption, title } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46
  function hatch(x: number, y: number, w: number, h: number, a: number) {
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip()
    ctx.strokeStyle = rgba(C.ink, 0.2 * a); ctx.lineWidth = 1
    for (let d = -h; d < w; d += 5) { ctx.beginPath(); ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); ctx.stroke() }
    ctx.restore()
  }

  /* ---------- scene 1: throughput against batch size ---------- */
  function sceneWhy(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const cx0 = pad + 80, cx1 = W - pad - 280, cy0 = top + 40, cy1 = top + avail - 40, fin = eout(clamp(p / 0.06))
    const lx = (b: number) => lerp(cx0, cx1, Math.log2(b) / Math.log2(512)), Y = (v: number) => lerp(cy1, cy0, Math.log10(v / 50) / Math.log10(40000 / 50))
    title('tokens per second · llama 3 8b on an a100', cx0, cy0 - 18, fin)
    ctx.strokeStyle = rgba(C.ink, 0.25 * fin); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy1); ctx.lineTo(cx1, cy1); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, fin)
    for (const v of [100, 1000, 10000]) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(v.toLocaleString('en-US'), cx0 - 6, Y(v)) }
    for (const b of [1, 4, 16, 64, 256]) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(String(b), lx(b), cy1 + 6) }
    caption('sequences in the batch', (cx0 + cx1) / 2, cy1 + 32, fin)
    const g = eio(clamp((p - 0.08) / 0.5)), tps = (b: number) => (b / stepMs(b)) * 1000
    ctx.strokeStyle = rgba(C.tok[4], fin); ctx.lineWidth = 2; ctx.beginPath()
    for (let s = 0; s <= 90 * g; s++) { const b = Math.pow(2, (s / 90) * 9); if (s) ctx.lineTo(lx(b), Y(tps(b))); else ctx.moveTo(lx(b), Y(tps(b))) }
    ctx.stroke()
    const pts = [1, 32, 128]
    pts.forEach((b, i) => {
      const a = eout(clamp((p - 0.4 - i * 0.1) / 0.1))
      if (a <= 0) return
      ctx.fillStyle = rgba(C.ink, a); ctx.beginPath(); ctx.arc(lx(b), Y(tps(b)), 4, 0, 7); ctx.fill()
      caption(`batch ${b}: ${Math.round(tps(b)).toLocaleString('en-US')} tok/s · ${stepMs(b).toFixed(1)} ms per step`, i === 2 ? lx(b) - 10 : lx(b) + 10, Y(tps(b)) + 20, a, C.ink, i === 2 ? 'right' : 'left')
    })
    const ra = eout(clamp((p - 0.75) / 0.1)), ridge = WEIGHT_MS / FLOP_MS
    if (ra > 0) {
      ctx.setLineDash([3, 4]); ctx.strokeStyle = rgba(C.ink2, ra); ctx.beginPath(); ctx.moveTo(lx(ridge), cy0); ctx.lineTo(lx(ridge), cy1); ctx.stroke(); ctx.setLineDash([])
      caption(`from about ${Math.round(ridge)} sequences`, cx1 + 20, cy0 + 150, ra, C.ink2, 'left')
      caption('the arithmetic, not memory,', cx1 + 20, cy0 + 166, ra, C.ink2, 'left')
      caption('sets the pace', cx1 + 20, cy0 + 182, ra, C.ink2, 'left')
    }
    mk.formula = { segs: [[`weights 16 GB ÷ 2 TB/s = ${WEIGHT_MS.toFixed(1)} ms`, C.ink2], ['     ·     ', C.mute], [`each token 16 GFLOPs ÷ 312 TFLOP/s = ${(FLOP_MS * 1000).toFixed(0)} µs`, C.ink2]], note: 'A step’s time is the larger of the two: reading the weights once, or doing the arithmetic for every token in the batch. Below about 156 sequences the reading dominates, so each extra sequence is nearly free. The KV cache reads, ignored here, grow with the batch too.' }
  }

  /* ---------- scenes 2 and 3: the Gantt charts ---------- */
  function gantt(s: Sched, p: number, label: string) {
    const { W } = stage
    const cw = Math.min(17, (W - pad - 120 - 20) / Math.max(STATIC.grid.length, CONT.grid.length)), x0 = pad + 90, y0 = top + 60, rh = 38
    const shown = Math.floor(eio(clamp((p - 0.05) / 0.8)) * s.grid.length)
    title(label, x0, y0 - 30, 1)
    for (let sl = 0; sl < SLOTS; sl++) {
      const y = y0 + sl * (rh + 8)
      ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, 1); ctx.fillText(`slot ${sl + 1}`, x0 - 10, y + rh / 2)
      ctx.strokeStyle = rgba(C.ink, 0.12); ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y + 0.5, s.grid.length * cw - 1, rh - 1)
      let runStart = 0
      for (let t = 0; t < shown; t++) {
        const id = s.grid[t][sl], x = x0 + t * cw
        if (id >= 0) { ctx.fillStyle = rgba(hue(id), s.start[id] === t ? 0.95 : 0.7); ctx.fillRect(x + 0.5, y + 1, cw - 1, rh - 2) }
        else if (id === -2) hatch(x, y + 1, cw, rh - 2, 1)
        const next = t + 1 < shown ? s.grid[t + 1][sl] : -99
        if (next !== id) {
          if (id >= 0 && (t - runStart + 1) * cw > 22) { ctx.font = F.mono(10.5, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.bg, 1); ctx.fillText(`r${id + 1}`, x0 + ((runStart + t + 1) / 2) * cw, y + rh / 2) }
          runStart = t + 1
        }
      }
    }
    // arrivals and the waiting queue
    const qy = y0 + SLOTS * (rh + 8) + 20, now = Math.max(0, shown - 1)
    ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, 1); ctx.fillText('arrive', x0 - 10, qy + 6)
    for (const q of REQS) {
      if (q.arrive > now) continue
      const x = x0 + q.arrive * cw + cw / 2, waiting = s.start[q.id] > now
      ctx.fillStyle = rgba(hue(q.id), waiting ? 1 : 0.45); ctx.beginPath(); ctx.moveTo(x, qy); ctx.lineTo(x - 5, qy + 10); ctx.lineTo(x + 5, qy + 10); ctx.closePath(); ctx.fill()
      if (s.start[q.id] > q.arrive) { const ly = qy + 16 + (q.id % 4) * 4, to = Math.min(s.start[q.id], now); ctx.strokeStyle = rgba(hue(q.id), 0.7); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, ly); ctx.lineTo(x0 + to * cw + cw / 2, ly); ctx.stroke(); ctx.setLineDash([]) }
    }
    const waiting = REQS.filter((q) => q.arrive <= now && s.start[q.id] > now).length
    ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, 1)
    ctx.fillText(`step ${now + 1} · ${waiting} waiting`, x0, qy + 50)
    caption('bright: the request’s first step (its prefill) · hatched: a finished request still holding its slot · dashed: time spent waiting', x0, qy + 70, 1, C.mute, 'left')
  }
  function sceneStatic(p: number) {
    gantt(STATIC, p, 'static batching · 4 slots')
    const st = stats(STATIC)
    mk.formula = { segs: [['a new batch starts only when every request in the last one is done', C.ink]], note: `Over the run ${Math.round(st.util * 100)}% of the slot-steps do useful work; requests wait ${st.wait.toFixed(1)} steps on average before they start.` }
  }
  function sceneContinuous(p: number) {
    gantt(CONT, p, 'continuous batching · 4 slots')
    const st = stats(CONT)
    mk.formula = { segs: [['after every step: finished requests leave, waiting ones join', C.ink]], note: `Every slot-step with a request in it does useful work; requests wait ${st.wait.toFixed(1)} steps on average. Here a slot sits empty only when nobody is waiting.` }
  }

  /* ---------- scene 4: the comparison ---------- */
  function sceneCompare(p: number) {
    const { W } = stage
    const a = stats(STATIC), b = stats(CONT), x0 = pad + 220, bw = W - pad - x0 - 200
    const rows: [string, number, number, string, (v: number) => string][] = [
      ['steps to serve all 12', a.steps, b.steps, 'fewer is better', (v) => String(v)],
      ['mean wait to start', a.wait, b.wait, 'steps', (v) => v.toFixed(1)],
      ['mean time to finish', a.latency, b.latency, 'steps from arrival', (v) => v.toFixed(1)],
    ]
    title('static · continuous', x0, top + 18, 1)
    rows.forEach(([name, va, vb, unit, f], r) => {
      const al = eout(clamp((p - 0.05 - r * 0.15) / 0.15)), g = eio(clamp((p - 0.08 - r * 0.15) / 0.2)), y = top + 50 + r * 96, max = Math.max(va, vb)
      if (al <= 0) return
      ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, al); ctx.fillText(name, pad, y + 14)
      caption(unit, pad, y + 32, al, C.mute, 'left')
      ;[[va, C.ink2], [vb, C.tok[4]]].forEach(([v, col], i) => {
        const w = ((v as number) / max) * bw * g
        ctx.fillStyle = rgba(col as RGB, 0.85 * al); ctx.fillRect(x0, y + i * 26, Math.max(2, w), 18)
        ctx.font = F.mono(12); ctx.fillStyle = rgba(C.ink, al); ctx.fillText(f(v as number), x0 + w + 8, y + i * 26 + 14)
      })
    })
    const ra = eout(clamp((p - 0.6) / 0.12))
    ctx.font = F.mono(13, 500); ctx.fillStyle = rgba(C.ink, ra); ctx.textAlign = 'left'
    ctx.fillText(`${(a.steps / b.steps).toFixed(1)} × the throughput, ${(a.latency / b.latency).toFixed(1)} × faster on average`, x0, top + 50 + 3 * 96 + 6)
    mk.formula = { segs: [['same 12 requests · same 4 slots', C.ink2]], note: 'The Orca paper reported up to 36.9 × the throughput of FasterTransformer at the same latency for GPT-3 175B; the gain in practice depends on how much request lengths vary. vLLM and every major serving system now schedule this way.' }
  }

  /* ---------- scene 5: chunked prefill ---------- */
  function sceneChunked(p: number) {
    const { W } = stage
    const x0 = pad + 180, sc = (W - pad - x0 - 30) / 60, rows: [string, number[]][] = [
      ['prefill at once', [stepMs(3), stepMs(3), stepMs(512 + 3), stepMs(4), stepMs(4)]],
      ['chunks of 128', [stepMs(3), stepMs(3), stepMs(128 + 3), stepMs(128 + 3), stepMs(128 + 3), stepMs(128 + 3), stepMs(4)]],
    ]
    title('step times · 3 requests decoding, then a 512-token prompt arrives', pad, top + 18, 1)
    rows.forEach(([name, steps], r) => {
      const a = eout(clamp((p - 0.05 - r * 0.3) / 0.15)), y = top + 70 + r * 130
      if (a <= 0) return
      ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(name, pad, y + 22)
      let x = x0
      steps.forEach((ms, i) => {
        const w = ms * sc, big = ms > WEIGHT_MS + 0.5, pre = r === 0 ? i === 2 : i >= 2 && i <= 5
        rr(x, y, w - 3, 34, 4); ctx.fillStyle = rgba(pre ? C.tok[3] : C.tok[0], (big ? 0.9 : 0.6) * a); ctx.fill()
        ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.bg, a); if (w > 34) ctx.fillText(`${ms.toFixed(0)} ms`, x + w / 2, y + 17.5)
        x += w
      })
      const worst = Math.max(...steps)
      caption(`longest gap between two tokens for the running requests: ${worst.toFixed(0)} ms`, x0, y + 58, a, C.ink2, 'left')
    })
    // a key with swatches rather than colour names, which would be wrong under the colour-blind palette
    const ky = top + 70 + 2 * 130 + 6
    ;[[C.tok[0], 'decode-only step'], [C.tok[3], 'step that also carries prompt tokens']].forEach(([col, lab], i) => {
      const kx = x0 + i * 220
      rr(kx, ky - 9, 14, 12, 3); ctx.fillStyle = rgba(col as RGB, 0.8); ctx.fill()
      caption(lab as string, kx + 22, ky + 1, 1, C.mute, 'left')
    })
    mk.formula = { segs: [['step = max(8.0 ms to read the weights, tokens × 51 µs)', C.ink]], note: 'A 128-token chunk plus three decodes needs 6.7 ms of arithmetic, less than the 8 ms the weight read takes anyway, so it rides along for free. The new request’s first token comes a little later (4 steps), but nobody else stalls. Estimates for LLaMA 3 8B on an A100, ignoring attention.' }
  }

  return { why: sceneWhy, static: sceneStatic, continuous: sceneContinuous, compare: sceneCompare, chunked: sceneChunked }
}
