import { F, fillRich, rr, type TokLike } from '../../core/draw'
import { C, rgba, type RGB } from '../../core/theme'
import { clamp, eio, eout, lerp, rng } from '../../core/util'
import { mountExhibit, type Env } from '../kit'
import type { Nav } from '../registry'

/*
 * PagedAttention (vLLM): the KV cache stored in fixed-size blocks allocated on demand and found
 * through a block table, instead of one contiguous reservation per request. Everything here is a toy
 * simulation (256 token slots, blocks of 4, requests up to 64 tokens), run the same way for both
 * allocators; the percentages quoted from the paper are labelled as such.
 */

const PHASES = [
  { id: 'waste', name: 'Reserving for the worst case', short: 'Contiguous', dur: 10 },
  { id: 'blocks', name: 'Blocks on demand', short: 'Blocks', dur: 12 },
  { id: 'kernel', name: 'Attention across blocks', short: 'Kernel', dur: 8 },
  { id: 'share', name: 'Sharing and copy-on-write', short: 'Sharing', dur: 11 },
  { id: 'sim', name: 'More requests at once', short: 'Throughput', dur: 9 },
]

const CAP = 256, BS = 4, NBLK = CAP / BS, MAXLEN = 64, NREQ = 16
interface Req { id: number; prompt: number; out: number; arrive: number; gen: number; start: number; end: number; table: number[] }
interface Snap { run: { id: number; len: number; table: number[] }[]; waiting: number[]; done: number; used: number; held: number }
/** Requests are the same for both allocators: a seeded list of prompt and output lengths, arriving every 1.5 steps. */
const REQS = (() => { const r = rng(1); return Array.from({ length: NREQ }, (_, i) => ({ id: i, prompt: 4 + Math.floor(r() * 14), out: 4 + Math.floor(r() * 34), arrive: Math.floor(i * 1.5) })) })()
function simulate(mode: 'contig' | 'paged') {
  const R: Req[] = REQS.map((q) => ({ ...q, gen: 0, start: -1, end: -1, table: [] }))
  // free blocks in a scattered order, so a request's blocks are not neighbours
  const order = (() => { const r = rng(7), a = Array.from({ length: NBLK }, (_, i) => i); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] } return a })()
  const free = [...order], rows = [0, 1, 2, 3], running: Req[] = [], snaps: Snap[] = []
  for (let step = 0; R.some((q) => q.end < 0) && step < 400; step++) {
    for (const q of R) {
      if (q.start >= 0 || q.arrive > step) continue
      if (mode === 'contig') { if (!rows.length) break; q.table = [rows.shift()!]; q.start = step; running.push(q) }
      else { const need = Math.ceil((q.prompt + 1) / BS); if (free.length - need < running.length) break; q.table = free.splice(0, need); q.start = step; running.push(q) }
    }
    for (const q of [...running]) {
      const len = q.prompt + q.gen + 1
      if (mode === 'paged' && Math.ceil(len / BS) > q.table.length) { if (!free.length) continue; q.table.push(free.shift()!) }
      q.gen++
      if (q.gen >= q.out) { q.end = step; running.splice(running.indexOf(q), 1); if (mode === 'contig') rows.push(q.table[0]); else free.unshift(...q.table) }
    }
    const used = running.reduce((s, q) => s + q.prompt + q.gen, 0)
    snaps.push({ run: running.map((q) => ({ id: q.id, len: q.prompt + q.gen, table: q.table.slice() })), waiting: R.filter((q) => q.start < 0 && q.arrive <= step).map((q) => q.id), done: R.filter((q) => q.end >= 0).length, used, held: mode === 'contig' ? running.length * MAXLEN : running.reduce((s, q) => s + q.table.length * BS, 0) })
  }
  return snaps
}
const CONTIG = simulate('contig'), PAGED = simulate('paged')
const util = (s: Snap[]) => s.reduce((a, b) => a + b.used, 0) / s.reduce((a, b) => a + b.held, 0)
const meanRun = (s: Snap[]) => s.reduce((a, b) => a + b.run.length, 0) / s.length
const TOKS: TokLike[] = REQS.map((q) => ({ text: `r${q.id + 1}`, c: q.id }))

const COMPARE: Record<string, [string, string]> = {
  waste: ['the KV cache’s size', 'serving/kv-cache?phase=size'], blocks: ['the KV cache', 'serving/kv-cache?phase=step'], kernel: ['GPT-2’s attention', 'anatomy/attention?phase=scores'],
  share: ['GPT-2’s sampling', 'anatomy/unembed?phase=sample'], sim: ['decode’s memory limit', 'serving/kv-cache?phase=bound'],
}

const CAPS: Record<string, [string, string]> = {
  waste: ['A server does not know how long each answer will be, so a simple one reserves room for the longest possible answer in one contiguous stretch of memory. Most of that room is never used, and new requests wait even though the memory is mostly empty. The vLLM paper measured only 20 to 38% of KV-cache memory holding tokens in such systems.', 'reserve max length per request'],
  blocks: ['PagedAttention splits the cache into small fixed-size blocks (16 tokens in vLLM, 4 here) and hands a request a new block only when its last one is full, from anywhere in memory. A block table per request maps its blocks in order to where they are, like virtual memory pages. At most one partly filled block per request is wasted.', 'logical block → physical block'],
  kernel: ['The attention kernel follows the block table: for each of the request’s blocks it fetches the keys and values from wherever that block lives, and computes the same attention as before. Only the memory layout changed.', 'for each block: K, V ← memory[table[b]]'],
  share: ['Two samples from one prompt can share the prompt’s blocks; a reference count says how many sequences use each block. When a sample must write into a shared block that is partly filled, it first copies it (copy-on-write), just like forked processes in an operating system.', 'shared blocks · ref counts · copy-on-write'],
  sim: ['The same 16 requests, the same memory. Paging fits more requests into the batch at once, so the whole queue finishes sooner; in the paper vLLM served 2 to 4 times more requests per second than earlier systems at the same latency.', 'more sequences per batch → more tokens per second'],
}

export function mountPagedAttention(root: HTMLElement, nav: Nav): () => void {
  return mountExhibit(root, nav, {
    frame: {
      formulaHint: 'The memory and block tables are a toy simulation; step through with the timeline.',
      eyebrow: 'Serving · Memory',
      title: 'PagedAttention',
      subtitle: 'the KV cache in blocks, as in vLLM',
      specs: [
        { label: 'block', value: '4 tokens here', real: 'vLLM 16', realLabel: '' },
        { label: 'memory', value: '256 token slots' },
        { label: 'requests', value: '16, up to 64 tokens' },
        { label: 'used, contiguous', value: `${Math.round(util(CONTIG) * 100)}%` },
        { label: 'used, paged', value: `${Math.round(util(PAGED) * 100)}%` },
      ],
    },
    size: [1040, 480],
    aria: 'PagedAttention: the KV cache is split into small blocks given out on demand and found through a block table, so far less memory is reserved but unused, blocks can be shared between sequences, and more requests fit in a batch.',
    phases: PHASES, learn: 'pagedattention', tokens: TOKS, compare: COMPARE, compareLabel: 'Related', caps: CAPS,
    still: ['blocks', 11],
    scenes,
  })
}

function scenes({ stage, ctx, mk, k }: Env) {
  const { caption, title, arrow } = k
  const hue = (i: number): RGB => C.tok[i % 7]
  const pad = 36, top = 56, bot = 46
  function hatch(x: number, y: number, w: number, h: number, a: number) {
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip()
    ctx.strokeStyle = rgba(C.ink, 0.16 * a); ctx.lineWidth = 1
    for (let d = -h; d < w; d += 6) { ctx.beginPath(); ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); ctx.stroke() }
    ctx.restore()
  }
  function queue(snap: Snap, x: number, y: number, a: number) {
    title('waiting', x, y, a)
    snap.waiting.slice(0, 12).forEach((id, n) => { rr(x + n * 30, y + 10, 26, 18, 4); ctx.fillStyle = rgba(hue(id), 0.25 * a); ctx.fill(); ctx.strokeStyle = rgba(hue(id), 0.8 * a); ctx.lineWidth = 1; ctx.stroke(); ctx.font = F.mono(10); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, a); ctx.fillText(`r${id + 1}`, x + n * 30 + 13, y + 19.5) })
    if (!snap.waiting.length) caption('none', x, y + 24, a, C.mute, 'left')
  }
  function stats(snap: Snap, x: number, y: number, a: number) {
    ctx.font = F.mono(13, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, a)
    ctx.fillText(`${snap.run.length} running · ${snap.done} done`, x, y)
    ctx.font = F.mono(12); ctx.fillStyle = rgba(C.ink2, a)
    ctx.fillText(`tokens stored ${snap.used} · held ${snap.held} (${snap.held ? Math.round((snap.used / snap.held) * 100) : 0}%)`, x, y + 22)
  }

  /* ---------- scene 1: contiguous reservations ---------- */
  function sceneWaste(p: number) {
    const { W } = stage
    const si = Math.min(CONTIG.length - 1, Math.floor(eio(clamp((p - 0.05) / 0.85)) * 40)), snap = CONTIG[si]
    const x0 = pad + 70, cw = Math.min(13, (W - x0 - pad - 20) / MAXLEN), rh = 40, y0 = top + 40
    title(`gpu kv memory · step ${si + 1}`, x0, y0 - 16, 1)
    for (let row = 0; row < CAP / MAXLEN; row++) {
      const y = y0 + row * (rh + 18), owner = snap.run.find((q) => q.table[0] === row)
      ctx.font = F.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.mute, 1); ctx.fillText(owner ? `r${owner.id + 1}` : 'free', x0 - 10, y + rh / 2)
      ctx.strokeStyle = rgba(C.ink, 0.2); ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y + 0.5, MAXLEN * cw - 1, rh - 1)
      if (!owner) continue
      ctx.fillStyle = rgba(hue(owner.id), 0.8); ctx.fillRect(x0 + 1, y + 1, owner.len * cw - 1, rh - 2)
      hatch(x0 + owner.len * cw, y + 1, (MAXLEN - owner.len) * cw - 1, rh - 2, 1)
      ctx.strokeStyle = rgba(hue(owner.id), 0.9); ctx.strokeRect(x0 + 0.5, y + 0.5, MAXLEN * cw - 1, rh - 1)
    }
    const by = y0 + 4 * (rh + 18) + 16
    queue(snap, x0, by, 1)
    stats(snap, x0 + 420, by + 10, 1)
    caption('filled: tokens stored · hatched: reserved but empty', x0, by + 62, 1, C.mute, 'left')
    mk.formula = { segs: [['each request reserves 64 slots', C.ink2], ['     ·     ', C.mute], [`over the whole run, ${Math.round(util(CONTIG) * 100)}% of the reserved slots hold a token`, C.ink]], note: 'Only 4 requests fit, whatever their lengths, so the rest wait in the queue. Real systems also lose memory to gaps between reservations (external fragmentation); here every reservation is the same size, so only the reserved-but-empty part shows.' }
  }

  /* ---------- scene 2: blocks and block tables ---------- */
  function drawBlocks(snap: Snap, x0: number, y0: number, bw: number, bh: number, a: number, hl = -1) {
    const owner = new Map<number, { id: number; fill: number; logical: number }>()
    for (const q of snap.run) q.table.forEach((b, n) => owner.set(b, { id: q.id, fill: Math.min(BS, q.len - n * BS), logical: n }))
    for (let b = 0; b < NBLK; b++) {
      const x = x0 + (b % 16) * (bw + 4), y = y0 + Math.floor(b / 16) * (bh + 6), o = owner.get(b)
      ctx.strokeStyle = rgba(C.ink, 0.18 * a); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, bw - 1, bh - 1)
      if (!o) continue
      const dim = hl >= 0 && o.id !== hl ? 0.3 : 1
      for (let s = 0; s < BS; s++) {
        const sx = x + 1 + s * ((bw - 2) / BS)
        if (s < o.fill) { ctx.fillStyle = rgba(hue(o.id), 0.8 * a * dim); ctx.fillRect(sx, y + 1, (bw - 2) / BS - 1, bh - 2) }
        else hatch(sx, y + 1, (bw - 2) / BS - 1, bh - 2, a * dim)
      }
      ctx.strokeStyle = rgba(hue(o.id), 0.9 * a * dim); ctx.strokeRect(x + 0.5, y + 0.5, bw - 1, bh - 1)
    }
    return (b: number) => [x0 + (b % 16) * (bw + 4), y0 + Math.floor(b / 16) * (bh + 6)] as [number, number]
  }
  function sceneBlocks(p: number) {
    const { W } = stage
    const si = Math.min(PAGED.length - 1, Math.floor(eio(clamp((p - 0.05) / 0.85)) * 40)), snap = PAGED[si]
    const bw = 40, bh = 30, x0 = W - pad - 16 * (bw + 4), y0 = top + 40
    title(`gpu kv memory · 64 blocks of 4 · step ${si + 1}`, x0, y0 - 16, 1)
    drawBlocks(snap, x0, y0, bw, bh, 1)
    // block tables of the running requests
    const tx = pad
    title('block tables', tx, y0 - 16, 1)
    snap.run.slice(0, 9).forEach((q, n) => {
      const y = y0 + n * 26 + 10
      ctx.font = F.mono(11, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(hue(q.id), 1); ctx.fillText(`r${q.id + 1}`, tx, y)
      ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.ink2, 1); ctx.fillText(q.table.map((b) => String(b)).join(' ').slice(0, 26), tx + 34, y)
    })
    const by = y0 + 4 * (bh + 6) + 20
    queue(snap, x0, by, 1)
    stats(snap, x0 + 420, by + 10, 1)
    caption('block numbers in each table: where that request’s blocks are, in order', tx, y0 + 9 * 26 + 30, 1, C.mute, 'left')
    mk.formula = { segs: [['token t of a request lives in block table[⌊t / 4⌋], slot t mod 4', C.ink]], note: `A request gets a block only when its last one fills up, from anywhere in memory. Over the whole run ${Math.round(util(PAGED) * 100)}% of the held slots hold a token; the rest are the unfilled ends of each request’s last block.` }
  }

  /* ---------- scene 3: the kernel follows the table ---------- */
  function sceneKernel(p: number) {
    const { W } = stage
    const snap = PAGED[24], q = snap.run.reduce((a, b) => (b.table.length > a.table.length ? b : a))
    const bw = 40, bh = 30, x0 = W - pad - 16 * (bw + 4), y0 = top + 40
    const pos = drawBlocks(snap, x0, y0, bw, bh, 1, q.id)
    const qx = pad + 40, qy = top + 262, n = q.table.length, reach = eio(clamp((p - 0.1) / 0.6)) * n, tx = pad + 110, ty = top + 250
    rr(qx - 30, qy - 14, 60, 28, 6); ctx.fillStyle = rgba(hue(q.id), 0.3); ctx.fill(); ctx.strokeStyle = rgba(hue(q.id), 1); ctx.lineWidth = 1.4; ctx.stroke()
    ctx.font = F.mono(11, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, 1); ctx.fillText(`q · r${q.id + 1}`, qx, qy + 0.5)
    title(`block table of r${q.id + 1}`, tx, ty + 66, 1)
    q.table.forEach((b, i) => {
      const x = tx + i * 44, y = ty, on = i < reach
      rr(x, y, 38, 26, 4); ctx.fillStyle = rgba(hue(q.id), on ? 0.35 : 0.08); ctx.fill(); ctx.strokeStyle = rgba(hue(q.id), 0.8); ctx.lineWidth = 1; ctx.stroke()
      ctx.font = F.mono(11); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, 1); ctx.fillText(String(b), x + 19, y + 13.5)
      ctx.font = F.mono(10); ctx.fillStyle = rgba(C.mute, 1); ctx.fillText(`#${i}`, x + 19, y + 36)
      if (i < reach) {
        const [bx, byy] = pos(b), a = clamp(reach - i)
        ctx.strokeStyle = rgba(hue(q.id), 0.7 * a); ctx.lineWidth = 1.3
        ctx.beginPath(); ctx.moveTo(x + 19, y - 2); ctx.bezierCurveTo(x + 19, y - 60, bx + bw / 2, byy + bh + 60, bx + bw / 2, byy + bh); ctx.stroke()
      }
    })
    caption(`${q.len} tokens in ${n} blocks, scattered through memory`, tx, ty + 88, 1, C.ink2, 'left')
    mk.formula = { segs: [['for b in 0 … n−1:', C.ink2], ['  K_b, V_b ← memory[table[b]]', C.ink], ['   s_b = q · K_bᵀ', C.ink2], ['   then softmax over all s, · V as usual', C.ink2]], note: 'The kernel gathers each block through the table while it computes; nothing is copied into one contiguous array first. Block size trades waste (a partly empty last block) against the overhead of more, smaller reads; vLLM’s default is 16 tokens.' }
  }

  /* ---------- scene 4: sharing and copy-on-write ---------- */
  function sceneShare(p: number) {
    const { W } = stage
    // one 10-token prompt in blocks 7, 2, 11 (the last one half full); two samples continue from it
    const bw = 56, bh = 34, gx = W / 2 - 40, gy = top + 60, cols = 8
    const stage3 = p < 0.3 ? 0 : p < 0.62 ? 1 : 2
    type B = { owner: number[]; fill: number[]; ref: number }
    const blocks = new Map<number, B>()
    const set = (b: number, owner: number[], fill: number[], ref: number) => blocks.set(b, { owner, fill, ref })
    // fill[i] per slot: -1 empty, 0 prompt token, 1 sample 1 token, 2 sample 2 token
    set(7, [0], [0, 0, 0, 0], 2); set(2, [0], [0, 0, 0, 0], 2)
    let t1 = [7, 2, 11], t2 = [7, 2, 11]
    if (stage3 === 0) set(11, [0], [0, 0, -1, -1], 2)
    else {
      set(11, [0], [0, 0, stage3 === 2 ? 2 : -1, -1], 1)
      set(5, [1], [0, 0, 1, -1], 1)
      t1 = [7, 2, 5]
    }
    const pos = (b: number) => [gx + (b % cols) * (bw + 8), gy + Math.floor(b / cols) * (bh + 30)] as [number, number]
    title('memory', gx, gy - 18, 1)
    for (let b = 0; b < 16; b++) {
      const [x, y] = pos(b), bl = blocks.get(b)
      ctx.strokeStyle = rgba(C.ink, 0.18); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, bw - 1, bh - 1)
      if (!bl) continue
      bl.fill.forEach((f, s) => {
        const sx = x + 1 + s * ((bw - 2) / 4)
        if (f < 0) { hatch(sx, y + 1, (bw - 2) / 4 - 1, bh - 2, 1); return }
        ctx.fillStyle = rgba(f === 0 ? C.ink : hue(f), f === 0 ? 0.45 : 0.85); ctx.fillRect(sx, y + 1, (bw - 2) / 4 - 1, bh - 2)
      })
      ctx.strokeStyle = rgba(C.ink, 0.8); ctx.strokeRect(x + 0.5, y + 0.5, bw - 1, bh - 1)
      ctx.font = F.mono(10.5); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = rgba(bl.ref > 1 ? C.ink : C.mute, 1)
      ctx.fillText(`#${b} · ref ${bl.ref}`, x + bw / 2, y + bh + 4)
    }
    // the two block tables
    const tables: [string, number[], number][] = [['sample 1', t1, 1], ['sample 2', t2, 2]]
    tables.forEach(([name, t, h], n) => {
      const y = top + 70 + n * 110
      ctx.font = F.mono(12, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(hue(h), 1); ctx.fillText(name, pad, y)
      t.forEach((b, i) => {
        const x = pad + i * 50, yy = y + 12
        rr(x, yy, 42, 26, 4); ctx.fillStyle = rgba(hue(h), 0.15); ctx.fill(); ctx.strokeStyle = rgba(hue(h), 0.8); ctx.lineWidth = 1; ctx.stroke()
        ctx.font = F.mono(11); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(C.ink, 1); ctx.fillText(String(b), x + 21, yy + 13.5)
        const [bx, by] = pos(b)
        ctx.strokeStyle = rgba(hue(h), 0.35); ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(x + 42, yy + 13); ctx.bezierCurveTo(x + 150, yy + 13, bx - 60, by + bh / 2, bx, by + bh / 2 + (h === 1 ? -4 : 4)); ctx.stroke()
      })
    })
    const notes = ['both samples point at the prompt’s 3 blocks: stored once, reference count 2', 'sample 1 writes its first token: block 11 is shared, so it is copied to block 5 first (copy-on-write)', 'sample 2 now owns block 11 alone (ref 1) and writes into it in place']
    ctx.font = F.mono(12); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink, 1)
    ctx.fillText(notes[stage3], pad, top + 330)
    caption('grey: prompt tokens · coloured: each sample’s own tokens · hatched: empty slots', pad, top + 352, 1, C.mute, 'left')
    if (stage3 === 1) arrow([[pos(11)[0] + bw / 2, pos(11)[1] + bh + 20], [pos(5)[0] + bw / 2, pos(5)[1] + bh + 20]], 1, true)
    mk.formula = { segs: [['write into block b:', C.ink2], ['  if ref(b) > 1: copy b to a new block, ref(b) −= 1, point the table there', C.ink], ['  then write', C.ink2]], note: 'Parallel sampling and beam search keep many sequences with a common prefix; with sharing, the prompt’s KV cache is stored once. The vLLM paper reports memory savings of up to 55% for beam search this way.' }
  }

  /* ---------- scene 5: the simulation, both ways ---------- */
  function sceneSim(p: number) {
    const { W, H } = stage, avail = H - top - bot
    const cx0 = pad + 60, cx1 = W - pad - 260, cy0 = top + 40, cy1 = top + avail - 50, maxT = Math.max(CONTIG.length, PAGED.length), maxR = 2 * Math.ceil(Math.max(...PAGED.map((s) => s.run.length)) / 2)
    const X = (t: number) => lerp(cx0, cx1, t / maxT), Y = (v: number) => lerp(cy1, cy0, v / maxR), g = eio(clamp((p - 0.06) / 0.6))
    title('requests running at each step', cx0, cy0 - 22, 1)
    ctx.strokeStyle = rgba(C.ink, 0.25); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy1); ctx.lineTo(cx1, cy1); ctx.stroke()
    ctx.font = F.mono(10.5); ctx.fillStyle = rgba(C.mute, 1)
    for (let v = 0; v <= maxR; v += 2) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(v), cx0 - 6, Y(v)) }
    for (const t of [0, 20, 40, 60, 80]) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(String(t), X(t), cy1 + 6) }
    caption('step', (cx0 + cx1) / 2, cy1 + 32, 1)
    const line = (s: Snap[], col: RGB, lab: string) => {
      const n = Math.floor(g * s.length)
      ctx.strokeStyle = rgba(col, 1); ctx.lineWidth = 2; ctx.beginPath()
      for (let t = 0; t < n; t++) { const x = X(t), y = Y(s[t].run.length); if (t) ctx.lineTo(x, y); else ctx.moveTo(x, y) }
      ctx.stroke()
      if (n >= s.length) { ctx.fillStyle = rgba(col, 1); ctx.beginPath(); ctx.arc(X(s.length), cy1, 4, 0, 7); ctx.fill(); caption(`${lab}: done at step ${s.length}`, X(s.length) + 8, cy1 - (s === PAGED ? 40 : 12), 1, col, 'left') }
    }
    line(CONTIG, C.ink2, 'contiguous')
    line(PAGED, C.tok[4], 'paged')
    const sa = eout(clamp((p - 0.7) / 0.12)), sx = cx1 + 30
    if (sa > 0) {
      const rows: [string, string, string][] = [['mean batch', meanRun(CONTIG).toFixed(1), meanRun(PAGED).toFixed(1)], ['memory used', `${Math.round(util(CONTIG) * 100)}%`, `${Math.round(util(PAGED) * 100)}%`], ['steps', String(CONTIG.length), String(PAGED.length)]]
      title('contiguous · paged', sx, cy0 - 16, sa)
      rows.forEach(([n, a, b], i) => {
        const y = cy0 + 20 + i * 44
        caption(n, sx, y, sa, C.mute, 'left')
        ctx.font = F.mono(14, 500); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = rgba(C.ink2, sa); ctx.fillText(a, sx, y + 20)
        ctx.fillStyle = rgba(C.tok[4], sa); ctx.fillText(b, sx + 100, y + 20)
      })
      ctx.font = F.small; ctx.fillStyle = rgba(C.ink, sa); fillRich(`${(CONTIG.length / PAGED.length).toFixed(1)} × the throughput`, sx, cy0 + 170)
    }
    mk.formula = { segs: [['same requests, same memory:', C.ink2], [`  ${CONTIG.length} steps contiguous · ${PAGED.length} paged`, C.ink]], note: 'Decoding is memory-bound, so a bigger batch costs little extra time per step: fitting more requests at once raises tokens per second. The toy admits a request only while a few free blocks remain for the running ones to grow into.' }
  }

  return { waste: sceneWaste, blocks: sceneBlocks, kernel: sceneKernel, share: sceneShare, sim: sceneSim }
}
