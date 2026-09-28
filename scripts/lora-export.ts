/*
 * How low-rank is a real fine-tuning update? Reads the same weight matrices from Qwen3-1.7B-Base and Qwen3-1.7B
 * (the base model after Qwen's post-training), takes their difference ΔW, and finds the largest singular values of
 * ΔW and of the base weight W by a randomized SVD, so the page can show how much of each is captured by rank r.
 * Only the needed tensors are read from the safetensors files. The browser never runs a model.
 *
 *   node scripts/lora-export.ts            (Node 23+, which runs TypeScript directly)
 *
 * Needs ~/.cache/token-trails/qwen3/ and ~/.cache/token-trails/qwen3base/ (see scripts/posttrain-export.ts).
 * Writes src/data/lora.json.
 */
import { closeSync, existsSync, openSync, readFileSync, readSync, writeFileSync } from 'node:fs'
import { frob2, topSingular, type Mat } from './svd.ts'

const HOME = `${process.env.HOME}/.cache/token-trails`
/** One bf16 tensor from a safetensors checkpoint (sharded or not), as float64 rows. */
function tensor(dir: string, name: string) {
  const file = existsSync(`${dir}/model.safetensors.index.json`) ? JSON.parse(readFileSync(`${dir}/model.safetensors.index.json`, 'utf8')).weight_map[name] : 'model.safetensors'
  const fd = openSync(`${dir}/${file}`, 'r'), lb = Buffer.alloc(8)
  readSync(fd, lb, 0, 8, 0)
  const hl = Number(lb.readBigUInt64LE(0)), hb = Buffer.alloc(hl)
  readSync(fd, hb, 0, hl, 8)
  const t = JSON.parse(hb.toString('utf8'))[name], [a, b] = t.data_offsets as [number, number], raw = Buffer.alloc(b - a)
  readSync(fd, raw, 0, b - a, 8 + hl + a)
  closeSync(fd)
  const u16 = new Uint16Array(raw.buffer, raw.byteOffset, (b - a) / 2), u32 = new Uint32Array(u16.length), f = new Float32Array(u32.buffer)
  for (let i = 0; i < u16.length; i++) u32[i] = u16[i] << 16
  const [m, n] = t.shape as [number, number]
  return { m, n, a: Float64Array.from(f) }
}

const K = 64, RANKS = [1, 2, 4, 8, 16, 32, 64]
const LAYER = 14, NAMES = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.down_proj']
const r4 = (v: number) => Number(v.toPrecision(4))
const mats = NAMES.map((nm) => {
  const name = `model.layers.${LAYER}.${nm}.weight`, t0 = Date.now()
  const W = tensor(`${HOME}/qwen3base`, name), Wt = tensor(`${HOME}/qwen3`, name)
  const dW: Mat = { m: W.m, n: W.n, a: Wt.a.map((v, i) => v - W.a[i]) }
  const fd = frob2(dW.a), fw = frob2(W.a), sd = topSingular(dW, K), sw = topSingular(W, K)
  const cum = (s: number[], tot: number) => RANKS.map((r) => r4(s.slice(0, r).reduce((a, v) => a + v * v, 0) / tot))
  const out = { name: nm, shape: [W.m, W.n], relChange: r4(Math.sqrt(fd / fw)), delta: { sigma: sd.map(r4), captured: cum(sd, fd) }, weight: { sigma: sw.map(r4), captured: cum(sw, fw) } }
  console.log(`${nm} ${W.m}×${W.n}: ‖ΔW‖/‖W‖ ${out.relChange} · ΔW captured ${out.delta.captured.join(' ')} · W captured ${out.weight.captured.join(' ')} (${((Date.now() - t0) / 1000).toFixed(0)} s)`)
  return out
})
writeFileSync(new URL('../src/data/lora.json', import.meta.url), JSON.stringify({ base: 'Qwen3-1.7B-Base', tuned: 'Qwen3-1.7B', layer: LAYER, k: K, ranks: RANKS, mats }))
console.log('wrote src/data/lora.json')
