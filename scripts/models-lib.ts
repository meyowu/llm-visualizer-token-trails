/*
 * Fetching model configs and checkpoint layouts from the Hugging Face Hub, for scripts/models-export.ts.
 * Only config.json, the shard index and each shard's safetensors header are read (the header lists every
 * tensor's name, dtype and shape); no weights are downloaded. Everything is cached in
 * ~/.cache/token-trails/models/<repo>/.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'

const ROOT = `${process.env.HOME}/.cache/token-trails/models`
const HUB = 'https://huggingface.co'

export interface TensorInfo { dtype: string; shape: number[] }
export type Layout = Record<string, TensorInfo>

const dirOf = (repo: string) => { const d = `${ROOT}/${repo.replace('/', '__')}`; mkdirSync(d, { recursive: true }); return d }

async function get(url: string, headers: Record<string, string> = {}): Promise<Response> {
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(url, { headers, redirect: 'follow' })
      if (r.status === 429 || r.status >= 500) throw new Error(`HTTP ${r.status}`)
      return r
    } catch (e) {
      if (attempt >= 4) throw new Error(`${url}: ${e}`)
      await new Promise((ok) => setTimeout(ok, 1000 * 2 ** attempt))
    }
  }
}

/** A JSON file from the repo, cached; null when the repo has no such file. */
export async function repoJson<T>(repo: string, file: string): Promise<T | null> {
  const path = `${dirOf(repo)}/${file.replace(/\//g, '__')}`
  if (existsSync(path)) { const s = readFileSync(path, 'utf8'); return s === 'null' ? null : JSON.parse(s) }
  const r = await get(`${HUB}/${repo}/resolve/main/${file}`)
  if (r.status === 404) { writeFileSync(path, 'null'); return null }
  if (!r.ok) throw new Error(`${repo}/${file}: HTTP ${r.status}`)
  const text = await r.text()
  writeFileSync(path, text)
  return JSON.parse(text)
}

async function range(url: string, a: number, b: number): Promise<Buffer> {
  const r = await get(url, { Range: `bytes=${a}-${b}` })
  if (r.status !== 206) throw new Error(`${url} ${a}-${b}: HTTP ${r.status}`)
  return Buffer.from(await r.arrayBuffer())
}

/** The tensors of one safetensors file: its header only, read with two range requests. */
async function header(repo: string, file: string): Promise<Layout> {
  const path = `${dirOf(repo)}/${file.replace(/\//g, '__')}.header.json`
  if (existsSync(path)) return JSON.parse(readFileSync(path, 'utf8'))
  const url = `${HUB}/${repo}/resolve/main/${file}`
  const n = Number((await range(url, 0, 7)).readBigUInt64LE(0))
  const h = JSON.parse((await range(url, 8, 8 + n - 1)).toString('utf8'))
  delete h.__metadata__
  const out: Layout = {}
  for (const [k, v] of Object.entries(h) as [string, { dtype: string; shape: number[] }][]) out[k] = { dtype: v.dtype, shape: v.shape }
  writeFileSync(path, JSON.stringify(out))
  return out
}

/** Every tensor of the checkpoint, across its shards. */
export async function layout(repo: string): Promise<Layout> {
  const index = await repoJson<{ weight_map: Record<string, string> }>(repo, 'model.safetensors.index.json')
  const files = index ? [...new Set(Object.values(index.weight_map))] : ['model.safetensors']
  const out: Layout = {}
  // a few shards at a time
  for (let i = 0; i < files.length; i += 8) {
    const parts = await Promise.all(files.slice(i, i + 8).map((f) => header(repo, f)))
    for (const p of parts) Object.assign(out, p)
  }
  return out
}

/** The Hub's own count of the checkpoint's parameters, by dtype (includes any quantization scales). */
export async function hubCount(repo: string): Promise<number | null> {
  const path = `${dirOf(repo)}/hub-count.json`
  if (existsSync(path)) return JSON.parse(readFileSync(path, 'utf8'))
  const r = await get(`${HUB}/api/models/${repo}?expand[]=safetensors`)
  const j = r.ok ? ((await r.json()) as { safetensors?: { total?: number } }) : {}
  const n = j.safetensors?.total ?? null
  writeFileSync(path, JSON.stringify(n))
  return n
}
