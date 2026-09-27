// Turns the single-file build into Artifact page content: the Artifact host supplies
// <!doctype>, <html>, <head>, <body>, charset and viewport, and scans the first 8KB for <title>.
import { readFileSync, writeFileSync } from 'node:fs'

const file = 'dist-artifact/index.html'
let html = readFileSync(file, 'utf8')
const title = html.match(/<title>[\s\S]*?<\/title>/)?.[0] ?? '<title>Token Trails</title>'
html = html
  .replace(/<title>[\s\S]*?<\/title>/, '')
  .replace(/<!doctype html>/i, '')
  .replace(/<\/?html[^>]*>/gi, '')
  .replace(/<\/?head>/gi, '')
  .replace(/<\/?body[^>]*>/gi, '')
  .replace(/<meta charset[^>]*>/i, '')
  .replace(/<meta name="viewport"[^>]*>/i, '')
writeFileSync(file, `${title}\n${html.trim()}\n`)
console.log(`artifact page: ${file} (${(html.length / 1024).toFixed(0)} KB)`)
