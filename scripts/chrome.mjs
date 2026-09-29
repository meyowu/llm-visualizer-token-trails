/*
 * A headless Chrome for build scripts, driven over the DevTools protocol through a pipe, so the build needs no
 * browser package: only a Chrome or Chromium binary (CHROME, else the usual install paths; GitHub's Ubuntu runners
 * have Google Chrome). Just what scripts/pages.mjs needs: open a page, wait for its network to go quiet, run a
 * function in it, and collect the errors it throws or logs.
 */
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const CANDIDATES = {
  linux: ['/opt/google/chrome/chrome', '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser', '/snap/bin/chromium',
    ...(process.env.PLAYWRIGHT_BROWSERS_PATH ? [path.join(process.env.PLAYWRIGHT_BROWSERS_PATH, 'chromium')] : [])],
  darwin: ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/Applications/Chromium.app/Contents/MacOS/Chromium'],
  win32: [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)'], process.env.LOCALAPPDATA].filter(Boolean).map((d) => path.join(d, 'Google/Chrome/Application/chrome.exe')),
}

export function findChrome() {
  const exe = process.env.CHROME || (CANDIDATES[process.platform] ?? []).find((p) => fs.existsSync(p))
  if (!exe) throw new Error('No Chrome or Chromium found; set CHROME to a browser binary.')
  return exe
}

/** Launch headless Chrome; `block` lists URL patterns (with *) that are never fetched, such as analytics. */
export async function launch({ block = [] } = {}) {
  const exe = findChrome(), dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tt-chrome-'))
  const proc = spawn(exe, [
    '--headless', '--remote-debugging-pipe', `--user-data-dir=${dir}`, '--no-sandbox', '--disable-dev-shm-usage',
    // quiet (no background calls, no preconnects to a search engine), and tabs in the background run at full speed
    '--no-first-run', '--no-default-browser-check', '--disable-background-networking', '--disable-component-update',
    '--disable-client-side-phishing-detection', '--disable-default-apps', '--disable-domain-reliability', '--no-pings',
    '--disable-breakpad', '--metrics-recording-only', '--disable-features=Translate,MediaRouter,OptimizationHints,PreconnectToSearch',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
    '--disable-extensions', '--disable-sync', '--mute-audio', '--hide-scrollbars', '--window-size=1280,860', 'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] })
  // Chrome reads commands from fd 3 and writes replies and events to fd 4, each message ending in a NUL byte
  const toChrome = proc.stdio[3], fromChrome = proc.stdio[4]
  let log = '', buf = '', id = 0, dead = null
  proc.stderr.setEncoding('utf8').on('data', (s) => { log = (log + s).slice(-4000) })
  const calls = new Map(), handlers = new Map()
  const fail = (err) => { dead ??= err; for (const c of calls.values()) c.reject(err); calls.clear() }
  proc.on('error', fail)
  toChrome.on('error', fail)
  fromChrome.on('error', fail)
  proc.on('exit', (code) => fail(new Error(`Chrome exited (${code}): ${log.trim().split('\n').slice(-3).join(' | ')}`)))
  fromChrome.setEncoding('utf8').on('data', (s) => {
    buf += s
    for (let i; (i = buf.indexOf('\0')) >= 0; buf = buf.slice(i + 1)) {
      const msg = JSON.parse(buf.slice(0, i))
      if (msg.id) {
        const c = calls.get(msg.id)
        calls.delete(msg.id)
        if (msg.error) c?.reject(new Error(`${c.method}: ${msg.error.message}`)); else c?.resolve(msg.result)
      } else handlers.get(msg.sessionId ?? '')?.(msg.method, msg.params)
    }
  })
  // every command gets a minute, so a stuck browser fails the build instead of hanging it
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    if (dead) { reject(dead); return }
    const n = ++id, timer = setTimeout(() => { calls.delete(n); reject(new Error(`${method} timed out`)) }, 60000)
    const settle = (f) => (v) => { clearTimeout(timer); f(v) }
    calls.set(n, { method, resolve: settle(resolve), reject: settle(reject) })
    toChrome.write(JSON.stringify({ id: n, method, params, ...(sessionId ? { sessionId } : {}) }) + '\0')
  })
  await send('Browser.getVersion').catch((err) => {
    fs.rmSync(dir, { recursive: true, force: true })
    throw new Error(`Chrome did not start (${exe}): ${err.message}. Set CHROME to a Chrome or Chromium binary.`)
  })

  /** A new tab: goto(url), idle(), evaluate(fn, ...args), send(method, params) for any other command, errors,
   *  close(); `media` emulates media features. */
  async function newPage({ media = [] } = {}) {
    const { targetId } = await send('Target.createTarget', { url: 'about:blank' })
    const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true })
    // requests in flight, by the document (loader) that made them: a navigation cancels the old document's requests,
    // and Chrome does not always report them as finished
    const errors = [], inflight = new Map(), loaded = new Set(), waiters = new Set()
    let lastNet = Date.now(), doc = ''
    handlers.set(sessionId, (method, p) => {
      if (method === 'Page.lifecycleEvent' && p.name === 'load') loaded.add(p.loaderId)
      else if (method === 'Runtime.exceptionThrown') errors.push(p.exceptionDetails.exception?.description ?? p.exceptionDetails.text)
      else if (method === 'Runtime.consoleAPICalled' && (p.type === 'error' || p.type === 'assert')) errors.push(p.args.map((a) => a.value ?? a.description).join(' '))
      else if (method === 'Network.requestWillBeSent') { inflight.set(p.requestId, p); lastNet = Date.now() }
      else if (method === 'Network.loadingFinished' || method === 'Network.loadingFailed') { inflight.delete(p.requestId); lastNet = Date.now() }
      else if (method === 'Fetch.requestPaused') on('Fetch.failRequest', { requestId: p.requestId, errorReason: 'BlockedByClient' }).catch(() => {})
      for (const w of waiters) w(method)
    })
    const on = (s, params) => send(s, params, sessionId)
    await Promise.all([on('Page.enable'), on('Page.setLifecycleEventsEnabled', { enabled: true }), on('Runtime.enable'), on('Network.enable')])
    // requests matching `block` are paused by Chrome and failed here
    if (block.length) await on('Fetch.enable', { patterns: block.map((urlPattern) => ({ urlPattern })) })
    if (media.length) await on('Emulation.setEmulatedMedia', { features: media })
    const until = (test, ms, what) => new Promise((resolve, reject) => {
      const done = (ok) => { clearTimeout(timer); clearInterval(poll); waiters.delete(check); ok ? resolve() : reject(new Error(`timed out waiting for ${what()}`)) }
      const check = (method) => { if (test(method)) done(true) }
      const timer = setTimeout(() => done(false), ms), poll = setInterval(check, 50)
      waiters.add(check)
    })
    return {
      errors,
      send: on,
      async goto(url, ms = 30000) {
        const r = await on('Page.navigate', { url })
        if (r.errorText) throw new Error(`${url}: ${r.errorText}`)
        doc = r.loaderId
        await until(() => loaded.has(r.loaderId), ms, () => `${url} to load`)
      },
      /** Resolves once no request has been in flight for `quiet` ms. */
      idle(quiet = 300, ms = 30000) {
        const loading = () => [...inflight.values()].filter((q) => q.loaderId === doc).map((q) => q.request.url)
        return until(() => !loading().length && Date.now() - lastNet >= quiet, ms, () => `the network to go quiet (still loading: ${loading().join(', ')})`)
      },
      async evaluate(fn, ...args) {
        const r = await on('Runtime.evaluate', { expression: `(${fn})(...${JSON.stringify(args)})`, awaitPromise: true, returnByValue: true })
        if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text)
        return r.result.value
      },
      async close() { handlers.delete(sessionId); await send('Target.closeTarget', { targetId }) },
    }
  }

  async function close() {
    const exited = new Promise((resolve) => { if (proc.exitCode !== null) resolve(); else proc.once('exit', resolve) })
    await send('Browser.close').catch(() => {})
    await Promise.race([exited, new Promise((resolve) => setTimeout(resolve, 5000))])
    if (proc.exitCode === null) proc.kill('SIGKILL')
    fs.rmSync(dir, { recursive: true, force: true })
  }
  return { newPage, close }
}
