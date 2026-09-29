# Token Trails

A portfolio of interactive, animated visualizations of AI concepts, built step by step (Transformer first, then architecture variants, inference engines and agents). Vite + TypeScript with no UI framework; every exhibit draws on a 2D canvas.

**Language:** the repo is English: code, identifiers, comments, docs, commit messages, and the source UI copy (captions, canvas labels, aria labels). The site itself is bilingual with an EN / 中文 switch; the only Chinese in the repo is translated UI strings, kept in locale files and keyed to the English source, which stays the reference. The user chats in Chinese; answer them in Chinese.

## Commands

- `npm run dev`: dev server on :5173.
- `npm run typecheck`: `tsc --noEmit` (TypeScript 7, strict, `noUnusedLocals`/`noUnusedParameters`).
- `npm run build`: typecheck, then a production build to `dist/`, then `scripts/pages.mjs`: one HTML file per page (`dist/anatomy/attention/index.html`, its own title, description, canonical URL and share card), `404.html`, `sitemap.xml`, `robots.txt`.
- `npm run data:opening`: regenerate `src/data/opening.json` (the home animation's slice of `gpt2.json`); run it after `scripts/gpt2-export.ts`.
- `npm run build:artifact`: a single-file page in `dist-artifact/index.html` (via `vite-plugin-singlefile` + `scripts/artifact.mjs`, which strips the doctype/html/head/body wrappers). Publish that file to the existing preview Artifact by its `url` rather than creating a new one.

There are no tests. To verify a change: typecheck, then open the page and step through the phases (← / → jump between phases, Space plays/pauses). Check the console for errors.

## Layout

```
src/main.ts                 rail nav (collapsible; a drawer on phones), router (/route/?phase=id; # routes in the preview build; old #/ links
                            rewritten), each page's code loaded on first visit (the next tour page and hovered rail links prefetched),
                            page titles, theme,
                            ?embed=1 and Present modes, Save frame,
                            tour order with previous/next (Shift+←/→), zoom transitions
src/styles.css              design tokens (dark-first, light via prefers-color-scheme / data-theme) + frame styles
src/core/stage.ts           Stage: DPR-aware canvas with a min size; scaled to fit a narrower host (phones get a full-size toggle);
                            runLoop(step, idle) skips frames while idle and nothing was poke()d
src/core/player.ts          Player: phases, t, play/pause, step buttons, Step/Auto pacing, speed, timeline, keys, All steps list
                            (player.describe), ?phase= in the URL; openAtPhase() for links
src/core/fonts.ts           registerFonts(): the three families as self-hosted woff2 (@fontsource files), Latin, Latin Ext, Greek
src/core/link.ts            routeHref()/readAddress(): /anatomy/unembed/?phase=… on the site, #/anatomy/unembed?phase=… in the single-file
                            preview (HASH_ROUTES); getParams()/setParams(): page state in the address, replaceState only
src/core/progress.ts        steps seen per page and the last place (rail ✓ / n of m, Resume on the start page)
src/core/prefs.ts           pref.get/set: reader preferences in localStorage (speed, pacing, questions, temperature, strategy, lang)
src/core/i18n.ts            EN / 中文: lang, loadLang (the Chinese is its own chunk), setLang/onLang, t() (English is the key; numbers and quoted spans are {} placeholders),
                            tf(template, …args) for text built around names (the template is the key),
                            localizeCanvas() (fillText/measureText translate), raw() for text that must stay as is, harvest mode
src/locales/zh/             Chinese: ZH (key → text with {0}, {1}…) and ZH_TERMS (glossary names, definitions, spellings), one file per part
src/core/frame.ts           createFrame(): header/specs (setSpecs), stage host, formula strip, caption line, controls; toggle(), stepper(),
                            select() (a dropdown with option groups), rich()
src/core/draw.ts            primitives: chips, plate(), bracketLabel(), mathName()/mathRun(), fonts F
src/core/matrix.ts          MatrixKit: drawMat (slabs), gemm() schedules, gemmOverlay, hover and pinned cells (mk.focus); formula → frame.setFormula
src/core/legend.ts          the visual-language legend (start page and the How to read panel under every exhibit)
src/core/glossary.ts        TERMS (term, spellings to mark, definition); withTerms() marks first mentions in captions
src/core/theme.ts           canvas palette C (read from CSS tokens), rgba/mixc/blend/pop
src/exhibits/home.ts        #/home, the default: one full-window animation on the real "The cat sat on the" run (tokens, 12 blocks mixing
                            the lanes, next-token bars, the pick appended), then the name and Start the tour / Resume; no rail,
                            click / key / Skip jumps to the end, reduced motion opens there; the rail's logo comes back to it
src/exhibits/start.ts       #/start, the first stop of the tour: live next-token demo, what a Transformer is, the path, legend
src/exhibits/glossary.ts    #/glossary: every term in TERMS
src/exhibits/foundations.ts #/foundations: dot product, matrix product layout, softmax, one-hot (small made-up numbers)
src/exhibits/learn.ts       per page: code lines (marked per phase), predict-then-reveal checks (asked only when the reader turns
                            Questions on; options shown in a fixed shuffled order per question), recap; teach(player, page)
src/exhibits/kit.ts         mountExhibit(): the shared frame of a scene-per-phase page, Architectures, Serving, Training and Agents (Compare button,
                            pills that jump to a phase, onFrame hooks) and canvas helpers (Kit: pill, lane, glass, arrow…)
src/exhibits/registry.ts    categories (Inside the model, Architectures, Training, Serving, Agents; routes anatomy/*, lineage/*, …) → entries: exhibits or sub-headings;
                            an exhibit may have `children` (its steps); live when it has `route` and `load` (a dynamic import of its
                            mount, so every page is its own chunk); no static imports of pages, so Node can read it (scripts/pages.mjs)
src/lib/bert/data.ts        types for src/data/bert.json (masked-word examples, a look-ahead head, a sentence pair)
src/lib/t5/data.ts          types for src/data/t5.json, and T5's relative-position bucket()
src/lib/vit/data.ts         decodes src/data/vit.json (int8 base64): posSim(i, j), filters, explained
src/lib/clip/data.ts        types for src/data/clip.json, preview(i) canvases of the drawn images, softmax
src/lib/dit/data.ts         decodes src/data/dit.json (meanAbs, heatAt) and DDPM's ALPHA_BAR schedule
src/lib/mamba/data.ts       types for src/data/mamba.json
src/lib/spec/data.ts        types for src/data/spec.json, plain() for GPT-2 spellings
src/lib/quant/data.ts       types for src/data/quant.json
src/lib/icl/data.ts         types for src/data/icl.json
src/lib/react/data.ts       types for src/data/react.json
src/lib/tools/data.ts       types for src/data/tools.json
src/lib/rag/data.ts         types for src/data/rag.json, plainText() for model output
src/lib/multi/data.ts       types for src/data/multi.json
src/lib/backprop/data.ts    types for src/data/backprop.json
src/lib/scaling/data.ts     types for src/data/scaling.json
src/lib/finetune/data.ts    types for src/data/finetune.json and src/data/lora.json
src/lib/sft/data.ts         types for src/data/sft.json
src/lib/models/data.ts      the model library (src/data/models.json): types, families() for the dropdowns, kvBytes(m, n), fmtParams/Bytes/Tokens
src/lib/gpt2/
  bpe.ts                    GPT-2 byte-level BPE (pre-split, merges by rank, ids), symbolText(); no imports, so Node can load it
  merges.txt                GPT-2's 50,000 merge rules, imported with ?raw only by the tokenizer page
  decode.ts                 decoding without the data: attention bytes, token texts, nextDistOn() at any T, mixing() (lane colours)
  data.ts                   decodes src/data/gpt2.json with decode.ts: presets, nextDist() at any T, headKind(),
                            wpeSlice() and streamNorms() for the 2017 Transformer page, leftOnly() for the BERT page,
                            kvSlice() (one head's real q, k, v) for the KV cache page
src/data/gpt2.json          real GPT-2 small activations for 3 prompts × 3 greedy passes, and one sentence scored per position
                            for training (made by scripts/gpt2-export.ts)
src/data/opening.json       the home animation's pass of gpt2.json, top 8 logits with the rest folded into the tail (exact), a few KiB
                            (made by scripts/opening-export.mjs), so the home page does not download gpt2.json
public/og.png               the 1200 × 630 share card: the home animation just before its title, with the name and address
scripts/pages.mjs           after vite build: a page file per route with its meta tags, 404.html, sitemap.xml, robots.txt
scripts/opening-export.mjs  cuts src/data/opening.json out of src/data/gpt2.json
scripts/i18n-harvest.mjs    opens every page and step on the dev server in harvest mode and prints the strings with no Chinese yet
scripts/gpt2-export.ts      offline GPT-2 small forward pass in plain TS (Node 23+); weights in ~/.cache/token-trails/gpt2
scripts/bert-export.ts      offline BERT-base (uncased) forward pass, WordPiece included; writes src/data/bert.json;
                            weights in ~/.cache/token-trails/bert (model.safetensors, vocab.txt)
scripts/t5-export.ts        offline T5-small encoder–decoder with greedy decoding, SentencePiece unigram included;
                            writes src/data/t5.json; weights in ~/.cache/token-trails/t5 (model.safetensors, tokenizer.json)
scripts/vit-export.ts       fetches two ViT-B/16 tensors by HTTP range (cached in ~/.cache/token-trails/vit): position
                            similarities and patch-filter principal components; writes src/data/vit.json
scripts/clip-export.ts      offline CLIP ViT-B/32 (both towers, CLIP BPE) on four drawn shapes and captions; writes
                            src/data/clip.json; weights in ~/.cache/token-trails/clip (safetensors from refs/pr/21)
scripts/dit-export.ts       fetches DiT-XL/2 embedder and adaLN tensors of blocks 1, 14, 28 by HTTP range (safetensors
                            from refs/pr/1, cached in ~/.cache/token-trails/dit); writes src/data/dit.json
scripts/mamba-export.ts     offline Mamba-130m (selective scan, GPT-NeoX BPE): next-token guesses and Δ per layer and
                            token; writes src/data/mamba.json; weights in ~/.cache/token-trails/mamba
scripts/spec-export.ts      speculative decoding for real: distilgpt2 drafts, GPT-2 small verifies; writes src/data/spec.json;
                            weights in ~/.cache/token-trails/gpt2 and ~/.cache/token-trails/distilgpt2
scripts/quant-export.ts     GPT-2 small with its linear weights rounded to int8/int4 (per tensor, channel, group of 128) and
                            rerun; writes src/data/quant.json
scripts/gpt2-model.ts       shared GPT-2-shaped forward pass (any GPT-2 width) for spec-, quant- and scaling-export
scripts/qwen-model.ts       offline chat model for the Agents exports: Qwen3-1.7B (or Qwen2.5-Instruct) in plain TS with worker
                            threads, byte-level BPE, the ChatML/tool-call template (thinking off), KV cache, greedy generation
                            with stop strings and top-k; weights in ~/.cache/token-trails/qwen3
scripts/icl-export.ts       GPT-2 small given 0–4 examples of three tasks, the copying head's attention row, and GPT-2 vs
                            Qwen3-1.7B on a bare instruction; writes src/data/icl.json
scripts/react-export.ts     a ReAct loop with Qwen3-1.7B: lookup and calculator tools, stop at "Observation", real results appended;
                            also the run left unstopped and one without the look-up rule; writes src/data/react.json
scripts/tools-export.ts     tool calling with Qwen3-1.7B: count_letter and tokenize as JSON schemas, the call parsed, validated and
                            run, the answer; also the answer without tools and the mask at the function name; writes src/data/tools.json
scripts/rag-export.ts       RAG for real: this site's glossary and legend embedded with all-MiniLM-L6-v2 (a 6-layer BERT, mean-pooled),
                            four questions ranked by cosine, Qwen3-1.7B answers with and without the top 3; writes src/data/rag.json
scripts/multi-export.ts     a multi-agent run with Qwen3-1.7B: an orchestrator with ask_worker hands pieces to workers that start empty and
                            must call lookup first; also one agent alone; every call's token counts; writes src/data/multi.json
scripts/backprop-export.ts  GPT-2 small's forward and backward pass in float64 on the Training sentence: every weight's gradient, a dW
                            slice, per-position reach, finite-difference checks; writes src/data/backprop.json
scripts/scaling-export.ts   GPT-2 small, medium and large (sizes via gpt2-model.ts's dims) scoring this site's glossary, which none of them
                            saw; losses per token and in the mean; writes src/data/scaling.json; weights in ~/.cache/token-trails/gpt2-medium, gpt2-large
scripts/finetune-export.ts  GPT-2 small fine-tuned in float64 with Adam (gpt2-grad.ts): rank-4 LoRA vs full fine-tuning on two sentences,
                            and DPO with the same adapters on three pairs; writes src/data/finetune.json
scripts/lora-export.ts      Qwen3-1.7B minus Qwen3-1.7B-Base for four matrices of layer 15 (only those tensors read): top singular
                            values of ΔW and W by randomized SVD; writes src/data/lora.json; base in ~/.cache/token-trails/qwen3base
scripts/gpt2-grad.ts        GPT-2 small's forward and backward pass in float64, weighted losses, gradients only where asked
                            (backprop-export, finetune-export)
scripts/svd.ts              randomized SVD (Halko et al.) for the rank exports
scripts/sft-export.ts       Qwen3-1.7B-Base and Qwen3-1.7B, each in a child process: per-token loss on one chat example, their answers,
                            each scored on both; writes src/data/sft.json
scripts/models-lib.ts       Hub access for the model library: config.json, the shard index and each shard's safetensors header by HTTP
                            range (no weights), cached in ~/.cache/token-trails/models
scripts/models-export.ts    the model library: 29 decoder LMs (GPT-2 to 2025, grouped by family); parameters per component counted from
                            the checkpoints' tensor shapes, active parameters, every layer's attention kind and KV size, checked against
                            the Hub's count and the published totals (per author's convention); writes src/data/models.json
src/exhibits/transformer/
  model.ts                  toy model: prompt ids, toy attention + MLP blocks, LayerNorm params, laneMix()
  overview.ts               forward pass with real GPT-2 numbers, tokenizer → greedy pick; plates open the detail views
  tokenizer.ts              tokenizer: real GPT-2 BPE on any text (merges.txt lazy-loaded): pre-split, bytes, lowest-rank lookups, ids
  embedding.ts              embedding detail view: onehot · W_E as a lookup, + W_P, into the stream; prompts in TOY_PROMPTS
  layernorm.ts              pre-LN stream schematic, then ln_1 as dots on number lines (μ, σ, γ, β)
  attention.ts              attention detail view, every GEMM animated cell by cell
  mlp.ts                    MLP detail view: up-projection, GELU curve, down-projection, residual
  unembed.ts                ln_f, logits = x · W_Eᵀ (tied), temperature, softmax, sampling strategies; real GPT-2 numbers
src/exhibits/training/
  loss.ts                   next-token loss on a real GPT-2 run (per-position p, −log p), gradient p − y, a toy step
  backprop.ts               backprop through GPT-2 small on the Training sentence (scripts/backprop-export.ts): the chain, dW = Xᵀ·dY as a GEMM
                            on real slices, the MLP backwards, the causal mask backwards, ‖dW‖ per block, finite-difference checks
  optimizer.ts              SGD, momentum and Adam on a toy valley (real arithmetic, computed in the page), AdamW's decoupled decay, LLaMA 2's
                            warmup + cosine schedule, bytes per weight for Adam training
  bpetrain.ts               BPE trained live in the page on this site's glossary (byteSymbols() from bpe.ts): bytes, pair counts, 300 merges
                            on sample words, symbols per word vs merges, our first merges next to GPT-2's (merges.txt, lazy-loaded)
  scaling.ts                scaling laws: GPT-2 small/medium/large on unseen text (scripts/scaling-export.ts) with a power-law fit, per-token
                            gains, the Chinchilla fit L(N, D), isoFLOP curves and compute-optimal sizes, published models vs D = 20 N
  lora.ts                   LoRA: W + A·B at toy size, trainable counts (GPT-2 here, LLaMA 3 8B r = 16), a real rank-4 run vs full fine-tuning
                            (scripts/finetune-export.ts), captured energy by rank for that fine-tune vs Qwen3's post-training (lora-export.ts)
  sft.ts                    SFT on Qwen3-1.7B-Base vs Qwen3-1.7B (scripts/sft-export.ts), drawn as token chips with probability bars:
                            chat bubbles unrolling into the template's tokens, a cursor predicting every token while only the answer's
                            −log p fills the loss bar, both models' replies, the tuned model's odds on another wording and on the template
  dpo.ts                    RLHF & DPO: preference pairs with GPT-2's own split between the answers, a reward model scoring six next
                            tokens (toy scores), the RL loop on GPT-2's real next-token probabilities computed in the page (converges to
                            π_ref·exp(r/β); β toggle), toy DPO on the same bars, the real GPT-2 DPO run (scripts/finetune-export.ts) as
                            answers riding a log-probability axis, the next token before and after
src/exhibits/lineage/
  transformer2017.ts        the 2017 Transformer vs GPT-2: RNN → attention, encoder + decoder, a toy EN → DE
                            translation, the three attentions, cross-attention GEMM, post-LN (real GPT-2 stream
                            lengths), sinusoids vs GPT-2's real W_P
  llama.ts                  LLaMA 3 vs GPT-2: blocks, RoPE, RMSNorm, SwiGLU, GQA (real numbers are LLaMA 3 8B)
  mamba.ts                  Mamba-130m vs GPT-2: one mixer per block, KV cache vs fixed state, a toy scan coloured by the tokens
                            each state number holds, real Δ of every layer, real next-token guesses next to GPT-2's
  dit.ts                    DiT-XL/2 vs GPT-2: exact noising with DiT's schedule (toy image), latent patches, real adaLN-Zero
                            gates and scales vs t, a 250-step sampling run with the true noise
  clip.ts                   CLIP ViT-B/32 vs GPT-2: two towers, real embeddings and their PCA (modality gap), real I · Tᵀ
                            with both softmaxes and the loss, real zero-shot, the learned scale of 100
  vit.ts                    ViT-B/16 vs GPT-2: toy image → 14 × 14 patches, patch-embedding GEMM + real filter components,
                            real position-embedding similarity (the learned 2D grid), only [CLS] classified
  t5.ts                     T5-small vs GPT-2 and 2017: real text-to-text answers, span corruption (paper figure 2 + a real fill),
                            blocks, relative position buckets, real learned biases per bucket and head
  bert.ts                   BERT-base vs GPT-2: same shape, real heads (GPT-2 previous-token vs BERT next-token),
                            WordPiece pair inputs, real masked-word predictions next to GPT-2's left-only guesses, [CLS] classifier (toy)
  deepseek.ts               DeepSeek-V3 vs GPT-2: MLA as three GEMMs through a latent, KV cache per token (MHA/GQA/MQA/MLA),
                            DeepSeekMoE (1 shared + top 8 of 256), bias balancing simulated on 16 toy experts
  mixtral.ts                Mixtral 8x7B vs GPT-2: MoE block, router GEMM + top-2, dispatch/combine, stored vs
                            active params, balancing loss (toy routing, d_model 4, ROUTE_SEED)
  compare.ts                Architecture diff: any two models of src/data/models.json (two dropdowns, ?a=&b=): the differing rows,
                            a layer map (attention kind, MLP or experts, no-RoPE layers), heads and KV per token, experts, where the
                            parameters are, the KV cache against context; captions via tf()
src/exhibits/serving/
  kvcache.ts                why a KV cache (n² → n), prefill, a real GPT-2 decode step (q · Kᵀ, softmax, · V for one head),
                            cache sizes across models, the roofline (prefill compute-bound, decode memory-bound)
  flashattention.ts         SRAM vs HBM, standard attention's HBM round trips, the online softmax on one row, the tiled loop
                            with real toy arithmetic checked against standard attention, HBM traffic and N × N memory
  paged.ts                  toy allocator simulation (256 slots, blocks of 4): contiguous reservations vs blocks on demand with
                            block tables, the kernel gathering blocks, sharing with copy-on-write, batch size and steps for both
  batching.ts               why batch (throughput vs batch size on real sizes), static vs continuous Gantt charts from a toy
                            scheduler, wait/latency/throughput, chunked prefill step times
  speculative.ts            a real run of distilgpt2 drafting for GPT-2 small (scripts/spec-export.ts): one round checked,
                            all rounds, the min(1, p/q) rule with real p and q, tokens per pass and speed-up vs draft cost
  quantization.ts           number formats, absmax rounding of a real GPT-2 weight column (int8, int4), per-tensor/channel/group
                            scales with real errors and perplexities, a real activation's outliers, real next-token effect
src/exhibits/agents/
  common.ts                 shared pieces: next-token bars (distRows), prompts as rows of chips, the model as plates, text runs
                            with ligatures off (JetBrains Mono joins <| and ->)
  incontext.ts              in-context learning on GPT-2 small (scripts/icl-export.ts): 0 → 3 examples, three tasks from one
                            question, the copying (induction) head, majority and recency bias, Qwen3 on a bare instruction
  react.ts                  the ReAct loop, a real Qwen3-1.7B run (scripts/react-export.ts): the prompt's sections, Thought → Action
                            with the model's choice, stop/parse/run/append, the whole trace, context per call, the run left unstopped
  toolcalling.ts            tool calling with Qwen3-1.7B's own format (scripts/tools-export.ts): the no-tool answer (wrong), tool schemas in the
                            system prompt, the <tool_call> with its probabilities, checks/run/<tool_response>, the answer, the vocabulary mask
  rag.ts                    RAG over this site's glossary and legend (scripts/rag-export.ts): passages, MiniLM embeddings as cells, a 2D PCA
                            map with hover, cosine ranking with the top-3 cut, the assembled prompt, answers with and without, two misses
  multiagent.ts             orchestrator and workers, a real Qwen3-1.7B run (scripts/multi-export.ts): one agent alone, the handoffs (ask_worker),
                            a worker's fresh context, the reports and the answer, a parallel timeline, tokens and contexts against one agent
```

## Adding an exhibit or detail view

1. Write `mountX(root, nav): () => void`: `createFrame` (with `formula: true` for a detail view) → `new Stage` → `new Player(PHASES, frame.controls)` → `new MatrixKit(stage, tokens, frame.setFormula)` → set `player.describe` (caption per phase, for All steps) → `teach(player, page)` with an entry in `learn.ts` → `runLoop(step, () => !player.playing)` that ticks, draws, updates the timeline UI and sets the caption. Only set an initial `player.t` when it is still 0 (a `?phase=` link may have placed it). Return a destroy that stops the loop and calls `player.destroy()` and `stage.destroy()`.
2. Register it in `registry.ts` (`route` + `load: () => import('./x/page').then((m) => m.mountX)`; never a static import, which would put the page in every visitor's download), and add it to `TOUR` in `main.ts` if it belongs to the reading order. Steps of an exhibit go in its `children` with a deeper route (`anatomy/mlp`), which opens with a zoom-in. Routes are paths (`/anatomy/attention/`, `/lineage/llama/`; `#/…` in the preview build); old `#/` links and renamed prefixes (`ALIASES`) keep working. A new route gets its page file, sitemap entry and meta tags from `scripts/pages.mjs` automatically.
   An Architectures (lineage/*), Serving, Training or Agents page (scenes per phase rather than one matrix walk-through) uses `mountExhibit()` from `exhibits/kit.ts` instead: pass the frame options, phases, captions, a Compare route per phase and a `scenes(env)` factory. Serving, Training and Agents pages pass `compareLabel: 'Related'`, since their links go to related pages rather than to the GPT-2 part they change. Give it `hints` per phase: the hint under the formula strip should offer only what that phase has (hover targets, labels to click, controls).
3. To open a detail view from the overview, add its plate to `plateAt` / `PLATE_ROUTES` in `overview.ts` (hover highlight and `drawOpenHint` follow). Mention the click in that phase's caption.

## Conventions

- **Everything is a function of time.** Drawing reads `player.t` / `player.prog(id)` and keeps no animation state of its own, so the timeline scrubs both ways. Hover state is the only exception.
- **Phases** have a `name` (caption title) and an optional `short` (timeline label, e.g. `QKᵀ`).
- **Detail-view pattern** (follow `attention.ts`):
  - One `sceneX(p)` per group of phases.
  - Each scene has a layout computed in `geom()`, laid out so that a GEMM's A row i lines up with C row i and B column j lines up with C column j (A left, B above, C at the intersection). Transitions lerp matrix rects between scene layouts.
  - Use `MatrixKit` from `core/matrix.ts`: `gemm(p, m, n, K, 'slow'|'fast')` schedules cells, `mk.resolve()` picks the hovered cell or the animated one, `mk.gemmOverlay()` draws the highlights and guides and sets the formula line. It supports a transposed B (`bT`, used for W_projᵀ in `mlp.ts`) and bias vectors.
  - Call `mk.begin()` each frame and register hoverable result matrices with `mk.hit()`. A click or tap pins a cell (`mk.pin`); read the inspected cell with `mk.focus` / `mk.hovered()`, never `mk.hover` alone.
  - Specs: a plain value is what the drawing uses; `real` renders as "· GPT-2 768" (`realLabel` changes the prefix).
- **Real vs toy numbers.** Never show a made-up number as GPT-2's. The overview and Unembed use a real GPT-2 small run (`src/lib/gpt2/data.ts`); when only part of a real vector fits, say how much is drawn (`8 drawn`, `dims 1–16 of 768`). Detail views that animate every GEMM compute real arithmetic at toy size (`TOY` in `model.ts`: d_model 8, d_head 4, 2 heads; `TOY_FF` 32), say `shown: toy` in the specs, and show the GPT-2 small shape next to each matrix (`real`, `N × 768`, `value / real`).
- **Changing the real data.** Edit the prompts or fields in `scripts/gpt2-export.ts`, run `node scripts/gpt2-export.ts` (needs model.safetensors, merges.txt, vocab.json from huggingface.co/openai-community/gpt2 in ~/.cache/token-trails/gpt2), then `npm run data:opening`, and commit the regenerated `src/data/gpt2.json` and `src/data/opening.json`. Keep them small: they are downloaded by the pages that use them.
- **Visual language is fixed.** The user approved it; don't redesign it.
  - Colors come only from `C` / the CSS tokens. Never hard-code hex in drawing code, and check both themes (and the opt-in `data-palette="cvd"` token hues). `--faint` / `C.faint` is for rules and grids only; any text uses `--mute` or stronger.
  - Each token keeps its hue (`--t0…--t6`, token i → `i % 7`) everywhere. Information mixing between tokens is shown by blending hues: the overview uses real attention (`mixing()` in `data.ts`, first-token sinks count as no-ops), the toy detail views use `laneMix()` in `model.ts`, averaged over heads.
  - Layers are tilted glass plates. Matrices are slabs with a 5px depth face. A filled cell is positive, an outlined cell is negative.
  - Fonts (self-hosted, see `core/fonts.ts`; no Google Fonts): EB Garamond italic for names and math (`serifAt()`, scaled up 10%), Geist for UI text, JetBrains Mono for labels and numbers (`MONO`, never under 10.5px on canvas). Group labels are uppercase mono over thin brackets. Use the constants in `core/draw.ts`, never a hard-coded font string.
  - New terms of art go in `TERMS` (`core/glossary.ts`) with a one- or two-sentence definition; captions then mark them. Give each a Chinese entry in `ZH_TERMS`.
- Keep canvas text minimal, using real units and terms of art (GPT-2 ids, Ġ, 768 → 3072, FLOPs). Explanations go in the caption line: one or two plain sentences per phase. The formula strip under the stage (HTML, via `mk.formula` / `frame.setFormula`) carries the focused cell's arithmetic and a short note.
- **Two languages.** Write every string in English; Chinese goes only in `src/locales/zh`, keyed by the English with numbers and quoted spans (“…”, ‘…’) as `{}`. Text set through the frame, player, kit and rail is translated for you, and so is all canvas text (the stage's context translates in fillText and measureText); HTML a page builds itself needs `t()`. Token text, math and model output must never be translated: draw tokens with `drawChip`/`tokText`, math with `mathRun`/`mathName`, prompts and model text with `runs`/`wrap` (`agents/common.ts`), and anything else verbatim inside `raw()`. A key applies wherever that exact string is drawn, so don't add keys for single common words that also appear as data outside `raw()` (the, with, of). Numbers written as .072 are placeholders too. Keep model outputs and real prompts in English. After adding strings, run `node scripts/i18n-harvest.mjs` with the dev server up and add the missing Chinese. Check pages in both languages: Chinese text is often wider per character.
- Seeds are chosen for readable patterns (`WEIGHT_SEED` in `model.ts`). Changing a seed changes every number shown, so re-check the attention patterns if you touch it.

## Gotchas

- The minifier turns any spelling of U+FFFD (`'\uFFFD'`, `String.fromCharCode(0xfffd)`) into the literal character, and the preview Artifact refuses a page that contains it. Don't use it as a fallback character.
- The production build has no `window.__ttPlayer` (dev only); in scripts against `dist/`, step with the arrow keys and read `.cap-title`.
- The browser pane is unreliable for checking drawings: screenshots of an emulated viewport can come out blank or half-scaled, and a hidden pane stops rAF and ResizeObserver (document.visibilityState is "hidden"), so sizes go stale. To inspect a frame, POST `canvas.toDataURL()` to a small local server and read the PNG.
- The browser pane's `preview_start` with the `dev` config has failed to serve before. If :5173 doesn't answer, run `npx vite --port 5174` in the background and navigate there. A hidden pane throttles rAF, so take a fresh screenshot before judging a frozen frame.
- Don't rely on Unicode subscript characters in the serif; use `mathRun()` or `fillRich()` for math with subscripts. A one- or two-letter name before `_` is math (W_Q, ln_f, π_ref, x_{t−1} with braces); a longer one is code (count_letter, <tool_call>) and stays as written.
- Phones: nothing in the page may be wider than the screen. Below 520px the frame's toggles, steppers, slider labels, caption titles and shape lines wrap (`styles.css`); keep phase names plus their short label under about 36 characters and caption shape lines under about 46. Check every phase at 390px wide (document.documentElement.scrollWidth must stay 390) after adding a page.
- The repo: github.com/meyowu/llm-visualizer-token-trails (renamed from token-trails on 2026-09-29; the old URL redirects).

## Shipping

When a piece of work is finished and verified, ship it without asking: commit on `main`, push, then `npm run build:artifact` and republish `dist-artifact/index.html` to the existing preview Artifact (https://claude.ai/artifact/RbScAmBiuDJhyQ9xvgG9vz) by its `url`.

A push to `main` also deploys the site to GitHub Pages at https://tokentrails.org (custom domain; DNS at Cloudflare: CNAME `@` and `www` to meyowu.github.io, DNS only) through `.github/workflows/deploy.yml` (`npm ci`, `npm run build`, upload `dist/`); check it with `gh run list --workflow deploy.yml`. Code is MIT, the explanatory content CC BY-NC 4.0 (`LICENSE-CONTENT.md`), third-party material is listed in `NOTICE.md`: add a row there when a new export script uses another model.
