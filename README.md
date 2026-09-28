# Token Trails

Follow the tokens through AI systems: animated, explorable walk-throughs of the Transformer, inference engines and agent workflows.

The atlas follows a model's life in six parts: **Anatomy** (GPT-2 taken apart), **Lineage** (other architectures as changes to GPT-2), **Training**, **Interpretability**, **Serving** and **Agents**.

## What's here

- **Start here**: what a language model does (a live next-token example with real GPT-2 numbers), what a Transformer is, the path through the Anatomy chapters, and how to read the pictures.
- **Anatomy / Forward pass**: one full forward pass of GPT-2 small with the numbers of a real run, for three prompts, each continued greedily for three passes: real token ids, embeddings, all 144 attention heads (with a label for what each head does), block 1's MLP neurons, the logit lens after every block, and the real next-token distribution.
  - **Tokenizer**: GPT-2's real byte-level BPE on any text you type (or four examples: subwords, a contraction, a repeat, non-ASCII): regex pre-split, UTF-8 bytes (space → Ġ), then inside each piece the lowest-ranked adjacent pair looked up and merged until none is left, each token's id (256 + merge rank) placed on the vocabulary, and decoding back to text, bytes that wait for the rest of a character included.
  - **Embedding** (with a repeated word, and one sentence in two word orders): a map of real GPT-2 rows where words of a kind sit together, ids as one-hot rows times W_E (a GEMM that is really a row lookup, on a log-scaled vocabulary axis), plus the first N rows of W_P, giving the residual stream.
  - **LayerNorm & Residual**: the pre-LN residual stream through one block (sub-layers read a normalised copy and add back), then ln_1 on the real toy numbers: each token's features as dots, centred by μ, scaled by σ, then γ ⊙ x̂ + β.
  - **Attention**: every matrix product in one attention head, computed at toy scale (d_model 8, d_head 4) and animated cell by cell: X·W_Q/K/V → Q·Kᵀ → ÷√d → mask → softmax → A·V → concat·W_O → residual add. Hover any result cell to see which row and column produced it.
  - **MLP**: the feed-forward block at toy scale (d_model 8 → d_ff 32, the same 4× as GPT-2): X·W_fc + b → GELU (each activation plotted on the curve) → ·W_proj + b → residual add.

  - **Unembed & Sampling**: real GPT-2 numbers: ln_f on the last position, logits as x · W_Eᵀ (tied to the embedding; 8 of the 768 dimensions drawn, each logit the full sum), the whole 50,257-token distribution at any temperature (top 256 exact, the rest summarised), softmax, and sampling strategies (sample, greedy, top-k, top-p) drawn as an inverse CDF.
- **Foundations**: a warm-up on the four pieces of math the rest uses: the dot product as a similarity score, the matrix product in the site's layout, softmax, and one-hot times a matrix.
- **Training / Next-token loss**: "The cat sat on the floor" scored the way training scores it, with GPT-2's real probabilities: five examples at once, −log p per position, the gradient p − y on the logits, and a toy step downhill.
- **Glossary**: every term of art in plain words; captions mark first mentions with their definition.
- **Lineage / Transformer (2017)**: the original encoder–decoder as a diff against GPT-2: why attention replaced RNNs (one step instead of n), the encoder and decoder stacks, translating “I have seen the cat.” token by token, the three attentions and their masks, cross-attention as a target × source GEMM, post-LN vs pre-LN with GPT-2's real residual-stream lengths, and sinusoidal positions next to GPT-2's real learned W_P.
- **Lineage / LLaMA**: LLaMA 3 as a diff against GPT-2: the two blocks side by side, then RoPE (rotating q/k pairs by position, and why only the offset matters), RMSNorm vs LayerNorm, the gated SwiGLU MLP, and MHA vs GQA vs MQA with real KV-cache sizes.
- **Lineage / Mamba**: no attention: one mixer per block around a selective state-space scan; memory as the text grows (GPT-2's KV cache against Mamba's fixed 1.3 MiB state); a toy scan whose state is coloured by the tokens it holds, driven by the real step sizes Δ of Mamba-130m (names get the largest steps); and real next-token guesses from Mamba-130m next to GPT-2 small's.
- **Serving / KV Cache**: why generation caches keys and values (n² rows of work become n), prefill filling the cache, one decode step with a real GPT-2 head (q · Kᵀ, softmax, · V), how big the cache gets for GPT-2, LLaMA 3 and DeepSeek-V3, and the roofline that makes decoding memory-bound.
- **Serving / FlashAttention**: the GPU memory hierarchy, standard attention's round trips through HBM, the online softmax (a running maximum and sum, rescaled when the maximum rises), the tiled loop computed for real at toy size and matching ordinary attention exactly, and the traffic and memory it saves.
- **Serving / PagedAttention**: a toy serving simulation run twice on the same 16 requests: contiguous worst-case reservations (35% of held memory used) against vLLM-style blocks allocated on demand through block tables (94%), the attention kernel gathering scattered blocks, prompt blocks shared between samples with copy-on-write, and the bigger batches that finish the queue in 49 steps instead of 85.
- **Serving / Continuous Batching**: why a batch is nearly free (throughput against batch size for LLaMA 3 8B on an A100), a toy scheduler drawn as Gantt charts for static and iteration-level batching, the waiting, latency and throughput it changes, and chunked prefill keeping every step under the memory-bound step time.
- **Serving / Speculative Decoding**: a real run of distilgpt2 drafting 4 tokens at a time for GPT-2 small: one round checked position by position, the whole run (12 tokens from 5 GPT-2 passes, identical to GPT-2's own greedy text), the sampling rule min(1, p/q) with the two models' real probabilities, and the expected speed-up, which needs a much cheaper draft than distilgpt2.
- **Serving / Quantization**: number formats and model sizes, absmax rounding of a real GPT-2 weight column onto the int8 and int4 grids, per-tensor against per-channel against group-of-128 scales with GPT-2's real weight error and perplexity after quantizing every linear layer, a real activation's outliers, and the real next-token guesses at each precision (int8 nearly free, int4 per tensor broken).
- **Agents / In-context learning**: GPT-2 small given 0 to 3 examples of country: capital lines (one example takes “ Cairo” from under 0.1% to 58%), three tasks computed from the same question by changing the examples, the induction head that copies from earlier answers (97% of its attention from the last colon), the majority and recency bias this brings, and Qwen3-1.7B answering from a bare instruction where GPT-2 rambles.
- **Agents / ReAct Loop**: a real run of Qwen3-1.7B with a lookup tool and a calculator: the prompt that teaches the format (362 tokens, most of it one worked example), a Thought and an Action whose tool name is just the likeliest next token, the program stopping the text at "Observation", parsing the action, running the tool and appending its result, the whole loop (two lookups, one subtraction, finish[20]), the context growing call by call (658 tokens kept in a cache against 2,200 re-sent), and what the model writes when it is not stopped (a made-up fact sheet: LLaMA 3 8B with 8 layers).
- **Agents / Tool Calling**: a real run of Qwen3-1.7B with its own tool-call format: without tools it spells "strawberry" out and still answers 2 (the word is a single token to it), the tools as JSON schemas in the system prompt (241 tokens for two), the <tool_call> it writes (p ≈ 1 at the first token and at the name), the program parsing, validating and running it and returning <tool_response>3</tool_response>, the correct answer, a question it answers without calling, and constrained decoding (8 of 151,669 tokens allowed at the function name).
- **Agents / RAG Pipeline**: retrieval-augmented generation over this site's own text, all real: 92 passages (the glossary and the legend) embedded with all-MiniLM-L6-v2, a question's 384-dimensional vector against a passage's, the passages on their top two principal components (hover to read one), cosine ranking with the top 3 kept (for one question the right passage only comes second), the prompt the passages are pasted into, Qwen3-1.7B guessing what a hatched cell means without them and answering from the legend with them, and two misses: a question worded unlike its passage, and one whose passage lacks the answer (without passages the model made an author up).
- **Agents / Multi-agent Handoff**: a real run with Qwen3-1.7B in every role: one agent answering alone (two lookups at once, 887 tokens), then an orchestrator that hands the request to 5 workers through an ask_worker tool (and drops one piece), a worker's context from empty (its first reply is required to be a lookup), the reports coming back and the answer (which drops one of them), a timeline with the workers in parallel, and the cost: 4,189 tokens against 887, with no smaller largest context on a task this small.
- **Training / Backprop**: the chain rule from the loss back through GPT-2 small, computed for real in float64 on the Training sentence: a linear layer's weight gradient dW = Xᵀ · dY animated on real slices (summed over the positions), the gradient passed down through GELU, LayerNorm and attention, the causal mask seen backwards (each position's loss reaches only earlier tokens), the gradient on every weight matrix of all twelve blocks with the residual stream keeping it alive, and three weights checked against finite differences to six digits.
- **Training / Optimizer**: plain gradient descent, momentum and Adam run step by step on the same toy loss valley (real arithmetic, made-up surface) with their loss curves, AdamW's weight decay applied apart from Adam's scaling, LLaMA 2's learning-rate schedule (2,000 warmup steps, cosine to 10% of 3 × 10⁻⁴ over about half a million steps), and why training LLaMA 3 8B needs about 16 bytes per weight against 2 to serve it.
- **Training / Learning the tokenizer**: byte-pair encoding trained for real, in the page, on this site's own glossary: every word as bytes, the pair counts, 300 merges learned one by one with sample words falling into larger pieces, symbols per word against vocabulary size, and our first merges next to GPT-2's real ones (the first four are the same).
- **Lineage / Diffusion Transformer**: a Transformer that predicts noise: the exact DDPM noising of a toy image with DiT's schedule, latent 2 × 2 patches, adaLN-Zero with DiT-XL/2's real gates and scales across timesteps (block 28's MLP is nearly switched off), and a 250-step sampling run that uses the true noise, next to the compute it costs.
- **Lineage / CLIP**: two encoders, an image ViT and a GPT-2-like text Transformer, meeting in one 512-dimensional space; everything real (CLIP ViT-B/32 on four drawn shapes and their captions): the embeddings and their principal directions (the modality gap), the image × caption matrix with both softmaxes and the loss, zero-shot classification by shape and by colour, and the learned temperature of 100.
- **Lineage / Vision Transformer**: an image cut into 14 × 14 patches that become the tokens, the patch embedding as one shared matrix product next to the real ViT-B/16 filters' principal components, the real position embeddings, whose similarities rediscover the 2D grid, and a head that reads only [CLS].
- **Lineage / T5**: every task as text in, text out, with real T5-small answers (translation, summary, a question, sentiment, similarity), span corruption with sentinels and a real fill, the blocks against GPT-2 and 2017 (scale-only pre-norm, no biases, no position vector), the relative-position buckets, and T5-small's real learned bias per bucket and head.
- **Lineage / BERT**: the encoder-only model with GPT-2 small's exact shape: a real GPT-2 previous-token head next to BERT's real look-ahead head, WordPiece inputs with segments, real bert-base-uncased fill-in-the-blank predictions next to GPT-2's real guesses from the left side alone, and a classifier on the [CLS] vector.
- **Lineage / DeepSeek**: DeepSeek-V3's two changes: multi-head latent attention as three GEMMs through a small cached latent, the KV cache per token for MHA, GQA, MQA and MLA at V3's size (3.8 MiB vs 69 KiB), DeepSeekMoE with a shared expert and 8 of 256 routed experts, and auxiliary-loss-free balancing simulated step by step.
- **Lineage / Mixtral**: the mixture of experts: the MLP becomes a router and 8 SwiGLU experts, the router's scores and top-2 pick as a small GEMM, six tokens dispatched to their two experts and combined, 46.7B stored vs 12.9B used per token, and the Switch Transformer balancing loss computed on the batch.

Every page has an All steps list, a Code drawer with the matching PyTorch, a Go deeper reading list, a question or two to answer before key steps, and a recap at the end. It pauses at the end of each step so there is time to read (switch to Auto to play straight through); ← and → move between steps. On the overview, click a part marked ↗ (the tokens, the embedding strips, the attn, mlp, ln_f and W_U plates, the next-token bars) to zoom into its detail view.

## Development

```bash
npm install
npm run dev              # http://localhost:5173
npm run build            # typecheck + production build to dist/
npm run build:artifact   # single-file build to dist-artifact/index.html, for publishing as a claude.ai Artifact
node scripts/gpt2-export.ts  # regenerate src/data/gpt2.json from GPT-2 small weights (see the script header)
node scripts/bert-export.ts  # regenerate src/data/bert.json from BERT-base weights (see the script header)
node scripts/t5-export.ts    # regenerate src/data/t5.json from T5-small weights (see the script header)
node scripts/vit-export.ts   # regenerate src/data/vit.json from two ViT-B/16 tensors (fetched by range)
node scripts/clip-export.ts  # regenerate src/data/clip.json from CLIP ViT-B/32 weights (see the script header)
node scripts/dit-export.ts   # regenerate src/data/dit.json from DiT-XL/2 tensors (fetched by range)
node scripts/mamba-export.ts # regenerate src/data/mamba.json from Mamba-130m weights (see the script header)
node scripts/spec-export.ts  # regenerate src/data/spec.json (distilgpt2 drafting for GPT-2 small)
node scripts/quant-export.ts # regenerate src/data/quant.json (GPT-2 small quantized to int8 and int4)
node scripts/icl-export.ts   # regenerate src/data/icl.json (GPT-2 small few-shot, Qwen3-1.7B)
node scripts/react-export.ts # regenerate src/data/react.json (a ReAct loop with Qwen3-1.7B)
node scripts/tools-export.ts # regenerate src/data/tools.json (tool calling with Qwen3-1.7B)
node scripts/rag-export.ts   # regenerate src/data/rag.json (MiniLM retrieval, Qwen3-1.7B answers)
node scripts/multi-export.ts # regenerate src/data/multi.json (an orchestrator and workers, Qwen3-1.7B)
node scripts/backprop-export.ts # regenerate src/data/backprop.json (GPT-2 small's gradients)
```

The browser never runs a model. `scripts/gpt2-export.ts` runs GPT-2 small once, offline, in plain TypeScript, and exports the slices the pages draw to `src/data/gpt2.json`; `scripts/bert-export.ts` and `scripts/t5-export.ts` do the same for BERT-base and T5-small (`src/data/bert.json`, `src/data/t5.json`). The detail views that animate every matrix product use a toy model at d_model 8 and say so.

## Structure

```
src/
  main.ts                 sidebar navigation, hash router, zoom transitions between views
  styles.css              design tokens (colors, fonts) and page frame styles
  core/
    stage.ts              DPR-aware canvas and rAF loop
    player.ts             phase timeline: play, scrub, speed, keyboard shortcuts
    frame.ts              exhibit page frame: header, specs, caption, controls
    draw.ts               shared drawing primitives: token chips, glass plates, labels, math text
    matrix.ts             matrices as 2.5D slabs, GEMM cell schedules, hover and the formula line
    theme.ts              canvas palette read from the CSS tokens; follows light/dark changes
  lib/gpt2/
    bpe.ts                GPT-2's byte-level BPE tokenizer
    data.ts               decodes the exported GPT-2 run: distributions at any T, head kinds, lane colours
  data/gpt2.json          real GPT-2 small activations for the overview and Unembed
  exhibits/
    start.ts              landing page
    registry.ts           categories, sub-headings and exhibits (with child steps); an exhibit goes live once it has a route
    transformer/
      model.ts            toy model for the detail views: attention and MLP arithmetic at d_model 8
      overview.ts         forward-pass overview
      tokenizer.ts        tokenizer detail view
      embedding.ts        embedding detail view
      layernorm.ts        layernorm & residual detail view
      attention.ts        attention detail view
      mlp.ts              MLP detail view
      unembed.ts          unembed & sampling detail view
    training/
      loss.ts             next-token loss on a real GPT-2 run
      backprop.ts         the chain rule through GPT-2, checked
      optimizer.ts        SGD, momentum, Adam, schedules
      bpetrain.ts         BPE learned in the page
    lineage/
      transformer2017.ts  the 2017 Transformer compared with GPT-2
      llama.ts            LLaMA 3 compared with GPT-2
      mixtral.ts          Mixtral 8x7B compared with GPT-2
      deepseek.ts         DeepSeek-V3 compared with GPT-2
      bert.ts             BERT-base compared with GPT-2
      t5.ts               T5 compared with GPT-2 and the 2017 Transformer
      vit.ts              ViT-B/16 compared with GPT-2
      clip.ts             CLIP compared with GPT-2
      dit.ts              DiT-XL/2 compared with GPT-2
      mamba.ts            Mamba-130m compared with GPT-2
    serving/
      kvcache.ts          the KV cache: prefill, decode, size, roofline
      flashattention.ts   exact attention in SRAM tiles
      paged.ts            the KV cache in blocks (vLLM)
      batching.ts         static vs continuous batching, chunked prefill
      speculative.ts      real distilgpt2 → GPT-2 speculative decoding
      quantization.ts     int8 and int4 on the real GPT-2
    agents/
      common.ts           next-token bars, prompts as chips, the model glyph
      incontext.ts        few-shot GPT-2, an induction head, Qwen3 on an instruction
      react.ts            Thought → Action → Observation with Qwen3
      toolcalling.ts      schemas, <tool_call>, <tool_response>, masks
      rag.ts              embeddings, cosine ranking, grounded answers
      multiagent.ts       orchestrator, workers, handoffs, cost
scripts/
  gpt2-export.ts          offline GPT-2 small forward pass that writes src/data/gpt2.json
  bert-export.ts          offline BERT-base forward pass that writes src/data/bert.json
  t5-export.ts            offline T5-small encoder–decoder that writes src/data/t5.json
  vit-export.ts           two ViT-B/16 tensors → src/data/vit.json
  clip-export.ts          offline CLIP ViT-B/32 on drawn shapes → src/data/clip.json
  dit-export.ts           DiT-XL/2 adaLN-Zero modulation → src/data/dit.json
  mamba-export.ts         offline Mamba-130m → src/data/mamba.json
  spec-export.ts          speculative decoding, distilgpt2 → GPT-2 → src/data/spec.json
  quant-export.ts         GPT-2 small quantized and rerun → src/data/quant.json
  gpt2-model.ts           the GPT-2 forward pass the two scripts above share
  qwen-model.ts           Qwen3-1.7B chat model (tokenizer, chat template, KV cache) for the Agents exports
  icl-export.ts           GPT-2 few-shot and Qwen3 on an instruction → src/data/icl.json
  react-export.ts         a real ReAct loop with Qwen3 → src/data/react.json
  tools-export.ts         Qwen3 tool calling → src/data/tools.json
  rag-export.ts           MiniLM retrieval + Qwen3 answers → src/data/rag.json
  multi-export.ts         orchestrator + workers with Qwen3 → src/data/multi.json
  backprop-export.ts      GPT-2 backprop in float64 → src/data/backprop.json
```

To add an exhibit, write a `mount(root, nav) => destroy` function under `exhibits/` and give its entry in `registry.ts` a `route` and `mount`.
