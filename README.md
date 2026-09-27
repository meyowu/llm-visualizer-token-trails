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
    lineage/
      transformer2017.ts  the 2017 Transformer compared with GPT-2
      llama.ts            LLaMA 3 compared with GPT-2
      mixtral.ts          Mixtral 8x7B compared with GPT-2
      deepseek.ts         DeepSeek-V3 compared with GPT-2
      bert.ts             BERT-base compared with GPT-2
      t5.ts               T5 compared with GPT-2 and the 2017 Transformer
      vit.ts              ViT-B/16 compared with GPT-2
scripts/
  gpt2-export.ts          offline GPT-2 small forward pass that writes src/data/gpt2.json
  bert-export.ts          offline BERT-base forward pass that writes src/data/bert.json
  t5-export.ts            offline T5-small encoder–decoder that writes src/data/t5.json
  vit-export.ts           two ViT-B/16 tensors → src/data/vit.json
```

To add an exhibit, write a `mount(root, nav) => destroy` function under `exhibits/` and give its entry in `registry.ts` a `route` and `mount`.
