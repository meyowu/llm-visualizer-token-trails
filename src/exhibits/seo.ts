/*
 * What each page is called in search results, browser tabs and share cards: a title in the words people search for
 * (the site's name is added after it, except on the home page) and one sentence on what the page shows. main.ts sets
 * both as a page opens, through t(); the Chinese is in src/locales/zh/pages.ts; scripts/pages.mjs writes them into
 * every page file and fails when a live route has none. No imports, so Node can read it.
 */
export const SEO: Record<string, [title: string, description: string]> = {
  home: ['Token Trails: how a language model works, token by token', 'Animated, explorable walk-throughs of how language models work, with the numbers of real models: GPT-2’s forward pass, attention, training, serving and agents.'],
  start: ['How a language model works: next-token prediction', 'Start here: what a Transformer language model does, a live GPT-2 next-token example, where GPT-2’s 124M parameters live, and how to read the animated pages.'],
  foundations: ['Dot product, matmul and softmax, visualized', 'The four pieces of math every Transformer page builds on, animated with small numbers: the dot product, the matrix product, softmax and a one-hot lookup.'],
  glossary: ['LLM glossary: Transformer terms in plain words', 'Plain-word definitions of over 100 terms behind language models, from attention, BPE and the KV cache to LoRA, DPO, RAG and mixture of experts.'],

  anatomy: ['How a Transformer works: GPT-2’s forward pass', 'Follow one prompt through GPT-2 small with its real activations: tokenizer, embeddings, attention and MLP in 12 blocks, then the next-token probabilities.'],
  'anatomy/tokenizer': ['BPE tokenizer explained: GPT-2’s byte-level BPE', 'Type any text and watch GPT-2’s real byte-level BPE tokenize it: the pre-split, UTF-8 bytes, merges by rank from its 50,000 rules, token ids and decoding.'],
  'anatomy/embedding': ['Token and position embeddings, visualized', 'How GPT-2 turns token ids into vectors: a one-hot row times the embedding matrix is a row lookup, a learned position vector is added, and the sum enters the residual stream.'],
  'anatomy/layernorm': ['LayerNorm and the residual stream, visualized', 'Pre-LN Transformers explained: the residual stream every block adds to, and LayerNorm step by step (subtract the mean, divide by the spread, then scale and shift).'],
  'anatomy/attention': ['Self-attention explained: Q, K, V step by step', 'Causal self-attention animated cell by cell: the Q, K and V projections, QKᵀ scores, scaling, the causal mask, softmax, the weighted sum and the output projection.'],
  'anatomy/mlp': ['The Transformer MLP: up-projection, GELU, down', 'GPT-2’s feed-forward layer animated as matrix products: the up-projection from 768 to 3,072, the GELU nonlinearity, the down-projection back and the residual add.'],
  'anatomy/unembed': ['Logits, temperature and sampling, explained', 'How a language model picks the next token, on a real GPT-2 run: final LayerNorm, logits from the tied embedding matrix, temperature, softmax, greedy, top-k and top-p.'],

  'lineage/compare': ['LLM architecture comparison: any two of 29 models', 'Compare the architectures of 29 language models, GPT-2 to 2025, read from their own checkpoints: layers, attention kinds, heads, KV cache per token, experts and parameters.'],
  'lineage/transformer-2017': ['Attention Is All You Need, visualized: 2017 vs GPT-2', 'The original encoder–decoder Transformer next to GPT-2: why attention replaced RNNs, a toy translation, three kinds of attention, cross-attention, post-LN and sinusoidal positions.'],
  'lineage/llama': ['LLaMA vs GPT-2: RoPE, RMSNorm, SwiGLU and GQA', 'What LLaMA 3 changed in GPT-2’s block, with LLaMA 3 8B’s real sizes: rotary position embeddings (RoPE), RMSNorm, the SwiGLU MLP and grouped-query attention.'],
  'lineage/mixtral': ['Mixture of experts explained: Mixtral vs GPT-2', 'Mixtral 8x7B’s mixture of experts in place of the MLP: the router, two experts per token, dispatch and combine, stored against active parameters, and load balancing.'],
  'lineage/deepseek': ['DeepSeek-V3 explained: MLA and DeepSeekMoE', 'DeepSeek-V3 against GPT-2: multi-head latent attention (MLA) and its KV cache per token next to MHA, GQA and MQA, shared and routed experts, and load balancing by bias.'],
  'lineage/bert': ['BERT vs GPT-2: bidirectional vs causal attention', 'BERT-base has GPT-2 small’s shape but reads both ways: real attention heads, WordPiece sentence pairs, real masked-word predictions next to GPT-2’s, and the [CLS] classifier.'],
  'lineage/t5': ['T5 explained: text-to-text and relative positions', 'T5-small, an encoder–decoder, next to GPT-2: real text-to-text answers, span corruption, relative position buckets and the position biases each head learned.'],
  'lineage/vit': ['How Vision Transformers work: images as patches', 'ViT-B/16 next to GPT-2: an image cut into 16 × 16 patches, the patch embedding and its real filters, learned 2D positions, and one label read from the [CLS] token.'],
  'lineage/clip': ['CLIP explained: images and text in one space', 'CLIP ViT-B/32 with real embeddings: two encoders, one shared space and its modality gap, the contrastive image–text matrix and loss, and zero-shot classification.'],
  'lineage/dit': ['Diffusion Transformer (DiT) explained', 'DiT-XL/2 next to GPT-2: noise added on DiT’s real schedule, latent patches as tokens, adaLN-Zero conditioning with real gates and scales, and a 250-step sampling run.'],
  'lineage/mamba': ['Mamba vs Transformer: selective state spaces', 'Mamba-130m next to GPT-2 small: one mixer per block, a fixed-size state instead of a KV cache, the selective scan, the real Δ of every layer and real next-token guesses.'],

  'training/loss': ['Next-token loss: cross-entropy, visualized', 'How a language model is scored in training, on a real GPT-2 run: every position predicts the next token, −log p is the loss, and p − y is the gradient on the logits.'],
  'training/backprop': ['Backpropagation through GPT-2, visualized', 'The chain rule from the loss back to all of GPT-2 small’s weights, with real gradients: dW = Xᵀ · dY, dX = dY · Wᵀ, the causal mask backwards and finite-difference checks.'],
  'training/optimizer': ['SGD, momentum, Adam and AdamW, visualized', 'Optimizers on a toy loss valley with real arithmetic: gradient descent, momentum, Adam’s step size per weight, AdamW’s decoupled weight decay, warmup and cosine decay.'],
  'training/tokenizer': ['How a BPE tokenizer is trained, step by step', 'Byte-pair encoding trained live in the page on this site’s glossary: start from bytes, count adjacent pairs, merge the most frequent, repeat, then compare with GPT-2’s merges.'],
  'training/scaling': ['Scaling laws and Chinchilla, visualized', 'GPT-2 small, medium and large on text none of them saw, with a power-law fit; the Chinchilla fit L(N, D), isoFLOP curves, compute-optimal sizes and where real models sit.'],
  'training/sft': ['Instruction tuning (SFT), visualized', 'Supervised fine-tuning with Qwen3-1.7B-Base and Qwen3-1.7B: the chat template as tokens, a loss on the answer only, and what the tuned model learns to say.'],
  'training/dpo': ['RLHF and DPO explained, with a real GPT-2 run', 'Learning from preferences: a reward model and the RL loop on GPT-2’s real next-token probabilities, then DPO without a reward model, and a real DPO run on GPT-2.'],
  'training/lora': ['LoRA explained: low-rank adapters, visualized', 'LoRA fine-tuning: a frozen W plus A · B, how few weights train, a real rank-4 run on GPT-2 against full fine-tuning, and whether real weight changes are low-rank.'],

  'serving/kv-cache': ['KV cache explained: prefill vs decode', 'Why language models cache keys and values: the cost with and without a cache, prefill, a real GPT-2 decode step, cache sizes across models, and why decoding is memory-bound.'],
  'serving/flashattention': ['FlashAttention explained: tiling and online softmax', 'How FlashAttention computes exact attention in on-chip SRAM: the HBM round trips of standard attention, the online softmax, the tiled loop with real arithmetic, and the memory saved.'],
  'serving/pagedattention': ['PagedAttention and vLLM explained', 'The KV cache in fixed-size blocks, as in vLLM: contiguous reservations against blocks on demand, block tables, attention across blocks, copy-on-write sharing and larger batches.'],
  'serving/continuous-batching': ['Continuous batching for LLM serving, explained', 'Why LLM servers batch, and how continuous batching beats static batching: schedules from a toy scheduler, waiting time, latency and throughput, and chunked prefill.'],
  'serving/speculative-decoding': ['Speculative decoding explained, with a real run', 'A real run of distilgpt2 drafting tokens for GPT-2 small to check in one pass: one round, a whole run, the min(1, p/q) acceptance rule, and the speed-up.'],
  'serving/quantization': ['LLM quantization explained: int8 and int4', 'Quantizing the real GPT-2 to 8 and 4 bits: number formats, absmax rounding, one scale per tensor, channel or group, activation outliers, and what it costs the model.'],

  'agents/in-context': ['In-context learning explained, with GPT-2', 'How a prompt programs a model whose weights stay fixed: GPT-2 small given 0 to 3 examples, one question read as three tasks, the copying head, and Qwen3 on an instruction.'],
  'agents/react': ['The ReAct agent loop explained: a real run', 'A real Qwen3-1.7B agent run step by step: the ReAct prompt, Thought and Action, stopping at Observation to run the tool, the growing context, and the run without a stop.'],
  'agents/tool-calling': ['LLM tool calling explained: a real run', 'How a model calls tools, in Qwen3-1.7B’s own format: tools as JSON schemas in the prompt, the tool call it writes, checking and running it, the answer, and constrained decoding.'],
  'agents/rag': ['RAG explained: embed, retrieve, read', 'Retrieval-augmented generation over this site’s glossary: MiniLM embeddings, a 2D map, ranking by cosine similarity, the assembled prompt, and answers with and without retrieval.'],
  'agents/multi-agent': ['Multi-agent handoffs explained: a real run', 'A real Qwen3-1.7B multi-agent run: one agent alone, an orchestrator handing work to workers that start with empty contexts, the reports, a parallel timeline and the token cost.'],
}
