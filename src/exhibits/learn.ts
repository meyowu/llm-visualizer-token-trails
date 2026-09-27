import type { Check, Code, Player } from '../core/player'

/*
 * What each page teaches beyond its drawing: the matching lines of real code (nanoGPT-style
 * PyTorch, and the Hugging Face names for LLaMA), predict-then-reveal questions, and a recap.
 */

interface Learn { code: Code; checks: Check[]; recap: string[]; refs?: [string, string][] }

const LEARN: Record<string, Learn> = {
  overview: {
    refs: [["Radford et al. 2019, Language Models are Unsupervised Multitask Learners (GPT-2)", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"], ["Karpathy, nanoGPT: the whole model in about 300 lines", "https://github.com/karpathy/nanoGPT"], ["nostalgebraist 2020, interpreting GPT: the logit lens", "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens"]],
    code: {
      lines: [
        'idx = torch.tensor([enc.encode("The cat sat on the")])   # (1, T) token ids',
        'x = wte(idx) + wpe(torch.arange(idx.size(1)))             # (1, T, 768)',
        'for block in h:                                           # 12 blocks, own weights',
        '    x = x + block.attn(block.ln_1(x))',
        '    x = x + block.mlp(block.ln_2(x))',
        'logits = lm_head(ln_f(x)[:, -1, :])                       # last position only',
        'idx_next = logits.argmax(-1, keepdim=True)                # greedy',
        'idx = torch.cat((idx, idx_next), dim=1)                   # append, next pass',
      ],
      at: { tokenize: [0], embed: [1], attn: [2, 3], mlp: [4], stack: [2, 3, 4], unembed: [5], pick: [6, 7] },
    },
    checks: [
      { phase: 'unembed', q: 'Which position’s vector is used to predict the next token?', options: ['Only the last one', 'All of them, averaged', 'The first one', 'The one with the largest norm'], answer: 0, why: 'Only the last position is unembedded. The other positions were still needed: attention read their keys and values.' },
    ],
    recap: [],
  },
  tokenizer: {
    refs: [["Sennrich et al. 2016, Neural Machine Translation of Rare Words with Subword Units (BPE)", "https://arxiv.org/abs/1508.07909"], ["OpenAI, GPT-2 encoder.py: the reference byte-level BPE", "https://github.com/openai/gpt-2/blob/master/src/encoder.py"]],
    code: {
      lines: [
        'import regex as re                  # pip install regex (\\p{L} needs it)',
        'pat = re.compile(r"""\'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+""")',
        '',
        'def encode(text):',
        '    ids = []',
        '    for piece in pat.findall(text):',
        '        word = [byte_char[b] for b in piece.encode("utf-8")]    # " " → "Ġ"',
        '        while len(word) > 1:',
        '            pairs = list(zip(word, word[1:]))',
        '            best = min(pairs, key=lambda p: ranks.get(p, float("inf")))',
        '            if best not in ranks: break                         # no rule left',
        '            word = merge(word, best)                            # every a, b → ab',
        '        ids += [vocab[s] for s in word]                         # 256 + merge rank',
        '    return ids',
        '',
        'def decode(ids):',
        '    data = b"".join(token_bytes[i] for i in ids)        # a lookup and a join',
        '    return data.decode("utf-8", errors="replace")',
      ],
      at: { split: [1, 5], bytes: [6], merge: [7, 8, 9, 10, 11], ids: [12, 13], decode: [15, 16, 17] },
    },
    checks: [
      { phase: 'ids', q: 'Why is Ġthe’s id (262) so much smaller than Ġcat’s (3797)?', options: ['It was merged earlier in training, because it is more common', 'Ids are in alphabetical order', 'Shorter tokens get smaller ids', 'Ids are assigned at random'], answer: 0, why: 'A merged token’s id is 256 + its merge rank, and merges were learned in order of frequency.' },
    ],
    recap: [
      'Text is split into pieces by a regex, then each piece into its UTF-8 bytes: 256 base symbols.',
      'Inside each piece the lowest-ranked adjacent pair is merged, again and again; common words become one token, rare ones several.',
      'A token’s id is 256 plus its merge rank, so common tokens have small ids.',
    ],
  },
  embedding: {
    refs: [["Vaswani et al. 2017, Attention Is All You Need, §3.4–3.5 (embeddings, positions)", "https://arxiv.org/abs/1706.03762"]],
    code: {
      lines: [
        '# idx: (B, T) token ids',
        '# wte(idx) equals F.one_hot(idx, 50257).float() @ wte.weight, without the multiply',
        'tok_emb = self.transformer.wte(idx)          # (B, T, 768)',
        'pos = torch.arange(0, T, device=idx.device)',
        'pos_emb = self.transformer.wpe(pos)          # (T, 768)',
        'x = self.transformer.drop(tok_emb + pos_emb) # the residual stream',
      ],
      at: { onehot: [0, 1], lookup: [2], pos: [3, 4, 5], stream: [5] },
    },
    checks: [
      { phase: 'pos', q: 'Without W_P, how would “The dog bit the man” and “The man bit the dog” look to the model?', options: ['The same five E rows, only in another order', 'Completely different rows', 'Identical in every way, order included', 'The model would refuse the input'], answer: 0, why: 'E depends only on which tokens appear. Attention on its own does not care about order, so position rows are what tell the two sentences apart.' },
    ],
    recap: [
      'An id selects one row of W_E: a learned 768-number vector for that token.',
      'Row i of W_P is added for position i, so word order survives.',
      'The sum is the residual stream that every block reads and adds to.',
    ],
  },
  layernorm: {
    refs: [["Ba et al. 2016, Layer Normalization", "https://arxiv.org/abs/1607.06450"], ["Xiong et al. 2020, On Layer Normalization in the Transformer Architecture (pre-LN vs post-LN)", "https://arxiv.org/abs/2002.04745"]],
    code: {
      lines: [
        'class Block(nn.Module):',
        '    def forward(self, x):',
        '        x = x + self.attn(self.ln_1(x))   # read a normalised copy, add back',
        '        x = x + self.mlp(self.ln_2(x))',
        '        return x',
        '',
        '# what ln_1 computes for each token\'s 768 features (F.layer_norm):',
        'mu = x.mean(-1, keepdim=True)',
        'var = x.var(-1, keepdim=True, unbiased=False)',
        'x_hat = (x - mu) / torch.sqrt(var + 1e-5)',
        'y = self.weight * x_hat + self.bias      # γ ⊙ x̂ + β',
      ],
      at: { stream: [0, 1, 2, 3, 4], mean: [7], scale: [8, 9], affine: [10] },
    },
    checks: [
      { phase: 'scale', q: 'After subtracting μ and dividing by σ, what is the spread (standard deviation) of each token’s features?', options: ['1', '0', '768', 'It depends on the token'], answer: 0, why: 'That is the point of dividing by σ: every token comes out with mean 0 and spread 1, before γ and β.' },
    ],
    recap: [
      'Each sub-layer reads a normalised copy of the stream and adds its output back.',
      'LayerNorm centres each token’s features (− μ) and scales them to unit spread (÷ σ).',
      'Learned γ and β then set each feature’s scale and offset.',
    ],
  },
  attention: {
    refs: [["Vaswani et al. 2017, Attention Is All You Need, §3.2", "https://arxiv.org/abs/1706.03762"], ["Olsson et al. 2022, In-context Learning and Induction Heads", "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html"]],
    code: {
      lines: [
        'B, T, C = x.size()                                    # x = ln_1(h)',
        'q, k, v = self.c_attn(x).split(self.n_embd, dim=2)    # one GEMM for all three',
        'q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, 12, T, 64)',
        'k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)',
        'v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)',
        'att = q @ k.transpose(-2, -1)                         # (B, 12, T, T)',
        'att = att * (1.0 / math.sqrt(k.size(-1)))             # ÷ √64',
        'att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))',
        'att = F.softmax(att, dim=-1)',
        'y = att @ v                                           # (B, 12, T, 64)',
        'y = y.transpose(1, 2).contiguous().view(B, T, C)      # heads side by side',
        'y = self.c_proj(y)                                    # · W_O',
      ],
      at: { qkv: [0, 1, 2, 3, 4], scores: [5], scale: [6], mask: [7], softmax: [8], av: [9], out: [10, 11] },
    },
    checks: [
      { phase: 'mask', q: 'Which cells of the score matrix will the causal mask set to −∞?', options: ['Those with j > i: later tokens', 'Those with j < i: earlier tokens', 'The diagonal', 'None, in GPT-2'], answer: 0, why: 'Position i predicts token i + 1, so it may look at itself and earlier positions only.' },
      { phase: 'softmax', q: 'After softmax, what does each row of A add up to?', options: ['1', '0', 'The number of tokens', 'It depends on the scores'], answer: 0, why: 'Softmax turns each row into positive weights that sum to 1; masked cells become exactly 0.' },
    ],
    recap: [
      'Queries are scored against keys, Q · Kᵀ, and divided by √d_head.',
      'The causal mask and softmax turn each row into weights over itself and earlier tokens.',
      'Each token takes the weighted mix of values; the heads are joined and projected by W_O.',
    ],
  },
  mlp: {
    refs: [["Hendrycks & Gimpel 2016, Gaussian Error Linear Units (GELU)", "https://arxiv.org/abs/1606.08415"], ["Geva et al. 2021, Transformer Feed-Forward Layers Are Key-Value Memories", "https://arxiv.org/abs/2012.14913"]],
    code: {
      lines: [
        'def forward(self, x):          # x = ln_2(h): (B, T, 768)',
        '    x = self.c_fc(x)           # (B, T, 3072)',
        '    x = self.gelu(x)           # GPT-2 used the tanh approximation',
        '    x = self.c_proj(x)         # (B, T, 768)',
        '    x = self.dropout(x)',
        '    return x',
        '# in the block: x = x + self.mlp(self.ln_2(x))',
      ],
      at: { up: [0, 1], gelu: [2], down: [3, 4, 5], resid: [6] },
    },
    checks: [
      { phase: 'gelu', q: 'Without GELU, what would W_fc followed by W_proj amount to?', options: ['A single 768 × 768 matrix', 'Nothing at all', 'A bigger, more powerful layer', 'A softmax'], answer: 0, why: 'Two linear maps in a row are one linear map. The bend in between is what lets the MLP compute more than a matrix could.' },
    ],
    recap: [
      'Each token is widened to 4 × (3,072 numbers), bent by GELU, and projected back to 768.',
      'Without the nonlinearity the two matrices would collapse into one.',
      'The MLP works on each token alone; its output is added to the residual stream.',
    ],
  },
  unembed: {
    refs: [["Press & Wolf 2017, Using the Output Embedding to Improve Language Models (weight tying)", "https://arxiv.org/abs/1608.05859"], ["Holtzman et al. 2020, The Curious Case of Neural Text Degeneration (top-p)", "https://arxiv.org/abs/1904.09751"]],
    code: {
      lines: [
        'x = self.transformer.ln_f(x)',
        'logits = self.lm_head(x[:, [-1], :])      # lm_head.weight is wte.weight (tied)',
        'logits = logits[:, -1, :] / temperature',
        'if top_k is not None:',
        '    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))',
        '    logits[logits < v[:, [-1]]] = -float("Inf")',
        'probs = F.softmax(logits, dim=-1)',
        'idx_next = torch.multinomial(probs, num_samples=1)   # greedy: probs.argmax(-1)',
        'idx = torch.cat((idx, idx_next), dim=1)              # append, run again',
      ],
      at: { lnf: [0], logits: [1], temp: [2], softmax: [3, 4, 5, 6], sample: [7, 8] },
    },
    checks: [
      { phase: 'temp', q: 'What happens to sampling as the temperature T goes towards 0?', options: ['It becomes greedy: always the top token', 'It becomes uniformly random', 'Nothing changes', 'It keeps only the top 3 tokens'], answer: 0, why: 'Dividing by a tiny T stretches every gap between logits, so the top token takes nearly all the probability.' },
    ],
    recap: [
      'Only the last position is scored: ln_f, then a dot product with every row of W_E.',
      'Temperature divides the logits; softmax turns them into probabilities over all 50,257 tokens.',
      'A strategy picks one token; it is appended and the model runs again.',
    ],
  },
  foundations: {
    refs: [["3Blue1Brown, Essence of linear algebra", "https://www.3blue1brown.com/topics/linear-algebra"]],
    code: {
      lines: [
        'a @ b                        # dot product: (a * b).sum()',
        'C = A @ B                    # (3, 4) @ (4, 3) → (3, 3); C[i, j] = A[i] @ B[:, j]',
        'p = torch.softmax(s, dim=-1) # exp(s) / exp(s).sum(), rows sum to 1',
        'F.one_hot(i, 4).float() @ M  # == M[i]',
      ],
      at: { dot: [0], matmul: [1], softmax: [2], onehot: [3] },
    },
    checks: [
      { phase: 'softmax', q: 'Softmax of the scores [2, 1, 0.2, −1]: which is true?', options: ['All four get some weight, most goes to the 2', 'The 2 gets everything', 'The −1 gets a negative weight', 'They all get 1/4'], answer: 0, why: 'exp is never 0 and never negative, so every score keeps a share, and larger scores get exponentially more.' },
    ],
    recap: [
      'A dot product multiplies and adds; it is large when two vectors point the same way.',
      'A matrix product is a grid of dot products: row i of A (left) meets column j of B (above).',
      'Softmax turns scores into weights that sum to 1, and a one-hot row times a matrix picks one row.',
    ],
  },
  loss: {
    refs: [["Karpathy, nanoGPT train.py: the training loop", "https://github.com/karpathy/nanoGPT/blob/master/train.py"], ["Radford et al. 2019, Language Models are Unsupervised Multitask Learners", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"]],
    code: {
      lines: [
        'x, y = idx[:, :-1], idx[:, 1:]             # inputs and targets, shifted by one',
        'logits = model(x)                           # (B, T, 50257): every position at once',
        'probs = F.softmax(logits, dim=-1)',
        'loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))   # mean of −log p',
        'loss.backward()                             # dloss/dlogits = (probs − onehot(y)) / N, then back through every layer',
        'optimizer.step(); optimizer.zero_grad()     # nudge all 124M weights',
      ],
      at: { shift: [0], predict: [1, 2], loss: [3], grad: [4], step: [5] },
    },
    checks: [
      { phase: 'loss', q: 'GPT-2 gives the right token 45% at one position and 0.07% at another. Which adds more to the loss?', options: ['The 0.07% one, by far', 'The 45% one', 'They add the same', 'Neither: only the last position counts'], answer: 0, why: '−ln 0.45 ≈ 0.8 but −ln 0.0007 ≈ 7.3: the loss punishes confident mistakes much more than mild ones.' },
    ],
    recap: [
      'One sentence gives one training example per position: each predicts the token after it.',
      'The loss is the average of −log p(right token), the cross-entropy.',
      'Its gradient on the logits is p − onehot(target); backpropagation carries it to every weight.',
    ],
  },
  transformer2017: {
    refs: [["Vaswani et al. 2017, Attention Is All You Need", "https://arxiv.org/abs/1706.03762"], ["Bahdanau et al. 2014, Neural Machine Translation by Jointly Learning to Align and Translate", "https://arxiv.org/abs/1409.0473"], ["Xiong et al. 2020, On Layer Normalization in the Transformer Architecture", "https://arxiv.org/abs/2002.04745"], ["Rush et al., The Annotated Transformer", "https://nlp.seas.harvard.edu/annotated-transformer/"]],
    code: {
      lines: [
        '# positions: fixed sine and cosine waves, added to embeddings scaled by √d_model',
        'pe[:, 0::2] = torch.sin(pos * 10000 ** (-i2 / d_model))',
        'pe[:, 1::2] = torch.cos(pos * 10000 ** (-i2 / d_model))',
        'x = embed(src) * math.sqrt(d_model) + pe[:len(src)]',
        '# encoder layer, 6 times: LayerNorm after each residual add (post-LN)',
        'x = norm1(x + self_attn(x, x, x))                     # no mask',
        'x = norm2(x + ffn(x))                                 # ReLU',
        '# decoder layer, 6 times',
        'y = norm1(y + self_attn(y, y, y, mask=causal))',
        'y = norm2(y + cross_attn(q=y, k=memory, v=memory))    # memory = encoder output',
        'y = norm3(y + ffn(y))',
        '# GPT-2, for comparison: pre-LN',
        'h = x + attn(ln_1(x)); out = h + mlp(ln_2(h))',
      ],
      at: { blocks: [4, 5, 6, 7, 8, 9, 10], translate: [3, 9], masks: [5, 8, 9], cross: [9], postln: [5, 6, 11, 12], pos: [0, 1, 2, 3] },
    },
    checks: [
      { phase: 'rnn', q: 'Why could the Transformer train much faster than an RNN on the same GPU?', options: ['It computes all positions at once instead of one step after another', 'It has fewer parameters', 'It uses a smaller vocabulary', 'It needs no position information'], answer: 0, why: 'Each RNN step needs the previous hidden state. Attention compares all positions in one matrix product, so the sequence is not a chain of steps.' },
      { phase: 'cross', q: 'In cross-attention, where do the queries, keys and values come from?', options: ['Queries from the decoder; keys and values from the encoder output', 'Queries from the encoder; keys and values from the decoder', 'All three from the decoder, under a causal mask', 'All three from the encoder'], answer: 0, why: 'Each target position asks (query) what it needs; the source offers keys and values. That is why the score matrix is target × source.' },
      { phase: 'postln', q: 'Where does the 2017 Transformer apply LayerNorm?', options: ['After each residual add, on the stream itself', 'Before each sub-layer, on the copy it reads', 'Only once, after the last layer', 'Only in the decoder'], answer: 0, why: 'Post-LN: x = LN(x + f(x)). GPT-2 uses pre-LN, x = x + f(LN(x)), plus one final ln_f.' },
    ],
    recap: [
      'The 2017 Transformer is an encoder–decoder: the encoder reads the source once, the decoder writes the target token by token.',
      'Cross-attention takes queries from the decoder and keys and values from the encoder output.',
      'It normalises after each residual add and adds fixed sinusoids; GPT-2 moved to pre-LN and learned positions.',
    ],
  },
  mixtral: {
    refs: [["Jiang et al. 2024, Mixtral of Experts", "https://arxiv.org/abs/2401.04088"], ["Shazeer et al. 2017, Outrageously Large Neural Networks: the Sparsely-Gated Mixture-of-Experts Layer", "https://arxiv.org/abs/1701.06538"], ["Fedus et al. 2021, Switch Transformers (load-balancing loss)", "https://arxiv.org/abs/2101.03961"]],
    code: {
      lines: [
        '# the block is LLaMA\'s, with a sparse MoE layer where the MLP was',
        'h = x + self_attn(rms_norm(x)); out = h + moe(rms_norm(h))',
        '# moe: score the 8 experts, keep the best 2, renormalise their weights',
        'router_logits = self.gate(x)                          # (tokens, 8)',
        'weights = F.softmax(router_logits, dim=-1)',
        'weights, experts = torch.topk(weights, 2, dim=-1)',
        'weights /= weights.sum(dim=-1, keepdim=True)            # = softmax over the top 2',
        'y = sum(w * self.experts[e](x) for w, e in zip(weights, experts))',
        '# each expert is a SwiGLU MLP',
        'expert_out = self.w2(F.silu(self.w1(x)) * self.w3(x))',
        '# training adds a balancing loss: share routed × mean probability',
        'aux = n_experts * (f * P).sum()',
      ],
      at: { blocks: [0, 1], route: [2, 3, 4, 5, 6], dispatch: [7, 8, 9], params: [7, 9], balance: [10, 11] },
    },
    checks: [
      { phase: 'dispatch', q: 'Mixtral has 8 experts per layer and runs 2 for each token. Roughly what share of the expert weights does one token use?', options: ['A quarter', 'An eighth', 'Two thirds', 'All of them'], answer: 0, why: '2 of 8 experts: a quarter of the expert weights, in every layer. The attention weights are shared and always used.' },
      { phase: 'params', q: 'Mixtral stores 46.7B parameters. What sets its compute per token?', options: ['The 12.9B parameters a token actually runs through', 'All 46.7B parameters', 'Only the router', 'The size of the vocabulary'], answer: 0, why: 'About 2 FLOPs per weight used: the shared parts plus 2 experts per layer, 12.9B. Memory still has to hold all 46.7B.' },
    ],
    recap: [
      'Mixtral keeps LLaMA’s attention and replaces each MLP with 8 expert MLPs and a router.',
      'The router scores the experts for each token; the top 2 run, mixed by a softmax over their two scores.',
      'It stores 46.7B parameters but uses 12.9B per token; a balancing loss keeps the experts evenly used.',
    ],
  },
  deepseek: {
    refs: [["DeepSeek-AI 2024, DeepSeek-V3 Technical Report", "https://arxiv.org/abs/2412.19437"], ["DeepSeek-AI 2024, DeepSeek-V2 (multi-head latent attention)", "https://arxiv.org/abs/2405.04434"], ["Dai et al. 2024, DeepSeekMoE", "https://arxiv.org/abs/2401.06066"], ["Wang et al. 2024, Auxiliary-Loss-Free Load Balancing for Mixture-of-Experts", "https://arxiv.org/abs/2408.15664"]],
    code: {
      lines: [
        '# MLA: cache one small latent per token instead of every head\'s K and V',
        'c_kv = kv_a_proj(h)                          # (tokens, 512)  cached',
        'k_rope = rope(k_rope_proj(h))                # (tokens, 64)   cached, shared by all heads',
        'k_nope, v = kv_b_proj(c_kv).split(...)       # each head\'s keys and values, rebuilt',
        'k = torch.cat([k_nope, k_rope.expand(heads)], dim=-1)',
        '# DeepSeekMoE: one shared expert plus the top 8 of 256 routed experts',
        's = torch.sigmoid(gate(x))                   # (tokens, 256) affinities',
        'idx = torch.topk(s + bias, 8).indices        # the bias only affects the choice',
        'w = s.gather(-1, idx); w = w / w.sum(-1, keepdim=True)',
        'y = shared_expert(x) + sum(w[..., j] * experts[idx[..., j]](x) for j in range(8))',
        '# after each step: nudge each expert\'s bias toward an even load',
        'bias += gamma * torch.sign(load.mean() - load)',
      ],
      at: { blocks: [0, 5], mla: [0, 1, 2, 3, 4], cache: [1, 2], moe: [5, 6, 7, 8, 9], balance: [7, 10, 11] },
    },
    checks: [
      { phase: 'cache', q: 'In MLA, what goes into the KV cache for each token and layer?', options: ['A 512-number latent plus a 64-number RoPE key', 'Every head’s keys and values', 'Only the query', 'Nothing: keys are recomputed from the text'], answer: 0, why: 'Keys and values are rebuilt from the latent by up-projections, so the latent (and the small shared RoPE key) is all that needs keeping.' },
      { phase: 'balance', q: 'DeepSeek-V3 adds a bias to each expert’s score. What does the bias change?', options: ['Only which experts are chosen', 'How much each expert’s output counts', 'The training loss', 'The size of each expert'], answer: 0, why: 'The bias steers the top-8 choice toward idle experts; the gate weights still come from the unbiased scores.' },
    ],
    recap: [
      'DeepSeek-V3 changes both halves of GPT-2’s block: MLA for attention and DeepSeekMoE for the MLP.',
      'MLA caches one small latent per token and rebuilds keys and values from it: 57× less cache than MHA.',
      'DeepSeekMoE runs 1 shared and 8 of 256 small experts, kept balanced by a per-expert bias instead of an extra loss.',
    ],
  },
  llama: {
    refs: [["Touvron et al. 2023, LLaMA", "https://arxiv.org/abs/2302.13971"], ["Llama Team 2024, The Llama 3 Herd of Models", "https://arxiv.org/abs/2407.21783"], ["Su et al. 2021, RoFormer (RoPE)", "https://arxiv.org/abs/2104.09864"], ["Zhang & Sennrich 2019, Root Mean Square Layer Normalization", "https://arxiv.org/abs/1910.07467"], ["Shazeer 2020, GLU Variants Improve Transformer (SwiGLU)", "https://arxiv.org/abs/2002.05202"], ["Ainslie et al. 2023, GQA", "https://arxiv.org/abs/2305.13245"]],
    code: {
      lines: [
        '# RMSNorm',
        'x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * self.weight',
        '# RoPE, inside attention: rotate each pair of q and k by position × θ',
        'q, k = apply_rotary_pos_emb(q, k, cos, sin)',
        '# GQA: 8 key/value heads serve 32 query heads',
        'k, v = repeat_kv(k, 4), repeat_kv(v, 4)',
        '# SwiGLU MLP',
        'y = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))',
        '# the block, shaped like GPT-2\'s',
        'h = x + self.self_attn(self.input_layernorm(x))',
        'out = h + self.mlp(self.post_attention_layernorm(h))',
      ],
      at: { blocks: [8, 9, 10], rope: [2, 3], rms: [0, 1], swiglu: [6, 7], gqa: [4, 5] },
    },
    checks: [
      { phase: 'gqa', q: 'With 32 query heads and 8 key/value heads, how much smaller is the KV cache than with one key/value head per query head?', options: ['4 ×', '8 ×', '32 ×', 'The same size'], answer: 0, why: 'The cache stores keys and values only: 8 heads instead of 32 is 4 × less.' },
    ],
    recap: [
      'LLaMA keeps GPT-2’s pre-norm block with residual adds.',
      'RoPE rotates q and k instead of adding position vectors; RMSNorm drops the mean.',
      'SwiGLU gates the MLP, and GQA shares keys and values to shrink the KV cache.',
    ],
  },
}

/** Give a page's player its code drawer, questions and recap. */
export function teach(player: Player, page: keyof typeof LEARN) {
  const l = LEARN[page]
  player.checks = l.checks
  player.recap = l.recap
  player.setCode(l.code)
  player.setRefs(l.refs ?? [])
}
