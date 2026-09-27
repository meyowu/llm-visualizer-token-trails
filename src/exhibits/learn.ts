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
  bert: {
    refs: [["Devlin et al. 2018, BERT: Pre-training of Deep Bidirectional Transformers", "https://arxiv.org/abs/1810.04805"], ["Clark et al. 2019, What Does BERT Look At?", "https://arxiv.org/abs/1906.04341"], ["google-bert/bert-base-uncased, the weights behind the real numbers", "https://huggingface.co/google-bert/bert-base-uncased"]],
    code: {
      lines: [
        '# BERT: GPT-2 small\'s shape, no causal mask, LayerNorm after each add',
        'x = layer_norm(tok_emb[ids] + seg_emb[segments] + pos_emb[positions])',
        'for layer in layers:                          # 12 × (768 wide, 12 heads, 3,072)',
        '    x = layer_norm(x + self_attn(x))            # every token sees every token',
        '    x = layer_norm(x + mlp(x))',
        '# pretraining: predict hidden tokens from both sides',
        'logits = mlm_head(x[masked_positions])        # 30,522 WordPiece scores',
        'loss = F.cross_entropy(logits, original_ids)',
        '# using it: a classifier on the [CLS] vector',
        'probs = F.softmax(x[:, 0] @ W_cls, dim=-1)',
      ],
      at: { blocks: [0, 2, 3, 4], masks: [3], inputs: [1], mlm: [5, 6, 7], use: [8, 9] },
    },
    checks: [
      { phase: 'masks', q: 'Why can BERT’s heads look at later tokens when GPT-2’s cannot?', options: ['BERT has no causal mask: it reads whole sentences, not left to right', 'BERT has more heads', 'BERT normalises after each residual add', 'BERT’s vocabulary is smaller'], answer: 0, why: 'GPT-2 must not see the token it is about to predict, so it masks later positions. BERT predicts hidden words instead, so it can look both ways.' },
      { phase: 'mlm', q: 'In “She played the [MASK] in the orchestra”, why does BERT guess violin while GPT-2 guesses role?', options: ['BERT sees “in the orchestra” after the blank; GPT-2 sees only “She played the”', 'BERT is bigger', 'GPT-2 has never seen the word violin', 'BERT was trained only on music'], answer: 0, why: 'Both are about the same size. The difference is what each can see: the words after the blank decide it here.' },
    ],
    recap: [
      'BERT has GPT-2 small’s exact shape but no causal mask: every token sees the whole sentence.',
      'It is trained to fill in hidden words from both sides, not to predict the next word.',
      'It is used to read and classify text, through the [CLS] vector, rather than to write it.',
    ],
  },
  t5: {
    refs: [["Raffel et al. 2019, Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)", "https://arxiv.org/abs/1910.10683"], ["Shaw et al. 2018, Self-Attention with Relative Position Representations", "https://arxiv.org/abs/1803.02155"], ["google-t5/t5-small, the weights behind the real numbers", "https://huggingface.co/google-t5/t5-small"]],
    code: {
      lines: [
        '# every task is a string in and a string out',
        'ids = tok("translate English to German: I have seen the cat.").input_ids',
        'out = model.generate(ids)          # "Ich habe die Katze gesehen."',
        '# pretraining: drop spans, mark them with sentinels, write them back',
        'inp = "Thank you <extra_id_0> me to your party <extra_id_1> week."',
        'tgt = "<extra_id_0> for inviting <extra_id_1> last <extra_id_2>"',
        '# the block: norm first, scale only (RMSNorm), no biases anywhere',
        'x = x + self_attn(rms_norm(x), position_bias)',
        'x = x + ffn(rms_norm(x))                        # ReLU in T5 v1.0',
        '# one learned number per (bucket, head), added to every score',
        'position_bias = rel_bias[bucket(key_pos - query_pos)]',
        'scores = q @ k.transpose(-1, -2) + position_bias  # no ÷ √d in T5',
      ],
      at: { tasks: [0, 1, 2], spans: [3, 4, 5], blocks: [6, 7, 8], buckets: [9, 10], bias: [9, 10, 11] },
    },
    checks: [
      { phase: 'spans', q: 'In span corruption, what does the decoder have to write?', options: ['Only the dropped spans, each after its sentinel', 'The whole sentence again', 'One word per sentinel', 'The next sentence'], answer: 0, why: 'The target lists each sentinel followed by the words it replaced, so it is much shorter than the input.' },
      { phase: 'bias', q: 'Why can T5 read inputs longer than any it was trained on?', options: ['Position is a bias per relative-distance bucket, and all far distances share one bucket', 'It has a learned vector for every position up to a million', 'It uses no position information at all', 'Its sinusoids repeat'], answer: 0, why: 'Nothing depends on the absolute position: any pair farther apart than 90 tokens simply uses the last bucket.' },
    ],
    recap: [
      'T5 casts every task as text in, text out, named by a prefix such as “translate English to German:”.',
      'It is pretrained by span corruption: sentinels replace dropped spans, and the decoder writes the spans back.',
      'Position enters as a learned bias per bucket of relative distance and per head, added to the attention scores.',
    ],
  },
  vit: {
    refs: [["Dosovitskiy et al. 2020, An Image is Worth 16x16 Words (ViT)", "https://arxiv.org/abs/2010.11929"], ["google/vit-base-patch16-224, the weights behind the real numbers", "https://huggingface.co/google/vit-base-patch16-224"]],
    code: {
      lines: [
        '# ViT: an image becomes a sequence of patch tokens',
        'patches = img.unfold(16)                        # (196, 16 · 16 · 3 = 768)',
        'x = patches @ W_patch + b                       # = Conv2d(3, 768, kernel=16, stride=16)',
        'x = torch.cat([cls_token, x]) + pos_emb         # (197, 768), learned positions',
        'for block in blocks:                            # GPT-2 small\'s shape, pre-LN',
        '    x = x + attn(ln_1(x))                       # no mask: every patch sees every patch',
        '    x = x + mlp(ln_2(x))',
        'logits = head(ln_f(x[0]))                       # [CLS] only → 1,000 classes',
      ],
      at: { blocks: [4, 5, 6], patches: [1], embed: [1, 2], positions: [3], head: [7] },
    },
    checks: [
      { phase: 'positions', q: 'ViT’s position embeddings are a plain list of 197 vectors, with no 2D layout given. What did they learn?', options: ['Nearby patches, and patches in the same row or column, get similar vectors: the 2D grid', 'All positions look alike', 'Only the left-to-right reading order', 'Random noise'], answer: 0, why: 'Neighbouring patches tend to look alike, so training pulls their position vectors together; the grid emerges from the data.' },
      { phase: 'head', q: 'Which output does ViT classify?', options: ['Only the [CLS] token’s', 'Every patch’s, one label each', 'The last patch’s', 'The average of the pixels'], answer: 0, why: 'The [CLS] token has no pixels of its own; through attention it gathers what the class head needs.' },
    ],
    recap: [
      'ViT turns an image into 196 patch tokens plus a [CLS] token and runs GPT-2-sized encoder blocks on them, with no mask.',
      'Each patch is flattened and multiplied by one shared matrix: a 16 × 16 convolution with stride 16.',
      'Its learned 1D positions rediscover the 2D grid, and only the [CLS] output is classified.',
    ],
  },
  clip: {
    refs: [["Radford et al. 2021, Learning Transferable Visual Models From Natural Language Supervision (CLIP)", "https://arxiv.org/abs/2103.00020"], ["Liang et al. 2022, Mind the Gap: the modality gap in contrastive models", "https://arxiv.org/abs/2203.02053"], ["openai/clip-vit-base-patch32, the weights behind the real numbers", "https://huggingface.co/openai/clip-vit-base-patch32"]],
    code: {
      lines: [
        '# two encoders, one shared space',
        'i = F.normalize(visual_projection(vit(images)[:, 0]), dim=-1)          # (N, 512)',
        't = F.normalize(text_projection(text_tf(ids)[range(N), eot_pos]), dim=-1)  # (N, 512)',
        '# every image against every caption, scaled by the learned temperature',
        'logits = logit_scale.exp() * i @ t.T                    # (N, N), scale capped at 100',
        'labels = torch.arange(N)                                # the diagonal holds the true pairs',
        'loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2',
        '# zero-shot: one caption per class, pick the closest',
        'classes = F.normalize(text_projection(text_tf(prompts)), dim=-1)',
        'pred = (i @ classes.T).argmax(dim=-1)',
      ],
      at: { towers: [0, 1, 2], space: [1, 2], matrix: [3, 4, 5, 6], zeroshot: [7, 8, 9], scale: [4] },
    },
    checks: [
      { phase: 'matrix', q: 'In a batch of 4 image–caption pairs, which cells of the 4 × 4 similarity matrix should training push up?', options: ['The diagonal: each image with its own caption', 'The first row', 'All of them equally', 'The ones off the diagonal'], answer: 0, why: 'Pair i sits at row i, column i. Cross-entropy over each row and each column rewards the diagonal and penalises the rest.' },
      { phase: 'zeroshot', q: 'How does CLIP classify an image into classes it was never trained on?', options: ['It compares the image with a caption written for each class and picks the most similar', 'It adds a new output layer and trains it', 'It generates the class name word by word', 'It looks the image up in its training set'], answer: 0, why: 'Class names are text, so the text encoder turns them into vectors in the same space as images.' },
    ],
    recap: [
      'CLIP pairs an image encoder (a ViT) with a GPT-2-like text encoder, both ending in a length-1 vector of 512.',
      'Training makes each image most similar to its own caption across the batch, in both directions.',
      'Classes can be written as captions, which gives zero-shot classification; a learned temperature of 100 sharpens the softmax.',
    ],
  },
  dit: {
    refs: [["Peebles & Xie 2022, Scalable Diffusion Models with Transformers (DiT)", "https://arxiv.org/abs/2212.09748"], ["Ho et al. 2020, Denoising Diffusion Probabilistic Models", "https://arxiv.org/abs/2006.11239"], ["Ho & Salimans 2022, Classifier-Free Diffusion Guidance", "https://arxiv.org/abs/2207.12598"], ["facebook/DiT-XL-2-256, the weights behind the real numbers", "https://huggingface.co/facebook/DiT-XL-2-256"]],
    code: {
      lines: [
        '# the DiT block: the timestep and class set LayerNorm\'s scale and shift, and a gate',
        'c = t_embedder(t) + y_embedder(y)                        # (1152,)',
        'shift1, scale1, gate1, shift2, scale2, gate2 = adaLN_modulation(c).chunk(6)',
        'x = x + gate1 * attn(layer_norm(x) * (1 + scale1) + shift1)',
        'x = x + gate2 * mlp(layer_norm(x) * (1 + scale2) + shift2)',
        '# training: add noise at a random t, predict it',
        'x_t = alpha_bar[t].sqrt() * x0 + (1 - alpha_bar[t]).sqrt() * eps',
        'loss = F.mse_loss(model(x_t, t, y), eps)',
        '# sampling: 250 steps, each a full forward pass (twice with guidance)',
        'for t in reversed(steps): x = p_sample(model, x, t, y)',
      ],
      at: { blocks: [0, 1, 2, 3, 4], noise: [5, 6, 7], latent: [], adaln: [0, 1, 2, 3, 4], sample: [8, 9] },
    },
    checks: [
      { phase: 'noise', q: 'What does DiT learn to predict?', options: ['The noise that was added to the image', 'The next token', 'The image’s class', 'A caption for the image'], answer: 0, why: 'The loss compares the model’s output with the noise ε that was mixed in; knowing the noise, you can remove it.' },
      { phase: 'adaln', q: 'In DiT, what sets each LayerNorm’s scale and shift?', options: ['A small network reading the timestep and the class', 'Fixed learned constants, as in GPT-2', 'The previous patch', 'Nothing: DiT has no LayerNorm'], answer: 0, why: 'adaLN-Zero computes γ, β and a gate α from c = emb(t) + emb(class), separately in every block.' },
    ],
    recap: [
      'DiT is a Transformer over latent image patches that predicts the noise in them, not the next token.',
      'The timestep and class enter through adaLN-Zero: they set every block’s LayerNorm scale and shift and a gate on each sub-layer.',
      'One image takes many full passes (250 here), each removing a little noise.',
    ],
  },
  mamba: {
    refs: [["Gu & Dao 2023, Mamba: Linear-Time Sequence Modeling with Selective State Spaces", "https://arxiv.org/abs/2312.00752"], ["Gu et al. 2021, Efficiently Modeling Long Sequences with Structured State Spaces (S4)", "https://arxiv.org/abs/2111.00396"], ["state-spaces/mamba-130m-hf, the weights behind the real numbers", "https://huggingface.co/state-spaces/mamba-130m-hf"]],
    code: {
      lines: [
        '# the Mamba block: one mixer where GPT-2 has attention and an MLP',
        'x, z = in_proj(rms_norm(u)).chunk(2, dim=-1)       # 768 → 2 × 1,536',
        'x = F.silu(causal_conv1d(x))                       # depthwise, width 4',
        'dt, B, C = x_proj(x).split([48, 16, 16], dim=-1)   # all three depend on the token',
        'dt = F.softplus(dt_proj(dt))                       # Δ: a step size per channel',
        'A = -torch.exp(A_log)                              # (1536, 16) learned decay rates',
        'for t in range(T):                                 # the selective scan',
        '    h = torch.exp(dt[t, :, None] * A) * h + dt[t, :, None] * B[t] * x[t, :, None]',
        '    y[t] = (h * C[t]).sum(-1) + D * x[t]',
        'u = u + out_proj(y * F.silu(z))',
      ],
      at: { blocks: [0, 1, 2, 9], cost: [6, 7], scan: [5, 6, 7, 8], select: [3, 4, 7], predict: [] },
    },
    checks: [
      { phase: 'cost', q: 'As the text gets longer, how much does Mamba keep in memory to predict the next token?', options: ['The same amount at any length: a fixed-size state', 'More with every token, like a KV cache', 'Nothing at all', 'The whole text'], answer: 0, why: 'Everything earlier tokens contributed is folded into the state, 16 numbers per channel plus 3 inputs for the convolution.' },
      { phase: 'select', q: 'In Mamba, what does a large step size Δ on a token do?', options: ['Writes that token strongly into the state and forgets more of the past', 'Skips the token', 'Makes the model wider', 'Nothing: Δ is fixed after training'], answer: 0, why: 'The old state is multiplied by e^(ΔA), which shrinks as Δ grows, while the input enters with weight Δ·B.' },
    ],
    recap: [
      'Mamba replaces attention with a selective state-space scan: a fixed-size state updated once per token.',
      'Its step size Δ and its B and C depend on the input, so it chooses per token what to write and what to keep.',
      'Its work grows linearly with the length of the text and its memory stays constant.',
    ],
  },
  kvcache: {
    refs: [["Pope et al. 2022, Efficiently Scaling Transformer Inference", "https://arxiv.org/abs/2211.05102"], ["Williams et al. 2009, Roofline: an insightful visual performance model", "https://doi.org/10.1145/1498765.1498785"]],
    code: {
      lines: [
        '# prefill: the whole prompt in one pass; every layer keeps its K and V',
        'k, v = ln_1(x) @ W_k + b_k, ln_1(x) @ W_v + b_v      # (prompt_len, 64) per head',
        'cache[layer] = (k, v)',
        '# decode: one token per step',
        'q, k_new, v_new = (ln_1(x_t) @ W) .split(768)          # just the new token',
        'K = torch.cat([cache_k, k_new]); V = torch.cat([cache_v, v_new])',
        'w = F.softmax(q @ K.T / 8, dim=-1)                    # one row of scores',
        'out = w @ V',
      ],
      at: { loop: [2, 5], prefill: [0, 1, 2], step: [3, 4, 5, 6, 7], size: [2], bound: [1, 4] },
    },
    checks: [
      { phase: 'step', q: 'During a decode step, how many rows of keys does each layer and head compute?', options: ['One, for the new token; the rest come from the cache', 'One for every token so far', 'None', 'One for each prompt token'], answer: 0, why: 'Earlier tokens’ keys and values do not change, so only the new token’s are computed and appended.' },
      { phase: 'bound', q: 'Why is a decode step at batch size 1 limited by memory rather than compute?', options: ['It reads every weight to produce one token, so it does few FLOPs per byte', 'It needs more FLOPs than prefill', 'The cache lives on disk', 'Softmax is slow'], answer: 0, why: 'About 2 FLOPs per 2-byte weight: 1 FLOP per byte, far below the 156 an A100 needs to be compute-bound.' },
    ],
    recap: [
      'A KV cache keeps every earlier token’s keys and values, so each new token computes only its own.',
      'Prefill runs the prompt in one pass; each decode step attends one new query over the whole cache.',
      'The cache grows with every token and layer, and decoding is limited by memory bandwidth, not arithmetic.',
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
