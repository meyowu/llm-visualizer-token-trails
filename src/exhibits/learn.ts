import type { Check, Code, Player } from '../core/player'

/*
 * What each page teaches beyond its drawing: the matching lines of real code (nanoGPT-style
 * PyTorch, and the Hugging Face names for LLaMA), predict-then-reveal questions, and a recap.
 */

interface Learn { code: Code; checks: Check[]; recap: string[] }

const LEARN: Record<string, Learn> = {
  overview: {
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
      ],
      at: { split: [1, 5], bytes: [6], merge: [7, 8, 9, 10, 11], ids: [12, 13] },
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
  llama: {
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
}
