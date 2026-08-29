# Lab 0 - Visual Data Flow with Token Example

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING STEP 0 WALKTHROUGH                  │
└─────────────────────────────────────────────────────────────────┘

┌─── INPUT: Raw TinyShakespeare Text ───┐
│                                        │
│ "First Citizen: Before we proceed      │
│  any further, hear me speak..."        │
│                                        │
└────────────────────────────────────────┘
        │
        ▼
┌─── CHARACTER TOKENIZATION ────────────────────┐
│                                               │
│ Vocabulary: ['\n', ' ', '!', '"', ..., 'z']  │
│ Size: 65 unique characters                   │
│                                               │
│ "First" → [20, 26, 40, 45, 47]              │
│            F   i   r   s   t                 │
└───────────────────────────────────────────────┘
        │
        ▼
┌─── BATCH SAMPLING (Step 0) ───────────────┐
│                                           │
│ Seed: 1234 + 0 = 1234                    │
│ Sample 32 random positions in corpus      │
│ Each sequence: 256 tokens                 │
│                                           │
│ Sequence 0:  [45, 47, 1, 20, 47, ...]    │
│              s   t   _  F   t             │
│ Sequence 1:  [12, 34, 56, ...]           │
│ ...                                       │
│ Sequence 31: [78, 90, 12, ...]           │
│                                           │
│ x shape: (32, 256)  ← input tokens       │
│ y shape: (32, 256)  ← target (x shifted) │
└───────────────────────────────────────────┘
        │
        ▼
┌─── EMBEDDING LAYER ────────────────────────┐
│                                            │
│ Token embeddings (wte): vocab × d_model   │
│   Input: (32, 256)                        │
│   wte[45] = [0.023, -0.015, 0.142, ...]  │
│   wte[47] = [0.045, 0.067, -0.023, ...]  │
│   Output: (32, 256, 512)                  │
│                                            │
│ Position embeddings (wpe): block × d_model│
│   wpe[0]   = [0.001, 0.002, -0.003, ...]  │
│   wpe[1]   = [0.002, 0.003, -0.001, ...]  │
│   Output: (1, 256, 512)                   │
│                                            │
│ Combine: x = token_embed + pos_embed      │
│   Output: (32, 256, 512)                  │
└────────────────────────────────────────────┘
        │
        ▼
┌────── TRANSFORMER BLOCK 0 ───────────────────┐
│                                              │
│  Input: (32, 256, 512)                      │
│                                              │
│  ┌─ LayerNorm ─────────────────────┐        │
│  │ Normalize each position to       │        │
│  │ mean=0, std=1                   │        │
│  │ Output: (32, 256, 512)          │        │
│  └─────────────────────────────────┘        │
│                 │                           │
│                 ▼                           │
│  ┌─ Self-Attention (8 heads) ──────┐        │
│  │                                  │        │
│  │ q_proj: (32, 256, 512)           │        │
│  │ k_proj: (32, 256, 512)           │        │
│  │ v_proj: (32, 256, 512)           │        │
│  │                                  │        │
│  │ Split into 8 heads (64 dims):   │        │
│  │ Head 0: (32, 256, 64)            │        │
│  │ Head 1: (32, 256, 64)            │        │
│  │ ...                              │        │
│  │ Head 7: (32, 256, 64)            │        │
│  │                                  │        │
│  │ For each head:                   │        │
│  │   scores = q @ k.T / sqrt(64)   │        │
│  │   scores = scores + causal_mask │        │
│  │   probs = softmax(scores)       │        │
│  │   output = probs @ v            │        │
│  │                                  │        │
│  │ Head example (pos 0, seq 0):    │        │
│  │   q = [0.12, -0.34, 0.56, ...]  │        │
│  │   k = [0.23, 0.45, -0.12, ...]  │        │
│  │   v = [0.01, 0.02, 0.03, ...]   │        │
│  │                                  │        │
│  │   scores = [0.23, -0.15, ...]   │        │
│  │   scores[1:] = -inf  ← causal!  │        │
│  │   probs = [1.0, 0.0, 0.0, ...]  │        │
│  │   out = 1.0 * v[0] = [0.01...]  │        │
│  │                                  │        │
│  │ Concat heads: (32, 256, 512)    │        │
│  │ o_proj: (32, 256, 512)           │        │
│  │                                  │        │
│  │ Output: (32, 256, 512)           │        │
│  └──────────────────────────────────┘        │
│                 │                           │
│                 ▼                           │
│  x = x + attn_output  ← residual connection │
│  Output: (32, 256, 512)                     │
│                 │                           │
│                 ▼                           │
│  ┌─ LayerNorm ─────────────────────┐        │
│  │ Output: (32, 256, 512)          │        │
│  └─────────────────────────────────┘        │
│                 │                           │
│                 ▼                           │
│  ┌─ MLP (Feed-Forward) ─────────────┐       │
│  │                                  │       │
│  │ up:   (512 → 2048) + GELU       │       │
│  │ down: (2048 → 512)              │       │
│  │                                  │       │
│  │ Output: (32, 256, 512)          │       │
│  └──────────────────────────────────┘       │
│                 │                           │
│                 ▼                           │
│  x = x + mlp_output  ← residual connection  │
│  Output: (32, 256, 512)                     │
│                                             │
│  ✓ Block 0 complete                         │
└─────────────────────────────────────────────┘
        │
        ▼
┌────── BLOCKS 1-7 (Same structure) ──┐
│                                      │
│ Apply blocks 1 through 7 identically │
│ Each: attn + mlp + residuals        │
│ Output: (32, 256, 512)              │
│                                      │
└──────────────────────────────────────┘
        │
        ▼
┌─── FINAL LAYER NORM ──────────────────┐
│                                       │
│ ln_f: normalize final hidden states  │
│ Output: (32, 256, 512)               │
│                                       │
└───────────────────────────────────────┘
        │
        ▼
┌─── LOGITS PROJECTION ──────────────────┐
│                                        │
│ lm_head: (512 → 65)                   │
│ Input:  (32, 256, 512)                │
│ Output: (32, 256, 65)                 │
│                                        │
│ Example: position [0, 0] (seq 0, tok 0)
│ Logits for next token:                │
│   vocab[0]:  0.12   ('\\n')           │
│   vocab[1]:  -0.45  (space)           │
│   vocab[2]:  0.89   ('!')             │
│   ...                                 │
│   vocab[47]: 0.34   ('t')  ← highest? │
│   ...                                 │
│   vocab[64]: -0.67  ('z')             │
│                                        │
└────────────────────────────────────────┘
        │
        ▼
┌─── CROSS-ENTROPY LOSS ────────────────────────┐
│                                               │
│ Targets (y): what we want to predict         │
│   y[0, 0] = 47  (character 't')              │
│   Next after 's' should be 't'               │
│                                               │
│ Predicted logits for pos[0, 0]:              │
│   logits = [0.12, -0.45, 0.89, ..., 0.34]  │
│                                               │
│ Convert to probabilities (softmax):          │
│   softmax = [0.001, 0.001, 0.15, ..., 0.05] │
│   (sums to 1.0)                             │
│                                               │
│ Loss for this position:                      │
│   target = 47                                │
│   pred_prob[47] = 0.05                       │
│   loss = -log(0.05) = 2.996                  │
│                                               │
│ Average across all 32 × 256 = 8192 positions│
│ LOSS = 4.402704                              │
│                                               │
└───────────────────────────────────────────────┘
        │
        ▼
┌─── BACKWARD PASS (Autograd) ───────────────┐
│                                             │
│ PyTorch computes ∂loss/∂param for all      │
│ parameters using the chain rule            │
│                                             │
│ Example gradients:                         │
│   lm_head[47, :].grad = [+0.0012, ...]    │
│   lm_head[0, :].grad  = [-0.0001, ...]    │
│   ...                                       │
│   blocks[0].attn.q_proj.weight.grad = ... │
│   ...                                       │
│                                             │
│ All 25.4M parameters have gradients        │
│                                             │
│ Gradient clipping:                         │
│   if ||grad|| > 1.0:                       │
│     grad = grad * (1.0 / ||grad||)        │
│                                             │
└─────────────────────────────────────────────┘
        │
        ▼
┌─── OPTIMIZER STEP (AdamW) ────────────────┐
│                                           │
│ For each parameter:                       │
│   param -= lr * (m / (sqrt(v) + eps))    │
│                                           │
│ Example: lm_head[47, 0]                  │
│   Before: 0.012345                       │
│   grad:   -0.00012                       │
│   m:      0.1 * grad = -0.000012         │
│   v:      0.05 * grad² = 7.2e-10         │
│   update: -3e-4 * (-1.2e-5) / 0.000027   │
│   After:  0.012345 + 0.00000000133       │
│                                           │
│ Weight decay:                             │
│   param -= weight_decay * param          │
│   After:  0.012344                       │
│                                           │
│ Result: All parameters updated!          │
│                                           │
└───────────────────────────────────────────┘
        │
        ▼
┌─── TRAINING COMPLETE FOR STEP 0 ────────┐
│                                          │
│ Loss:    4.402704                       │
│ Action:  Step 0 of 100 done             │
│                                          │
│ Next:    Repeat for step 1, 2, ..., 99 │
│                                          │
└──────────────────────────────────────────┘
```

---

## Attention Mechanism - Detailed Example

### Causal Self-Attention at Position 0

**Input sequence:**
```
Pos 0: 45 ('s')
Pos 1: 47 ('t')
Pos 2: 1  (' ')
Pos 3: 20 ('F')
Pos 4: 47 ('t')
...
Pos 255: <last token>
```

**Attention for Position 0:**

```
┌─ Query at pos 0 ──────────────┐
│ q[0] = W_q @ x[0]            │
│      = W_q @ embed('s')       │
│      = [0.12, -0.34, ...]     │ (512 dims → 64 dims per head)
└───────────────────────────────┘
        │
        ├─ Key at pos 0:  k[0] = [0.23, 0.45, ...]
        ├─ Key at pos 1:  k[1] = [0.78, -0.12, ...]
        ├─ Key at pos 2:  k[2] = [-0.34, 0.56, ...]
        └─ ... (all keys)
        │
        ▼
┌─ Attention Scores (one head) ──────────────────┐
│                                                 │
│ score[0] = q[0] · k[0] / sqrt(64)              │
│          = [0.12,-0.34,...] · [0.23,0.45,...]  │
│          / 8.0                                 │
│          = +0.23 (pos 0: can attend to self)  │
│                                                 │
│ score[1] = q[0] · k[1] / sqrt(64)              │
│          = ... / 8.0                           │
│          = -0.15 (pos 1: future token!)       │
│                                                 │
│ scores before mask:                            │
│   [+0.23, -0.15, +0.45, -0.67, ..., +0.12]   │
│                                                 │
│ CAUSAL MASK: set future positions to -inf     │
│   [+0.23, -inf, -inf, -inf, ..., -inf]        │
│                                                 │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─ Softmax (Convert to Probabilities) ──────┐
│                                            │
│ probabilities = softmax([0.23, -∞, -∞, ...])
│               = [1.0, 0.0, 0.0, ..., 0.0] │
│                                            │
│ At position 0, can ONLY attend to itself! │
│                                            │
└────────────────────────────────────────────┘
        │
        ▼
┌─ Apply Attention to Values ──────────────┐
│                                          │
│ output = 1.0 * v[0] + 0.0 * v[1] + ... │
│        = v[0]                           │
│        = [0.01, 0.02, 0.03, ...]        │
│                                          │
│ Position 0 attends only to itself!      │
│                                          │
└──────────────────────────────────────────┘
```

**Position 100 (middle of sequence):**

```
Scores (causal mask already applied):
  [+0.34, +0.12, -0.45, +0.23, -0.01, ..., -inf, -inf]
   pos 0   pos 1   pos 2   pos 3   pos 4   pos 101 pos 255

Softmax → probabilities:
  [0.18, 0.15, 0.05, 0.12, 0.08, ..., 0.0, 0.0]

Output = weighted sum of values:
  = 0.18 * v[0] + 0.15 * v[1] + ... + 0.0 * v[255]
  
Position 100 can attend to positions 0-100 (past and self)
Cannot attend to positions 101-255 (future) ← This is "causal"!
```

---

## Loss Progression Over 100 Steps

```
Step 0:   █████████████████████░░░░░░░░░░░░░░░░░░ 4.403
Step 10:  ████████████████████░░░░░░░░░░░░░░░░░░░░ 3.759
Step 20:  ███████████████████░░░░░░░░░░░░░░░░░░░░░ 3.270
Step 30:  ██████████████████░░░░░░░░░░░░░░░░░░░░░░ 3.043
Step 40:  █████████████████░░░░░░░░░░░░░░░░░░░░░░░ 2.789
Step 50:  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░ 2.696
Step 60:  ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░ 2.588
Step 70:  ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░ 2.548
Step 80:  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 2.541
Step 90:  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 2.523
Step 99:  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 2.516

Probe:    ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 2.498
```

**What's happening:**
- Steep drop (steps 0-30): Model learns basic character patterns
- Slower decline (steps 30-80): Fine-tuning on context
- Plateau (steps 80-99): Convergence approaching, smaller improvements
- Probe loss: Confirms generalization to unseen data

---

## Memory & Compute

```
┌─ Forward Pass ──────────────────────────────┐
│ Stores:                                     │
│  • Input embeddings:    32 × 256 × 512 fp32 │
│  • Each block outputs:  32 × 256 × 512 fp32 │
│  • Attention scores:    32 × 8 × 256 × 256  │
│  • All activations (for backward)           │
│                                             │
│ Peak during forward:    ~1.2 GiB            │
└─────────────────────────────────────────────┘

┌─ Backward Pass ─────────────────────────────┐
│ Stores:                                     │
│  • All gradients:      same size as weights │
│  • Optimizer states:   m and v for AdamW    │
│                                             │
│ Peak during backward:   ~2.1 GiB            │
└─────────────────────────────────────────────┘

Total Peak Memory: 2.1 GiB on GB300 (277 GiB available)
Throughput: 27.8 steps/sec (after 10-step warmup)
Time for 100 steps: 3.24 seconds
```

---

## Key Insights

1. **Embeddings** transform tokens (65 possible values) into dense vectors (512-dimensional)
2. **Attention** lets each position learn from previous positions
3. **Causal masking** prevents "cheating" by looking at future tokens
4. **Residual connections** help gradients flow back through 8 blocks
5. **Cross-entropy loss** measures prediction accuracy
6. **Backprop** computes gradients efficiently using chain rule
7. **AdamW** updates weights with adaptive learning rate
8. **Loss curve** shows learning progress

---

## Why This Is The Oracle

Every Lab 1-7 must produce **exactly** this loss curve (to ±1e-6):
```
[4.402704, 3.759218, 3.269731, ..., 2.516004]
```

Why so strict?
- ✓ A wrong tensor-parallel split still trains
- ✓ A wrong gradient synchronization still trains
- ✓ Many subtle bugs only appear as tiny numerical divergence
- ✓ The oracle catches them because it's unforgiving

Bugs caught by oracle:
- Gradient not synchronized correctly (appears as loss drift)
- Tensor slicing off by one (subtle shape mismatches)
- Incorrect reduce-scatter order (small but consistent error)
- Missing synchronization barriers (race conditions in timing)

**Lesson:** In distributed training, "runs and loss goes down" is not sufficient evidence of correctness.
