# Lab 0 - Detailed Step-by-Step Walkthrough with Examples

## Overview
Lab 0 trains a GPT model on a single GPU to establish the "oracle" - a ground truth that all distributed training labs must match exactly.

---

## Step 1: Load Data & Tokenize

### Input: Raw Text (TinyShakespeare)
```
"First Citizen:
Before we proceed..."
```

### Tokenization: Character → Integer
Create a vocabulary of unique characters sorted alphabetically:
```python
chars = sorted(set(text))
# Result: ['\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', '0', '1', ..., 'y', 'z']
# Total vocab_size = 65 characters

stoi = {c: i for i, c in enumerate(chars)}  # string-to-index mapping
# Examples:
# ' ' (space) → 1
# 'F' → 20
# 'i' → 26
# 'r' → 40
# 's' → 45
# 't' → 47
```

### Encoding the Text
Text: `"First C..."`
Tokens: `[20, 26, 40, 45, 47, 1, 20, 26, 47, 26, ...]`
           F   i   r   s   t   _  C  i  t  i

**Result:** 1,115,394 token tensor on GPU

---

## Step 2: Build Model

### Model Architecture (GPTConfig):
```
vocab_size      = 65          (our character vocabulary)
n_layer         = 8           (8 transformer blocks)
n_head          = 8           (8 attention heads)
d_model         = 512         (embedding dimension)
block_size      = 256         (max context window)
head_dim        = 512 / 8 = 64
```

### Model Layers:
```
GPT(
  ├── wte (word embedding): (65, 512)           [embed each token]
  ├── wpe (position embedding): (256, 512)      [embed position 0-255]
  ├── blocks[0-7]:                               [8 identical blocks]
  │   ├── ln1: (512)                            [layer norm]
  │   ├── attn: CausalSelfAttention             [8 heads, 64 dim each]
  │   │   ├── q_proj: (512, 512)                [query projection]
  │   │   ├── k_proj: (512, 512)                [key projection]
  │   │   ├── v_proj: (512, 512)                [value projection]
  │   │   ├── o_proj: (512, 512)                [output projection]
  │   ├── ln2: (512)
  │   └── mlp:                                   [feed-forward]
  │       ├── up: (512, 2048)                   [expand 4x]
  │       └── down: (2048, 512)                 [project back]
  ├── ln_f: (512)                               [final layer norm]
  └── lm_head: (512, 65)                        [project to vocab logits]
)

Total: 25.4M parameters
```

---

## Step 3: Get a Training Batch

### Global Batch Setup:
```python
step = 0
global_batch = 32       # 32 sequences
block_size = 256        # 256 tokens per sequence
```

### Random Batch Selection (Deterministic):
```python
# Use fixed seed for reproducibility
g = torch.Generator().manual_seed(seed=1234)  # seed + step=0
ix = torch.randint(0, len(data)-257, (32,), generator=g)
# Result: random start indices [424502, 789123, 312456, ...]
```

### Example: Create One Sequence from the Batch

**Selected start position:** `idx=50000` (somewhere in the middle of TinyShakespeare)

**Raw tokens from data[50000:50256]:**
```
[45, 47, 1, 20, 47, 47, 45, 19, 1, 45, 47, 19, 19, 47, ...]
 s   t   _  F   t   t   s   e   _  s   t   e   e   t   ...
```

**Input sequence (x) - what the model sees:**
```python
x = data[50000:50256]   # 256 tokens
# Shape: (1, 256)
# Values: [45, 47, 1, 20, 47, ..., <256th token>]
```

**Target sequence (y) - what we want to predict:**
```python
y = data[50001:50257]   # Next 256 tokens (shifted by 1)
# Shape: (1, 256)
# Values: [47, 1, 20, 47, 47, ..., <257th token>]
```

**Full batch shape:**
```
x: (batch=32, seq_length=256)
y: (batch=32, seq_length=256)
```

---

## Step 4: Forward Pass Through Model

### Stage 1: Token & Position Embeddings
Input `x`: shape `(32, 256)` with token IDs

```python
# Embed each token using wte
token_embed = self.wte(x)  # (32, 256) → (32, 256, 512)
#
# Embed each position 0-255 using wpe
positions = torch.arange(256)  # [0, 1, 2, ..., 255]
pos_embed = self.wpe(positions)  # (256) → (256, 512)
#
# Combine them
x = token_embed + pos_embed  # (32, 256, 512)
```

**Example values in embeddings (first token of first sequence):**
```
Token ID = 45 (character 's')
wte[45] = [0.023, -0.015, 0.142, ..., 0.089]  (512 dims)

Position = 0
wpe[0] = [0.001, 0.002, -0.003, ..., 0.045]  (512 dims)

Combined = [0.024, -0.013, 0.139, ..., 0.134]  (512 dims)
```

### Stage 2: Process Through 8 Transformer Blocks

Each block does: `x = x + attention(norm(x)) + mlp(norm(x))`

#### Block 0 Example (trace one token):

**Input to Block 0:** shape `(32, 256, 512)`

**LayerNorm 1:**
```python
x_norm = self.ln1(x)  # Normalize to mean=0, std=1
# Each of 32×256 positions gets normalized independently
```

**Self-Attention:**
```python
q = self.q_proj(x_norm)      # (32, 256, 512) → (32, 256, 512)
k = self.k_proj(x_norm)      # (32, 256, 512) → (32, 256, 512)
v = self.v_proj(x_norm)      # (32, 256, 512) → (32, 256, 512)

# Split into 8 heads (each head has dim=64)
q = q.view(32, 256, 8, 64).transpose(1, 2)  # (32, 8, 256, 64)
k = k.view(32, 256, 8, 64).transpose(1, 2)  # (32, 8, 256, 64)
v = v.view(32, 256, 8, 64).transpose(1, 2)  # (32, 8, 256, 64)

# Compute attention scores (simplified: show 1 head, 1 sequence)
# Head 0, Sequence 0, Token 0 wants to attend to other tokens
scores = q[0, 0, 0, :] @ k[0, 0, :, :].T  # (64) @ (256, 64).T = (256)
# scores = [0.23, -0.15, 0.87, ..., <small because causal mask>]
# Divide by sqrt(64)=8
scores = scores / 8  # [-0.02, -0.019, 0.109, ..., -inf, -inf, ...]

# Apply causal mask (cannot attend to future tokens)
scores[1:256] = -inf  # Only token 0 can attend to itself!

# Softmax: convert to probabilities
att_probs = softmax(scores)  # [1.0, 0.0, 0.0, ..., 0.0]

# Apply to values: weighted sum of values
out = att_probs @ v[0, 0, :, :]  # (256) @ (256, 64) = (64)
```

**MLP (Feed-Forward):**
```python
x_norm2 = self.ln2(x + attn_output)  # Residual connection + norm
# Expand 4x
up = self.up(x_norm2)  # (32, 256, 512) → (32, 256, 2048)
up = gelu(up)          # Apply activation
# Contract back
down = self.down(up)   # (32, 256, 2048) → (32, 256, 512)

# Add residual
x = (x + attn_output) + down  # (32, 256, 512)
```

#### Blocks 1-7: Same structure, repeated 7 more times

**After all 8 blocks:** shape `(32, 256, 512)`

### Stage 3: Final Layer Norm & Logits

```python
x = self.ln_f(x)              # (32, 256, 512) → (32, 256, 512)
logits = self.lm_head(x)      # (32, 256, 512) → (32, 256, 65)
```

**Logits example (first token of first sequence, first 10 vocab items):**
```
logits[0, 0, :] = [0.12, -0.45, 0.89, -0.23, 0.56, -0.34, 0.78, -0.12, 0.45, -0.67]
                   [0]   [1]    [2]   [3]    [4]   [5]    [6]   [7]    [8]   [9]    ...
                   
# These are raw scores (not probabilities yet)
# The model is predicting which character comes next
```

---

## Step 5: Compute Loss (Cross-Entropy)

### Target: What we want the model to predict
```python
y[0, 0] = 47  # First token's target is '47' (character 't')
              # Because input[0,0] = 45 ('s'), we want model to predict next = 't'
```

### Cross-Entropy Loss
```python
# Cross-entropy compares logits to targets
# Reshape for loss computation
logits_flat = logits.view(-1, 65)      # (32*256, 65) = (8192, 65)
targets_flat = targets.reshape(-1)     # (32*256) = (8192)

# Loss function:
# loss = -log(softmax(logits[target]))
# 
# For position [0,0]:
# softmax_logits = [0.001, 0.001, 0.15, 0.001, ..., 0.05, ...]  (sums to 1)
# target = 47
# loss[0,0] = -log(softmax_logits[47]) = -log(0.05) = 2.996
#
# Average across all 8192 positions in batch

loss = F.cross_entropy(logits_flat, targets_flat)
# Result: loss = 4.402704 (at step 0)
```

**Interpretation:**
- Loss ≈ 4.4 means the model assigns ~1% probability to the correct next token
- Random guess on 65-char vocab would give -log(1/65) ≈ 4.17
- The model is barely better than random on step 0

---

## Step 6: Backward Pass (Compute Gradients)

### Zero Gradients
```python
opt.zero_grad(set_to_none=True)  # Clear old gradients
```

### Backward Pass
```python
loss.backward()  # PyTorch autograd computes ∂loss/∂param for all params
```

**Example: Gradient for lm_head (512, 65):**
```python
# lm_head connects hidden states to logit predictions
lm_head.weight.grad = [
  [-0.00012, 0.00045, -0.00023, ..., 0.00067],  # grad for vocab[0]
  [0.00034, -0.00089, 0.00012, ..., -0.00034],  # grad for vocab[1]
  ...
  [0.00156, -0.00201, 0.00089, ..., 0.00423],   # grad for vocab[47] (target)
]
# Shape: (512, 65) - same as weights
```

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# Limit max gradient norm to 1.0 to prevent exploding gradients
# This rescales all gradients if their norm exceeds 1.0
```

---

## Step 7: Optimizer Step (Update Weights)

### AdamW Optimizer
```python
opt = torch.optim.AdamW(
    model.parameters(), 
    lr=3e-4,           # learning rate
    betas=(0.9, 0.95), # momentum parameters
    weight_decay=0.1   # L2 regularization
)
```

### Weight Update Example (lm_head[0, 0])
```python
# Before update:
old_weight = 0.012345

# Gradient:
grad = -0.00012

# Optimizer maintains momentum (m) and second moment (v):
m = beta1 * m + (1 - beta1) * grad       # m = 0.9 * 0 + 0.1 * (-0.00012) = -0.000012
v = beta2 * v + (1 - beta2) * grad²     # v = 0.95 * 0 + 0.05 * 1.44e-8 = 7.2e-10

# Update:
weight = old_weight - lr * m / (sqrt(v) + eps)
       = 0.012345 - 0.0003 * (-0.000012) / sqrt(7.2e-10)
       = 0.012345 + 0.00000000036 / 0.0000268
       = 0.0123452

# Tiny change on step 0, but accumulated over 100 steps → trained model
```

### Weight Decay (L2 Regularization)
```python
# Also subtract weight_decay * weight
weight = weight - weight_decay * weight
       = 0.0123452 - 0.1 * 0.0123452
       = 0.0111107  # Small additional decay
```

**After step 0 update:**
```
loss: 4.402704 → model parameters updated (slightly)
```

---

## Step 8: Repeat for 100 Steps

### Step 20:
```
Get next batch of sequences (different token indices via random seed)
Forward pass → logits
Compute loss: 3.269731  (better! model is learning)
Backward → gradients
Update → weights improved
```

### Step 40:
```
loss: 2.789218  (still improving)
```

### Step 99 (final step):
```
loss: 2.516004  (converged to ~2.5)
```

---

## Step 9: Probe Loss (Validation)

After training, run one final forward pass on a fixed batch (never seen during training):

```python
# Fixed batch at step 10_000:
xr, yr = get_batch(data, step=10_000, global_batch=8, ...)

# Forward pass (no gradient computation)
with torch.no_grad():
    ref_logits, ref_loss = model(xr, yr)

# Result: ref_loss = 2.497602
# Confirms model generalizes to unseen data
```

---

## Complete Training Curve

```
Step  Loss      Interpretation
----  --------  ---------------------
0     4.402704  Random initialization
10    3.759218  Learning
20    3.269731  Better predictions
30    3.043291  
40    2.789218  Significant progress
50    2.695832  
60    2.587731  
70    2.548219  
80    2.540576  Convergence starting
90    2.523401  
99    2.516004  Final (steady state)

Probe  2.497602  Generalization
```

---

## Key Metrics

```
Model:        25.4M parameters
Training:     100 steps
Batch size:   32 sequences
Seq length:   256 tokens
Vocabulary:   65 characters
Execution:    ~3.24 seconds (after 10-step warmup)
Throughput:   27.8 steps/sec
Memory:       ~2.1 GiB peak
```

---

## What's Being Learned?

1. **Character statistics:** P(next='t' | prev='s')
2. **Common patterns:** "the", "and", "to"
3. **Language structure:** Punctuation, capitalization, dialogue format
4. **Context window:** Predicting based on up to 256 previous tokens

---

## The Oracle Principle

This curve becomes the "ground truth" because:
- ✓ It's bit-reproducible (fp32 + deterministic algorithms)
- ✓ All distributed labs must match it to ±1e-6
- ✓ Subtle bugs that don't crash still diverge from it
- ✓ No "loss goes down = correct" false positives

**Example bug that would be caught:**
```
Wrong tensor-parallel split:
- Lab 0 final loss: 2.516004
- Lab 3 (buggy):   2.516001 ✓ looks good
- Difference:      0.000003 < tolerance!

Wait... but only if we compare EXACTLY.
Without oracle comparison, divergence of 1e-4 could hide forever.
```

---

## Next: What Distributed Training Changes

All 8 labs follow this same training loop, but change:

| Lab | Change | Benefit |
|-----|--------|---------|
| 0   | 1 GPU  | Baseline |
| 1   | Collectives only | Learn communication |
| 2   | Data parallel | 2-4x speedup |
| 3   | Tensor parallel | Fits larger models |
| 4   | Pipeline parallel | Fits even larger models |
| 5   | ZeRO stages | 2-16x memory savings |
| 6   | 2D mesh | Combine everything |
| 7   | FSDP2/DTensor | Production tools |

**Critical requirement:** Loss curve must match Lab 0 exactly.
