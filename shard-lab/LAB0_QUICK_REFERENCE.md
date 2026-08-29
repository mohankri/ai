# Lab 0 - Quick Reference Guide

## One-Minute Summary

**Lab 0 trains a GPT model on a single GPU to create the "oracle"** - a ground-truth loss curve that all distributed training must match exactly.

```
Data (1.1M tokens) → Tokenize (65 vocab) → Train GPT (25.4M params)
                                                    ↓
                                            Loss: 4.4 → 2.5
                                                    ↓
                                            Save as oracle
```

---

## The 7 Key Concepts

### 1. **Tokenization: Text → Numbers**
```
"First" → [20, 26, 40, 45, 47]  (character IDs)
```
Each unique character gets an ID (0-64). The model predicts which character comes next.

### 2. **Embeddings: Numbers → Vectors**
```
Token 45 ('s') → [0.023, -0.015, 0.142, ..., 0.089]  (512 dimensions)
Position 0 →     [0.001, 0.002, -0.003, ..., 0.045]  (512 dimensions)
```
Combine them: `embedded = token_vec + position_vec`

### 3. **Attention: "Look at Past"**
```
Position 0 can attend to: [position 0]  (only itself, causal)
Position 100 can attend to: [0, 1, 2, ..., 100]  (past and self)
Position 256 can attend to: [0, 1, 2, ..., 256]  (all positions so far)
```
Each position learns which past positions are important.

### 4. **Transformer Block: Attention + MLP**
```
x = x + attention(norm(x))      ← Look at past
x = x + mlp(norm(x))            ← Process information
```
Repeat 8 times for 8 transformer blocks.

### 5. **Logits: Model Outputs**
```
After all blocks: (batch=32, seq_len=256, hidden=512)
Logits:          (batch=32, seq_len=256, vocab=65)

Logits for position 0:
  [0.12, -0.45, 0.89, -0.23, ..., 0.34]
   '\n'  ' '    '!'   '"'          't'
```
Raw scores for each character. Which is most likely next?

### 6. **Cross-Entropy Loss: "How Wrong?"**
```
Model predicts: [0.12, -0.45, 0.89, -0.23, ..., 0.34]
Actual target:  45 (character 's')
Softmax probs:  [0.001, 0.001, 0.15, ..., 0.05]

Loss = -log(softmax[45]) = -log(0.05) = 2.996
Average across 8192 positions = ~4.4 at step 0
```

### 7. **Optimization: Update Weights**
```
new_param = param - learning_rate × gradient
new_param = param - weight_decay × param  (L2 regularization)
```
Small updates repeat 100 times → loss decreases.

---

## The Data Flow (Simple Version)

```
Step 0:
  x[0] = "s t _ F t ..."  (256 chars, batch size 32)
  y[0] = "t _ F t t ..."  (targets: next character)
  
  Forward: x → embed → attention(8×) → mlp(8×) → logits
  
  Loss: model predicts "t" after "s" → CORRECT (good logit)
        model predicts "X" after " " → WRONG (bad logit)
        → Average loss = 4.4 (early, mostly guessing)
  
  Backward: ∇loss/∇param for all 25.4M parameters
  
  Update: All weights improve slightly
  
Step 1: Repeat with different batch
  Loss: 4.33 (tiny improvement)
  
Step 20: Still learning
  Loss: 3.27 (bigger improvement)
  
Step 99: Converged
  Loss: 2.52 (model learned patterns)
```

---

## Model Architecture (ASCII)

```
Sequence "First C..."
     ↓
┌─ Token Embedding ──┐
│  (65, 512)         │  Each token → 512-dim vector
└─ Position Embed ───┘  Each position → 512-dim vector
     ↓ (add)
┌─────────────────────────┐
│  Input: (32, 256, 512)  │
│  32 sequences, 256 tokens, 512-dim embeddings
└─────────────────────────┘
     ↓
┌─────────────────────────┐
│  Transformer Block 0    │
│  ├─ Self-Attention(×8)  │
│  └─ MLP                 │
└─────────────────────────┘  Block 0 output: (32, 256, 512)
     ↓
┌─────────────────────────┐
│  Transformer Block 1    │  Same structure
└─────────────────────────┘
     ↓
  ... (blocks 2-6)
     ↓
┌─────────────────────────┐
│  Transformer Block 7    │
└─────────────────────────┘
     ↓
┌─────────────────────────┐
│  Final LayerNorm        │
│  Output: (32, 256, 512) │
└─────────────────────────┘
     ↓
┌─────────────────────────┐
│  Logits Projection      │
│  (512 → 65)             │
│  Output: (32, 256, 65)  │
└─────────────────────────┘
     ↓
Predict next character!
```

---

## Hyperparameters Explained

| Parameter | Value | Why? |
|-----------|-------|------|
| `vocab_size` | 65 | 65 unique characters in TinyShakespeare |
| `n_layer` | 8 | Moderate depth for 25M params |
| `n_head` | 8 | 8 attention heads, 64 dims each |
| `d_model` | 512 | 512-dim hidden size (good for this task) |
| `block_size` | 256 | Max context window |
| `batch_size` | 32 | 32 sequences per step |
| `steps` | 100 | Train for 100 steps |
| `lr` | 3e-4 | Learning rate (0.0003) |
| `weight_decay` | 0.1 | L2 regularization strength |

---

## The Loss Curve Explained

```
Loss
4.5  ▲
     │ █                               Step 0: Random weights
4.0  │ ▓█
     │ ▓▓█
3.5  │ ▓▓▓█
     │ ▓▓▓▓█                          Step 0-30: Steep learning
3.0  │ ▓▓▓▓▓█
     │ ▓▓▓▓▓▓█
2.5  │ ▓▓▓▓▓▓▓███                     Step 30-99: Convergence
     │ ▓▓▓▓▓▓▓▓▓▓██
2.0  └──────────────────→ Steps
     0   20   40   60   80   99

Step  Loss    Meaning
─────────────────────────────────────
0     4.40   Random initialization
20    3.27   Learning basic patterns
40    2.79   Major improvements
60    2.59   Slowing down
80    2.54   Nearly converged
99    2.52   Final trained weights
```

**Why the shape?**
- Initial steep drop: Model learns obvious patterns ("th", "er", etc.)
- Slower decline: Fine-tuning on context
- Plateau: Convergence (diminishing returns)

---

## Memory & Speed

```
┌─ Sizes ───────────────┐
│ Model weights: 100 MB │  All parameters
│ Batch:      ~400 MB   │  32 sequences of 256 tokens
│ Activations: ~900 MB  │  Hidden states during forward
│ Gradients:   ~100 MB  │  Gradients during backward
│ Optimizer:   ~500 MB  │  AdamW state (m and v)
│ ──────────────────────│
│ Peak memory: ~2.1 GiB │  All needed at once during backward
└────────────────────────┘

┌─ Speed ────────────────┐
│ Warmup (10 steps): 1.4s │ NCCL setup, kernel compilation
│ 90 steps:          2.3s │ Steady state
│ ──────────────────────  │
│ Throughput: 27.8 s/step │ Very fast (single GPU)
│ Total time: 3.24s       │
└────────────────────────┘
```

---

## The Oracle Principle (Critical!)

**Why save the loss curve?**

Imagine two implementations:

**Implementation A (Correct):**
```
Step 0: 4.402704
Step 1: 4.328461
...
Step 99: 2.516004
```

**Implementation B (Bug - slightly wrong gradient):**
```
Step 0: 4.402704  ← Same start (uses same random seed)
Step 1: 4.328463  ← Tiny difference (0.000002)
...
Step 99: 2.516008  ← Same convergence (~matches)
```

**Without oracle:**
- "Both trained to loss 2.5, must both be correct!" ❌ WRONG

**With oracle:**
- "Difference of 0.000004 at step 99, exceeds tolerance!" ✓ CATCHES BUG

**Real bugs found:**
1. Gradient synchronization off-by-one → diverged by step 10
2. Wrong reduce-scatter order → diverged by step 5
3. Missing barrier between GPU steps → random divergence
4. Incorrect tensor slicing → consistent +0.00001 error

**Lesson:** In distributed training, "loss goes down" is not evidence of correctness.

---

## What Each Distributed Lab Changes

| Lab | Focus | Change from Lab 0 |
|-----|-------|------------------|
| 0 | Baseline | Single GPU only |
| 1 | Collectives | Learn communication (no model) |
| 2 | Data Parallel | Same model on 4 GPUs, split data |
| 3 | Tensor Parallel | Split model across GPUs |
| 4 | Pipeline Parallel | Split layers across GPUs |
| 5 | ZeRO | Split parameters, gradients, optimizer |
| 6 | 2D Mesh | Combine TP and DP on grid |
| 7 | Production | FSDP2 / DTensor (real tools) |

**Critical:** Loss curve must match Lab 0 ±1e-6 for all of them.

---

## Common Mistakes (What Could Go Wrong)

| Mistake | Result | How Oracle Catches It |
|---------|--------|----------------------|
| Wrong gradient reduce | Loss diverges quickly | Mismatch by step 5 |
| Incorrect broadcast | Different weights on GPUs | Mismatch by step 1 |
| Missing synchronization | Occasional random divergence | Inconsistent errors |
| Off-by-one slicing | Subtle shape mismatches | Mismatch ~1e-5 range |
| Wrong loss reduction | Systematic error | Consistent drift |

---

## Quick Run Commands

```bash
# Navigate to lab directory
cd ~/ai/shard-lab

# Run Lab 0 (single GPU, creates oracle)
.venv/bin/python lab0_reference.py

# Expected: Loss goes 4.4 → 2.5 in ~3 seconds
# Creates: out/reference.pt (the oracle)

# Verify output
ls -lh out/reference.pt

# To run with timing details
time .venv/bin/python lab0_reference.py
```

---

## Understanding the Numbers

**Why loss ~4.4 at start?**
- Random guess on 65-char vocab: -log(1/65) ≈ 4.17
- Model is barely better than random

**Why loss ~2.5 at end?**
- Model learned common patterns ("the", "and", dialogue)
- Still can't perfectly predict (Shakespeare is complex)
- Loss = -log(P(next_char | context)) ≈ 0.08 probability on average

**Why probe loss (2.497) < training loss (2.516)?**
- Training uses many different contexts
- Probe uses fixed batch (may be easier/harder)
- Slight difference is normal

---

## What to Focus On

1. **Input/Output shapes** - know what shape flows at each step
2. **Attention mechanism** - how it implements causality
3. **Residual connections** - why `x = x + f(x)` matters
4. **Loss calculation** - cross-entropy on logits
5. **Reproducibility** - fixed seed makes training deterministic
6. **Oracle principle** - exact loss comparison catches bugs

---

## Key Insight

> **A correct model and a wrong model both train and reduce loss. The oracle catches the wrong one because it's unforgiving.**

This is why Labs 1-7 must match Lab 0 exactly. No amount of "loss decreasing" makes a subtle bug invisible.

---

## Next: Lab 1

After understanding Lab 0, Lab 1 teaches the 5 collective operations that make distributed training work:
- `all_reduce`: sync gradients
- `reduce_scatter`: distribute reduction
- `all_gather`: reassemble data
- `broadcast`: distribute weights
- `send`/`recv`: direct GPU-GPU messages

All of these show up in Labs 2-7 in different combinations.
