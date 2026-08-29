# DDP vs Tensor Parallel - Complete Visual Comparison

## One-Minute Summary

| Feature | DDP (Lab 2) | Tensor Parallel (Lab 3) |
|---------|------------|------------------------|
| **Idea** | Same model on each GPU, different data | Split model across GPUs, same data |
| **Memory** | O(M) per GPU (M = model size) | O(M/N) per GPU (sharded) |
| **Data** | Each GPU sees different batch | All GPUs see same batch |
| **Sync** | All-reduce gradients after backward | All-gather weights during forward |
| **Communication** | After each step (gradients) | During forward/backward (activations) |
| **Best for** | Large batches, small model | Small batch, huge model |
| **Scaling** | Good (linear) up to ~32 GPUs | Requires high-bandwidth inter-GPU |

---

## Visual Comparison

### DDP (Data Parallel)

```
Step 1: Replicate Model
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  GPU 0          │  │  GPU 1          │  │  GPU 2          │  │  GPU 3          │
│ Model: 25.4M    │  │ Model: 25.4M    │  │ Model: 25.4M    │  │ Model: 25.4M    │
│ Batch: 8        │  │ Batch: 8        │  │ Batch: 8        │  │ Batch: 8        │
│                 │  │                 │  │                 │  │                 │
│ Forward: 8 seq  │  │ Forward: 8 seq  │  │ Forward: 8 seq  │  │ Forward: 8 seq  │
│ Gradients:      │  │ Gradients:      │  │ Gradients:      │  │ Gradients:      │
│ [g0, g1, ...]   │  │ [g0, g1, ...]   │  │ [g0, g1, ...]   │  │ [g0, g1, ...]   │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
        │                   │                    │                    │
        └───────────────────┴────────────────────┴────────────────────┘
                              ▼
Step 2: Synchronize Gradients
        ┌──────────────────────────────────────────────┐
        │  All-Reduce (SUM) on Gradients              │
        │  All ranks sum [g0, g1, ...] from all GPUs   │
        │  Result: [sum_g0, sum_g1, ...]              │
        │  sent to ALL ranks                           │
        └──────────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  GPU 0          │  │  GPU 1          │  │  GPU 2          │  │  GPU 3          │
│ Avg gradients:  │  │ Avg gradients:  │  │ Avg gradients:  │  │ Avg gradients:  │
│ [sum_g0/4,...]  │  │ [sum_g0/4,...]  │  │ [sum_g0/4,...]  │  │ [sum_g0/4,...]  │
│ ↓ Update params │  │ ↓ Update params │  │ ↓ Update params │  │ ↓ Update params │
│ Model identical │  │ Model identical │  │ Model identical │  │ Model identical │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Key Points:**
- Every GPU has full model (25.4M params)
- Each GPU trains on different data batch
- Gradients computed independently
- **Sync:** All-reduce gradients (full reduction to all ranks)
- After update, all models identical

---

### Tensor Parallel (Model Parallel)

```
Step 1: Shard Model
Model Layer: (d_model=512) → (d_model/4=128) per GPU

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  GPU 0          │  │  GPU 1          │  │  GPU 2          │  │  GPU 3          │
│ Model slice:    │  │ Model slice:    │  │ Model slice:    │  │ Model slice:    │
│ (512 → 128)     │  │ (512 → 128)     │  │ (512 → 128)     │  │ (512 → 128)     │
│ ~6.35M params   │  │ ~6.35M params   │  │ ~6.35M params   │  │ ~6.35M params   │
│                 │  │                 │  │                 │  │                 │
│ Batch: 32       │  │ Batch: 32       │  │ Batch: 32       │  │ Batch: 32       │
│ (ALL see same)  │  │ (ALL see same)  │  │ (ALL see same)  │  │ (ALL see same)  │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Key insight:** Model is split, but data is NOT split.

---

## Training Step Comparison

### DDP Training Step

```
Step T (Batch size 32, split 8 per GPU):

GPU 0: batch[0:8]
  ↓
  Forward pass: x (8, 256) → logits (8, 256, 65)
  Loss: compute cross-entropy on logits
  ↓
  Backward: compute gradients for ALL params
  Gradients: ∂loss/∂w for each w in model
  
GPU 1: batch[8:16]        GPU 2: batch[16:24]      GPU 3: batch[24:32]
  ↓                         ↓                          ↓
  Forward pass              Forward pass               Forward pass
  Loss                      Loss                       Loss
  Backward                  Backward                   Backward
  
All GPUs:
  ↓
  ALL-REDUCE SYNC
  ┌──────────────────────────────────┐
  │ All GPUs compute:                │
  │ avg_grad = (g0 + g1 + g2 + g3)/4 │
  │ (sum then divide by 4)           │
  │                                  │
  │ Result: all GPUs have same       │
  │ avg_grad in memory               │
  └──────────────────────────────────┘
  ↓
GPU 0: optimizer.step(avg_grad)
GPU 1: optimizer.step(avg_grad)
GPU 2: optimizer.step(avg_grad)
GPU 3: optimizer.step(avg_grad)
  ↓
  Models are IDENTICAL now

Loss curves are identical on all GPUs (same data distribution via averaging)
```

---

### Tensor Parallel Training Step

```
Step T (Batch size 32, ALL GPUs see ALL sequences):

GPU 0: batch[0:32], columns 0-127       ← My weight columns
GPU 1: batch[0:32], columns 128-255
GPU 2: batch[0:32], columns 256-383
GPU 3: batch[0:32], columns 384-511

FORWARD PASS (requires communication):

GPU 0: x @ W[0:128, :] = y0 (32, 256, 128)
         ↓ all-gather columns
         Need full W: all-gather y0, y1, y2, y3
         ↓
         y_full = [y0, y1, y2, y3] = (32, 256, 512) ← Full activation

GPU 1: x @ W[128:256, :] = y1 (32, 256, 128)
         ↓ (same all-gather happening)

GPU 2: x @ W[256:384, :] = y2 (32, 256, 128)
         ↓ (same all-gather happening)

GPU 3: x @ W[384:512, :] = y3 (32, 256, 128)
         ↓ (same all-gather happening)

All GPUs:
  Now have y_full = (32, 256, 512)
  
  Continue forward with full activation
  → attention, MLP, etc.
  → logits (32, 256, 65)
  → loss

BACKWARD PASS (requires communication):

GPU 0: ∂loss/∂W[0:128, :] = g0
       ∂loss/∂x needed for prev layer
         ↓
         Requires ∂loss/∂y from other GPUs!
         Need all-reduce-scatter: sum ∂loss/∂y0 from all GPUs
         Each GPU keeps its slice
         
       g_x = ∂loss/∂y0 @ W[0:128, :].T (32, 256, 128)
       
GPU 1: Similar, gets g_x for columns 128-255
       g_x1 (32, 256, 128)

GPU 2, 3: Similar

To pass to prev layer, need full ∂loss/∂x:
  all-gather: [g_x, g_x1, g_x2, g_x3] = (32, 256, 512)

OPTIMIZER STEP (NO synchronization needed):

GPU 0: optimizer.step(g0)  ← Updates only MY weights (128 dims)
GPU 1: optimizer.step(g1)  ← Updates only MY weights
GPU 2: optimizer.step(g2)  ← Updates only MY weights
GPU 3: optimizer.step(g3)  ← Updates only MY weights

Models are SHARDED (not identical, but complementary)
```

---

## Memory Footprint

### DDP Memory

```
GPU 0 memory:
  ├─ Model weights:        25.4M params × 4 bytes = 101.6 MB
  ├─ Model gradients:      25.4M params × 4 bytes = 101.6 MB
  ├─ Optimizer state (AdamW):
  │  ├─ m (momentum):      25.4M × 4 bytes = 101.6 MB
  │  └─ v (second moment): 25.4M × 4 bytes = 101.6 MB
  ├─ Batch (32 sequences × 256 tokens):
  │  ├─ Input tokens:      32 × 256 × int64 = 512 KB
  │  ├─ Embeddings:        32 × 256 × 512 × 4 = 16.8 MB
  │  ├─ Hidden activations (8 blocks): 8 × 16.8 = 134.4 MB
  │  └─ Attention buffers:           ~100 MB
  └─ Misc (buffers, etc):            ~50 MB
  ─────────────────────────────────────────────
  TOTAL per GPU:                    ~710 MB

4 GPUs: 4 × 710 MB = 2,840 MB (2.84 GiB) ✓ Within GB300 capacity
```

### Tensor Parallel Memory

```
GPU 0 memory (TP sharded):
  ├─ Model weights:        6.35M params × 4 bytes = 25.4 MB   (1/4 of DDP)
  ├─ Model gradients:      6.35M params × 4 bytes = 25.4 MB   (1/4 of DDP)
  ├─ Optimizer state (AdamW):
  │  ├─ m (momentum):      6.35M × 4 bytes = 25.4 MB          (1/4 of DDP)
  │  └─ v (second moment): 6.35M × 4 bytes = 25.4 MB          (1/4 of DDP)
  ├─ Batch (ALL 32 sequences, not sliced!):
  │  ├─ Input tokens:      32 × 256 × int64 = 512 KB
  │  ├─ Embeddings:        32 × 256 × 512 × 4 = 16.8 MB
  │  ├─ Hidden activations (8 blocks): 8 × 16.8 = 134.4 MB
  │  └─ Attention buffers:           ~100 MB
  └─ Misc (buffers, etc):            ~50 MB
  ─────────────────────────────────────────────
  TOTAL per GPU:                    ~460 MB

4 GPUs: 4 × 460 MB = 1,840 MB (1.84 GiB) ✓ Better!
  - But requires ALL GPUs see full batch
  - Only beneficial if batch doesn't fit on 1 GPU
```

**Key insight:** TP saves model memory but requires larger activations (full batch on each GPU).

---

## Communication Comparison

### DDP Communication Pattern

```
Forward Pass:
  GPU 0: forward(batch[0:8])    No communication needed
  GPU 1: forward(batch[8:16])
  GPU 2: forward(batch[16:24])
  GPU 3: forward(batch[24:32])
  
Backward Pass:
  GPU 0: backward()             No communication needed
  GPU 1: backward()
  GPU 2: backward()
  GPU 3: backward()
  
Gradient Sync:
  ┌────────────────────────────────────────┐
  │ ALL-REDUCE: sum gradients from 4 GPUs  │
  │ Bytes: 25.4M × 4 × 3 = 305 MB on wire  │ (ring all-reduce)
  └────────────────────────────────────────┘
  
Optimizer Step:
  GPU 0: step(avg_grad)         No communication needed
  GPU 1: step(avg_grad)
  GPU 2: step(avg_grad)
  GPU 3: step(avg_grad)

Total bytes per step: ~305 MB (after computation)
Timing: Compute ~0.8s, Comm ~0.3s (DDP scales OK)
```

### Tensor Parallel Communication Pattern

```
Forward Pass:
  GPU 0-3: forward(batch[0:32], my_weights)
    ↓
    ALL-GATHER weights/activations
    Bytes: 25.4M × 4 × 3 = 305 MB ← During forward!
    ↓
  GPU 0-3: continue forward with full weights

Backward Pass:
  GPU 0-3: backward(my_grad_slice)
    ↓
    ALL-REDUCE-SCATTER gradients
    Bytes: 25.4M × 4 × 3 = 305 MB ← During backward!
    ↓
  GPU 0-3: continue backward
    ↓
    ALL-GATHER gradients from prev layer
    Bytes: 25.4M × 4 × 3 = 305 MB ← During backward!
    ↓

Optimizer Step:
  GPU 0: step(my_grad_slice)    No communication needed
  GPU 1: step(my_grad_slice)
  GPU 2: step(my_grad_slice)
  GPU 3: step(my_grad_slice)

Total bytes per step: ~915 MB (spread through forward/backward)
Timing: Compute ~0.4s, Comm ~0.9s ← Worse! (TP is communication-heavy)
```

**Why TP is slower on 4 GPUs:**
- DDP: all-reduce happens after done, can overlap
- TP: all-gather happens **during** forward/backward, blocking

TP only wins on **massive models** where communication is smaller % of total time.

---

## When to Use Which

### Use DDP When:
```
✓ Model fits on one GPU
✓ Have large batch size (can split across GPUs)
✓ Limited inter-GPU bandwidth
✓ Want to scale to 16+ GPUs easily

Example:
  Model: 7B params (fits on 1 GPU)
  Batch: 256 sequences
  → Split batch to 64 per GPU, replicate model
  → DDP: 4x speedup with minimal communication
```

### Use Tensor Parallel When:
```
✓ Model does NOT fit on one GPU
✓ Can afford high inter-GPU bandwidth (NVLink)
✓ Must fit entire batch on each GPU
✓ Have 2-8 GPUs (more = worse scaling)

Example:
  Model: 70B params (does NOT fit on 1 GPU)
  Batch: 32 sequences
  → Each GPU gets full batch + model slice
  → TP: Necessary evil to fit model at all
```

---

## Side-by-Side Code Comparison

### DDP Code (Lab 2)

```python
import torch.distributed as dist

# Setup
dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()

# Build model (FULL model on every GPU)
model = build_model()  # 25.4M params

# Get batch (DIFFERENT batch on each GPU)
x, y = get_batch(data, step, global_batch=32, rank=rank, world=world)

# Forward
logits, loss = model(x, y)

# Backward
loss.backward()

# Sync gradients
for param in model.parameters():
    if param.grad is not None:
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world)  # Average

# Step
optimizer.step()
```

### Tensor Parallel Code (Lab 3)

```python
import torch.distributed as dist

# Setup
dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()

# Build SHARDED model (1/world size on each GPU)
model = build_sharded_model(rank, world)  # 25.4M / 4 params per GPU

# Get batch (SAME batch on all GPUs!)
x, y = get_batch(data, step, global_batch=32, rank=rank, world=world, no_shard=True)

# Forward
logits, loss = model(x, y)  # Forward contains all-gather internally

# Backward
loss.backward()  # Backward contains all-reduce/all-gather internally

# NO sync needed - shards already synchronized via all-reduce during backward

# Step
optimizer.step()
```

---

## Detailed Data Flow

### DDP Forward Pass

```
Input batch: (32, 256)

GPU 0 (8 sequences):
  x_0 = (8, 256)
  token_embed_0 = (8, 256, 512)
  pos_embed = (256, 512)
  x_embed = token_embed_0 + pos_embed
  
  for block in blocks:
    x_embed = block(x_embed)  ← Uses FULL model weights
  
  logits_0 = (8, 256, 65)

GPU 1 (8 sequences):
  x_1 = (8, 256)
  token_embed_1 = (8, 256, 512)
  x_embed = token_embed_1 + pos_embed
  
  for block in blocks:
    x_embed = block(x_embed)  ← Uses SAME full model weights
  
  logits_1 = (8, 256, 65)

GPU 2, 3: Similar

⚠ No communication in forward!
  Each GPU uses its slice of data independently
```

### TP Forward Pass

```
Input batch: (32, 256) ← FULL batch on each GPU!

GPU 0 (my weight slice: 128 dims):
  x = (32, 256)
  token_embed = (32, 256, 512)
  pos_embed = (256, 512)
  x_embed = token_embed + pos_embed  ← (32, 256, 512)
  
  block 0:
    q = q_proj(x_embed)  ← (32, 256, 512) → (32, 256, 512)
        ↓ split heads + compute attention
        ← (32, 256, 128) my slice only!
    
    all-gather: q_full = [q_gpu0, q_gpu1, q_gpu2, q_gpu3]
    ← (32, 256, 512) communication happens!
    
    ... continue with full tensor

GPU 1-3: Same pattern, each with different weight slice
         All see full activation after all-gather

✓ Communication embedded in forward!
  Causes synchronization points between GPUs
```

---

## The Oracle Principle Applied

### DDP Oracle Match

```
Lab 0 (single GPU): loss curve [4.40, 3.76, 3.27, ..., 2.52]

Lab 2 (DDP, 4 GPUs):
  GPU 0 batch: seq[0:8]   gradients: g0
  GPU 1 batch: seq[8:16]  gradients: g1
  GPU 2 batch: seq[16:24] gradients: g2
  GPU 3 batch: seq[24:32] gradients: g3
  
  All-reduce: avg_grad = (g0+g1+g2+g3)/4
  
  Update: param -= lr * avg_grad
  
  → Loss should match Lab 0 exactly!
    ✓ Same data (just different order per GPU)
    ✓ Same averaged gradients
    ✓ Same model updates
```

### TP Oracle Match (Harder!)

```
Lab 0 (single GPU): loss curve [4.40, 3.76, 3.27, ..., 2.52]

Lab 3 (TP, 4 GPUs):
  All GPUs: full batch (32 sequences)
  Each GPU: weight slice (128 dims)
  
  Forward:
    GPU 0: y0 = x @ W[0:128]    (32, 256, 128)
    GPU 1: y1 = x @ W[128:256]  (32, 256, 128)
    GPU 2: y2 = x @ W[256:384]  (32, 256, 128)
    GPU 3: y3 = x @ W[384:512]  (32, 256, 128)
    
    all-gather: y_full = [y0, y1, y2, y3]  (32, 256, 512)
    
  Continue forward with y_full...
  
  → Logits should be identical to Lab 0
    But careful! Shape mismatches or communication errors
    appear as tiny divergence (~1e-5) that oracle catches

Why TP is trickier:
  - Must manage communication during forward/backward
  - Weight sharding creates new bugs (off-by-one slicing)
  - Gradient reduction must account for sharding
  - Oracle is unforgiving: 1e-6 tolerance catches everything
```

---

## Performance Characteristics

### DDP Scaling

```
Speedup vs 1 GPU:

GPUs:  1    2    3    4    8    16   32
─────────────────────────────────────────
Ideal: 1.0  2.0  3.0  4.0  8.0  16.0 32.0
Actual:1.0  1.95 2.85 3.7  6.8  12.5 18.0

Why:
  - All-reduce overhead grows with #GPUs
  - At 32 GPUs: all-reduce = 30% of time
  - Scaling: ~95% efficient up to 8 GPUs
  - After 16 GPUs: communication becomes bottleneck
```

### Tensor Parallel Scaling

```
Speedup vs 1 GPU (on 4 GPUs, splits 4 ways):

Batch: 1   2    4    8    16   32   64
────────────────────────────────────────
Ideal: 1.0 1.0  1.0  1.0  1.0  1.0  1.0  (no speedup!)
Actual:0.8 1.1  1.4  1.8  2.2  2.7  3.1

Why:
  - Each GPU only has 1/4 of model
  - But communication is heavy (all-gather during forward)
  - Larger batches = better amortization of communication
  - Never achieves linear speedup
  
TP is slower than DDP on small models!
Only wins on massive models (70B+) where:
  - Can't fit on 1 GPU anyway
  - Communication << computation time
```

---

## Summary Table

| Aspect | DDP | TP |
|--------|-----|-----|
| **Model Location** | Full on each GPU | Sharded across GPUs |
| **Data per GPU** | Subset (batch/world) | Full batch |
| **Sync Timing** | After backward only | During forward/backward |
| **Sync Method** | All-reduce gradients | All-gather/all-reduce activations |
| **Ideal Use** | Normal models, large batch | Huge models only |
| **Scaling** | Linear up to ~16 GPUs | Sub-linear, even fewer GPUs |
| **Memory/GPU** | O(M) | O(M/N) where N=world_size |
| **Communication/step** | ~305 MB (after compute) | ~915 MB (during compute) |
| **Ease** | Simple | Complex (requires careful sharding) |
| **Bug Detection** | Oracle catches gradient bugs | Oracle catches communication bugs |

---

## Real Example: Training a 25.4M Model

### Scenario 1: DDP on 4 GPUs

```
Config:
  Model: 25.4M params (fits on 1 GPU: 100MB)
  Batch per GPU: 8 sequences
  Total batch: 32 sequences
  GPUs: 4

Memory per GPU:
  Model + gradients + optimizer: ~400 MB
  Batch activations: ~300 MB
  Total: ~700 MB per GPU ✓ Plenty of room

Performance:
  Compute time: ~0.8s per step
  All-reduce: ~0.1s per step
  Total: ~0.9s per step
  
  4 GPU speedup: 3.8x (95% efficient) ✓

Result: Excellent scaling!
```

### Scenario 2: TP on 4 GPUs

```
Config:
  Model: 25.4M params (already fits on 1 GPU!)
  Batch per GPU: 32 sequences (full batch)
  Total batch: 32 sequences
  GPUs: 4

Memory per GPU:
  Model shards: ~100 MB (vs ~400 MB in DDP)
  Batch activations: ~600 MB (full batch, vs ~300 MB in DDP)
  Total: ~700 MB per GPU ← Same!

Performance:
  Compute time: ~0.4s per step (1/4 weights)
  All-gather forward: ~0.3s per step
  All-reduce backward: ~0.3s per step
  All-gather backward: ~0.2s per step
  Total: ~1.2s per step
  
  4 GPU speedup: 2.7x (67% efficient) ✗

Result: WORSE than DDP!
```

**Lesson:** Don't use TP when DDP works. TP is for models that don't fit on a single GPU.
