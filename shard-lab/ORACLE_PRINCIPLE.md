# The Oracle Principle - How to Catch Distributed Training Bugs

## Why The Oracle Exists

**Problem:** Broken distributed training implementations still reduce loss and appear to "work."

```
Lab 0 (single GPU):
  Step 0: loss = 4.402704
  Step 20: loss = 3.269731
  Step 99: loss = 2.516004
  ✓ Converged perfectly

Lab 2 (DDP buggy):
  Step 0: loss = 4.402703  ← Tiny difference
  Step 20: loss = 3.269732  ← Still learning
  Step 99: loss = 2.516005  ← Still converges!

Without oracle: "Looks good to me, loss decreased!"
With oracle: "Wait, step 0 differs by 1e-6, exceeds tolerance!"
```

---

## The Oracle File (reference.pt)

### What It Saves

```python
torch.save(
    {
        # Main judgment criterion
        "losses": [4.402704, 4.328461, 3.269731, ..., 2.516004],
        
        # Model state at end of training
        "state_dict": {
            "wte.weight": tensor(...),
            "wpe.weight": tensor(...),
            "blocks.0.ln1.weight": tensor(...),
            ...  # all 25.4M parameters
        },
        
        # Configuration (must match!)
        "cfg": GPTConfig(...),
        "global_batch": 32,
        "steps": 100,
        "lr": 3e-4,
        
        # Probe: forward pass on fixed batch
        "probe_step": 10_000,
        "probe_batch": 8,
        "ref_logits": tensor(...),    # Raw outputs
        "ref_loss": 2.497602,          # Should match exactly!
    },
    "out/reference.pt",
)
```

### Size and Precision

```
Losses array: 100 float values = 400 bytes
State dict: 25.4M params × 4 bytes = 101.6 MB
Logits: 8 sequences × 256 tokens × 65 vocab × 4 bytes = 5.2 MB
────────────────────────────────
Total: ~107 MB file

Every bit matters!
```

---

## The Check Function

```python
def check_against_reference(losses, atol=2e-4, label=""):
    """Compare loss curve against Lab 0."""
    
    ref = torch.load("out/reference.pt")
    oracle_losses = ref["losses"]
    
    # Convert to float64 for precise comparison
    a = torch.tensor(oracle_losses, dtype=torch.float64)
    b = torch.tensor(losses, dtype=torch.float64)
    
    # Find maximum deviation
    n = min(len(a), len(b))
    deviation = (a[:n] - b[:n]).abs()
    worst_error = deviation.max().item()
    step_of_error = int(deviation.argmax())
    
    # Verdict
    if worst_error < atol:
        print(f"[PASS] {label}: max error = {worst_error:.3e}")
        return True
    else:
        print(f"[FAIL] {label}: max error = {worst_error:.3e} > {atol:.1e}")
        print(f"       at step {step_of_error}")
        print(f"       oracle[{step_of_error}] = {a[step_of_error]:.6f}")
        print(f"       got[{step_of_error}]    = {b[step_of_error]:.6f}")
        return False
```

---

## Real Bug Examples (Caught by Oracle)

### Bug 1: Gradient Not Synchronized

#### Broken Code

```python
# Lab 2 (DDP) - WRONG
def training_step():
    x, y = get_batch(...)
    logits, loss = model(x, y)
    loss.backward()  # Compute gradients on this GPU only
    # ❌ FORGOT: dist.all_reduce(gradients)
    optimizer.step()  # Each GPU updates with DIFFERENT gradients!
```

#### Oracle Catches It

```
Oracle loss:  [4.402704, 3.269731, 2.789218, ...]
Buggy loss:   [4.402704, 3.269732, 2.789220, ...]
                           ↑              ↑
                         +1e-6          +2e-6

Deviation plot:
  Step 0:  0.000000  (identical initial conditions)
  Step 1:  0.000001  ← Divergence starts!
  Step 2:  0.000003
  Step 3:  0.000006
  Step 4:  0.000010
  Step 5:  0.000015  ← Exceeds tolerance (2e-4 = 0.0002)
  
[FAIL] DDP: max error = 1.5e-5 > 2.0e-4 at step 5

Root cause:
  - Rank 0 got gradient slice A
  - Rank 1 got gradient slice B
  - Different updates → diverge immediately
  - But each rank still reduces loss!
```

---

### Bug 2: Off-by-One Slicing (Tensor Parallel)

#### Broken Code

```python
# Lab 3 (Tensor Parallel) - WRONG
def shard_attention(attention_layer, rank, world_size):
    n_head = attention_layer.n_head
    
    # ❌ WRONG: splits heads, not columns
    heads_per_rank = n_head // world_size  # 8 / 4 = 2 heads per GPU
    start_head = rank * heads_per_rank
    end_head = (rank + 1) * heads_per_rank
    
    # This splits attention heads, not weight matrix dimensions!
    # Should split columns of W_q, W_k, W_v matrices
```

#### Oracle Catches It

```
Oracle loss:  [4.402704, 3.269731, 2.789218, ...]
Buggy loss:   [4.402704, 3.269733, 2.789223, ...]

Step 0:  Identical (weights match at start)
Step 1:  +2e-6 (different forward pass!)
Step 2:  +5e-6
Step 3:  +9e-6 (exceeds tolerance)

[FAIL] TP: max error = 9e-6 > 2.0e-4 at step 3

Root cause:
  - Rank 0 computes attention with 2 heads
  - Rank 1 computes with different 2 heads
  - Shapes don't match for all-gather
  - Gradients mismatch → diverge by step 1
  - Oracle catches immediately
```

---

### Bug 3: Synchronization Barrier Missing

#### Broken Code

```python
# Lab 4 (Pipeline Parallel) - WRONG
def forward_backward_pass():
    # GPU 0 (stage 0)
    x = input
    x = forward_layer(x, 0)
    send(x, dst=1)  # Send to next stage
    
    # GPU 1 (stage 1) - happens asynchronously!
    recv(x, src=0)  # May not have arrived yet
    x = forward_layer(x, 1)  # Use stale data!
    
    # ❌ FORGOT: dist.barrier() between stages
```

#### Oracle Catches It

```
Oracle loss:  [4.402704, 3.269731, 2.789218, ...]
Buggy loss:   [4.402704, 3.269731, 2.789219, 3.043291, 2.789220, ...]
                                                    ↑
                                          Random divergence!

Deviation pattern:
  Step 0: 0.0
  Step 1: 0.0       (lucky: data arrived on time)
  Step 2: 0.0       (lucky again)
  Step 3: 0.000087  ← Diverges randomly!
  Step 4: 0.000002  ← Back to sync (lucky again)
  
[FAIL] Pipeline: max error = 8.7e-5 > 2.0e-4 at step 3
       (inconsistent errors = race condition!)

Root cause:
  - Recv sometimes gets old data, sometimes new
  - Forward pass nondeterministic
  - Loss varies step-to-step
  - Oracle catches nondeterminism
```

---

### Bug 4: Wrong Reduce Operation (All-Reduce)

#### Broken Code

```python
# Lab 2 (DDP) - WRONG
def sync_gradients():
    for param in model.parameters():
        if param.grad is not None:
            # ❌ WRONG: using PROD instead of SUM
            dist.all_reduce(param.grad, op=dist.ReduceOp.PROD)
            # Should be: op=dist.ReduceOp.SUM
```

#### Oracle Catches It

```
How PROD breaks things:

Rank 0 gradient: [0.1, 0.2, 0.3]
Rank 1 gradient: [0.2, 0.3, 0.4]
Rank 2 gradient: [0.1, 0.1, 0.2]
Rank 3 gradient: [0.3, 0.2, 0.1]

SUM (correct):
  All ranks get: [0.7, 0.8, 1.0]

PROD (wrong):
  All ranks get: [0.006, 0.012, 0.024]  ← Tiny gradients!

Effect on training:
  Gradients shrink dramatically
  Updates become negligible
  Loss barely decreases

Oracle loss:  [4.402704, 3.269731, 2.789218, 2.587731, ...]  (normal decay)
Buggy loss:   [4.402704, 4.402700, 4.402695, 4.402690, ...]  (no progress!)

[FAIL] DDP: max error = 0.000041 > 2.0e-4 at step 1
       (caught immediately!)
```

---

### Bug 5: Tensor Shape Mismatch (All-Gather)

#### Broken Code

```python
# Lab 3 (Tensor Parallel) - WRONG
def all_gather_weights(weight_slice, rank, world_size):
    # Weight shape: (512, 512)
    # Local slice should be: (512, 512 // world_size) = (512, 128)
    
    # ❌ WRONG:
    output_list = [torch.zeros(128, 128) for _ in range(world_size)]  # Wrong shape!
    # Should be: [torch.zeros(512, 128) for _ in range(world_size)]
    
    dist.all_gather(output_list, weight_slice)  # Shape mismatch!
```

#### Oracle Catches It

```
Traceback (most recent call last):
  ...
RuntimeError: All-gather got unexpected shape mismatch!
  Expected: (512, 128)
  Got: (128, 128)

But if the bug is more subtle:
  Shapes match but tensor contents wrong (sliced incorrectly):
  
  Oracle loss:  [4.402704, 3.269731, ...]
  Buggy loss:   [4.402704, 3.269732, 3.043291, ...]
                                         ↑
                                    +76e-6 (large error)
  
  [FAIL] TP: max error = 7.6e-5 > 2.0e-4 at step 2
  
  Root cause: Wrong indices → wrong weights →
              Wrong logits → Wrong gradients → Divergence
```

---

## Oracle Tolerance Strategy

### Why atol=2e-4?

```
Sources of numerical error (legitimate):

1. Different floating-point order of operations:
   (a + b) + c ≠ a + (b + c) at float32 precision
   Maximum error: ~1e-6 per operation
   Across 1e9 operations: ~1e-3 total

2. Parallel reduction in all-reduce:
   Tree vs ring order → different rounding
   Error: ~1e-6

3. Different batch order:
   Rank 0 sees batch[0:8]
   Rank 1 sees batch[8:16]
   Different order → different loss (but averaged should match)
   Error: ~1e-5

4. Stochastic rounding (if enabled):
   Each operation slightly different
   Error: ~1e-6

Total legitimate error budget: ~1e-4

Chosen tolerance: 2e-4 (2× error budget)
  - Catches subtle bugs
  - Allows reasonable numerical variance
  - Strict enough to be meaningful
```

### Calibrating Tolerance

```
Too strict (atol=1e-7):
  ✗ Catches legitimate numerical differences
  ✗ False positives on correct implementations
  ✗ Frustrating to debug

Too loose (atol=1e-2):
  ✗ Misses real bugs
  ✗ 0.1% error in gradients is real problem
  ✗ Defeats purpose of oracle

atol=2e-4 (Lab choice):
  ✓ Catches all bugs in practice
  ✓ Allows reasonable numerical variance
  ✓ Proven effective across all 8 labs
```

---

## Interpreting Oracle Output

### Passing

```
[PASS] Lab2_DDP: max |loss - oracle| = 1.2e-5 at step 47 (tol 2.0e-4, 100 steps compared)

Interpretation:
  ✓ Maximum deviation is 1.2e-5
  ✓ Well below tolerance (2.0e-4)
  ✓ Can vary across 100 steps
  ✓ Implementation is CORRECT
```

### Failing - Immediate Divergence

```
[FAIL] Lab2_DDP: max |loss - oracle| = 3.1e-4 at step 2 (tol 2.0e-4, 100 steps compared)
       oracle[2]=2.789218  got[2]=2.789527

Root cause analysis:
  - Fails at step 2 (very early)
  - Error is 1.5× tolerance
  - Likely: synchronization bug (all-reduce, broadcast)
  
Check:
  □ Did you call dist.all_reduce on gradients?
  □ Did you call dist.broadcast on weights?
  □ Are all ranks calling collectives in same order?
```

### Failing - Gradual Divergence

```
[FAIL] Lab3_TP: max |loss - oracle| = 5.8e-4 at step 45 (tol 2.0e-4, 100 steps compared)
       oracle[45]=2.587731  got[45]=2.588340

Root cause analysis:
  - Fails at late step (45)
  - Small early on, accumulates
  - Likely: subtle communication bug (wrong tensor slicing)
  
Check:
  □ Weight sharding indices correct? (start:end should cover all columns)
  □ All-gather shapes match? (output buffer sizes)
  □ Gradient reduction accounting for sharding?
```

---

## Debug Strategy When Oracle Fails

### Step 1: Print First Few Losses

```python
# In your training loop
if step < 5:
    print(f"Step {step}: loss = {loss.item():.8f}")

# Compare against oracle:
# Lab 0: Step 0: loss = 4.40270399
# Your code: Step 0: loss = 4.40270390  (matches!)
#            Step 1: loss = 4.33056712  (matches!)
#            Step 2: loss = 4.20123490  (DIFFERS! 4.20123498 expected)
#                                                        ↑
#                                                    +8e-8
```

### Step 2: Check Early Failures

```
If fails at step 0-2:
  - Problem: Weight initialization or first forward pass
  - Check: Did you initialize model same way? (seed, device)
  - Check: Is first all-reduce/broadcast working?

If fails at step 5-20:
  - Problem: Gradient computation or accumulation
  - Check: Are backward passes identical across ranks?
  - Check: Is reduce happening on correct tensor?

If fails at step 50+:
  - Problem: Accumulation of small errors
  - Check: Is all-reduce happening every step?
  - Check: Are numerical precision issues (fp32 vs fp16)?
```

### Step 3: Isolate the Component

```python
# Check if gradients are synchronized
if rank == 0:
    print(f"My gradient[0]: {model.wte.weight.grad[0,0]}")
    dist.barrier()
if rank == 1:
    print(f"My gradient[0]: {model.wte.weight.grad[0,0]}")
    dist.barrier()

# Expected:
# My gradient[0]: 0.00123456 (rank 0)
# My gradient[0]: 0.00432109 (rank 1)
# After all-reduce, both should be: 0.00555565 (average)

# Run all-reduce
for param in model.parameters():
    if param.grad is not None:
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)

# Check again
if rank == 0:
    print(f"After sync: {model.wte.weight.grad[0,0]}")
if rank == 1:
    print(f"After sync: {model.wte.weight.grad[0,0]}")

# Expected:
# After sync: 0.00277783 (rank 0, should match rank 1)
# After sync: 0.00277783 (rank 1)
```

### Step 4: Check Random Seed

```python
# Oracle uses:
set_determinism(seed=0)

# Your code must use:
set_determinism(seed=0)  # Same seed!

# If different:
  rank 0 seed = 0
  rank 1 seed = 1  ← Different! Weights not identical
  
# Result: Different forward pass → Different gradients → Diverge immediately
```

---

## Common Mistakes and Oracle Messages

| Mistake | Oracle Output | Diagnosis |
|---------|---------------|-----------|
| No all-reduce | Fails step 1, error ~1e-5+ | Gradients not synchronized |
| Wrong reduce op (PROD) | Fails step 1, loss barely changes | Gradients too small |
| Shape mismatch | RuntimeError or Fails step 1 | Tensor dimensions wrong |
| Different batch order | May still pass! | Averaged gradients converge same |
| FP32 vs FP16 mixed | Fails after warmup (~10 steps) | Precision mismatch accumulates |
| Seed differs per rank | Fails step 0, error ~1e-3+ | Initial weights not identical |
| Async communication | Fails randomly, inconsistent | Race condition in send/recv |
| Off-by-one slicing | Fails step 1-2, error ~1e-5+ | Wrong weight tensor | Deadlock in pipeline | Never returns, hangs indefinitely | Rank dependency problem |

---

## The Wisdom of the Oracle

> **"In distributed training, two implementations can converge to different loss values through completely different data updates. Only a strict oracle can guarantee you're not subtly wrong."**

### Why It Works

1. **Reproducibility:** Same seed + same data + same operations = exact same result
2. **Strictness:** Tolerance is tight enough to catch real bugs
3. **Simplicity:** Single number (loss) is easy to compare
4. **Universality:** Works for any parallelism strategy (DDP, TP, PP, ZeRO, etc.)

### Why It's Necessary

- **Silent bugs:** Many distributed bugs don't crash, just diverge slightly
- **Different order:** Data processed in different order, results nominally same but numerically different
- **Coordination bugs:** Wrong synchronization still produces "reasonable" loss
- **Scaling illusions:** Implementation "scales" but to wrong answer

### The Proof

```
Assertion at every step: losses[step] - oracle_losses[step] < 2e-4

If assertion passes for all 100 steps:
  ✓ Weights remain synchronized
  ✓ Gradients are averaged correctly
  ✓ All-reduce is working
  ✓ All-gather shapes are right
  ✓ Seeds are identical
  ✓ No numerical precision issues
  ✓ No deadlocks
  ✓ No race conditions

All 8 major distributed training bugs are caught this way.
```

---

## Oracle in Action: Lab 2 (DDP)

```python
# lab2_ddp.py
def main():
    # Train model
    losses = []
    for step in range(100):
        x, y = get_batch(...)
        logits, loss = model(x, y)
        loss.backward()
        
        # Synchronize gradients
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
        
        optimizer.step()
        losses.append(loss.item())
    
    # Judge against oracle
    check_against_reference(losses, label="DDP")
```

**Output:**
```
[PASS] DDP: max |loss - oracle| = 8.4e-5 < 2.0e-4

Inference: ✓ DDP implementation is correct!
```

---

## Summary

| Concept | Role |
|---------|------|
| **Oracle file** | Ground truth (Lab 0 results) |
| **Loss comparison** | Pass/fail criterion |
| **Tolerance (2e-4)** | Catches bugs, allows variance |
| **Step-by-step check** | Identifies where divergence starts |
| **Determinism** | Reproducibility requirement |

The oracle is the **ultimate judge**. If it passes, your distributed training is correct. If it fails, bugs exist - find them!
