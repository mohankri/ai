# Complete AI Model Training Guide - Index & Roadmap

## What You Now Have

I've created 8 comprehensive guides covering everything from basic AI model training to advanced distributed techniques. Here's the complete package:

---

## 📚 Learning Sequence (Recommended Order)

### 1. **Core Model Training (Lab 0)**
- [LAB0_QUICK_REFERENCE.md](LAB0_QUICK_REFERENCE.md) ← **Start here** (5 min)
  - Quick summary of model training concepts
  - Hyperparameters explained
  - One-minute overview of loss curve

- [LAB0_DETAILED_WALKTHROUGH.md](LAB0_DETAILED_WALKTHROUGH.md) (20 min)
  - Complete token-by-token example
  - "First Citizen" → tokens → embeddings → predictions
  - Loss computation step-by-step

- [LAB0_VISUAL_FLOW.md](LAB0_VISUAL_FLOW.md) (15 min)
  - ASCII diagrams of data flowing through model
  - Attention mechanism visualization
  - Memory & compute breakdown
  - Loss progression over 100 steps

- [LAB0_CODE_WALKTHROUGH.md](LAB0_CODE_WALKTHROUGH.md) (20 min)
  - Python code with actual tensor operations
  - See shapes at each layer
  - Complete training loop example

---

### 2. **Understanding Communication (Lab 1)**
- [LAB1_COLLECTIVES_EXPLAINED.md](LAB1_COLLECTIVES_EXPLAINED.md) (25 min)
  - All 5 collective operations with examples
  - **broadcast:** one-to-many
  - **all_reduce:** sum from all, result to all
  - **reduce_scatter:** sum then split
  - **all_gather:** reassemble pieces
  - **all_to_all:** permutation routing
  - **send/recv:** direct GPU-to-GPU
  
  Key insight: reduce_scatter + all_gather = all_reduce (oracle principle)

---

### 3. **Comparison: DDP vs Tensor Parallel (Labs 2 & 3)**
- [DDP_VS_TENSORPARALLEL.md](DDP_VS_TENSORPARALLEL.md) (30 min)
  - When to use each approach
  - Memory footprint comparison
  - Communication patterns
  - Performance scaling
  - DDP: same model, different data ✓ Scales linearly up to ~32 GPUs
  - TP: split model, same data ✗ Only for huge models that don't fit

---

### 4. **Deep Dive: Attention Mechanism**
- [ATTENTION_DEEP_DIVE.md](ATTENTION_DEEP_DIVE.md) (30 min)
  - Attention intuition (what it learns)
  - Query/Key/Value mathematics
  - Complete example with "hello" → predict next character
  - Multi-head attention (why multiple?)
  - Causal masking (can't look at future)
  - Why better than RNN
  - Complexity analysis (O(T²) is bottleneck)
  - How Lab 3 shards attention across GPUs

---

### 5. **Quality Assurance: The Oracle Principle**
- [ORACLE_PRINCIPLE.md](ORACLE_PRINCIPLE.md) (25 min)
  - Why oracle exists (catch subtle bugs)
  - Real bug examples (5 common mistakes)
  - How each bug is detected
  - Oracle tolerance strategy (atol=2e-4)
  - Debug strategy when oracle fails
  - Common mistakes and their signatures

---

## 🎯 Quick Navigation

### By Topic

**Want to understand models?**
→ LAB0_QUICK_REFERENCE → LAB0_DETAILED_WALKTHROUGH → ATTENTION_DEEP_DIVE

**Want to understand distributed training?**
→ LAB1_COLLECTIVES_EXPLAINED → DDP_VS_TENSORPARALLEL → ORACLE_PRINCIPLE

**Want to see code examples?**
→ LAB0_CODE_WALKTHROUGH → LAB1_COLLECTIVES_EXPLAINED (has Python)

**Want visual explanations?**
→ LAB0_VISUAL_FLOW → DDP_VS_TENSORPARALLEL → ATTENTION_DEEP_DIVE

**Want to debug failing code?**
→ ORACLE_PRINCIPLE (entire doc is debugging guide)

---

## 🚀 Key Concepts Summary

### Model Training (Lab 0)

```
Data → Tokenize → Embed → Transformer (8 blocks) → Logits → Loss → Backward → Update

Loss: 4.4 (start, barely better than random) → 2.5 (end, learned patterns)
Memory: ~2.1 GiB on single GPU
Speed: 27.8 steps/sec (after warmup)
```

### Communication Primitives (Lab 1)

| Operation | Purpose | Lab Use | Bytes |
|-----------|---------|---------|-------|
| Broadcast | One→All | DDP init | same on all |
| All-reduce | Sum then all get | DDP gradient sync | 3× tensor size |
| Reduce-scatter | Sum then split | ZeRO stage 2 | 3/4× tensor size |
| All-gather | Gather pieces | FSDP, TP fwd | 3/4× tensor size |
| All-to-all | Permute | Expert routing | 1× tensor size |
| Send/recv | Direct | Pipeline stages | custom |

### Data Parallel vs Model Parallel (Labs 2 & 3)

| Aspect | DDP (Lab 2) | TP (Lab 3) |
|--------|-----------|-----------|
| Model | Full on each GPU | Sharded across GPUs |
| Data | Different batches | Same batch on all |
| Speedup | Linear (95% eff 4-8 GPU) | Sub-linear (60% eff 4 GPU) |
| When | Normal models | Huge models only |
| Communication | After each step (sync) | During forward/backward |

### Attention Mechanism (Deep Dive)

```
Query (what I want) · Key (what you are) = Score
Score → Softmax → Weights
Output = Weights · Values

Causal mask: can't attend to future
Multi-head: 8 heads learn different patterns
```

### Quality Assurance (Oracle Principle)

```
Lab 0: Single GPU → loss curve (ground truth)
Lab 1-7: Distributed → must match Lab 0 to ±2e-4

If diverges:
  Step 1-2: initialization/synchronization bug
  Step 5-20: gradient computation error
  Step 50+: accumulation of small errors
```

---

## 🔧 Practical Workflow

### Running Lab 0 (Single GPU)

```bash
cd ~/ai/shard-lab
.venv/bin/python lab0_reference.py
# Creates: out/reference.pt (the oracle)
```

### Running Lab 1 (Collectives, 4 GPUs)

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 lab1_collectives.py
# Tests: broadcast, all_reduce, reduce_scatter, all_gather, all_to_all, send/recv
```

### Running Lab 2 (DDP, 4 GPUs)

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 lab2_ddp.py
# Must match Lab 0 loss curve exactly
```

### Running Lab 3 (Tensor Parallel, 4 GPUs)

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 lab3_tp.py
# Trickier! Must match Lab 0 with all-gather during forward/backward
```

---

## 🐛 Debugging Checklist

When oracle fails:

- [ ] **Step 0 matches:** ✓ Initialization is correct
- [ ] **Step 1 matches:** ✓ First forward pass is correct
- [ ] **Step 2-5 match:** ✓ Gradient computation is correct
- [ ] **Later steps match:** ✓ No accumulation of errors
- [ ] **All 100 steps match:** ✓ Implementation is CORRECT!

If fails at step N:
1. Print `loss[N]` and `oracle_loss[N]`
2. Check what changed at step N (collective call? data? model update?)
3. Compare against Lab 0 code for that exact operation
4. Look for: wrong tensor shape, wrong reduce op, missing sync, wrong slice

---

## 📖 How to Use This Package

### For Learning

1. Start with **LAB0_QUICK_REFERENCE** (5 min orientation)
2. Read **LAB0_DETAILED_WALKTHROUGH** (understand one complete step)
3. Study **LAB0_VISUAL_FLOW** (see the big picture)
4. Reference **LAB0_CODE_WALKTHROUGH** (as needed for specific operations)
5. Continue to **LAB1_COLLECTIVES_EXPLAINED** (learn communication)
6. Then **DDP_VS_TENSORPARALLEL** (understand tradeoffs)
7. Deep dive **ATTENTION_DEEP_DIVE** (master the mechanism)
8. Finally **ORACLE_PRINCIPLE** (understand quality assurance)

### For Implementing

1. Write code for Lab 2 (DDP) - simplest distributed case
2. Reference **LAB1_COLLECTIVES_EXPLAINED** for collective calls
3. Reference **DDP_VS_TENSORPARALLEL** for architecture differences
4. Check against **ORACLE_PRINCIPLE** to validate
5. Debug failures using ORACLE_PRINCIPLE debugging section

### For Debugging

1. Run code until oracle fails
2. Go to **ORACLE_PRINCIPLE**, find similar failure pattern
3. Check root causes listed
4. Use debug checklist
5. Print intermediate values
6. Compare step-by-step with Lab 0

---

## 🧠 Mental Models

### Model Training Stages

```
Stage 1: Random (step 0)
  - Loss ≈ random guess on 65 chars = 4.17
  - Model predicts wrong answers

Stage 2: Learning Patterns (steps 0-30)
  - Loss drops steeply (4.4 → 3.2)
  - Model learns common character pairs ("th", "er")

Stage 3: Fine-tuning (steps 30-80)
  - Loss declines slowly (3.2 → 2.5)
  - Model learns context (position matters)

Stage 4: Convergence (steps 80-99)
  - Loss plateau (2.5 → 2.5)
  - Model has learned what it can from data
```

### Distributed Training Scaling

```
DDP (Data Parallel):
  1 GPU:   baseline
  2 GPUs:  ~1.9x speedup (good!)
  4 GPUs:  ~3.8x speedup (excellent)
  8 GPUs:  ~7.6x speedup (still good)
  16 GPUs: ~14x speedup (good)
  32 GPUs: ~22x speedup (overhead starts)

TP (Tensor Parallel):
  1 GPU:   baseline
  2 GPUs:  ~1.3x speedup (poor!)
  4 GPUs:  ~2.7x speedup (okay)
  8 GPUs:  ~4.2x speedup (meh)

Key: DDP is much better for models that fit on one GPU!
```

### Communication Cost

```
All-reduce (ring algorithm):

  Phase 1 (reduce-scatter):
    GPU 0 → GPU 1: data[0:256/4]
    GPU 1 → GPU 2: data[256/4:512/4]
    GPU 2 → GPU 3: data[512/4:768/4]
    GPU 3 → GPU 0: data[768/4:1024/4]
    Cost: 3/4 of tensor size

  Phase 2 (all-gather):
    (same cost)
    
  Total: 3/2 of tensor size on network

Implication:
  - 4 GPUs, 100MB tensor → 150 MB on network per step
  - At 50 GB/s bandwidth → 3ms per step
  - If forward pass is 800ms → 0.4% overhead (negligible)
```

---

## 🎓 Learning Outcomes

After working through this package, you'll understand:

✓ How transformer models work (tokens → embeddings → attention → logits)
✓ How loss is computed and gradients flow backward
✓ How to use AdamW optimizer and weight decay
✓ How all 5 collective operations work (broadcast, all_reduce, reduce_scatter, all_gather, all_to_all)
✓ Why DDP scales linearly but TP doesn't
✓ When to use DDP vs TP
✓ How attention mechanism works mathematically
✓ Why multi-head attention is better than single-head
✓ How causal masking prevents cheating
✓ Why oracle principle catches subtle bugs
✓ How to debug distributed training failures

---

## 🔗 Cross-References

### Collective Operations Appear In

- **Broadcast:** Lab 2 init, Lab 6 coordination
- **All-reduce:** Lab 2 (every step!), Lab 6 (partial)
- **Reduce-scatter:** Lab 5 (stage 2), Lab 6 (hybrid)
- **All-gather:** Lab 3 (forward), Lab 5 (every backward), Lab 6 (often)
- **Send/recv:** Lab 4 (stage boundaries)

### Attention Appears In

- **Lab 0:** 8 transformer blocks, 8 heads each = 64 attention heads total
- **Lab 3:** Sharded attention (why it's complex)
- **Later labs:** Attention is just a layer, other optimizations focus on it

### Oracle Principle Applies To

- **Lab 0:** Creates oracle (loss curve)
- **Lab 1:** No model, so no oracle check (but concepts used)
- **Labs 2-7:** MUST match Lab 0 oracle to ±2e-4

---

## 📊 Complexity at a Glance

| Document | Time | Difficulty | Code? | Math? |
|----------|------|-----------|-------|-------|
| LAB0_QUICK_REFERENCE | 5 min | Easy | No | Light |
| LAB0_DETAILED_WALKTHROUGH | 20 min | Medium | No | Medium |
| LAB0_VISUAL_FLOW | 15 min | Medium | No | Light |
| LAB0_CODE_WALKTHROUGH | 20 min | Hard | Yes | Medium |
| LAB1_COLLECTIVES_EXPLAINED | 25 min | Medium | Yes | Light |
| DDP_VS_TENSORPARALLEL | 30 min | Hard | Yes | Medium |
| ATTENTION_DEEP_DIVE | 30 min | Hard | Yes | Hard |
| ORACLE_PRINCIPLE | 25 min | Medium | Yes | Light |
| **TOTAL** | **170 min** | **~6 hours** | **Majority** | **Moderate** |

---

## 🎬 Next Steps

### Immediate (Understanding)
1. Read LAB0_QUICK_REFERENCE (you should do this in 5 min)
2. Skim LAB0_VISUAL_FLOW (get mental image)
3. Read LAB1_COLLECTIVES_EXPLAINED (understand communication)

### Short Term (Implementation)
1. Run Lab 0: `.venv/bin/python lab0_reference.py`
2. Run Lab 1: `.venv/bin/torchrun --standalone --nproc_per_node=4 lab1_collectives.py`
3. Try to implement Lab 2 (DDP) based on guides
4. Check with oracle: `check_against_reference(losses, label="DDP")`

### Medium Term (Mastery)
1. Implement Lab 3 (Tensor Parallel) - hardest!
2. Use ATTENTION_DEEP_DIVE to understand sharding
3. Use ORACLE_PRINCIPLE to debug failures
4. Implement Labs 4-7 (pipeline, ZeRO, etc.)

### Long Term (Application)
1. Apply to real models (LLaMA, GPT)
2. Combine multiple strategies (2D mesh: DDP + TP)
3. Implement ZeRO optimizations
4. Use production frameworks (FSDP2, DTensor)

---

## 📝 Quick Facts

- **Model size:** 25.4M parameters
- **Vocabulary:** 65 characters (TinyShakespeare)
- **Sequence length:** 256 tokens (context window)
- **Batch size:** 32 sequences
- **Number of blocks:** 8 transformer layers
- **Number of heads:** 8 attention heads per block
- **Hidden dimension:** 512 (d_model)
- **Training steps:** 100
- **Initial loss:** 4.402704 (barely better than random)
- **Final loss:** 2.516004 (trained model)
- **Probe loss:** 2.497602 (generalization)
- **Time:** 3.24 seconds for 90 steps (27.8 steps/sec)
- **Memory:** 2.1 GiB peak (single GPU)
- **Oracle tolerance:** 2e-4 (strict but fair)

---

## 🏆 The Big Picture

This package teaches **step-by-step** what happens when you train a model and why distributed approaches work:

1. **Lab 0:** Single GPU baseline (understand the process)
2. **Lab 1:** Learn communication primitives (collectives)
3. **Lab 2:** Replicate model, split data (DDP)
4. **Lab 3:** Split model, same data (TP)
5. **Lab 4:** Chain stages together (Pipeline)
6. **Lab 5:** Shard gradients and optimizer (ZeRO)
7. **Lab 6:** Combine TP and DDP (2D Mesh)
8. **Lab 7:** Use production frameworks (FSDP2/DTensor)

Each adds complexity, but **everything must match Lab 0**.

The oracle principle is the key: *exact* numerical agreement proves correctness.

---

## 💡 Remember

> **"A broken distributed training implementation still reduces loss. The oracle catches it because it's unforgiving."**

This is the insight that makes this course powerful. Without the oracle, subtle bugs hide forever. With it, they can't escape.

Use these guides as reference while implementing. When stuck, search the relevant document for that concept. When oracle fails, check the debugging section in ORACLE_PRINCIPLE.

Good luck! 🚀
