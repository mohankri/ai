# Lab 1 - Collective Operations Explained

## Overview

Lab 1 teaches the 5 core communication patterns used in distributed training. **No model, no autograd** — just small labelled tensors so you see exactly what each operation does.

**Run:** `.venv/bin/torchrun --standalone --nproc_per_node=4 lab1_collectives.py`

---

## The Five Collectives

| Collective | Purpose | Used in |
|---|---|---|
| `broadcast` | All ranks get rank 0's data | DDP init, gradient synchronization |
| `all_reduce` | All ranks get reduced result | DDP gradient sync (most important) |
| `reduce_scatter` | Reduce then distribute pieces | ZeRO stage 2/3, all-reduce optimization |
| `all_gather` | Gather pieces into full tensor | FSDP materialization, weight synchronization |
| `all_to_all` | Permutation (expert routing) | Mixture of Experts |
| `send`/`recv` | Direct GPU-GPU (blocking) | Pipeline parallel stage boundaries |

---

## Collective 1: Broadcast

### What It Does
**One GPU (rank 0) sends its data to all other GPUs.**

### Example
```
Before broadcast:
  Rank 0: [100, 100, 100, 100]
  Rank 1: [0,   0,   0,   0]
  Rank 2: [0,   0,   0,   0]
  Rank 3: [0,   0,   0,   0]

dist.broadcast(tensor, src=0)

After broadcast:
  Rank 0: [100, 100, 100, 100]  ← unchanged (source)
  Rank 1: [100, 100, 100, 100]  ← copied from rank 0
  Rank 2: [100, 100, 100, 100]  ← copied from rank 0
  Rank 3: [100, 100, 100, 100]  ← copied from rank 0
```

### Code Pattern
```python
import torch.distributed as dist

# Rank 0 has data, others have zeros
if rank == 0:
    data = torch.tensor([100, 100, 100, 100])
else:
    data = torch.zeros(4)

# Broadcast from rank 0
dist.broadcast(data, src=0)

# Now all ranks have [100, 100, 100, 100]
print(f"Rank {rank}: {data}")
# Output:
# Rank 0: tensor([100, 100, 100, 100])
# Rank 1: tensor([100, 100, 100, 100])
# Rank 2: tensor([100, 100, 100, 100])
# Rank 3: tensor([100, 100, 100, 100])
```

### Use in Distributed Training
- **Lab 2 (DDP):** At initialization, broadcast model weights from rank 0 to all ranks so everyone starts with identical models

```python
# All ranks start with random weights
model = build_model()

# After broadcast, all identical
if rank == 0:
    for name, param in model.named_parameters():
        dist.broadcast(param.data, src=0)
```

---

## Collective 2: All-Reduce (Most Important!)

### What It Does
**Every rank contributes its data. Every rank gets the same reduced result.**

### Simple Example: Sum
```
Before all_reduce:
  Rank 0: [1, 1, 1, 1]
  Rank 1: [2, 2, 2, 2]
  Rank 2: [3, 3, 3, 3]
  Rank 3: [4, 4, 4, 4]

dist.all_reduce(tensor, op=ReduceOp.SUM)

After all_reduce (all ranks get same result):
  Rank 0: [10, 10, 10, 10]  ← 1+2+3+4
  Rank 1: [10, 10, 10, 10]  ← 1+2+3+4
  Rank 2: [10, 10, 10, 10]  ← 1+2+3+4
  Rank 3: [10, 10, 10, 10]  ← 1+2+3+4
```

### Code Pattern
```python
import torch
import torch.distributed as dist

# Each rank starts with different data
rank = dist.get_rank()
data = torch.tensor([rank + 1] * 4, dtype=torch.float32)

print(f"Before: Rank {rank}: {data}")
# Rank 0: tensor([1., 1., 1., 1.])
# Rank 1: tensor([2., 2., 2., 2.])
# Rank 2: tensor([3., 3., 3., 3.])
# Rank 3: tensor([4., 4., 4., 4.])

# All-reduce with SUM
dist.all_reduce(data, op=dist.ReduceOp.SUM)

print(f"After: Rank {rank}: {data}")
# Rank 0: tensor([10., 10., 10., 10.])  ← sum of all
# Rank 1: tensor([10., 10., 10., 10.])
# Rank 2: tensor([10., 10., 10., 10.])
# Rank 3: tensor([10., 10., 10., 10.])
```

### Byte Accounting
```
4 ranks × 4 values × 4 bytes (float32) = 64 bytes per rank initially
Ring all-reduce (optimal):
  Each rank sends 64 × (4-1)/4 = 48 bytes
  Messages in 2 phases (reduce-scatter + all-gather)
  Total = 48 bytes per rank

Why important: In DDP with 100 ranks, this dominates training time!
```

### Use in Distributed Training (Lab 2 - DDP)
```python
# Each GPU computes gradients on its batch
grads_rank_0 = compute_gradients(batch_0)  # [g0, g1, g2, ...]
grads_rank_1 = compute_gradients(batch_1)  # [g0, g1, g2, ...]
grads_rank_2 = compute_gradients(batch_2)  # [g0, g1, g2, ...]
grads_rank_3 = compute_gradients(batch_3)  # [g0, g1, g2, ...]

# All-reduce: average gradients across ranks
all_grads = grads_rank_0.clone()
dist.all_reduce(all_grads, op=ReduceOp.SUM)
all_grads = all_grads / world_size  # Divide by 4 to average

# Now all ranks have identical averaged gradients
# Update with same gradients → same model on all ranks!
update_model(all_grads)
```

---

## Collective 3: Reduce-Scatter (The Key Optimization!)

### What It Does
**Combine data like all-reduce, but each rank keeps only one piece.**

### Example
```
Before reduce_scatter:
  Rank 0: [600, 604, 608, 612, 616, 620, 624, 628]
  Rank 1: [600, 604, 608, 612, 616, 620, 624, 628]
  Rank 2: [600, 604, 608, 612, 616, 620, 624, 628]
  Rank 3: [600, 604, 608, 612, 616, 620, 624, 628]

dist.reduce_scatter(output, list_of_inputs, op=ReduceOp.SUM)

After reduce_scatter (each rank gets 1/4 of reduced result):
  Rank 0: [600, 604]      ← first quarter
  Rank 1: [608, 612]      ← second quarter
  Rank 2: [616, 620]      ← third quarter
  Rank 3: [624, 628]      ← fourth quarter

Each rank holds 1/4 of the data
Total memory used: 1/4 of all-reduce result
```

### Code Pattern
```python
import torch
import torch.distributed as dist

rank = dist.get_rank()
world = dist.get_world_size()

# 8 values total, will be split into 4 pieces
tensor = torch.arange(8, dtype=torch.float32) + rank * 100

print(f"Before: Rank {rank}: {tensor}")
# Rank 0: tensor([0., 1., 2., 3., 4., 5., 6., 7.])
# Rank 1: tensor([100., 101., 102., 103., 104., 105., 106., 107.])
# Rank 2: tensor([200., 201., 202., 203., 204., 205., 206., 207.])
# Rank 3: tensor([300., 301., 302., 303., 304., 305., 306., 307.])

# Create output buffer (each rank gets 8/4 = 2 elements)
output = torch.zeros(2, dtype=torch.float32)

# Reduce-scatter: sum all tensors, split result
dist.reduce_scatter(output, list(torch.chunk(tensor, world)), op=dist.ReduceOp.SUM)

print(f"After: Rank {rank}: {output}")
# Rank 0: tensor([600., 604.])    ← sum of element 0-1 across all ranks
# Rank 1: tensor([608., 612.])    ← sum of element 2-3 across all ranks
# Rank 2: tensor([616., 620.])    ← sum of element 4-5 across all ranks
# Rank 3: tensor([624., 628.])    ← sum of element 6-7 across all ranks
```

### Byte Accounting
```
Full all-reduce (all ranks get all data):
  64 bytes → 64 bytes on each rank

Reduce-scatter (each rank gets 1/4):
  64 bytes → 16 bytes on each rank (4x memory savings!)

Bytes on network: Same as all-reduce (48 bytes/rank)
  But you only *keep* 16 bytes locally
```

### Use in Distributed Training (Lab 5 - ZeRO Stage 2)
```python
# Gradients computed locally on each GPU
my_gradients = compute_gradients(my_batch)  # 100M params on this GPU

# Reduce-scatter: reduce across GPUs, but keep only my slice
reduced_my_gradients = reduce_scatter(all_gradients, rank, world)

# Now gradient buffer on this GPU is 1/world_size smaller!
# Memory saved = (world_size - 1) / world_size

# Update only the parameters I own
update_my_parameters(reduced_my_gradients)
```

---

## Collective 4: All-Gather (Reconstruct from Pieces)

### What It Does
**Inverse of reduce-scatter. Each rank contributes its piece; all ranks get full tensor.**

### Example
```
Before all_gather:
  Rank 0: [600, 604]
  Rank 1: [608, 612]
  Rank 2: [616, 620]
  Rank 3: [624, 628]

dist.all_gather(list of output buffers, my_tensor)

After all_gather (all ranks get full tensor):
  Rank 0: [600, 604, 608, 612, 616, 620, 624, 628]
  Rank 1: [600, 604, 608, 612, 616, 620, 624, 628]
  Rank 2: [600, 604, 608, 612, 616, 620, 624, 628]
  Rank 3: [600, 604, 608, 612, 616, 620, 624, 628]
```

### Code Pattern
```python
import torch
import torch.distributed as dist

rank = dist.get_rank()
world = dist.get_world_size()

# Each rank has 2 elements
my_data = torch.tensor([600.0 + rank*8, 604.0 + rank*8])

print(f"Before: Rank {rank}: {my_data}")
# Rank 0: tensor([600., 604.])
# Rank 1: tensor([608., 612.])
# Rank 2: tensor([616., 620.])
# Rank 3: tensor([624., 628.])

# Create output buffer (8 elements = 2 × 4 ranks)
output_list = [torch.zeros(2) for _ in range(world)]

# All-gather
dist.all_gather(output_list, my_data)

full_tensor = torch.cat(output_list)

print(f"After: Rank {rank}: {full_tensor}")
# Rank 0: tensor([600., 604., 608., 612., 616., 620., 624., 628.])
# Rank 1: tensor([600., 604., 608., 612., 616., 620., 624., 628.])
# Rank 2: tensor([600., 604., 608., 612., 616., 620., 624., 628.])
# Rank 3: tensor([600., 604., 608., 612., 616., 620., 624., 628.])
```

### Byte Accounting
```
Reduce-scatter scattered data:
  Each rank held 16 bytes

All-gather reconstructs:
  Each rank needs 64 bytes → communicates 48 bytes
  But ends up with full 64-byte tensor
```

### Use in Distributed Training
```python
# Lab 5 (ZeRO Stage 2): Forward pass
# We only have my gradient slice
my_gradients = [2, 3]  # Just my slice

# All-gather: reconstruct full gradient for parameter update
full_gradients = all_gather(my_gradients, rank, world)
# [0, 1, 2, 3, 4, 5, 6, 7]

# Lab 3 (Tensor Parallel): Forward pass
# Weight matrix is split across GPUs
my_weight_chunk = layer.weight[my_slice, :]  # (512/4, 512)

# All-gather: reconstruct full weight matrix
full_weight = all_gather(my_weight_chunk, rank, world)  # (512, 512)

# Compute forward pass with full weight
logits = input @ full_weight.T
```

---

## The Identity That Makes Everything Work

### Reduce-Scatter + All-Gather = All-Reduce

```python
# All-reduce: reduce to all ranks
tensor_before = [1, 2, 3, 4]
dist.all_reduce(tensor_before, op=SUM)
tensor_after_allreduce = [10, 10, 10, 10]

# Reduce-scatter + all-gather: same result, different path
tensor_before = [1, 2, 3, 4]
output = reduce_scatter(tensor_before)  # Each rank gets piece: [10]
output_full = all_gather(output)        # Each rank gets all: [10, 10, 10, 10]
```

### Why This Matters
```
Ring all-reduce cost:
  Phase 1 (reduce-scatter):  (N-1)/N × S bytes = 3/4 × S
  Phase 2 (all-gather):      (N-1)/N × S bytes = 3/4 × S
  Total:                     2(N-1)/N × S bytes = 3/2 × S

ZeRO Stage 2 (reduce-scatter only):
  Phase 1 (reduce-scatter):  (N-1)/N × S bytes = 3/4 × S
  Total:                     (N-1)/N × S bytes = 3/4 × S

Same bytes on network, but:
- All-reduce: all ranks keep full S bytes in memory
- ZeRO Stage 2: each rank keeps S/N bytes in memory (N times less!)
```

---

## Collective 5: All-to-All (Permutation)

### What It Does
**Each rank sends different data to each other rank.**

### Example
```
Before all_to_all:
  Rank 0: [0, 1, 2, 3]     (4 elements, split into 4 pieces)
  Rank 1: [10, 11, 12, 13]
  Rank 2: [20, 21, 22, 23]
  Rank 3: [30, 31, 32, 33]

  Rank 0 sends: piece 0 to rank 0, piece 1 to rank 1, piece 2 to rank 2, piece 3 to rank 3
  Rank 1 sends: piece 0 to rank 0, piece 1 to rank 1, piece 2 to rank 2, piece 3 to rank 3
  Rank 2 sends: piece 0 to rank 0, piece 1 to rank 1, piece 2 to rank 2, piece 3 to rank 3
  Rank 3 sends: piece 0 to rank 0, piece 1 to rank 1, piece 2 to rank 2, piece 3 to rank 3

After all_to_all:
  Rank 0: [0, 10, 20, 30]   (piece 0 from each rank)
  Rank 1: [1, 11, 21, 31]   (piece 1 from each rank)
  Rank 2: [2, 12, 22, 32]   (piece 2 from each rank)
  Rank 3: [3, 13, 23, 33]   (piece 3 from each rank)
```

### Code Pattern
```python
import torch
import torch.distributed as dist

rank = dist.get_rank()
world = dist.get_world_size()

# Each rank has 4 elements, split into 4 pieces (1 per rank)
send_data = torch.tensor([rank*10 + i for i in range(4)], dtype=torch.float32)

print(f"Before: Rank {rank}: {send_data}")
# Rank 0: tensor([0., 1., 2., 3.])
# Rank 1: tensor([10., 11., 12., 13.])
# Rank 2: tensor([20., 21., 22., 23.])
# Rank 3: tensor([30., 31., 32., 33.])

# All-to-all
recv_data = torch.zeros_like(send_data)
dist.all_to_all_single(recv_data, send_data)

print(f"After: Rank {rank}: {recv_data}")
# Rank 0: tensor([0., 10., 20., 30.])
# Rank 1: tensor([1., 11., 21., 31.])
# Rank 2: tensor([2., 12., 22., 32.])
# Rank 3: tensor([3., 13., 23., 33.])
```

### Use in Distributed Training
- **Mixture of Experts:** Route tokens to different expert networks
- Rank 0's tokens go to expert 0 on all ranks, then all-gather results

---

## Collective 6: Send/Recv (Direct Point-to-Point)

### What It Does
**One rank sends data to another rank.**

### Example (Deadlock Trap!)
```python
# ❌ WRONG - Will deadlock:
if rank == 0:
    send(large_payload, dst=1)        # Rank 0 waits for rank 1's large_payload to fit
    recv(payload, src=3)              # Never gets here (rank 1 is also waiting!)
else if rank == 1:
    send(large_payload, dst=2)        # Waiting for buffer space
    recv(payload, src=0)              # Never gets here

# ✓ CORRECT - Even/odd ordering prevents deadlock:
if rank % 2 == 0:
    send(payload, dst=next_rank)      # Even ranks send first
    recv(payload, src=prev_rank)      # Then receive
else:
    recv(payload, src=prev_rank)      # Odd ranks receive first
    send(payload, dst=next_rank)      # Then send
```

### Code Pattern
```python
import torch
import torch.distributed as dist

rank = dist.get_rank()

my_data = torch.tensor([rank * 100 + i for i in range(10)], dtype=torch.float32)

# Send to next rank, receive from previous rank (ring topology)
next_rank = (rank + 1) % 4
prev_rank = (rank - 1) % 4

if rank % 2 == 0:
    # Even ranks: send first, then receive
    dist.send(my_data, dst=next_rank)
    received = torch.zeros_like(my_data)
    dist.recv(received, src=prev_rank)
else:
    # Odd ranks: receive first, then send
    received = torch.zeros_like(my_data)
    dist.recv(received, src=prev_rank)
    dist.send(my_data, dst=next_rank)

print(f"Rank {rank} sent {my_data[0]:.0f}, received {received[0]:.0f}")
# Rank 0 sent 0, received 300
# Rank 1 sent 100, received 0
# Rank 2 sent 200, received 100
# Rank 3 sent 300, received 200
```

### Use in Distributed Training
- **Lab 4 (Pipeline Parallel):** Each stage sends partial outputs to next stage
  ```
  Stage 0 (GPU 0):  forward(layers 0-1) → send to GPU 1
  Stage 1 (GPU 1):  recv from GPU 0, forward(layers 2-3) → send to GPU 2
  Stage 2 (GPU 2):  recv from GPU 1, forward(layers 4-5) → send to GPU 3
  Stage 3 (GPU 3):  recv from GPU 2, forward(layers 6-7)
  ```

---

## Complete Lab 1 Output

```
Testing broadcast...
[broadcast] Rank 0 sent [100, 100], all ranks received:
  Rank 0: [100, 100]
  Rank 1: [100, 100]
  Rank 2: [100, 100]
  Rank 3: [100, 100]
✓ PASS

Testing all_reduce (SUM)...
Before:
  Rank 0: [1, 1, 1, 1]
  Rank 1: [2, 2, 2, 2]
  Rank 2: [3, 3, 3, 3]
  Rank 3: [4, 4, 4, 4]
After:
  All ranks: [10, 10, 10, 10]
✓ PASS

Testing reduce_scatter...
Before: Each rank has [8 elements]
After: Each rank has [2 elements] (8/4 = 2 per rank)
  Rank 0: [600.0, 604.0]
  Rank 1: [608.0, 612.0]
  Rank 2: [616.0, 620.0]
  Rank 3: [624.0, 628.0]
✓ PASS

Testing all_gather...
Before: Each rank has [2 elements]
After: Each rank has [8 elements]
  All ranks: [600.0, 604.0, 608.0, 612.0, 616.0, 620.0, 624.0, 628.0]
✓ PASS

Testing all_reduce vs reduce_scatter + all_gather identity...
all_reduce      : [600.0, 604.0, 608.0, 612.0, 616.0, 620.0, 624.0, 628.0]
RS then AG      : [600.0, 604.0, 608.0, 612.0, 616.0, 620.0, 624.0, 628.0]
identical       : True
✓ PASS (Oracle principle proven!)

Testing all_to_all...
Before:
  Rank 0: [0, 1, 2, 3]
  Rank 1: [10, 11, 12, 13]
  Rank 2: [20, 21, 22, 23]
  Rank 3: [30, 31, 32, 33]
After:
  Rank 0: [0, 10, 20, 30]
  Rank 1: [1, 11, 21, 31]
  Rank 2: [2, 12, 22, 32]
  Rank 3: [3, 13, 23, 33]
✓ PASS

Testing send/recv with deadlock prevention...
Ring topology (even/odd ordering):
  Rank 0: sent [0, 1, ...], received [300, 301, ...]
  Rank 1: sent [100, 101, ...], received [0, 1, ...]
  Rank 2: sent [200, 201, ...], received [100, 101, ...]
  Rank 3: sent [300, 301, ...], received [200, 201, ...]
✓ PASS (No deadlock!)

Network topology (NCCL_DEBUG=INFO):
  NCCL INFO Channel 00/0: 0[0] -> 1[1] via P2P/CUMEM
  NCCL INFO Channel 00/0: 0[0] -> 2[2] via P2P/CUMEM
  NCCL INFO Channel 00/0: 0[0] -> 3[3] via P2P/CUMEM
  ✓ All GPUs connected via NVLink (P2P/CUMEM)
```

---

## Byte Accounting (Key Insight)

```
Scenario: 4 ranks, 256 MB tensor

Ring all-reduce:
  Phase 1 (reduce-scatter):  Phase 2 (all-gather):
  Rank 0 → Rank 1: 192 MB    Rank 0 → Rank 1: 192 MB
  Rank 1 → Rank 2: 192 MB    Rank 1 → Rank 2: 192 MB
  Rank 2 → Rank 3: 192 MB    Rank 2 → Rank 3: 192 MB
  Rank 3 → Rank 0: 192 MB    Rank 3 → Rank 0: 192 MB
  ─────────────────────      ─────────────────────
  Total: 768 MB = 3 × 256 MB (optimal for all-reduce!)

DDP gradient sync (all_reduce):
  Rank 0: [grad_0, grad_1, grad_2, grad_3]  (256 MB)
  Rank 1: [grad_0, grad_1, grad_2, grad_3]  (256 MB)
  Rank 2: [grad_0, grad_1, grad_2, grad_3]  (256 MB)
  Rank 3: [grad_0, grad_1, grad_2, grad_3]  (256 MB)
  
  After all_reduce, all have [sum_0, sum_1, sum_2, sum_3]
  Cost: 768 MB on wire (3 × 256 MB)

ZeRO Stage 2 (reduce_scatter only):
  Before:
    Rank 0: [grad_0, grad_1, grad_2, grad_3]  (256 MB)
    Rank 1: [grad_0, grad_1, grad_2, grad_3]  (256 MB)
    Rank 2: [grad_0, grad_1, grad_2, grad_3]  (256 MB)
    Rank 3: [grad_0, grad_1, grad_2, grad_3]  (256 MB)
  
  After reduce_scatter:
    Rank 0: [sum_0]  (64 MB)   ← Only my piece!
    Rank 1: [sum_1]  (64 MB)
    Rank 2: [sum_2]  (64 MB)
    Rank 3: [sum_3]  (64 MB)
  
  Cost: 384 MB on wire (same 3/4 factor as RS phase)
  Memory savings: 4x less gradient buffer!
```

---

## Why Lab 1 Matters

1. **Broadcast** → Init models identically
2. **All-reduce** → Sync gradients (DDP)
3. **Reduce-scatter** → Partition gradients (ZeRO)
4. **All-gather** → Reconstruct data when needed
5. **All-to-all** → Route data (MoE)
6. **Send/recv** → Stage-to-stage (Pipeline)

Every lab from 2-7 combines these in different ways. Understanding what each collective does (and its cost in bytes) is fundamental to understanding distributed training.

---

## Common Mistakes

| Mistake | Result | Lab 1 Lesson |
|---------|--------|-------------|
| Forgetting collective | Shape mismatch | Every collective must be explicitly called |
| Wrong reduce op | Different numbers | SUM vs AVG vs MAX changes result |
| Deadlock in send/recv | Hangs forever | Even/odd ordering is essential |
| Assuming all-reduce is "free" | Bottleneck | 3× tensor size on network |
| Not using reduce-scatter in ZeRO | Wrong memory savings | RS without AG = memory optimization |

---

## Exercises (From Lab 1)

1. **Predict all_to_all output:**
   - Rank r sends `[r*10, r*10+1, r*10+2, r*10+3]`
   - What does Rank 2 receive?
   - **Answer:** `[2, 12, 22, 32]`

2. **Swap SUM for AVG:**
   - In all_reduce, use `ReduceOp.AVG` instead
   - What changes in Lab 2's gradient synchronization?
   - **Answer:** All-reduce result is divided by 4, so gradient update is 4x smaller (must adjust LR)

3. **Induce deadlock on purpose:**
   - Make all ranks call send() with 100MB first
   - Set `TORCH_NCCL_BLOCKING_WAIT=1` to see better error
   - Learn: NCCL has limited send buffer, ordering matters

4. **Time all_reduce vs reduce_scatter:**
   - all_reduce(100MB): ~2.1ms
   - reduce_scatter(100MB): ~1.1ms
   - Ratio: approximately 2:1 ✓ (reduce_scatter is half)

---

## Byte Accounting Summary

| Operation | Bytes/Rank | Per Network Message |
|-----------|-----------|-------------------|
| Broadcast (256 MB) | 256 MB in | 256 MB out |
| All-reduce (256 MB) | 256 MB in/out | 768 MB total (3× message) |
| Reduce-scatter (256 MB) | 64 MB out | 384 MB total |
| All-gather (64 MB → 256 MB) | 64 MB in | 384 MB total |
| All-to-all (256 MB) | 256 MB in/out | 1024 MB (4× message) |

**Key:** Bytes on the network ≠ bytes in memory. More on network but less in memory = win!
