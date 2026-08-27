# Results

Everything measured on `lambda-gpu`: 4× NVIDIA GB300, 284208 MiB each, aarch64
Grace host, driver 580.173.02, `torch 2.13.0+cu130`, NCCL 2.29.7.

Model: 25.4M params, 8 blocks, `d_model` 512, 8 heads, 256 context. Loss goes
4.403 → 2.516 over 100 steps at global batch 32, fp32.

## Verification

Every fp32 implementation reproduces the single-GPU oracle:

| Implementation | max abs deviation from oracle |
|---|---|
| Hand-written DDP | 4.768e-07 |
| torch DDP | 9.537e-07 |
| Hand-written TP 4 | 4.768e-07 |
| Pipeline, 1 micro-batch | 4.768e-07 |
| ZeRO-1 | 4.768e-07 |
| ZeRO-2 | 4.768e-07 |
| ZeRO-3 | 9.537e-07 |
| 2D mesh TP2 × DP2 | 4.768e-07 |
| FSDP2 fp32 | 9.537e-07 |
| DTensor TP | 4.768e-07 |
| **FSDP2 bf16** | **5.438e-03** |

Sharding is a change of layout, never a change of mathematics. Only bf16
breaks that, and it breaks it by four orders of magnitude.

### Determinism

A fresh oracle regenerated in a separate process is **bit-identical** to the
archived one: loss curve, every weight, and probe logits all at exactly `0.0`
deviation. The whole oracle discipline rests on this and it is worth verifying
rather than assuming.

### NCCL topology

All four GB300s connect via `P2P/CUMEM` across 32 channels, with direct
rank-to-rank paths alongside the ring. No `SHM`, no `NET` — full NVLink peer
access.

## Timing

All timings discard the first 10 steps. See "the measurement bug" below for why
that is not optional.

| Implementation | 90 timed steps | vs 1 GPU |
|---|---|---|
| **Single GPU** | **3.24s** | 1.00× |
| torch DDP | 3.15s | 1.03× |
| Hand-written TP 4 | 4.25s | 0.76× |
| Hand-written DDP | 4.54s | 0.71× |
| FSDP2 fp32 | 5.6s | 0.58× |
| 2D mesh TP2 × DP2 | 5.95s | 0.54× |
| Hand-written ZeRO-3 | 15.0s | 0.22× |
| DTensor TP | 15.4s | 0.21× |

**Four GB300s buy essentially nothing at this scale.** Only torch DDP beats one
GPU, and only by 3%.

This is the single most useful performance result here. At 25M parameters with
a global batch of 32, per-rank compute is far too small to hide collective
latency. These labs demonstrate correctness; they cannot demonstrate speedup.
Scaling the model up would fix that, at the cost of the fast iteration that
makes the labs teachable.

Two comparisons that *are* meaningful, because they are like-for-like:

- torch DDP is **1.44×** the hand-written version — the value of gradient
  bucketing and communication/compute overlap.
- FSDP2 is **2.7×** the hand-written ZeRO-3 — the value of prefetching,
  flat buffers, and overlap.

And one that went the other way: DTensor TP is **3.6× slower** than
hand-written TP. Per-op dispatch overhead plus the forced `foreach=False`
optimizer path dominate at this size. "Use the library version" is a default,
not a law.

## Memory (Lab 5, persistent training state per rank)

| Stage | params | grads | optim | total | vs DDP | theory |
|---|---|---|---|---|---|---|
| DDP | 96.96 | 96.96 | 193.92 | 387.84 MiB | 1.00× | 1.00× |
| ZeRO-1 | 96.96 | 96.96 | 49.00 | 242.92 MiB | 1.60× | 1.60× |
| ZeRO-2 | 96.96 | 24.50 | 49.00 | 170.46 MiB | 2.28× | 2.29× |
| ZeRO-3 | 24.81 | 24.81 | 49.62 | 99.23 MiB | 3.91× | 4.00× |

Within 2% of theory at every stage.

**Reshard-after-forward:** peak 285.2 MiB with it on, 911.6 MiB with it off.
Without recomputation the gathered weights stay pinned by the autograd graph and
you save nothing — resharding and recomputation are one decision.

## Pipeline bubble (Lab 4)

| micro-batches | ms/step | NCCL share of device time | formula predicts busy |
|---|---|---|---|
| 1 | 39.4 | 53–57% | 25% |
| 2 | 33.0 | — | 40% |
| 4 | 41.2 | — | 57% |
| 8 | 67.2 | 50–60% | 73% |

**Micro-batching does not shrink the bubble here.** Only micro=2 helps at all,
and micro=8 is 70% worse than no micro-batching. The absolute NCCL time roughly
doubles from micro=1 to micro=8.

The textbook formula assumes per-stage compute dwarfs per-transfer latency. At
micro=8 each transfer carries 4 sequences, so eight small transfers cost more
than one large one. The formula is an upper bound you approach only when each
stage has real work.

## Bugs found during development

None of these raised an exception on their own. Every one was caught by the
oracle disagreeing — which is the entire argument for having an oracle.

### 1. Gradient clipping under tensor parallelism
`torch.nn.utils.clip_grad_norm_` norms over `model.parameters()`, which on each
rank is only that rank's shards. Norm too small → under-clipped → larger steps
than the oracle. **7.1e-02 off.** Forward and gradients were both exact to
1e-06, so only the third verification stage caught it. Under-clipping makes the
loss fall *faster*, so the bug makes your training curve look better.

Recurs in three variants:

| Layout | Correct rule |
|---|---|
| TP | sum sharded across TP; count replicated exactly once |
| PP | plain SUM across stages — parameters are disjoint |
| TP×DP | sum sharded across TP only; DP is already averaged |

### 2. `opt.zero_grad()` with a subset-owning optimizer
Under ZeRO the optimizer holds only this rank's parameters, so the rest are
never cleared. In ZeRO-1 they accumulate for 80 steps and reach **NaN**. Use
`model.zero_grad()`.

### 3. Round-robin parameter ownership
Balances tensor *count*, not bytes. With sizes spanning 512 to 1,048,576
elements the heaviest rank got **1.99×** its fair share. Greedy largest-first
brings it to 1.01×. Production shards one flat buffer instead.

### 4. A deadlock from guarding a collective
`if rank in (0, 2): describe_shards(...)` hung two ranks for the full ten-minute
store timeout, because `describe_shards` contains a barrier. Filter the output,
never the participation. This is the exact failure the debugging notes warn
about, and it still happened.

### 5. Partially-DTensor models break every foreach kernel
`parallelize_module` converts only the named modules. Both `clip_grad_norm_` and
AdamW's fused path then raise `got mixed torch.Tensor and DTensor`. Needs a
hand-written norm and `foreach=False`. This is why `fully_shard` converts the
whole model.

### 6. The measurement bug — worse than the rest
Found only by re-running the oracle from a clean slate and watching it drop from
23.7s to 4.9s. Timing from step 0 charges one-off JIT compilation, NCCL
communicator setup, cuBLAS workspace allocation and allocator growth to whatever
runs first, systematically flattering whatever runs second.

Every performance claim in the first pass was affected. The Lab 4 conclusion was
not merely imprecise but **qualitatively wrong**: the reported "utilisation
climbs 17% to 58% as predicted" was almost entirely warmup, and the corrected
data shows micro-batching barely helps and the bubble does not shrink at all.

Compounding it, the original micro-batch sweep timed a 100-step run against
30-step runs and reported raw wall clock, making micro-batching appear to cut
time from 3.5s to 0.7s. Normalising to ms/step reverses the conclusion.

**The lesson:** correctness had an oracle guarding it and survived intact.
Performance had nothing guarding it, and that is exactly where every mistake
lived. If you take one thing from this exercise, take that.

## Error magnitudes, for calibration

| Source | Magnitude |
|---|---|
| Correct implementation, fp32 | 1e-06 |
| bf16 numerical noise | 5.4e-03 |
| Missing `f` operator | 1.8e-02 |
| TP local-norm clipping | 7.1e-02 |
| DDP sum instead of mean | 1.0e-01 |
| Row-parallel bias before reduce | 4.5e-01 (logits) |
| `n_head` not divided, silent variant | 2.0e+00 (logits) |

Note how close bf16 noise sits to real bugs. A tolerance loose enough to accept
bf16 is within a factor of 3 of the missing-`f` bug. Verify in fp32.

## Corrections to the original plan

- Model is 25.4M params, not the ~50M estimated.
- The `n_head` bug is only silent when `head_dim` is derived from the local
  width. With a fixed `head_dim` it raises immediately.
- The NGC container route was unavailable (no Docker); `uv` plus a venv was used.
- `lm_head` is untied from `wte`, because tying complicates tensor-parallel
  sharding without teaching anything relevant.
- Production does not always win: FSDP2 beat the hand-written ZeRO-3, but
  DTensor TP lost badly to hand-written TP.
- Micro-batching does not reclaim the pipeline bubble at this scale.
