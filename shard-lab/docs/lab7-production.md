# Lab 7 — Compare against production

**File:** `lab7_production.py`
**Run:** `.venv/bin/torchrun --standalone --nproc_per_node=4 lab7_production.py`
**Time:** ~3 minutes

Now that you have written ZeRO-3 and tensor parallelism by hand, meet the
versions PyTorch ships.

| Yours | Theirs |
|---|---|
| Lab 5 `ShardedBlock` | `fully_shard` (FSDP2) |
| Lab 3 `f` / `g` operators | `parallelize_module` (DTensor) |

## FSDP2 — the production form of Lab 5

```python
for blk in model.blocks:
    fully_shard(blk, mp_policy=mp)      # shard each block FIRST
fully_shard(model, mp_policy=mp)        # then the root
```

**The ordering matters.** Sharding each block is what lets FSDP free a block's
gathered weights the moment it is done with it. Sharding only the root would
gather the entire model at once — the same trap you saw in Lab 5's
`reshard_after_forward=False` experiment.

```
[PASS] FSDP2 fp32: max |loss - oracle| = 9.537e-07
wall 5.6s   peak 837.1 MiB
```

Against your hand-written ZeRO-3 at **15.0s**, so production genuinely wins
here — nearly 3×. The gap is the list of optimisations you skipped:

- the all-gather for block *n+1* is issued while block *n* is still computing
- gradients are reduce-scattered in flat buckets, not per tensor
- shards live in one contiguous buffer instead of 100+ separate tensors

Note also that `torch.nn.utils.clip_grad_norm_` works unmodified here. The
parameters are DTensors, so it dispatches to a mesh-aware implementation and
computes the true global norm. That is exactly the bug you had to fix by hand in
Lab 3 — DTensor knows a shard is a shard, so it cannot be got wrong.

## DTensor TP — the production form of Lab 3

```python
plan = {
    "attn.q_proj": ColwiseParallel(),
    "attn.k_proj": ColwiseParallel(),
    "attn.v_proj": ColwiseParallel(),
    "attn.o_proj": RowwiseParallel(input_layouts=Shard(-1)),
    "mlp.up":      ColwiseParallel(),
    "mlp.down":    RowwiseParallel(input_layouts=Shard(-1)),
}
for blk in model.blocks:
    parallelize_module(blk, mesh, plan)
    blk.attn.n_head //= tp_size          # still your responsibility
```

**What disappeared:** `_CopyToTP` and `_ReduceFromTP`. Declaring a weight
`Shard(0)` or `Shard(1)` is enough — the dispatcher derives which collective
each operation needs, in both directions. The `f`/`g` operators still exist,
they are just generated rather than written.

**What did not disappear:**

1. You still divide `n_head` yourself.
2. You still compute the global gradient norm yourself.
3. You must pass `foreach=False` to the optimizer.

Items 2 and 3 share a cause worth understanding.

## Partially-converted models break every foreach kernel

`parallelize_module` converts only the modules named in the plan. Embeddings,
LayerNorms and `lm_head` remain ordinary tensors. Every batched kernel then
refuses to span both worlds:

```
aten._foreach_norm.Scalar got mixed torch.Tensor and DTensor
aten._foreach_mul_.Scalar got mixed torch.Tensor and DTensor
```

The first comes from `clip_grad_norm_`, the second from AdamW's fused path. Both
required workarounds — a hand-written norm (`clip_mixed_`) and
`foreach=False`.

This is why `fully_shard` converts the *whole* model rather than part of it,
and it is why FSDP2 above needed neither workaround.

Note the improvement over Lab 3 though: these are **loud exceptions** on step
one, not a silent wrong answer discovered 26 steps into a loss curve.

## DTensor is 3.6× slower here — and that surprised me

```
FSDP2 fp32     5.6s      837.1 MiB   matches oracle: yes
DTensor TP    15.4s     1283.9 MiB   matches oracle: yes
```

Against the hand-written TP at **4.25s**. The plan assumed production always
wins; for a model this small, DTensor's per-operation dispatch overhead plus the
forced `foreach=False` optimizer path dominate completely.

That gap should invert at realistic scale — but it was not measured here, so do
not assume it. This is a good reminder that "use the library version" is a
default, not a law.

## bf16, and why every previous lab ran in fp32

```
[FAIL] FSDP2 bf16 @ atol 2e-04: max |loss - oracle| = 5.438e-03
[PASS] FSDP2 bf16 @ atol 1e-02: max |loss - oracle| = 5.438e-03
[PASS] FSDP2 bf16 @ atol 1e-01: max |loss - oracle| = 5.438e-03
```

bf16 has 8 mantissa bits against fp32's 23. Agreement with the oracle collapses
from ~1e-07 to **5.4e-03** — four orders of magnitude.

That is fine for training and fatal for verification. Line it up against the
real bugs found in these labs:

| Error source | Magnitude |
|---|---|
| correct implementation, fp32 | 1e-06 |
| **bf16 numerical noise** | **5.4e-03** |
| missing `f` (Lab 3) | 1.8e-02 |
| TP local-norm clipping (Lab 3) | 7.1e-02 |
| DDP sum-not-mean (Lab 2) | 1.0e-01 |

A tolerance loose enough to accept bf16 is within a factor of 3 of the missing
`f` bug. Establish correctness in fp32, then turn on mixed precision for speed.

The memory saving is real though: 554.7 MiB against 837.1 MiB for fp32.

## Expected summary

```
  implementation        wall    peak MiB   matches oracle
  FSDP2 fp32             5.6s      837.1   yes
  DTensor TP            15.4s     1283.9   yes
```

## Exercises

1. Convert the *whole* model to DTensor — wrap embeddings, norms and `lm_head`
   with `distribute_tensor(..., [Replicate()])` — and confirm both
   `clip_grad_norm_` and the default foreach optimizer then work unmodified.
2. Compose FSDP2 with DTensor TP on a 2D mesh: `fully_shard` over `dp` on top of
   `parallelize_module` over `tp`. This is the production form of Lab 6.
3. Profile FSDP2 and confirm the all-gather for block *n+1* really does overlap
   block *n*'s compute. Compare that timeline against your Lab 5 ZeRO-3.
4. Scale `d_model` to 2048 and `n_layer` to 24, then re-measure DTensor TP
   against hand-written TP. Find the size where the library implementation
   starts winning.
