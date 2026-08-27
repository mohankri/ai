# Lab 6 — Composing into a 2D mesh

**File:** `lab6_2d.py`
**Run:** `.venv/bin/torchrun --standalone --nproc_per_node=4 lab6_2d.py`
**Time:** ~1 minute

Reuses Lab 3's parallel layers **completely unchanged** and adds a second axis.

```python
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
```

Row-major, so `rank = dp_index * 2 + tp_index`:

```
                tp=0    tp=1
      dp=0    rank 0  rank 1     <- these two form a TP group
      dp=1    rank 2  rank 3     <- and these two

              ranks 0,2 form a DP group;  ranks 1,3 form the other
```

Confirmed at runtime:

```
[rank 0] mesh coords: dp=0 tp=0   tp peers [0, 1]   dp peers [0, 2]
[rank 1] mesh coords: dp=0 tp=1   tp peers [0, 1]   dp peers [1, 3]
[rank 2] mesh coords: dp=1 tp=0   tp peers [2, 3]   dp peers [0, 2]
[rank 3] mesh coords: dp=1 tp=1   tp peers [2, 3]   dp peers [1, 3]
```

## The point of the lab

The two axes are genuinely independent. Lab 3's `ColumnParallelLinear` and
`RowParallelLinear` need **no modification whatsoever** to work inside a
data-parallel replica — they simply operate on a smaller process group.

- Activations all-reduce across `tp`, inside the layers, via `f` and `g`
- Gradients all-reduce across `dp`, after backward, in the training loop
- Neither collective knows the other exists

That orthogonality is the whole idea behind 3D and 4D parallelism.

## What the layout looks like

TP is now 2 instead of 4, so shards are twice as large:

```
blocks.0.attn.q_proj.weight   local=(256, 512)     global=(512, 512)     <-- SHARDED
blocks.0.mlp.up.weight        local=(1024, 512)    global=(2048, 512)    <-- SHARDED
blocks.0.mlp.down.weight      local=(512, 1024)    global=(512, 2048)    <-- SHARDED
local params: 12,820,480
```

Compare with Lab 3's 6,521,856 — exactly double, because TP 2 instead of TP 4.
Ranks 0 and 2 hold *identical* shards; they are DP replicas of each other.

## Two things that reliably trip people up

### 1. The global batch is `micro × dp_size`, not `micro × world_size`

```python
x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device,
                 rank=dp_rank, world=dp_size)     # dp only!
```

Both ranks of a TP group must see **exactly the same tokens** — they are
cooperating on one forward pass, not processing different data. Splitting the
batch four ways here would silently quarter your effective batch size.

### 2. The gradient norm must be correct along both axes at once

```python
def clip_2d_(model, max_norm, tp_group):
    for p in model.parameters():
        sq = p.grad.detach().pow(2).sum()
        if getattr(p, "_tp_sharded", False):
            sharded_sq += sq
        else:
            repl_sq += sq
    dist.all_reduce(sharded_sq, op=dist.ReduceOp.SUM, group=tp_group)
    total = torch.sqrt(sharded_sq + repl_sq)
```

**Only the TP axis needs reducing.** Gradients were already averaged across DP
before this runs, so every rank in a DP group holds identical values — reducing
over DP as well would inflate the norm by `dp_size`.

This is the third variant of the same bug from Lab 3, and each parallelism
layout needs its own reduction rule:

| Layout | Rule |
|---|---|
| TP (Lab 3) | sum sharded across TP; count replicated once |
| PP (Lab 4) | plain SUM across stages — parameters are disjoint |
| TP×DP (Lab 6) | sum sharded across TP only; DP already averaged |

## Expected output

```
  [PASS] 2D mesh (TP2 x DP2): max |loss - oracle| = 4.768e-07

steady state 5.95s over 90 timed steps  (single GPU: 3.24s)
[rank 0] TP2 x DP2   peak=  1051.74 MiB  current=   163.81 MiB
```

## Why a mesh at all

TP 2 × DP 2 costs about half the per-rank parameter memory of pure DP, and half
the TP communication of Lab 3's TP 4, while consuming the same global batch as
both.

That trade — less chatter per collective, more replicas — is why real training
runs use a mesh rather than a single axis. At scale the rule is:

```
within a node (NVLink)  ->  TP  (chatty: all-reduce per block per token)
across nodes (IB)       ->  PP  (cheap: activations at stage boundaries)
outermost               ->  DP / FSDP
```

Total GPUs = `TP × PP × DP`.

## The deadlock this lab caused during development

The original code was:

```python
if rank in (0, 2):            # one representative per DP replica
    describe_shards(probe, full_shapes)
dist.barrier()
```

This hung ranks 1 and 3 for the full ten-minute store timeout. `describe_shards`
calls `rank_print`, which contains a `dist.barrier()` — so ranks 1 and 3 never
arrived at barriers that ranks 0 and 2 were waiting on.

The fix keeps every rank participating and filters only the printing:

```python
describe_shards(probe, full_shapes, only={0, 2})
```

**Filter the output, never the participation.** This is exactly the failure mode
the README warns about, and it still happened.

## Exercises

1. Build TP 4 × DP 1 and TP 1 × DP 4 and compare all three against TP 2 × DP 2
   for time and per-rank memory. Which shape wins, and why?
2. Add a third axis with `init_device_mesh("cuda", (2, 1, 2),
   mesh_dim_names=("dp", "pp", "tp"))` and wire Lab 4's stages into it.
3. Deliberately pass `rank=rank, world=world` to `get_batch` instead of the DP
   coordinates. Watch the loss diverge from the oracle and work out why from
   first principles before reading the answer above.
