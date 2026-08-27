# Lab 5 — ZeRO by hand

**File:** `lab5_zero.py`
**Run:** `.venv/bin/torchrun --standalone --nproc_per_node=4 lab5_zero.py`
**Time:** ~4 minutes

DDP replicates everything. With AdamW in fp32 the cost per parameter is about
16 bytes:

| Component | Bytes |
|---|---|
| parameter | 4 |
| gradient | 4 |
| Adam `m` | 4 |
| Adam `v` | 4 |

ZeRO removes the replication one tier at a time:

| Stage | Shards | Per-param bytes at world=4 |
|---|---|---|
| DDP | nothing | 16 |
| ZeRO-1 | optimizer state | 4 + 4 + 2 = 10 |
| ZeRO-2 | + gradients | 4 + 3 = 7 |
| ZeRO-3 | + parameters | 4 |

Stage 3 is FSDP. Having written it, you will understand why per-block wrapping
matters and why "reshard after forward" is inseparable from recomputation.

## Measuring the right thing

Peak allocator readings on a 277 GiB GB300 are dominated by activations and
noise. This lab accounts for the bytes of **persistent training state**
directly — that is the number ZeRO is designed to reduce, and the ratios come
out exact.

## Expected output

```
persistent training state per rank (lower is better):
  DDP (baseline)         params   96.96  grads   96.96  optim  193.92  TOTAL   387.84 MiB
  ZeRO-1 (optim)         params   96.96  grads   96.96  optim   49.00  TOTAL   242.92 MiB
  ZeRO-2 (+grads)        params   96.96  grads   24.50  optim   49.00  TOTAL   170.46 MiB
  ZeRO-3 (+params)       params   24.81  grads   24.81  optim   49.62  TOTAL    99.23 MiB

  reduction vs DDP:  ZeRO-1 1.60x (theory 1.60x)   ZeRO-2 2.28x (theory 2.29x)   ZeRO-3 3.91x (theory 4.00x)
```

All four stages `[PASS]` against the oracle. Every stage lands within 2% of
theory, which is the point — you can predict these numbers before running.

## Stage 1 vs Stage 2: one line apart

```python
if stage == 1:
    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)   # everyone gets everything
else:
    dist.reduce(p.grad, dst=own, op=dist.ReduceOp.SUM)  # only the owner
if own == rank:
    p.grad /= world
elif stage >= 2:
    p.grad = None                                    # stage 1 keeps it
```

Both then step only their own slice and `broadcast` the result, so the weights
are identical to DDP's either way.

**What stage 2 does not fix here:** autograd still *allocates* the full gradient
before it is reduced away, so the transient peak barely moves — only the
steady-state holding drops. Production ZeRO-2 reduce-scatters into a
preallocated flat bucket via backward hooks, so the full gradient is never
materialised at all. That is what turns this accounting win into a real
peak-memory win.

## Stage 3 is FSDP, in one autograd function

```python
class _AllGatherParam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, group, numel, shape):
        flat = torch.empty(world * shard.numel(), ...)
        dist.all_gather_into_tensor(flat, shard, group=group)
        return flat[:numel].view(shape)

    @staticmethod
    def backward(ctx, grad):
        out = torch.empty(ctx.shard_numel, ...)
        dist.reduce_scatter_tensor(out, padded_grad, group=ctx.group)
        return out / world, None, None, None
```

Forward all-gathers the shards into the full weight; backward reduce-scatters
the gradient so each rank comes away holding only the slice it owns. That is
the whole of FSDP's parameter handling.

The block is then run functionally, with the gathered weights substituted in:

```python
torch.func.functional_call(self.block, self._gather(), (x,))
```

## Resharding and recomputation are the same decision

This is the most interesting result in the lab:

```
reshard-after-forward ON :  peak  285.2 MiB
reshard-after-forward OFF:  peak  911.6 MiB
```

With resharding off, every block's gathered weights stay pinned by the autograd
graph until backward finishes — `F.linear` saves its weight for the backward
pass. So all 8 blocks are materialised at once and you have saved **nothing**.
That is ZeRO-2 wearing a stage-3 costume.

Freeing them requires the forward graph to be discarded and rebuilt during
backward, which is exactly activation checkpointing:

```python
def forward(self, x):
    if self.reshard:
        return torch.utils.checkpoint.checkpoint(self._run, x, use_reentrant=False)
    return self._run(x)
```

You cannot have reshard-after-forward without recomputation. They are one
decision, and this experiment shows why.

## Two bugs worth knowing about

### `opt.zero_grad()` with a subset-owning optimizer

Under ZeRO the optimizer holds only the parameters this rank owns, so
`opt.zero_grad()` never clears the rest. In stage 2 you get away with it because
non-owned gradients are set to `None` anyway. In **stage 1** they are kept, grow
without bound across steps, get fed into the all-reduce, and reach **NaN around
step 80**.

Use `model.zero_grad()`. A subset-owning optimizer and `opt.zero_grad()` are a
bad pairing.

### Round-robin ownership balances nothing

```
roundrobin   heaviest/ideal = 1.99x   per-rank MiB: ['48.3', '0.6', '48.0', '0.1']
balanced     heaviest/ideal = 1.01x   per-rank MiB: ['24.5', '24.2', '24.2', '24.2']
```

Assigning parameter `i` to rank `i % world` balances the *count* of tensors, not
their size. This model's parameters span 512 to 1,048,576 elements, so one rank
gets double its fair share and your measured saving falls well short of theory.

Greedy largest-first assignment fixes it. Production implementations sidestep
the problem entirely by flattening every parameter into one buffer and splitting
that, which is balanced by construction.

## Performance

```
wall clock:  DDP 5.1s  Z1 6.3s  Z2 6.1s  Z3 15.0s
```

ZeRO-3 is 3× slower than DDP here. It gathers and re-gathers weights per block
per step, and this model is small enough that the collectives dominate. You pay
time to buy memory — which is exactly the trade, and worth it only when you
actually need the memory. Lab 7's FSDP2 does the same job in 5.6s.

## Exercises

1. Implement stage 2 properly with backward hooks and a flat bucket, so the
   full gradient is never allocated. Measure the peak, not just the accounting.
2. Add CPU offload to stage 1: keep Adam states in pinned host memory and
   stream them in. How much slower?
3. Shard a single flat buffer instead of whole tensors and confirm the
   imbalance ratio drops to exactly 1.00x.
4. Set `reshard_after_forward=True` but checkpoint only every *other* block.
   Plot peak memory against recompute cost.
