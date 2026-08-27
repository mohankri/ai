# Lab 3 — Tensor parallel by hand

**File:** `lab3_tp.py`
**Run:** `.venv/bin/torchrun --standalone --nproc_per_node=4 lab3_tp.py`
**Time:** ~2 minutes

The core of the course. Spend real time here — this is where the non-obvious
ideas live.

Unlike DDP, this splits the matrices *inside* a layer. Every rank holds a
different slice of every weight matrix, and they cooperate on a single forward
pass over the **same** batch. There is no data parallelism at all.

## The two operators everything is built from

```python
class _CopyToTP(torch.autograd.Function):        # f
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x                                  # identity
    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, group=ctx.group)    # all-reduce
        return grad, None

class _ReduceFromTP(torch.autograd.Function):    # g
    @staticmethod
    def forward(ctx, x, group):
        x = x.clone()                             # never all_reduce in place
        dist.all_reduce(x, group=group)           # all-reduce
        return x
    @staticmethod
    def backward(ctx, grad):
        return grad, None                         # identity
```

They are adjoints of each other. `f` is identity forward / all-reduce backward;
`g` is all-reduce forward / identity backward.

**Why `f` needs the backward all-reduce:** every rank received the *same* input
`x` and used it to produce a *different* slice of the output. The true gradient
with respect to `x` is therefore the sum of every rank's contribution.

**Why `g` needs the forward all-reduce:** each rank computed a partial sum over
its slice of the contraction dimension. Backward is identity because every rank
receives the same incoming gradient and needs only its own slice.

The `.clone()` in `g` is not decoration. `all_reduce` is in-place, and mutating
an autograd input in place silently corrupts the graph.

## Column-parallel and row-parallel

**Column-parallel** splits the weight by *output* features. Input replicated,
output sharded. Needs `f` on the input.

**Row-parallel** splits by *input* features. Input already sharded, output is a
partial sum that must be reduced. Needs `g` on the output, and **no `f`**,
because its input arrives already sharded.

```python
class RowParallelLinear(nn.Module):
    def forward(self, x):
        y = F.linear(x, self.weight)          # partial sum on this rank
        y = _ReduceFromTP.apply(y, self.group)
        return y + self.bias                  # AFTER the reduce
```

The bias must be added **after** the all-reduce. Add it before and the reduce
sums it `world` times.

## Why pairing them matters

Build the MLP as column-parallel up-projection → GeLU → row-parallel
down-projection. GeLU is elementwise and needs no communication because each
rank owns whole output columns.

The result: **one all-reduce forward and one backward per MLP.** The wide
4×`d_model` intermediate activation never has to be gathered. Attention works
the same way — shard q/k/v column-wise so each rank owns whole heads, make
`o_proj` row-parallel.

## What the sharding actually looks like

```
blocks.0.attn.q_proj.weight   local=(128, 512)    global=(512, 512)    <-- SHARDED
blocks.0.attn.o_proj.weight   local=(512, 128)    global=(512, 512)    <-- SHARDED
blocks.0.attn.o_proj.bias     local=(512,)        global=(512,)
blocks.0.mlp.up.weight        local=(512, 512)    global=(2048, 512)   <-- SHARDED
blocks.0.mlp.down.weight      local=(512, 512)    global=(512, 2048)   <-- SHARDED
wte.weight                    local=(65, 512)     global=(65, 512)
local params: 6,521,856
```

Note `q_proj` splits on dim 0 and `o_proj` on dim 1 — column versus row. The
biases of row-parallel layers stay full size because they are replicated.
Embeddings and `lm_head` are left replicated here; Megatron makes `lm_head`
column-parallel with a vocab-parallel cross-entropy, which with a vocab of 65
would be all cost and no lesson.

## Three escalating checks

This structure is what makes the lab worth doing.

| Check | Catches | Result |
|---|---|---|
| 1. Logits vs oracle | forward-pass errors | `3.815e-06` PASS |
| 2. Gradients vs oracle | missing `f` — invisible to check 1 | `1.490e-08` PASS |
| 3. 100-step loss curve | optimizer-level errors | `4.768e-07` PASS |

Check 2 loads the oracle's weights, slices them onto each rank, runs one
forward/backward, and compares each parameter's gradient against the matching
slice of the unsharded model's gradient.

## The five bugs, and how each announces itself

### (a) Omitting `f` — the nastiest

```
[PASS] no-f logits:     max|delta| = 3.815e-06     <- identical to correct
[FAIL] no-f gradients:  max|delta| = 1.813e-02  (worst on wte.weight)
```

The forward pass is **perfect**. Only gradients are wrong, and only *upstream*
of the sharded layers — which is why the failure surfaces on the embedding.
This single case justifies checking gradients separately from logits.

### (b) Row-parallel bias before the reduce

`[FAIL] bias-early logits: max|delta| = 4.540e-01`. Bias summed 4×.

### (c) Forgetting to divide `n_head`, derived `head_dim`

`[FAIL] n_head-silent logits: max|delta| = 2.012e+00`

The reshape **succeeds** with the wrong grouping — 8 heads of dim 16 instead of
2 heads of dim 64 — and computes plausible-looking nonsense. This is the
dangerous variant, and it is what most real codebases would hit, because they
derive `head_dim` from the local width.

### (d) The same mistake, fixed `head_dim`

`[RAISED] shape '[8, 256, 8, 64]' is invalid for input of size 262144`

Raises immediately. Strictly the friendlier failure.

> The original plan claimed this bug is always silent. That was wrong: whether
> it is silent depends entirely on where `head_dim` comes from.

### (e) `torch.nn.utils.clip_grad_norm_` — found the hard way

```
[FAIL] TP with local-norm clipping: max |loss - oracle| = 7.133e-02
       oracle[26]=3.236362  got[26]=3.165036
```

This one was not in the plan. It was found because check 3 failed while checks
1 and 2 passed — forward and gradients were both exact to 1e-06.

`clip_grad_norm_` norms over `model.parameters()`, which on each rank is only
that rank's **shards**. The norm comes out too small, the clip coefficient too
large, the model is under-clipped, and it takes bigger steps than the oracle.

Note the failure direction: under-clipping makes the loss fall **faster**. The
bug makes your training curve look *better*. The correct version:

```python
def tp_clip_grad_norm_(model, max_norm, group):
    sharded_sq, repl_sq = 0, 0
    for p in model.parameters():
        sq = p.grad.detach().pow(2).sum()
        if getattr(p, "_tp_sharded", False):
            sharded_sq += sq        # distinct slices: must be summed across TP
        else:
            repl_sq += sq           # identical copies: count exactly ONCE
    dist.all_reduce(sharded_sq, op=dist.ReduceOp.SUM, group=group)
    total = torch.sqrt(sharded_sq + repl_sq)
```

All-reducing the replicated parameters too would inflate the norm by `world`.
This same class of bug reappears in Lab 4 and Lab 6, each needing a different
reduction rule.

## Performance

```
steady state 4.25s over 90 timed steps (single GPU: 3.24s)
```

TP 4 is **slower** than one GPU. Two all-reduces per block per step cost more
than the tiny per-rank matmuls they enable. Tensor parallelism buys memory, not
speed — and at this size it does not buy much of that either.

TP is also the most communication-intensive form of parallelism, which is why
the rule is to keep it inside one NVLink domain. On this box that is all four
GPUs (see Lab 1's topology dump); across PCIe or a network it would be far
worse than not sharding at all.

## Exercises

1. Fuse q/k/v into one `Linear(d, 3d)` and shard it column-wise. You will need
   to interleave the slices so each rank still owns whole heads. This is the
   index arithmetic the lab deliberately avoids — worth doing once.
2. Make `lm_head` column-parallel and implement vocab-parallel cross-entropy.
   The logits are now sharded along vocab, so the softmax needs a max and a sum
   all-reduce before the loss.
3. Count the all-reduces per step: 8 blocks × (1 attention + 1 MLP) × (forward +
   backward). Compare against the measured NCCL time from Lab 4's profiler.
4. Run with TP=2 (`--nproc_per_node=2`) and compare timing to TP=4. Does halving
   the communication help more than halving the parallelism hurts?
