# Lab 1 — Collectives in isolation

**File:** `lab1_collectives.py`
**Run:** `.venv/bin/torchrun --standalone --nproc_per_node=4 lab1_collectives.py`
**Time:** ~30 seconds

No model, no autograd. Small labelled tensors so you can see exactly what each
collective does to them.

**Predict each result before you read it.** That is the entire exercise. If you
can predict all six, the rest of the course is mechanics.

## The five collectives, and what each becomes later

| Collective | Where it shows up |
|---|---|
| `broadcast` | making DDP replicas identical at init |
| `all_reduce` | gradient synchronisation in DDP; the `f`/`g` operators in Lab 3 |
| `reduce_scatter` | ZeRO-2/3 gradient reduction |
| `all_gather` | FSDP materialising a layer's weights before forward |
| `all_to_all` | MoE expert routing |
| `send`/`recv` | pipeline parallel stage boundaries |

## Expected output

Working through the interesting ones:

**`all_reduce(SUM)`** — every rank starts with `[r+1, r+1, r+1, r+1]`, everyone
ends with `[10, 10, 10, 10]` because `1+2+3+4 = 10`.

**`reduce_scatter`** — input `arange(8) + rank*100` on each rank. Rank 0 keeps
`[600.0, 604.0]`, rank 1 `[608.0, 612.0]`, rank 2 `[616.0, 620.0]`, rank 3
`[624.0, 628.0]`. Each rank holds one quarter of the reduced answer, using one
quarter of the memory.

**`all_gather`** — reassembles those quarters back into the full
`[600, 604, 608, 612, 616, 620, 624, 628]`.

## The identity that the rest of the course rests on

```
reduce_scatter  then  all_gather   ==   all_reduce
```

The lab asserts this holds **exactly**, not approximately:

```
all_reduce      : [600.0, 604.0, 608.0, 612.0, 616.0, 620.0, 624.0, 628.0]
RS then AG      : [600.0, 604.0, 608.0, 612.0, 616.0, 620.0, 624.0, 628.0]
identical       : True
```

And the byte accounting:

```
ring all_reduce moves      48.0 B/rank
reduce_scatter moves       24.0 B/rank
all_gather moves           24.0 B/rank
```

A ring all-reduce *is* a reduce-scatter followed by an all-gather. Each half
moves `(N-1)/N × S` bytes; together they cost `2(N-1)/N × S`, which is exactly
what a ring all-reduce costs.

**This is why ZeRO-2 is free.** It stops after the reduce-scatter, because each
rank only needs gradients for the parameters it owns. Same bytes on the wire as
DDP, less memory held. You will implement precisely this in Lab 5.

## The deadlock trap

`send`/`recv` in this lab is deliberately ordered:

```python
if rank % 2 == 0:
    dist.send(payload, dst=nxt)
    dist.recv(got, src=prv)
else:
    dist.recv(got, src=prv)
    dist.send(payload, dst=nxt)
```

If every rank called `send` first with a large enough payload, this would
deadlock — NCCL `send` is not guaranteed to buffer. The even/odd split breaks
the cycle. Lab 4 relies on the same discipline at every pipeline boundary.

## Inspect the NCCL topology

The plan calls for doing this once. It tells you what your interconnect
actually is, as opposed to what you assume it is:

```bash
NCCL_DEBUG=INFO .venv/bin/torchrun --standalone --nproc_per_node=4 \
  lab1_collectives.py 2>&1 | grep -E "via (P2P|SHM|NET)" | sort -u
```

On this box:

```
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM
NCCL INFO Channel 00/1 : 0[0] -> 2[2] via P2P/CUMEM
NCCL INFO Channel 00/1 : 0[0] -> 3[3] via P2P/CUMEM
...
```

Every path is `P2P/CUMEM` across 32 channels, with direct rank-to-rank links
as well as the ring. No `SHM` and no `NET` anywhere, which confirms full NVLink
peer access between all four GB300s.

This is worth checking on any new machine. If you see `SHM` you are going
through host memory, and if you see `NET` you are going over the network —
either would make Lab 3's tensor parallelism a very bad idea.

## Exercises

1. Predict the `all_to_all` output before running. Rank `r` sends
   `[r*10+0, r*10+1, r*10+2, r*10+3]`; what does rank 2 receive?
2. Swap `ReduceOp.SUM` for `ReduceOp.AVG` in the all-reduce and work out what
   changes in Lab 2's gradient synchronisation as a result.
3. Make the `send`/`recv` section deadlock on purpose: have every rank send
   first, with a payload of a few hundred MB. Then set
   `TORCH_NCCL_BLOCKING_WAIT=1` and see how much more useful the failure is.
4. Time an `all_reduce` of 100 MB against a `reduce_scatter` of the same
   tensor. Confirm the second is roughly half the first.
