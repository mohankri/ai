# Lab 2 — Data parallel by hand

**File:** `lab2_ddp.py`
**Run:** `.venv/bin/torchrun --standalone --nproc_per_node=4 lab2_ddp.py`
**Time:** ~1 minute

Every GPU holds a complete replica. DDP saves no memory at all — it only buys
throughput. The whole mechanism is two steps:

1. `broadcast` the parameters once, so replicas start identical
2. `all_reduce` the gradients after every backward, and divide by world size

That is it. About ten lines.

## Why divide by world size

Each rank computes the mean loss over its own 8 samples. The gradient of the
*global* mean over 32 samples is the mean of the four per-rank gradients — but
only because every rank has an equal number of samples. `all_reduce(SUM)` then
`/= world` gives exactly that.

The lab demonstrates what happens when you get this wrong.

## Expected output

```
  [PASS] hand-written DDP (mean): max |loss - oracle| = 4.768e-07
  [FAIL] hand-written DDP (sum) -- EXPECTED TO FAIL: max |loss - oracle| = 1.023e-01
  [PASS] torch DDP: max |loss - oracle| = 9.537e-07

  steady-state over 90 timed steps (10 warmup steps discarded):
    single GPU   :  3.24s   (Lab 0 baseline)
    hand-written :  4.54s   0.71x vs 1 GPU
    torch DDP    :  3.15s   1.03x vs 1 GPU, 1.44x vs hand-written

Lab 2 PASSED
```

## The deliberate bug — read this one carefully

Using `SUM` instead of `MEAN` leaves gradients 4× too large. Look at what
happens:

```
step   0  loss 4.402704     <- identical to the oracle
step  20  loss 3.264856     <- oracle says 3.269731
step  99  loss 2.510812     <- oracle says 2.516004
```

It still trains. The loss still goes down. It even ends up *slightly lower*
than the oracle, because a 4× gradient behaves like a 4× learning rate. Nothing
crashes, nothing warns, and if you were staring at a loss curve you would call
this a successful run.

The deviation is 1.0e-01. Remember that number — in Lab 7 you will see bf16
introduce 5.4e-03 of noise all on its own, which is why establishing
correctness in fp32 first is not optional.

## Verify the replicas really are identical

```python
def verify_replicas_identical(model, tag):
    total = torch.zeros((), device=...)
    for p in model.parameters():
        ref = p.detach().clone()
        dist.broadcast(ref, src=0)
        total += (p.detach() - ref).abs().sum()
    dist.all_reduce(total, op=dist.ReduceOp.MAX)
```

Expected: `0.000e+00` both after the broadcast and after 100 steps.

`build_model` already seeds identically on every rank, so the initial broadcast
is a no-op here. Write it anyway. The day someone adds a rank-dependent init,
it is the only thing standing between you and silently training four different
models — which, again, would not crash.

## What torch DDP does that yours does not

1.44× faster, from two things:

- **Gradient bucketing** — gradients are grouped into buckets of a few MB
  instead of one all-reduce per tensor. 128 small collectives become a handful
  of large ones, and collective latency dominates at this size.
- **Overlap** — each bucket's all-reduce is launched as soon as that bucket's
  gradients are ready, during the backward pass. The hand-written loop waits
  for the entire backward to finish before communicating anything.

## Now look at the single-GPU column

```
single GPU   :  3.24s
hand-written :  4.54s   0.71x
torch DDP    :  3.15s   1.03x
```

Four GB300s, and the best result is 3% faster than one. The hand-written
version is 29% *slower* than not parallelising at all.

This is honest and it is the point. At 25M parameters with a global batch of
32, each rank does so little work that collective latency and Python overhead
dominate completely. Data parallelism pays off when per-rank compute is large
enough to hide the communication behind it — this model is deliberately far too
small for that.

If a tutorial shows you a big speedup on a toy model, check whether they
included warmup in the timing. The original version of this lab reported
"1.76× faster" purely because the hand-written implementation ran first and
absorbed the NCCL communicator setup cost.

## Exercises

1. Raise `GLOBAL_BATCH` to 256 and re-measure. At what batch size does DDP
   finally beat one GPU by a worthwhile margin?
2. Replace the per-parameter `all_reduce` loop with a single flattened buffer
   (`torch._utils._flatten_dense_tensors`) and re-measure. How much of torch
   DDP's 1.44× do you recover from bucketing alone, without overlap?
3. Delete the initial `broadcast` and add a rank-dependent init such as
   `torch.manual_seed(rank)`. Confirm the loss still looks reasonable while
   `verify_replicas_identical` reports a large deviation.
