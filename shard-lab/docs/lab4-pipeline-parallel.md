# Lab 4 — Pipeline parallel by hand

**File:** `lab4_pp.py`
**Run:** `.venv/bin/torchrun --standalone --nproc_per_node=4 lab4_pp.py`
**Time:** ~3 minutes

Split the model by **depth**: blocks 0–1 on GPU0, 2–3 on GPU1, and so on. Only
activations cross the boundary, by point-to-point `send`/`recv`.

This is by far the cheapest form of communication, which is exactly why it is
what you use *across nodes* — and why it matters least on a single NVLink box
like this one.

> **This lab is the most skippable of the three parallelism strategies** on this
> hardware. PP exists to tolerate slow inter-node links you do not have. It is
> here to be understood, not used.

## Parameters are disjoint, which simplifies everything

Unlike tensor parallelism, no parameter is split or shared. Every parameter
lives on exactly one stage, so:

- the optimizer is trivially correct — each stage owns its own parameters
- gradient clipping just sums squares across stages with a plain `all_reduce`,
  with no double-counting to worry about

```python
def pp_clip_grad_norm_(stage, max_norm, group):
    sq = sum(p.grad.pow(2).sum() for p in stage.parameters() if p.grad is not None)
    dist.all_reduce(sq, op=dist.ReduceOp.SUM, group=group)   # disjoint: plain SUM
    total = torch.sqrt(sq)
```

Compare against Lab 3, which needed to distinguish sharded from replicated. It
is still wrong to use torch's `clip_grad_norm_`, which would norm over one
stage only.

## The schedule

GPipe: run every micro-batch forward, then every micro-batch backward. 1F1B
interleaves them to cut peak activation memory; the bubble is the same.

Forward on a middle stage:

```python
inp = torch.empty(shape, device=device, requires_grad=True)
dist.recv(inp, src=rank - 1)      # blocks here — this IS the bubble
out = stage(inp)
dist.send(out.contiguous(), dst=rank + 1)
```

Backward runs in reverse, sending `inp.grad` back to `rank - 1`. Only the last
stage knows the loss, so it is broadcast so every rank can log it.

## Expected output

```
[rank 0] busy 3.2s of 3.5s wall  ->  utilisation  89.5%
[rank 1] busy 2.7s of 3.5s wall  ->  utilisation  75.3%
[rank 2] busy 2.2s of 3.5s wall  ->  utilisation  61.1%
[rank 3] busy 1.7s of 3.5s wall  ->  utilisation  47.5%

  [PASS] pipeline, 1 micro-batch: max |loss - oracle| = 4.768e-07

  micro-batches  1:    39.4 ms/step  busy  89.2%   (bubble formula predicts  25.0% busy)
  micro-batches  2:    33.0 ms/step  busy  85.8%   (bubble formula predicts  40.0% busy)
  micro-batches  4:    41.2 ms/step  busy  87.3%   (bubble formula predicts  57.1% busy)
  micro-batches  8:    67.2 ms/step  busy  90.3%   (bubble formula predicts  72.7% busy)

[rank 0] micro=1: GPU compute   234.5 ms, NCCL   292.3 ms  -> 55.5% of device time is waiting
[rank 0] micro=8: GPU compute   452.7 ms, NCCL   619.5 ms  -> 57.8% of device time is waiting
```

## The textbook says micro-batching fixes the bubble. It does not here.

The standard formula:

```
bubble = (stages - 1) / (stages - 1 + microbatches)
```

predicts utilisation should climb from 25% to 73% as you go from 1 to 8
micro-batches. **The measurement disagrees.**

Per-step time goes 39.4 → 33.0 → 41.2 → 67.2 ms. Only micro=2 helps at all, and
micro=8 is 70% *worse* than not micro-batching.

The profiler confirms it: NCCL is 53–57% of device time at micro=1 and still
50–60% at micro=8, with the absolute NCCL time roughly **doubling**.

The reason is that the formula counts only pipeline fill and drain, and assumes
per-micro-batch compute dwarfs per-transfer latency. At micro=8 each transfer
carries 4 sequences. Eight small transfers cost far more than one large one,
and that swamps the idle time micro-batching was supposed to reclaim.

Pipeline parallelism needs stages with enough compute to amortise the boundary
transfers. A 25M-param model on NVLink has neither enough compute nor a slow
enough link.

## Two instruments that disagree — and which to trust

The wall-clock "busy" fraction reads 86–90%. The profiler reads ~55% of device
time in NCCL. Both are in the output and they tell different stories.

They measure different things. The busy fraction times a Python region that
contains a `cuda.synchronize()`; the profiler attributes actual device kernels.
**For bubble analysis, trust the profiler.** The wall-clock number is too easy
to fool.

## Look at the timeline yourself

The lab writes a Chrome trace per rank:

```
out/pp_micro1_rank0.json
out/pp_micro8_rank0.json
...
```

Copy one back and open it in [Perfetto](https://ui.perfetto.dev) or
`chrome://tracing`:

```bash
scp lambda-gpu:~/shard-lab/out/pp_micro1_rank0.json .
```

At micro=1 you see a single compute burst per rank separated by long NCCL
waits — the staircase of a pipeline with one item in flight. At micro=8 the
bursts pack together, but the NCCL bands get *wider*, which is the whole
finding above rendered visually.

## A note on the earlier, wrong version of this lab

The first version of this lab reported utilisation climbing 17% → 58% "exactly
as predicted". That was wrong twice over:

1. It timed from step 0, so the 17% figure was mostly one-time warmup.
2. It compared a 100-step run against 30-step runs and reported raw wall clock,
   making micro-batching look like it cut time from 3.5s to 0.7s.

Normalising to ms/step and discarding warmup reverses the conclusion entirely.
Correctness had an oracle guarding it; performance had nothing, and that is
precisely where the mistake survived.

## Exercises

1. Implement 1F1B: interleave forward and backward instead of doing all
   forwards first. The bubble is unchanged but peak activation memory should
   drop noticeably — measure it.
2. Raise `block_size` to 1024 or `d_model` to 2048 so each stage has real work,
   then rerun the micro-batch sweep. Does the formula start to hold?
3. Deliberately deadlock: make every stage `send` before `recv`. Then set
   `TORCH_NCCL_BLOCKING_WAIT=1` and compare the diagnostics.
4. Compare against `torch.distributed.pipelining`'s `ScheduleGPipe` and
   `Schedule1F1B`.
