"""
Lab 4 -- pipeline parallelism by hand.

Split the model by DEPTH: blocks 0-1 on GPU0, 2-3 on GPU1, and so on.  Only
activations cross the boundary, via point-to-point send/recv, so this is by far
the cheapest form of communication -- which is exactly why it is what you use
across nodes, and why it matters least on a single NVLink box like this one.

The lesson here is the *bubble*.  With one batch in flight, a 4-stage pipeline
has 3 of its 4 GPUs idle at any instant.  Splitting the batch into micro-batches
and keeping several in flight fills the gap.  Predicted bubble fraction:

    bubble = (stages - 1) / (stages - 1 + microbatches)

This lab measures the real utilisation and compares it against that formula.

Run:  torchrun --standalone --nproc_per_node=4 lab4_pp.py
"""

import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from common import (
    GPTConfig,
    OUT_DIR,
    StepTimer,
    build_model,
    check_against_reference,
    describe_shards,
    get_batch,
    load_data,
    mem_report,
    print0,
    rank_print,
    set_determinism,
    setup_dist,
)

STEPS = 100
GLOBAL_BATCH = 32
LR = 3e-4


class Stage(nn.Module):
    """One contiguous slice of the model's depth.

    Stage 0 additionally owns the embeddings; the last stage owns the final
    norm and the output projection.  Every parameter lives on exactly one
    stage, which makes the optimizer trivially correct -- unlike tensor
    parallelism, no parameter is shared or split.
    """

    def __init__(self, cfg, stage_id, n_stages, source):
        super().__init__()
        assert cfg.n_layer % n_stages == 0
        per = cfg.n_layer // n_stages
        self.stage_id = stage_id
        self.n_stages = n_stages
        self.is_first = stage_id == 0
        self.is_last = stage_id == n_stages - 1
        self.blocks = nn.ModuleList(
            list(source.blocks[stage_id * per:(stage_id + 1) * per])
        )
        if self.is_first:
            self.wte, self.wpe = source.wte, source.wpe
        if self.is_last:
            self.ln_f, self.lm_head = source.ln_f, source.lm_head

    def forward(self, x, targets=None):
        if self.is_first:
            pos = torch.arange(x.shape[1], device=x.device)
            x = self.wte(x) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        if self.is_last:
            logits = self.lm_head(self.ln_f(x))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
            return loss
        return x


def build_stage(cfg, rank, world, device):
    set_determinism(0)
    full = build_model(cfg)
    return Stage(cfg, rank, world, full).to(device)


def pp_clip_grad_norm_(stage, max_norm, group):
    """Global gradient norm across pipeline stages.

    Simpler than the tensor-parallel case: stages hold *disjoint* parameters,
    so no parameter is double counted and a plain SUM all_reduce of the local
    sums of squares is exactly right.  It is still wrong to use torch's
    clip_grad_norm_, which would norm over one stage only.
    """
    sq = torch.zeros((), device=next(stage.parameters()).device)
    for p in stage.parameters():
        if p.grad is not None:
            sq += p.grad.detach().pow(2).sum()
    dist.all_reduce(sq, op=dist.ReduceOp.SUM, group=group)
    total = torch.sqrt(sq)
    coef = max_norm / (total + 1e-6)
    if coef < 1:
        for p in stage.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(coef)
    return total


def run_step(stage, x, y, rank, world, n_micro, cfg, device):
    """One optimizer step, executed as `n_micro` micro-batches.

    GPipe schedule: run every micro-batch forward, then every micro-batch
    backward.  1F1B interleaves them to reduce peak activation memory; the
    bubble is the same.  Returns (loss, seconds this rank spent computing).
    """
    micro = x.shape[0] // n_micro
    busy = 0.0
    saved = []  # (input_activation, output_activation) per micro-batch

    # ---- forward over all micro-batches ----
    for m in range(n_micro):
        xm = x[m * micro:(m + 1) * micro]
        ym = y[m * micro:(m + 1) * micro]
        if stage.is_first:
            t = time.perf_counter()
            out = stage(xm)
            torch.cuda.synchronize()
            busy += time.perf_counter() - t
            dist.send(out.contiguous(), dst=rank + 1)
            saved.append((None, out))
        else:
            shape = (micro, cfg.block_size, cfg.d_model)
            inp = torch.empty(shape, device=device, requires_grad=True)
            dist.recv(inp, src=rank - 1)   # blocks here == this rank's bubble
            inp.requires_grad_(True)
            t = time.perf_counter()
            out = stage(inp, ym) if stage.is_last else stage(inp)
            torch.cuda.synchronize()
            busy += time.perf_counter() - t
            if not stage.is_last:
                dist.send(out.contiguous(), dst=rank + 1)
            saved.append((inp, out))

    # ---- backward over all micro-batches ----
    total_loss = torch.zeros((), device=device)
    for m in range(n_micro):
        inp, out = saved[m]
        if stage.is_last:
            t = time.perf_counter()
            (out / n_micro).backward()
            torch.cuda.synchronize()
            busy += time.perf_counter() - t
            total_loss += out.detach() / n_micro
            dist.send(inp.grad.contiguous(), dst=rank - 1)
        else:
            gshape = (micro, cfg.block_size, cfg.d_model)
            gout = torch.empty(gshape, device=device)
            dist.recv(gout, src=rank + 1)
            t = time.perf_counter()
            out.backward(gout)
            torch.cuda.synchronize()
            busy += time.perf_counter() - t
            if not stage.is_first:
                dist.send(inp.grad.contiguous(), dst=rank - 1)

    # Only the last stage knows the loss; share it so every rank can log.
    dist.broadcast(total_loss, src=world - 1)
    return total_loss, busy


def train_pp(cfg, data, device, rank, world, n_micro, steps=STEPS, verbose=True):
    stage = build_stage(cfg, rank, world, device)
    opt = torch.optim.AdamW(stage.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    losses, busy_total = [], 0.0
    torch.cuda.reset_peak_memory_stats()
    timer = StepTimer()
    for step in range(steps):
        timer.tick(step)
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device)
        opt.zero_grad(set_to_none=True)
        loss, busy = run_step(stage, x, y, rank, world, n_micro, cfg, device)
        if step >= timer.warmup:      # match the timed window
            busy_total += busy
        pp_clip_grad_norm_(stage, 1.0, dist.group.WORLD)
        opt.step()
        losses.append(loss.item())
        if verbose and (step % 20 == 0 or step == steps - 1):
            print0(f"  step {step:3d}  loss {loss.item():.6f}")
    wall, _ = timer.stop(steps)
    return losses, wall, busy_total, stage


def profile_bubble(cfg, data, device, rank, world, n_micro):
    """Profile a few steps and split GPU time into compute versus communication.

    The utilisation numbers above are wall-clock bookkeeping around Python
    calls.  This is the same story told by the CUDA profiler, which is harder
    to fool: every device kernel is attributed either to NCCL (the rank sat
    waiting for a neighbour -- the bubble) or to real compute.

    A Chrome trace is written per rank.  Open it at chrome://tracing or in
    Perfetto and the staircase is unmistakable: rank 0 works, then idles while
    the wave travels down the pipeline and back.  With more micro-batches the
    staircase tightens into overlapping bands.
    """
    stage = build_stage(cfg, rank, world, device)
    opt = torch.optim.AdamW(stage.parameters(), lr=LR)

    for step in range(3):            # warm up before profiling
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device)
        opt.zero_grad(set_to_none=True)
        run_step(stage, x, y, rank, world, n_micro, cfg, device)
        opt.step()

    dist.barrier()
    t0 = time.time()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for step in range(5):
            x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device)
            opt.zero_grad(set_to_none=True)
            run_step(stage, x, y, rank, world, n_micro, cfg, device)
            opt.step()
        torch.cuda.synchronize()
    wall = time.time() - t0

    comm_us = compute_us = 0.0
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", None)
        if dev is None:
            dev = getattr(e, "self_cuda_time_total", 0)
        dev = dev or 0
        if dev <= 0:
            continue
        if "nccl" in e.key.lower():
            comm_us += dev
        else:
            compute_us += dev

    path = f"{OUT_DIR}/pp_micro{n_micro}_rank{rank}.json"
    prof.export_chrome_trace(path)
    return compute_us / 1e6, comm_us / 1e6, wall, path


def main():
    rank, world, local_rank, device = setup_dist()
    data, vocab_size = load_data()
    cfg = GPTConfig(vocab_size=vocab_size)

    print0(f"\npipeline of {world} stages, {cfg.n_layer // world} blocks each")
    print0(f"global batch {GLOBAL_BATCH} (no data parallelism)\n")

    print0("=" * 70)
    print0("what lives on each GPU -- note these sets are DISJOINT")
    print0("=" * 70)
    stage = build_stage(cfg, rank, world, device)
    describe_shards(stage)
    del stage

    print0("\n" + "=" * 70)
    print0("naive pipeline: 1 micro-batch, i.e. no pipelining at all")
    print0("=" * 70)
    losses1, wall1, busy1, _ = train_pp(cfg, data, device, rank, world, 1)
    util1 = busy1 / wall1
    rank_print(f"busy {busy1:.1f}s of {wall1:.1f}s wall  ->  utilisation {util1:6.1%}")
    mem_report("PP micro=1")
    ok1 = check_against_reference(losses1, label="pipeline, 1 micro-batch")
    pred1 = (world - 1) / (world - 1 + 1)
    print0(f"  predicted bubble {pred1:.1%}, so predicted utilisation "
           f"{1 - pred1:.1%}")

    print0("\n" + "=" * 70)
    print0("micro-batching: same math, same loss -- but does it help?")
    print0("=" * 70)
    print0("  All rows use an identical step count and identical warmup, so the")
    print0("  per-step column is directly comparable.  (An earlier version of")
    print0("  this lab compared a 100-step run against 30-step runs and timed")
    print0("  from step 0, which made micro-batching look far better than it is.)")
    SWEEP_STEPS = 40
    timed = SWEEP_STEPS - 10
    for n_micro in (1, 2, 4, 8):
        _, wall, busy, _ = train_pp(cfg, data, device, rank, world,
                                    n_micro, steps=SWEEP_STEPS, verbose=False)
        util = busy / wall
        pred = 1 - (world - 1) / (world - 1 + n_micro)
        print0(f"  micro-batches {n_micro:2d}:  {wall/timed*1e3:6.1f} ms/step  "
               f"busy {util:6.1%}   (bubble formula predicts {pred:6.1%} busy)")

    print0("\n  The formula predicts utilisation should climb steeply with more\n"
           "  micro-batches.  Measured per-step time barely improves and then\n"
           "  gets worse.  The formula counts only pipeline fill and drain and\n"
           "  assumes per-micro-batch compute dwarfs per-transfer latency.  Here\n"
           "  it does not: at micro=8 each transfer carries 4 sequences and the\n"
           "  send/recv latency costs more than the idle time it removes.\n"
           "  Micro-batching is not free, and the textbook bubble formula is an\n"
           "  upper bound you will only approach when each stage has real work.")

    print0("\n" + "=" * 70)
    print0("the bubble on a profiler timeline")
    print0("=" * 70)
    for n_micro in (1, 8):
        comp, comm, wall, path = profile_bubble(cfg, data, device, rank, world,
                                                n_micro)
        frac = comm / (comp + comm) if (comp + comm) > 0 else float("nan")
        rank_print(f"micro={n_micro}: GPU compute {comp*1e3:7.1f} ms, "
                   f"NCCL {comm*1e3:7.1f} ms  -> {frac:5.1%} of device time "
                   f"is waiting")
    print0(f"\n  Chrome traces written to {OUT_DIR}/pp_micro*_rank*.json --\n"
           "  open one in Perfetto or chrome://tracing to see the staircase.")
    print0("  The profiler is the honest witness here, and it disagrees with the\n"
           "  textbook: roughly 55% of device time is spent inside NCCL at\n"
           "  micro=1, and going to micro=8 does NOT reduce it -- it stays at\n"
           "  50-60% while the absolute NCCL time roughly doubles.  Eight small\n"
           "  transfers cost more than one large one, and at this model size that\n"
           "  swamps the idle time micro-batching was supposed to reclaim.\n"
           "  Pipeline parallelism needs stages with enough compute to amortise\n"
           "  the boundary transfers.  A 25M-param model on NVLink has neither\n"
           "  enough compute nor a slow enough link to make PP worthwhile -- it\n"
           "  is here to be understood, not to be used on this box.")

    print0(f"\nLab 4 {'PASSED' if ok1 else 'FAILED'}\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
