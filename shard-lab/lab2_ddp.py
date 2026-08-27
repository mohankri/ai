"""
Lab 2 -- data parallel by hand.

Every GPU holds a complete replica.  The only distributed machinery is:
  1. broadcast the parameters once, so replicas start identical
  2. all_reduce the gradients after every backward, and divide by world size

That is the whole of DDP.  It saves no memory at all -- it only buys throughput.

Three things this lab demonstrates:
  * the hand-written version matches the single-GPU oracle exactly
  * using SUM instead of MEAN breaks it, in a specific and instructive way
  * torch's DDP gets the same numbers but is faster, because it overlaps the
    all_reduce with the backward pass instead of waiting for it

Run:  torchrun --standalone --nproc_per_node=4 lab2_ddp.py
"""

import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from common import (
    GPTConfig,
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
# Lab 0's steady-state time over the same 90 timed steps, for reference.
SINGLE_GPU_SECONDS = 3.24


def verify_replicas_identical(model, tag):
    """Confirm every rank really does hold the same weights.

    Worth doing explicitly: divergent replicas do not crash, they just quietly
    train a different model on every GPU while the loss still looks plausible.
    """
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        ref = p.detach().clone()
        dist.broadcast(ref, src=0)
        total += (p.detach() - ref).abs().sum()
    dist.all_reduce(total, op=dist.ReduceOp.MAX)
    print0(f"  {tag}: max deviation between replicas = {total.item():.3e}")
    return total.item()


def train_manual(data, cfg, device, rank, world, reduction="mean"):
    """Hand-rolled DDP.  `reduction='sum'` is the deliberate bug."""
    set_determinism(0)
    model = build_model(cfg).to(device)

    # Step 1 of DDP: make the replicas identical.  Our build_model is already
    # seeded identically on every rank, so this is a no-op here -- but it is a
    # no-op you should always write, because the day someone adds a
    # rank-dependent init it is the only thing standing between you and silently
    # training four different models.
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    verify_replicas_identical(model, "after broadcast")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)

    losses = []
    model.train()
    torch.cuda.reset_peak_memory_stats()
    timer = StepTimer()
    for step in range(STEPS):
        timer.tick(step)
        # Each rank takes its slice of the SAME global batch the oracle used.
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device,
                         rank=rank, world=world)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Step 2 of DDP: synchronise gradients.
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                if reduction == "mean":
                    p.grad /= world
                # reduction == "sum" leaves gradients world-times too large.
                # The loss will still fall -- it just follows a different
                # trajectory, as if the learning rate were 4x higher.  Nothing
                # crashes.  This is why you compare against an oracle.

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Report the GLOBAL loss so it is comparable to the oracle.  Because
        # every rank has an equal number of samples, the mean of the per-rank
        # means is the mean over the global batch.
        lt = loss.detach().clone()
        dist.all_reduce(lt, op=dist.ReduceOp.SUM)
        losses.append((lt / world).item())

        if rank == 0 and (step % 20 == 0 or step == STEPS - 1):
            print0(f"  step {step:3d}  loss {losses[-1]:.6f}")
    dt, _ = timer.stop(STEPS)
    return losses, dt, model


def train_torch_ddp(data, cfg, device, rank, world, local_rank):
    set_determinism(0)
    model = build_model(cfg).to(device)
    ddp = DistributedDataParallel(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(ddp.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)

    losses = []
    ddp.train()
    timer = StepTimer()
    for step in range(STEPS):
        timer.tick(step)
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device,
                         rank=rank, world=world)
        _, loss = ddp(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # DDP all-reduces here, overlapped with backward
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        opt.step()
        lt = loss.detach().clone()
        dist.all_reduce(lt, op=dist.ReduceOp.SUM)
        losses.append((lt / world).item())
    dt, _ = timer.stop(STEPS)
    return losses, dt


def main():
    rank, world, local_rank, device = setup_dist()
    data, vocab_size = load_data()
    cfg = GPTConfig(vocab_size=vocab_size)

    print0(f"\nworld size {world}, global batch {GLOBAL_BATCH}, "
           f"per-rank batch {GLOBAL_BATCH // world}\n")

    print0("=" * 70)
    print0("hand-written DDP (correct: MEAN reduction)")
    print0("=" * 70)
    losses, dt, model = train_manual(data, cfg, device, rank, world, "mean")
    print0(f"\nwall clock {dt:.1f}s  ({STEPS/dt:.1f} steps/s)")
    describe_shards(model)
    mem_report("hand DDP")
    verify_replicas_identical(model, "after 100 steps")
    ok = check_against_reference(losses, label="hand-written DDP (mean)")

    print0("\n" + "=" * 70)
    print0("the deliberate bug: SUM instead of MEAN")
    print0("=" * 70)
    bad, _, _ = train_manual(data, cfg, device, rank, world, "sum")
    check_against_reference(bad, label="hand-written DDP (sum) -- EXPECTED TO FAIL")
    print0(f"  note it still 'trains': {bad[0]:.4f} -> {bad[-1]:.4f}.  A loss that\n"
           f"  goes down is not evidence that your parallelism is correct.")

    print0("\n" + "=" * 70)
    print0("torch DistributedDataParallel, for comparison")
    print0("=" * 70)
    tl, tdt = train_torch_ddp(data, cfg, device, rank, world, local_rank)
    check_against_reference(tl, label="torch DDP")
    print0(f"\n  steady-state over 90 timed steps (10 warmup steps discarded):")
    print0(f"    single GPU   : {SINGLE_GPU_SECONDS:5.2f}s   (Lab 0 baseline)")
    print0(f"    hand-written : {dt:5.2f}s   {SINGLE_GPU_SECONDS/dt:.2f}x vs 1 GPU")
    print0(f"    torch DDP    : {tdt:5.2f}s   {SINGLE_GPU_SECONDS/tdt:.2f}x vs 1 GPU, "
           f"{dt/tdt:.2f}x vs hand-written")
    print0("  torch DDP buckets gradients and starts each all_reduce as soon as\n"
           "  that bucket's grads are ready, so communication overlaps backward\n"
           "  compute.  The hand-written loop waits for the entire backward pass\n"
           "  to finish before it communicates anything.")
    print0("  Read the 1-GPU column before celebrating: at 25M parameters and a\n"
           "  global batch of 32, each rank does so little work that collective\n"
           "  latency and Python overhead dominate.  Four GB300s barely beat one.\n"
           "  These labs demonstrate correctness, not speedup -- the speedup\n"
           "  arrives when per-rank compute is large enough to hide the\n"
           "  communication, which this model is deliberately too small to do.")

    print0(f"\nLab 2 {'PASSED' if ok else 'FAILED'}\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
