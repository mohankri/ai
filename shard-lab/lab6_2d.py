"""
Lab 6 -- composing tensor parallelism with data parallelism.

Reuses Lab 3's parallel layers unchanged and adds a second axis.  On 4 GPUs:

    mesh = (dp=2, tp=2), laid out row-major, so rank = dp_index * 2 + tp_index

      tp groups : {0,1} and {2,3}     activations all_reduce here
      dp groups : {0,2} and {1,3}     gradients all_reduce here

The point of the lab is that these two axes are genuinely independent.  The
tensor-parallel code from Lab 3 needs no modification whatsoever to work inside
a data-parallel replica -- it simply operates on a smaller process group.  Each
collective is confined to its own axis and neither knows the other exists.

Two things reliably trip people up here and both are checked below:
  * the global batch is now micro x dp_size, NOT micro x world_size
  * the gradient norm has to be correct along both axes at once

Run:  torchrun --standalone --nproc_per_node=4 lab6_2d.py
"""

import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from common import (
    GPTConfig,
    StepTimer,
    build_model,
    check_against_reference,
    describe_shards,
    get_batch,
    global_shapes_of,
    load_data,
    mem_report,
    print0,
    rank_print,
    set_determinism,
    setup_dist,
)
from lab3_tp import build_tp_model

STEPS = 100
GLOBAL_BATCH = 32
LR = 3e-4


def clip_2d_(model, max_norm, tp_group):
    """Global gradient norm over a 2D mesh.

    Only the TP axis needs reducing.  Gradients have already been averaged
    across the DP axis by the time this runs, so every rank in a DP group holds
    identical values -- reducing over DP as well would inflate the norm by
    `dp_size`.  Along TP, sharded parameters hold distinct slices and must be
    summed, while replicated parameters must be counted exactly once.
    """
    dev = next(model.parameters()).device
    sharded_sq = torch.zeros((), device=dev)
    repl_sq = torch.zeros((), device=dev)
    for p in model.parameters():
        if p.grad is None:
            continue
        sq = p.grad.detach().pow(2).sum()
        if getattr(p, "_tp_sharded", False):
            sharded_sq += sq
        else:
            repl_sq += sq
    dist.all_reduce(sharded_sq, op=dist.ReduceOp.SUM, group=tp_group)
    total = torch.sqrt(sharded_sq + repl_sq)
    coef = max_norm / (total + 1e-6)
    if coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(coef)
    return total


def train_2d(cfg, data, device, mesh, steps=STEPS):
    tp_group = mesh["tp"].get_group()
    dp_group = mesh["dp"].get_group()
    dp_rank, dp_size = dist.get_rank(dp_group), dist.get_world_size(dp_group)

    model = build_tp_model(cfg, device, tp_group)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)

    losses = []
    model.train()
    torch.cuda.reset_peak_memory_stats()
    timer = StepTimer()
    for step in range(steps):
        timer.tick(step)
        # The batch is split along DP only.  Both ranks of a TP group must see
        # exactly the SAME tokens -- they are cooperating on one forward pass,
        # not processing different data.
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device,
                         rank=dp_rank, world=dp_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Synchronise gradients along DP only.  The TP axis was already handled
        # inside the layers by the f/g operators during backward.
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
                p.grad /= dp_size

        clip_2d_(model, 1.0, tp_group)
        opt.step()

        lt = loss.detach().clone()
        dist.all_reduce(lt, op=dist.ReduceOp.SUM, group=dp_group)
        losses.append((lt / dp_size).item())
        if step % 20 == 0 or step == steps - 1:
            print0(f"  step {step:3d}  loss {losses[-1]:.6f}")
    dt, _ = timer.stop(steps)
    return losses, dt, model


def main():
    rank, world, local_rank, device = setup_dist()
    assert world == 4, "this lab assumes a 2x2 mesh on 4 GPUs"
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    tp_group, dp_group = mesh["tp"].get_group(), mesh["dp"].get_group()
    dp_rank = dist.get_rank(dp_group)
    tp_rank = dist.get_rank(tp_group)

    data, vocab_size = load_data()
    cfg = GPTConfig(vocab_size=vocab_size)

    print0(f"\nmesh (dp=2, tp=2) over {world} GPUs")
    print0(f"global batch {GLOBAL_BATCH} = {GLOBAL_BATCH // 2} per DP replica")
    print0(f"heads per rank: {cfg.n_head} / 2 = {cfg.n_head // 2}\n")
    rank_print(f"mesh coords: dp={dp_rank} tp={tp_rank}   "
               f"tp peers {dist.get_process_group_ranks(tp_group)}   "
               f"dp peers {dist.get_process_group_ranks(dp_group)}")

    set_determinism(0)
    full_shapes = global_shapes_of(build_model(cfg))

    print0("\n" + "=" * 70)
    print0("parameter layout -- sharded along TP, replicated along DP")
    print0("=" * 70)
    # EVERY rank calls describe_shards -- it contains a barrier, so guarding
    # the call with `if rank in (0, 2)` would hang ranks 1 and 3 forever.  The
    # `only` argument filters the output instead of the participation.
    probe = build_tp_model(cfg, device, tp_group)
    describe_shards(probe, full_shapes, only={0, 2})  # one per DP replica
    del probe

    print0("\n" + "=" * 70)
    print0("training on the 2D mesh")
    print0("=" * 70)
    losses, dt, model = train_2d(cfg, data, device, mesh)
    print0(f"\nsteady state {dt:.2f}s over 90 timed steps  (single GPU: 3.24s)")
    mem_report("TP2 x DP2")
    ok = check_against_reference(losses, label="2D mesh (TP2 x DP2)")

    print0("\n  Note that TP 2 x DP 2 costs about half the per-rank parameter\n"
           "  memory of pure DP, and half the TP communication of Lab 3's TP 4,\n"
           "  while consuming the same global batch as both.  That trade -- less\n"
           "  chatter per collective, more replicas -- is the whole reason real\n"
           "  training runs use a mesh rather than a single axis.")

    print0(f"\nLab 6 {'PASSED' if ok else 'FAILED'}\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
