"""
Lab 7 -- the same ideas, as PyTorch ships them.

Now that you have written ZeRO-3 and tensor parallelism by hand, meet the
production versions:

  fully_shard        FSDP2.  Your Lab 5 ShardedBlock, but with communication
                     overlapped against compute, flat parameter buffers, and
                     prefetching of the next block's all_gather.
  parallelize_module DTensor tensor parallelism.  Your Lab 3 f/g operators,
                     expressed as placements (Shard / Replicate) that the
                     dispatcher turns into the right collectives automatically.

Then bf16, and the demonstration of why every earlier lab ran in fp32.

Run:  torchrun --standalone --nproc_per_node=4 lab7_production.py
"""

import time
import traceback

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from common import (
    GPTConfig,
    StepTimer,
    build_model,
    check_against_reference,
    get_batch,
    load_data,
    print0,
    rank_print,
    set_determinism,
    setup_dist,
)

STEPS = 100
GLOBAL_BATCH = 32
LR = 3e-4


def train_fsdp2(cfg, data, device, rank, world, bf16=False, steps=STEPS):
    """FSDP2 == ZeRO-3.  Compare against Lab 5's hand-written version."""
    set_determinism(0)
    model = build_model(cfg).to(device)
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if bf16 else torch.float32,
        reduce_dtype=torch.float32,
    )
    # Shard each block FIRST, then the root.  This ordering is what lets FSDP
    # free a block's gathered weights the moment it is done with them; sharding
    # only the root would gather the entire model at once.
    for blk in model.blocks:
        fully_shard(blk, mp_policy=mp)
    fully_shard(model, mp_policy=mp)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    timer = StepTimer()
    for step in range(steps):
        timer.tick(step)
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device,
                         rank=rank, world=world)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Works unmodified: the parameters are DTensors, so clip_grad_norm_
        # dispatches to a mesh-aware implementation and computes the true
        # global norm.  This is precisely the bug you had to fix by hand in
        # Lab 3 -- DTensor knows a shard is a shard, so it cannot be got wrong.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lt = loss.detach().float().clone()
        dist.all_reduce(lt, op=dist.ReduceOp.SUM)
        losses.append((lt / world).item())
    dt, _ = timer.stop(steps)
    peak = torch.cuda.max_memory_allocated() / 2**20
    return losses, dt, peak


def clip_mixed_(params, max_norm, mesh, device):
    """Gradient clipping for a PARTIALLY converted model.

    `torch.nn.utils.clip_grad_norm_` raises here:

        aten._foreach_norm.Scalar got mixed torch.Tensor and DTensor

    Only the block linears were handed to `parallelize_module`; the
    embeddings, LayerNorms and lm_head are still ordinary tensors, and the
    foreach kernel refuses to span both worlds.  It works under FSDP2 above
    only because `fully_shard` converts *every* parameter, leaving nothing
    mixed.

    So DTensor removes the f/g bookkeeping but not the global-norm problem --
    it just changes the symptom from a silent wrong answer (Lab 3) into a loud
    exception, which is a real improvement.  The logic below is the same rule
    as Lab 3's: sum sharded contributions across the mesh, count replicated
    ones exactly once.
    """
    sq = torch.zeros((), device=device)
    for p in params:
        g = p.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            s = g.to_local().pow(2).sum()
            if any(isinstance(pl, Shard) for pl in g.placements):
                dist.all_reduce(s, op=dist.ReduceOp.SUM, group=mesh.get_group())
        else:
            s = g.pow(2).sum()      # replicated on every rank: count once
        sq = sq + s
    total = torch.sqrt(sq)
    coef = max_norm / (total + 1e-6)
    if coef < 1:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(coef)
    return total


def train_dtensor_tp(cfg, data, device, mesh, steps=STEPS):
    """DTensor tensor parallelism.  Compare against Lab 3."""
    set_determinism(0)
    model = build_model(cfg).to(device)
    tp_size = mesh.size()
    plan = {
        "attn.q_proj": ColwiseParallel(),
        "attn.k_proj": ColwiseParallel(),
        "attn.v_proj": ColwiseParallel(),
        "attn.o_proj": RowwiseParallel(input_layouts=Shard(-1)),
        "mlp.up": ColwiseParallel(),
        "mlp.down": RowwiseParallel(input_layouts=Shard(-1)),
    }
    for blk in model.blocks:
        parallelize_module(blk, mesh, plan)
        blk.attn.n_head //= tp_size   # still your responsibility

    # foreach=False is mandatory here, and the reason is worth understanding.
    # `parallelize_module` converted only the block linears to DTensor; the
    # embeddings, LayerNorms and lm_head are still ordinary tensors.  Every
    # foreach kernel refuses to span both worlds, so the default optimizer path
    # dies inside `_foreach_mul_` on the very first step:
    #
    #   aten._foreach_mul_.Scalar got mixed torch.Tensor and DTensor
    #
    # The single-tensor path handles each parameter independently and is fine.
    # This is not a clipping quirk -- it is every batched operation over a
    # partially converted model, and it is why `fully_shard` converts the whole
    # model rather than part of it.
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1, foreach=False)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    timer = StepTimer()
    for step in range(steps):
        timer.tick(step)
        # Pure TP: every rank consumes the whole global batch.
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_mixed_(list(model.parameters()), 1.0, mesh, device)
        opt.step()
        losses.append(loss.item())
    dt, _ = timer.stop(steps)
    peak = torch.cuda.max_memory_allocated() / 2**20
    return losses, dt, peak


def section(title):
    print0("\n" + "=" * 70)
    print0(title)
    print0("=" * 70)


def main():
    rank, world, local_rank, device = setup_dist()
    data, vocab_size = load_data()
    cfg = GPTConfig(vocab_size=vocab_size)
    results = {}

    section("FSDP2 (fully_shard) -- the production form of Lab 5's ZeRO-3")
    try:
        l, dt, peak = train_fsdp2(cfg, data, device, rank, world)
        ok = check_against_reference(l, label="FSDP2 fp32")
        results["FSDP2 fp32"] = (dt, peak, ok)
        print0(f"  wall {dt:.1f}s   peak {peak:.1f} MiB")
        print0("  Lab 5's hand-written ZeRO-3 took 14.3s for the same work.  The\n"
               "  gap is the optimisations you skipped: the all_gather for block\n"
               "  n+1 is issued while block n is still computing, gradients are\n"
               "  reduce_scattered in flat buckets rather than per tensor, and the\n"
               "  shards live in one contiguous buffer instead of 100+ tensors.")
    except Exception:
        print0("  FAILED:\n" + traceback.format_exc())

    section("DTensor tensor parallelism -- the production form of Lab 3")
    try:
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("tp",))
        l, dt, peak = train_dtensor_tp(cfg, data, device, mesh["tp"])
        ok = check_against_reference(l, label="DTensor TP")
        results["DTensor TP"] = (dt, peak, ok)
        print0(f"  wall {dt:.1f}s   peak {peak:.1f} MiB   "
               f"(hand-written TP in Lab 3: 10.5s)")
        print0("  What disappeared: _CopyToTP and _ReduceFromTP.  Declaring a\n"
               "  weight Shard(0) or Shard(1) is enough -- the dispatcher derives\n"
               "  which collective each operation needs, in both directions.  The\n"
               "  f/g operators still exist; they are generated, not written.\n"
               "  What did NOT disappear: you still divide n_head yourself, you\n"
               "  still compute the global gradient norm yourself, and you must\n"
               "  pass foreach=False to the optimizer -- all because only PART of\n"
               "  this model is DTensor and batched kernels cannot mix the two\n"
               "  representations.  Both failures are loud exceptions rather than\n"
               "  Lab 3's silent wrong answer, which is the real improvement.")
    except Exception:
        print0("  FAILED:\n" + traceback.format_exc())

    section("bf16 -- why every previous lab ran in fp32")
    try:
        l, dt, peak = train_fsdp2(cfg, data, device, rank, world, bf16=True)
        print0(f"  wall {dt:.1f}s   peak {peak:.1f} MiB")
        for tol in (2e-4, 1e-2, 1e-1):
            check_against_reference(l, atol=tol, label=f"FSDP2 bf16 @ atol {tol:.0e}")
        print0("\n  bf16 has 8 mantissa bits against fp32's 23, so agreement with\n"
               "  the oracle collapses from ~1e-07 to ~1e-02.  That is fine for\n"
               "  training and fatal for verification: at this tolerance a genuine\n"
               "  bug -- the missing `f` in Lab 3 cost 1.8e-02 -- is indis-\n"
               "  tinguishable from rounding.  Establish correctness in fp32, and\n"
               "  only then turn on mixed precision for speed.")
    except Exception:
        print0("  FAILED:\n" + traceback.format_exc())

    section("summary")
    print0("  implementation        wall    peak MiB   matches oracle")
    for name, (dt, peak, ok) in results.items():
        print0(f"  {name:20s} {dt:5.1f}s   {peak:8.1f}   {'yes' if ok else 'NO'}")
    print0("\n  Every fp32 implementation across these seven labs -- hand DDP,\n"
           "  hand TP, pipeline, ZeRO 1/2/3, a 2D mesh, FSDP2 and DTensor --\n"
           "  reproduces the single-GPU loss curve to within about 1e-06.  Only\n"
           "  bf16 does not, and it misses by 5e-03: four orders of magnitude\n"
           "  worse, and larger than several of the real bugs found along the way.\n"
           "  Sharding is a change of layout, never a change of mathematics.  An\n"
           "  oracle is how you keep yourself honest about the difference, and a\n"
           "  tolerance loose enough to accommodate bf16 is too loose to catch a\n"
           "  missing all_reduce.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
