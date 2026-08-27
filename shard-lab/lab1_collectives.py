"""
Lab 1 -- collectives in isolation.

No model, no autograd.  Just small labelled tensors so you can see exactly what
each collective does to them.

Before reading each result, predict it.  The one that matters most is the last
check: reduce_scatter followed by all_gather equals all_reduce.  That identity
is the entire reason ZeRO-2 can use less memory than DDP without sending more
bytes, and once you have seen it numerically the later labs stop being
mysterious.

Run:  torchrun --standalone --nproc_per_node=4 lab1_collectives.py
"""

import torch
import torch.distributed as dist

from common import print0, rank_print, setup_dist


def banner(title):
    print0(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    rank, world, local_rank, device = setup_dist()

    # ---------------------------------------------------------------- broadcast
    banner("broadcast -- rank 0's value overwrites everyone else's")
    t = torch.full((4,), float(rank), device=device)
    rank_print(f"before: {t.tolist()}")
    dist.broadcast(t, src=0)
    rank_print(f"after : {t.tolist()}")
    print0("  -> this is how you guarantee identical initial weights in DDP")

    # --------------------------------------------------------------- all_reduce
    banner("all_reduce(SUM) -- every rank ends up with the total")
    t = torch.full((4,), float(rank + 1), device=device)
    rank_print(f"before: {t.tolist()}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(r + 1 for r in range(world))
    rank_print(f"after : {t.tolist()}   (expected all {expected})")
    print0("  -> this is the gradient synchronisation in DDP")

    # ----------------------------------------------------------- reduce_scatter
    banner("reduce_scatter -- reduce, then keep only YOUR slice")
    full = torch.arange(world * 2, dtype=torch.float32, device=device) + rank * 100
    out = torch.empty(2, device=device)
    rank_print(f"input : {full.tolist()}")
    dist.reduce_scatter_tensor(out, full, op=dist.ReduceOp.SUM)
    rank_print(f"output: {out.tolist()}  <- slice {rank} of the reduced vector")
    print0("  -> each rank holds 1/world of the answer, using 1/world the memory")

    # -------------------------------------------------------------- all_gather
    banner("all_gather -- reassemble the slices")
    gathered = torch.empty(world * 2, device=device)
    dist.all_gather_into_tensor(gathered, out)
    rank_print(f"gathered: {gathered.tolist()}")
    print0("  -> this is how FSDP materialises a layer's weights before forward")

    # ----------------------------------------------------------- THE IDENTITY
    banner("THE IDENTITY:  reduce_scatter + all_gather  ==  all_reduce")
    src = torch.arange(world * 2, dtype=torch.float32, device=device) + rank * 100

    via_allreduce = src.clone()
    dist.all_reduce(via_allreduce, op=dist.ReduceOp.SUM)

    shard = torch.empty(2, device=device)
    dist.reduce_scatter_tensor(shard, src.clone(), op=dist.ReduceOp.SUM)
    via_two_step = torch.empty(world * 2, device=device)
    dist.all_gather_into_tensor(via_two_step, shard)

    same = torch.equal(via_allreduce, via_two_step)
    rank_print(f"all_reduce      : {via_allreduce.tolist()}")
    rank_print(f"RS then AG      : {via_two_step.tolist()}")
    rank_print(f"identical       : {same}")
    assert same, "the identity must hold exactly -- both are the same sum order"

    S = src.numel() * src.element_size()
    ring = 2 * (world - 1) / world * S
    half = (world - 1) / world * S
    print0(
        f"\n  ring all_reduce moves    {ring:8.1f} B/rank\n"
        f"  reduce_scatter moves     {half:8.1f} B/rank\n"
        f"  all_gather moves         {half:8.1f} B/rank\n"
        f"  -> an all_reduce IS a reduce_scatter followed by an all_gather.\n"
        f"     ZeRO-2 simply stops after the reduce_scatter, because each rank\n"
        f"     only needs gradients for the parameters it owns.  Same bytes on\n"
        f"     the wire, less memory held."
    )

    # --------------------------------------------------------------- all_to_all
    banner("all_to_all -- transpose across ranks (this is MoE routing)")
    send = torch.arange(world, dtype=torch.float32, device=device) + rank * 10
    recv = torch.empty(world, device=device)
    rank_print(f"send: {send.tolist()}  (element i goes to rank i)")
    dist.all_to_all_single(recv, send)
    rank_print(f"recv: {recv.tolist()}  (element i came from rank i)")

    # --------------------------------------------------------------- send/recv
    banner("send/recv -- point to point, the basis of pipeline parallelism")
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    payload = torch.full((3,), float(rank), device=device)
    got = torch.empty(3, device=device)
    # Even ranks send first, odd ranks receive first.  If everyone called send()
    # simultaneously with a large enough payload this would deadlock -- NCCL
    # send is not guaranteed to buffer.
    if rank % 2 == 0:
        dist.send(payload, dst=nxt)
        dist.recv(got, src=prv)
    else:
        dist.recv(got, src=prv)
        dist.send(payload, dst=nxt)
    rank_print(f"sent {rank} -> rank {nxt};  received {got.tolist()} from rank {prv}")

    print0("\nall collectives behaved as expected.\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
