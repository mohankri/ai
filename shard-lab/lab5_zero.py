"""
Lab 5 -- ZeRO by hand.

DDP replicates everything.  For a model trained with AdamW in fp32 the per
-parameter cost is about 16 bytes: 4 for the parameter, 4 for its gradient, and
8 for Adam's two moments.  ZeRO removes the replication one tier at a time:

    stage 1   shard the optimizer state          (8 of the 16 bytes)
    stage 2   also shard the gradients           (4 more)
    stage 3   also shard the parameters          (the last 4)

Stage 3 is FSDP.  Having written it you will understand why per-block wrapping
matters, and why "reshard after forward" is inseparable from recomputation.

Rather than relying on peak allocator readings -- which on a 277 GiB GB300 are
dominated by activations and noise -- this lab accounts for the bytes of
persistent training state directly.  That is the number ZeRO is designed to
reduce, and the ratios are exact.

Run:  torchrun --standalone --nproc_per_node=4 lab5_zero.py
"""

import time

import torch
import torch.distributed as dist
import torch.nn as nn

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
ADAMW = dict(lr=LR, betas=(0.9, 0.95), weight_decay=0.1)


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------
def state_bytes(params, opt, grads_held):
    """Bytes of persistent training state on THIS rank."""
    p_bytes = sum(p.numel() * p.element_size() for p in params)
    g_bytes = sum(g.numel() * g.element_size() for g in grads_held)
    o_bytes = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                o_bytes += v.numel() * v.element_size()
    return p_bytes, g_bytes, o_bytes


def report(tag, p, g, o):
    mib = lambda b: b / 2**20
    print0(f"  {tag:22s} params {mib(p):7.2f}  grads {mib(g):7.2f}  "
           f"optim {mib(o):7.2f}  TOTAL {mib(p + g + o):8.2f} MiB")
    return p + g + o


def plan_partition(params, world, mode="balanced"):
    """Decide which rank owns which parameter.

    The obvious scheme -- round-robin by index -- balances the *number* of
    tensors, not their size, which is useless here: this model's parameters run
    from 512 floats (a LayerNorm bias) to 1,048,576 (an MLP projection).
    Round-robin hands some ranks about twice their fair share of bytes, so the
    saving you measure is well short of what the arithmetic promises.

    Greedy longest-processing-time assignment -- walk the tensors largest first,
    give each to whichever rank is currently lightest -- lands within a few
    percent of even.  Production implementations dodge the problem entirely by
    flattening every parameter into one buffer and splitting that, which is
    balanced by construction but makes ownership much harder to see.
    """
    if mode == "roundrobin":
        return [i % world for i in range(len(params))]
    load = [0] * world
    owners = [0] * len(params)
    for i in sorted(range(len(params)), key=lambda j: -params[j].numel()):
        r = min(range(world), key=lambda k: load[k])
        owners[i] = r
        load[r] += params[i].numel()
    return owners


def partition_imbalance(params, owners, world):
    """Ratio of the heaviest rank's share to the ideal even share."""
    load = [0] * world
    for p, o in zip(params, owners):
        load[o] += p.numel() * p.element_size()
    ideal = sum(load) / world
    return max(load) / ideal, [b / 2**20 for b in load]


# ---------------------------------------------------------------------------
# baseline: plain DDP, nothing sharded
# ---------------------------------------------------------------------------
def train_ddp(cfg, data, device, rank, world, steps=STEPS):
    set_determinism(0)
    model = build_model(cfg).to(device)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, **ADAMW)
    losses = []
    timer = StepTimer()
    for step in range(steps):
        timer.tick(step)
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device,
                         rank=rank, world=world)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in params:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        lt = loss.detach().clone()
        dist.all_reduce(lt, op=dist.ReduceOp.SUM)
        losses.append((lt / world).item())
    wall, _ = timer.stop(steps)
    p, g, o = state_bytes(params, opt, [p.grad for p in params if p.grad is not None])
    return losses, wall, (p, g, o)


# ---------------------------------------------------------------------------
# ZeRO stages 1 and 2
# ---------------------------------------------------------------------------
def train_zero12(cfg, data, device, rank, world, stage, steps=STEPS,
                 mode="balanced"):
    """Stage 1 and stage 2 differ by how gradients are reduced, and by whether
    non-owned gradients are kept.

    stage 1   all_reduce, keep everything -- every rank still holds every
              gradient.  Only the optimizer state is sharded.
    stage 2   reduce to the owner, then drop what you do not own, so non-owners
              stop holding those gradients at all.

    Both step only their own slice and broadcast the result, so the weights are
    identical to DDP's either way.

    Note what stage 2 does NOT fix here: the gradient is still *allocated* in
    full by autograd before being reduced away, so the transient peak barely
    moves.  Only the steady-state holding drops.  Production ZeRO-2
    reduce_scatters into a preallocated flat bucket via backward hooks, so the
    full gradient is never materialised in the first place -- that is what
    turns this accounting win into a real peak-memory win.
    """
    set_determinism(0)
    model = build_model(cfg).to(device)
    params = list(model.parameters())
    owners = plan_partition(params, world, mode)
    mine = [p for i, p in enumerate(params) if owners[i] == rank]
    opt = torch.optim.AdamW(mine, **ADAMW)

    losses = []
    timer = StepTimer()
    for step in range(steps):
        timer.tick(step)
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device,
                         rank=rank, world=world)
        _, loss = model(x, y)
        # model.zero_grad, NOT opt.zero_grad.  The optimizer only knows about
        # the parameters this rank owns, so opt.zero_grad() leaves every other
        # gradient untouched and they accumulate step after step.  In stage 2
        # you get away with it because non-owned grads are set to None anyway;
        # in stage 1 they are kept, quietly grow without bound, and are then
        # fed into the all_reduce -- which reaches inf and then NaN around step
        # 80.  A subset-owning optimizer and zero_grad() are a bad pairing.
        model.zero_grad(set_to_none=True)
        loss.backward()

        for i, p in enumerate(params):
            own = owners[i]
            if stage == 1:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            else:
                dist.reduce(p.grad, dst=own, op=dist.ReduceOp.SUM)
            if own == rank:
                p.grad /= world
            elif stage >= 2:
                p.grad = None   # stage 1 deliberately keeps it

        # Gradient clipping must use the GLOBAL norm.  Each rank now holds a
        # disjoint subset of gradients, so a SUM all_reduce is exactly right.
        sq = torch.zeros((), device=device)
        for p in mine:
            if p.grad is not None:
                sq += p.grad.detach().pow(2).sum()
        dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        coef = 1.0 / (torch.sqrt(sq) + 1e-6)
        if coef < 1:
            for p in mine:
                if p.grad is not None:
                    p.grad.detach().mul_(coef)

        opt.step()

        # Publish the updated slices so every rank has whole weights again.
        for i, p in enumerate(params):
            dist.broadcast(p.data, src=owners[i])

        lt = loss.detach().clone()
        dist.all_reduce(lt, op=dist.ReduceOp.SUM)
        losses.append((lt / world).item())
    wall, _ = timer.stop(steps)
    held = [p.grad for p in params if p.grad is not None]
    p_b, g_b, o_b = state_bytes(params, opt, held)
    ratio, _ = partition_imbalance(params, owners, world)
    return losses, wall, (p_b, g_b, o_b), ratio


# ---------------------------------------------------------------------------
# ZeRO stage 3 == FSDP
# ---------------------------------------------------------------------------
def _pad_numel(n, world):
    return n + (-n) % world


class _AllGatherParam(torch.autograd.Function):
    """Materialise a full parameter from shards; reduce_scatter its gradient.

    This single autograd function is the whole of FSDP's parameter handling.
    Forward all_gathers the shards into the full weight; backward takes the
    gradient of that full weight and reduce_scatters it, so each rank comes
    away holding the gradient for precisely the slice it owns -- never the
    whole thing.
    """

    @staticmethod
    def forward(ctx, shard, group, numel, shape):
        ctx.group = group
        ctx.shard_numel = shard.numel()
        world = dist.get_world_size(group)
        flat = torch.empty(world * shard.numel(), device=shard.device,
                           dtype=shard.dtype)
        dist.all_gather_into_tensor(flat, shard.contiguous(), group=group)
        return flat[:numel].view(shape)

    @staticmethod
    def backward(ctx, grad):
        world = dist.get_world_size(ctx.group)
        flat = grad.reshape(-1)
        want = ctx.shard_numel * world
        if flat.numel() < want:
            flat = torch.cat([flat, flat.new_zeros(want - flat.numel())])
        out = torch.empty(ctx.shard_numel, device=grad.device, dtype=grad.dtype)
        dist.reduce_scatter_tensor(out, flat.contiguous(), group=ctx.group)
        return out / world, None, None, None


class ShardedBlock(nn.Module):
    """A transformer block whose parameters live only as 1/world shards.

    `reshard_after_forward` is the interesting knob.  If the gathered weights
    stay referenced by the autograd graph they cannot be freed until backward
    finishes, so every block's full weights are alive at once and you have
    saved nothing -- this is ZeRO stage 2 wearing a stage 3 costume.  Wrapping
    the block in activation checkpointing means the forward graph is discarded
    and rebuilt during backward, so the gathered weights really are freed and
    really are re-gathered.  Resharding and recomputation are the same
    decision.
    """

    def __init__(self, block, group, device, reshard_after_forward=True):
        super().__init__()
        self.group = group
        self.reshard = reshard_after_forward
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        self.meta = {}
        shards = {}
        for name, p in list(block.named_parameters()):
            flat = p.detach().reshape(-1)
            padded = _pad_numel(flat.numel(), world)
            if flat.numel() < padded:
                flat = torch.cat([flat, flat.new_zeros(padded - flat.numel())])
            chunk = padded // world
            key = name.replace(".", "__")
            shards[key] = nn.Parameter(
                flat[rank * chunk:(rank + 1) * chunk].clone().to(device))
            self.meta[name] = (key, p.numel(), tuple(p.shape))
        self.shards = nn.ParameterDict(shards)
        # Detach the real parameters: the module keeps its structure and its
        # buffers (the causal mask), but owns no parameter storage.
        for mod in block.modules():
            for pname in list(mod._parameters.keys()):
                mod._parameters[pname] = None
        self.block = block.to(device)

    def _gather(self):
        return {
            name: _AllGatherParam.apply(self.shards[key], self.group, numel, shape)
            for name, (key, numel, shape) in self.meta.items()
        }

    def _run(self, x):
        return torch.func.functional_call(self.block, self._gather(), (x,))

    def forward(self, x):
        if self.reshard:
            return torch.utils.checkpoint.checkpoint(self._run, x,
                                                     use_reentrant=False)
        return self._run(x)


def train_zero3(cfg, data, device, rank, world, reshard=True, steps=STEPS):
    set_determinism(0)
    model = build_model(cfg).to(device)
    group = dist.group.WORLD
    model.blocks = nn.ModuleList(
        [ShardedBlock(b, group, device, reshard) for b in model.blocks]
    )
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, **ADAMW)

    losses = []
    timer = StepTimer()
    for step in range(steps):
        timer.tick(step)
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device,
                         rank=rank, world=world)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Block shards already carry reduce_scattered (mean) gradients.  The
        # embeddings, norms and lm_head were never sharded, so they still need
        # an ordinary DDP all_reduce.
        for name, p in model.named_parameters():
            if ".shards." not in name and p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world
        sq = torch.zeros((), device=device)
        repl = torch.zeros((), device=device)
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            (sq if ".shards." in name else repl).add_(p.grad.detach().pow(2).sum())
        dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        coef = 1.0 / (torch.sqrt(sq + repl) + 1e-6)
        if coef < 1:
            for p in params:
                if p.grad is not None:
                    p.grad.detach().mul_(coef)
        opt.step()
        lt = loss.detach().clone()
        dist.all_reduce(lt, op=dist.ReduceOp.SUM)
        losses.append((lt / world).item())
    wall, _ = timer.stop(steps)
    p_b, g_b, o_b = state_bytes(params, opt,
                                [p.grad for p in params if p.grad is not None])
    return losses, wall, (p_b, g_b, o_b)


def main():
    rank, world, local_rank, device = setup_dist()
    data, vocab_size = load_data()
    cfg = GPTConfig(vocab_size=vocab_size)
    print0(f"\nworld {world}, global batch {GLOBAL_BATCH}, fp32 AdamW\n")
    print0("persistent training state per rank (lower is better):")

    l0, w0, s0 = train_ddp(cfg, data, device, rank, world)
    base = report("DDP (baseline)", *s0)
    ok0 = check_against_reference(l0, label="DDP baseline")

    l1, w1, s1, imb = train_zero12(cfg, data, device, rank, world, stage=1)
    t1 = report("ZeRO-1 (optim)", *s1)
    ok1 = check_against_reference(l1, label="ZeRO-1")

    l2, w2, s2, _ = train_zero12(cfg, data, device, rank, world, stage=2)
    t2 = report("ZeRO-2 (+grads)", *s2)
    ok2 = check_against_reference(l2, label="ZeRO-2")

    l3, w3, s3 = train_zero3(cfg, data, device, rank, world, reshard=True)
    t3 = report("ZeRO-3 (+params)", *s3)
    ok3 = check_against_reference(l3, label="ZeRO-3", atol=5e-4)

    ideal = {"ZeRO-1": (4 + 4 + 8 / world), "ZeRO-2": (4 + (4 + 8) / world),
             "ZeRO-3": (4 + 4 + 8) / world}
    per_p = 16.0
    print0(f"\n  reduction vs DDP:  ZeRO-1 {base/t1:.2f}x (theory "
           f"{per_p/ideal['ZeRO-1']:.2f}x)   ZeRO-2 {base/t2:.2f}x (theory "
           f"{per_p/ideal['ZeRO-2']:.2f}x)   ZeRO-3 {base/t3:.2f}x (theory "
           f"{per_p/ideal['ZeRO-3']:.2f}x)")
    print0(f"  wall clock:        DDP {w0:.1f}s  Z1 {w1:.1f}s  "
           f"Z2 {w2:.1f}s  Z3 {w3:.1f}s")

    print0("\n" + "=" * 70)
    print0("partition balance -- why ownership scheme matters")
    print0("=" * 70)
    set_determinism(0)
    probe = list(build_model(cfg).parameters())
    for mode in ("roundrobin", "balanced"):
        owners = plan_partition(probe, world, mode)
        ratio, loads = partition_imbalance(probe, owners, world)
        print0(f"  {mode:12s} heaviest/ideal = {ratio:.2f}x   "
               f"per-rank MiB: {[f'{m:.1f}' for m in loads]}")
    print0("  Parameter sizes here span 512 to 1,048,576 elements, so balancing\n"
           "  tensor COUNT balances nothing.  Sharding a flat buffer instead of\n"
           "  whole tensors makes this failure mode impossible, which is why\n"
           "  every production implementation does exactly that.")

    print0("\n" + "=" * 70)
    print0("why resharding needs recomputation")
    print0("=" * 70)
    torch.cuda.reset_peak_memory_stats()
    _, _, _ = train_zero3(cfg, data, device, rank, world, reshard=True, steps=10)
    peak_on = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    _, _, _ = train_zero3(cfg, data, device, rank, world, reshard=False, steps=10)
    peak_off = torch.cuda.max_memory_allocated() / 2**20
    rank_print(f"reshard-after-forward ON : peak {peak_on:8.1f} MiB")
    rank_print(f"reshard-after-forward OFF: peak {peak_off:8.1f} MiB")
    print0("  With resharding off, every block's gathered weights stay pinned by\n"
           "  the autograd graph until backward ends, so nothing is saved -- it is\n"
           "  ZeRO-2 wearing a stage-3 costume.  Freeing them means the forward\n"
           "  graph must be discarded and rebuilt, which is exactly activation\n"
           "  checkpointing.  Resharding and recomputation are one decision.")

    allok = ok0 and ok1 and ok2 and ok3
    print0(f"\nLab 5 {'PASSED' if allok else 'FAILED'}\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
