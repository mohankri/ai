"""
Lab 3 -- tensor parallelism by hand.  The core of the course.

Unlike DDP, this splits the matrices *inside* a layer.  Every rank holds a
different slice of every weight matrix and they cooperate on a single forward
pass over the *same* batch.

Everything follows from two adjoint operators:

    f : identity forward, all_reduce backward
    g : all_reduce forward, identity backward

`f` sits in front of a column-parallel layer, `g` behind a row-parallel one.
Pair them (column then row) and a whole MLP costs exactly one all_reduce
forward and one backward, because the wide intermediate activation never has to
be gathered.

Verification is in three escalating stages, which is what makes this lab useful
rather than merely runnable:
  1. logits vs the oracle       -- catches forward-pass errors
  2. gradients vs the oracle    -- catches missing `f`, which forward misses
  3. a 100-step loss curve      -- catches optimizer-level errors

Run:  torchrun --standalone --nproc_per_node=4 lab3_tp.py
"""

import copy
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from common import (
    GPTConfig,
    REFERENCE,
    StepTimer,
    build_model,
    check_against_reference,
    describe_shards,
    get_batch,
    global_shapes_of,
    load_data,
    mem_report,
    print0,
    set_determinism,
    setup_dist,
)

STEPS = 100
GLOBAL_BATCH = 32
LR = 3e-4


# ---------------------------------------------------------------------------
# the two operators everything else is built from
# ---------------------------------------------------------------------------
class _CopyToTP(torch.autograd.Function):
    """f -- identity forward, all_reduce backward.

    Forward is identity because every rank already has the full input.
    Backward must all_reduce because every rank used that same input to produce
    a *different* slice of the output, so the true gradient w.r.t. the input is
    the sum of every rank's contribution.

    Omit this and the forward pass still looks perfect.  Only the gradients are
    wrong, and only for everything *upstream* of the layer.  This is the single
    nastiest bug in tensor parallelism and the reason check #2 below exists.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromTP(torch.autograd.Function):
    """g -- all_reduce forward, identity backward.

    Forward must reduce because each rank computed a partial sum over its slice
    of the contraction dimension.  Backward is identity because every rank
    receives the same incoming gradient and needs only its own slice.
    """

    @staticmethod
    def forward(ctx, x, group):
        # Clone first: all_reduce is in-place, and mutating an autograd input
        # in place is how you get silently corrupted graphs.
        x = x.clone()
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


# ---------------------------------------------------------------------------
# parallel layers
# ---------------------------------------------------------------------------
class ColumnParallelLinear(nn.Module):
    """Split the weight by OUTPUT features.  Input replicated, output sharded."""

    def __init__(self, d_in, d_out, group, bias=True, apply_f=True):
        super().__init__()
        self.group = group
        self.world = dist.get_world_size(group)
        assert d_out % self.world == 0, f"{d_out} not divisible by TP size {self.world}"
        self.apply_f = apply_f  # False only to demonstrate the bug
        self.weight = nn.Parameter(torch.empty(d_out // self.world, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out // self.world)) if bias else None
        # Tag which parameters are shards rather than replicas.  Anything that
        # needs a global reduction later (gradient clipping, norm logging,
        # checkpoint consolidation) needs to know the difference.
        self.weight._tp_sharded = True
        if self.bias is not None:
            self.bias._tp_sharded = True

    def forward(self, x):
        if self.apply_f:
            x = _CopyToTP.apply(x, self.group)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Split the weight by INPUT features.  Input sharded, output reduced."""

    def __init__(self, d_in, d_out, group, bias=True, bias_before_reduce=False):
        super().__init__()
        self.group = group
        self.world = dist.get_world_size(group)
        assert d_in % self.world == 0
        self.bias_before_reduce = bias_before_reduce  # True only to show the bug
        self.weight = nn.Parameter(torch.empty(d_out, d_in // self.world))
        # The bias is NOT sharded: it is a single vector added once to the
        # reduced result, and every rank holds an identical copy.
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.weight._tp_sharded = True
        if self.bias is not None:
            self.bias._tp_sharded = False

    def forward(self, x):
        y = F.linear(x, self.weight)  # partial sum on this rank
        if self.bias_before_reduce and self.bias is not None:
            # THE BUG: adding here means the all_reduce sums the bias `world`
            # times.  Nothing crashes; the model just trains slightly wrong.
            y = y + self.bias
            return _ReduceFromTP.apply(y, self.group)
        y = _ReduceFromTP.apply(y, self.group)
        return y + self.bias if self.bias is not None else y


# ---------------------------------------------------------------------------
# converting a normal model into a tensor-parallel one
# ---------------------------------------------------------------------------
def _to_col(lin, group, rank, world, apply_f=True):
    d_out, d_in = lin.weight.shape
    new = ColumnParallelLinear(d_in, d_out, group,
                               bias=lin.bias is not None, apply_f=apply_f)
    per = d_out // world
    new.weight.data.copy_(lin.weight.data[rank * per:(rank + 1) * per])
    if lin.bias is not None:
        new.bias.data.copy_(lin.bias.data[rank * per:(rank + 1) * per])
    return new


def _to_row(lin, group, rank, world, bias_before_reduce=False):
    d_out, d_in = lin.weight.shape
    new = RowParallelLinear(d_in, d_out, group, bias=lin.bias is not None,
                            bias_before_reduce=bias_before_reduce)
    per = d_in // world
    new.weight.data.copy_(lin.weight.data[:, rank * per:(rank + 1) * per])
    if lin.bias is not None:
        new.bias.data.copy_(lin.bias.data)  # replicated
    return new


def shard_model_(model, group, bug=None):
    """Convert a replicated model into a tensor-parallel one, in place.

    Embeddings and lm_head are left replicated.  Megatron makes lm_head
    column-parallel with a vocab-parallel cross-entropy; with a vocab of 65
    that is all cost and no lesson, so it is skipped here.
    """
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    for blk in model.blocks:
        a = blk.attn
        # q/k/v are column-parallel: each rank ends up owning whole heads.
        a.q_proj = _to_col(a.q_proj, group, rank, world, apply_f=(bug != "no_f"))
        a.k_proj = _to_col(a.k_proj, group, rank, world, apply_f=(bug != "no_f"))
        a.v_proj = _to_col(a.v_proj, group, rank, world, apply_f=(bug != "no_f"))
        # o_proj is row-parallel: its input is already sharded along the head
        # dimension, so no `f` is needed here.
        a.o_proj = _to_row(a.o_proj, group, rank, world,
                           bias_before_reduce=(bug == "bias"))
        # Tell attention it now owns fewer heads.
        #
        # How this bug announces itself depends on how head_dim is obtained,
        # which is worth knowing because the two cases could not be more
        # different:
        #
        #   bug="n_head"        head_dim is a fixed attribute (as here), so
        #                       n_head*head_dim no longer matches the sharded
        #                       width and view() raises.  Loud, easy.
        #
        #   bug="n_head_silent" head_dim is derived from the local width (as it
        #                       is in most real codebases, `hidden // n_head`).
        #                       The reshape then SUCCEEDS with the wrong
        #                       grouping -- 8 heads of dim 16 instead of 2 heads
        #                       of dim 64 -- and you get a plausible-looking
        #                       model that is quietly computing the wrong
        #                       attention.  This is the dangerous one.
        if bug == "n_head":
            pass  # leave n_head alone; head_dim stays 64 -> view() will raise
        elif bug == "n_head_silent":
            a.head_dim = a.q_proj.weight.shape[0] // a.n_head
        else:
            assert a.n_head % world == 0, "TP size must divide the head count"
            a.n_head = a.n_head // world

        blk.mlp.up = _to_col(blk.mlp.up, group, rank, world,
                             apply_f=(bug != "no_f"))
        blk.mlp.down = _to_row(blk.mlp.down, group, rank, world,
                               bias_before_reduce=(bug == "bias"))
    return model


def tp_clip_grad_norm_(model, max_norm, group):
    """Gradient clipping that is correct under tensor parallelism.

    `torch.nn.utils.clip_grad_norm_` is WRONG here, and wrong in a way no
    exception will ever tell you about.  It computes the norm over
    `model.parameters()`, which on each rank is only that rank's *shards*.  The
    resulting norm is too small, so the clip coefficient is too large, so the
    model is under-clipped and takes bigger steps than the oracle.  It trains
    fine.  It just trains a different model.

    The correct global norm has two parts, which must be handled differently:

      sharded parameters    each rank holds a distinct slice, so the sums of
                            squares must be all_reduced across the TP group

      replicated parameters every rank holds an identical copy, so their
                            contribution must be counted exactly ONCE.
                            all_reducing them would inflate the norm by a
                            factor of `world`.
    """
    sharded_sq = torch.zeros((), device=next(model.parameters()).device)
    repl_sq = torch.zeros((), device=sharded_sq.device)
    for p in model.parameters():
        if p.grad is None:
            continue
        sq = p.grad.detach().pow(2).sum()
        if getattr(p, "_tp_sharded", False):
            sharded_sq += sq
        else:
            repl_sq += sq
    dist.all_reduce(sharded_sq, op=dist.ReduceOp.SUM, group=group)
    total = torch.sqrt(sharded_sq + repl_sq)
    coef = max_norm / (total + 1e-6)
    if coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(coef)
    return total


def build_tp_model(cfg, device, group, load_reference=False, bug=None):
    set_determinism(0)
    model = build_model(cfg)
    if load_reference:
        ref = torch.load(REFERENCE, map_location="cpu", weights_only=False)
        model.load_state_dict(ref["state_dict"])
    shard_model_(model, group, bug=bug)
    return model.to(device)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_forward(cfg, data, device, group, bug=None, label=""):
    """Check #1: do the TP logits match the oracle's logits?"""
    ref = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    model = build_tp_model(cfg, device, group, load_reference=True, bug=bug)
    model.eval()
    x, y = get_batch(data, ref["probe_step"], ref["probe_batch"],
                     cfg.block_size, device)
    with torch.no_grad():
        logits, loss = model(x, y)
    ref_logits = ref["ref_logits"].to(device)
    dev = (logits - ref_logits).abs().max().item()
    ok = dev < 1e-4
    print0(f"  [{'PASS' if ok else 'FAIL'}] {label} logits: "
           f"max|delta| = {dev:.3e}   loss {loss.item():.6f} "
           f"vs oracle {ref['ref_loss']:.6f}")
    return ok


def check_gradients(cfg, data, device, group, bug=None, label=""):
    """Check #2: do the TP gradients match the oracle's gradients?

    This is the check that earns its keep.  A missing `f` sails through the
    forward check and fails here, because the error is confined to gradients
    flowing *upstream* of the sharded layers -- i.e. into the embeddings.
    """
    ref = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    rank, world = dist.get_rank(group), dist.get_world_size(group)

    # Reference gradients from the unsharded model.
    set_determinism(0)
    full = build_model(cfg)
    full.load_state_dict(ref["state_dict"])
    full = full.to(device)
    x, y = get_batch(data, ref["probe_step"], ref["probe_batch"],
                     cfg.block_size, device)
    _, loss = full(x, y)
    loss.backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in full.named_parameters()}
    del full

    tp = build_tp_model(cfg, device, group, load_reference=True, bug=bug)
    _, tp_loss = tp(x, y)
    tp_loss.backward()

    worst_name, worst = "", 0.0
    for name, p in tp.named_parameters():
        g_ref = ref_grads[name]
        g_tp = p.grad
        if g_tp.shape != g_ref.shape:
            # Sharded: compare against the matching slice of the full gradient.
            if "o_proj" in name or "mlp.down" in name:  # row-parallel: dim 1
                per = g_ref.shape[1] // world
                g_ref = g_ref[:, rank * per:(rank + 1) * per]
            else:  # column-parallel: dim 0
                per = g_ref.shape[0] // world
                g_ref = g_ref[rank * per:(rank + 1) * per]
        d = (g_tp - g_ref).abs().max().item()
        if d > worst:
            worst, worst_name = d, name
    t = torch.tensor([worst], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    worst = t.item()
    ok = worst < 1e-4
    print0(f"  [{'PASS' if ok else 'FAIL'}] {label} gradients: "
           f"max|delta| = {worst:.3e}  (worst on rank {rank}: {worst_name})")
    return ok


def train_tp(cfg, data, device, group, global_clip=True, verbose=True):
    """Check #3: a full training run must reproduce the oracle's loss curve.

    Note the batching: under pure tensor parallelism there is NO data
    parallelism, so every rank consumes the entire global batch.  Passing
    rank/world here would be wrong.

    `global_clip=False` reverts to the naive local-norm clipping, which is the
    fourth classic bug.
    """
    model = build_tp_model(cfg, device, group)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    losses = []
    model.train()
    torch.cuda.reset_peak_memory_stats()
    timer = StepTimer()
    for step in range(STEPS):
        timer.tick(step)
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if global_clip:
            tp_clip_grad_norm_(model, 1.0, group)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if verbose and (step % 20 == 0 or step == STEPS - 1):
            print0(f"  step {step:3d}  loss {loss.item():.6f}")
    dt, _ = timer.stop(STEPS)
    return losses, dt, model


def main():
    rank, world, local_rank, device = setup_dist()
    group = dist.group.WORLD
    data, vocab_size = load_data()
    cfg = GPTConfig(vocab_size=vocab_size)

    print0(f"\ntensor parallel size {world} (no data parallelism: every rank "
           f"sees the full batch of {GLOBAL_BATCH})")
    print0(f"heads per rank: {cfg.n_head} / {world} = {cfg.n_head // world}\n")

    set_determinism(0)
    full_shapes = global_shapes_of(build_model(cfg))

    print0("=" * 70)
    print0("what actually lives on each GPU")
    print0("=" * 70)
    model = build_tp_model(cfg, device, group)
    describe_shards(model, full_shapes)
    del model

    print0("\n" + "=" * 70)
    print0("check 1 -- forward pass against the oracle")
    print0("=" * 70)
    f_ok = check_forward(cfg, data, device, group, label="correct TP")

    print0("\n" + "=" * 70)
    print0("check 2 -- gradients against the oracle")
    print0("=" * 70)
    g_ok = check_gradients(cfg, data, device, group, label="correct TP")

    print0("\n" + "=" * 70)
    print0("check 3 -- 100-step training run")
    print0("=" * 70)
    losses, dt, model = train_tp(cfg, data, device, group)
    print0(f"\nsteady state {dt:.2f}s over 90 timed steps "
           f"(single GPU: 3.24s) -- TP 4 is SLOWER than one GPU here, because\n"
           f"  two all_reduces per block per step cost more than the tiny\n"
           f"  per-rank matmuls they enable.  TP buys memory, not speed, and on\n"
           f"  a model this small it does not even buy much of that.")
    mem_report("TP")
    t_ok = check_against_reference(losses, label="tensor parallel")
    del model

    # Bug demos come last, so that a bug which raises cannot prevent the three
    # real checks above from reporting.
    print0("\n" + "=" * 70)
    print0("the classic bugs, and how each announces itself")
    print0("=" * 70)

    print0("\n(a) omitting `f`.  Forward is perfect; only gradients are wrong,")
    print0("    and only upstream of the sharded layers -- note it lands on wte:")
    check_forward(cfg, data, device, group, bug="no_f", label="no-f")
    check_gradients(cfg, data, device, group, bug="no_f", label="no-f")

    print0("\n(b) adding the row-parallel bias before the all_reduce, so the")
    print0(f"    all_reduce sums it {world} times:")
    check_forward(cfg, data, device, group, bug="bias", label="bias-early")

    print0("\n(c) forgetting to divide n_head, with head_dim derived from the")
    print0("    local width -- the reshape succeeds and attention is silently")
    print0("    wrong (8 heads of dim 16 instead of 2 heads of dim 64):")
    check_forward(cfg, data, device, group, bug="n_head_silent",
                  label="n_head-silent")

    print0("\n(d) the same mistake with a fixed head_dim, which cannot reshape")
    print0("    and therefore fails loudly.  Strictly the friendlier failure:")
    try:
        check_forward(cfg, data, device, group, bug="n_head", label="n_head")
        print0("       unexpectedly did NOT raise")
    except RuntimeError as e:
        first = str(e).split("\n")[0]
        print0(f"  [RAISED] {first}")

    print0("\n(e) using torch's clip_grad_norm_, which sees only this rank's")
    print0("    shards and so computes a norm that is too small.  Forward and")
    print0("    gradients are both perfect; only the optimizer step is wrong:")
    bad, _, _ = train_tp(cfg, data, device, group, global_clip=False,
                         verbose=False)
    check_against_reference(bad, label="TP with local-norm clipping")
    print0("    Under-clipping means bigger steps, so the loss actually falls")
    print0("    FASTER than the oracle.  A parallelism bug that makes your")
    print0("    training curve look better is the worst kind there is.")

    allok = f_ok and g_ok and t_ok
    print0(f"\nLab 3 {'PASSED' if allok else 'FAILED'}\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
