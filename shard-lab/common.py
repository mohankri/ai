"""
common.py -- shared scaffolding for the sharding labs.

Deliberately explicit: no HuggingFace, no Lightning, no Accelerate.  The whole
point of these labs is that you can read every line that touches a GPU.

Three design decisions worth understanding before you read the code:

1. The model is tiny and hand-written.  You cannot hand-shard someone else's
   attention implementation without fighting it; you can hand-shard this.

2. Everything runs in true fp32.  See `set_determinism` for why TF32 is
   explicitly disabled -- this is the difference between "is my tensor-parallel
   implementation correct?" being checkable and being unanswerable.

3. The batch sampler builds the *global* batch identically regardless of world
   size, then hands each rank its slice.  That is what lets a 4-GPU run be
   compared against the 1-GPU oracle step for step.
"""

from __future__ import annotations

import os

# Must be set before the first CUDA context is created.  Required for
# deterministic cuBLAS reductions.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "input.txt")
OUT_DIR = os.path.join(HERE, "out")
REFERENCE = os.path.join(OUT_DIR, "reference.pt")


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def set_determinism(seed: int = 0) -> None:
    """Make runs reproducible so numerical comparison is meaningful.

    TF32 is disabled on purpose.  On Blackwell a `float32` matmul will happily
    use TF32 internally (10-bit mantissa) unless told not to, costing roughly
    three decimal digits.  That is more than enough to make a *correct*
    tensor-parallel implementation fail an atol=1e-4 check against the oracle,
    which would teach exactly the wrong lesson.

    `warn_only=True` because a couple of backward kernels (embedding
    scatter-add in particular) have no deterministic implementation.  We want a
    warning, not a crash; the residual nondeterminism is far below our
    comparison tolerance.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ---------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world() -> int:
    return dist.get_world_size() if is_dist() else 1


def setup_dist(backend: str = "nccl"):
    """Initialise the process group and bind this process to exactly one GPU.

    Binding the device before `init_process_group` matters: NCCL infers device
    affinity from the current device, and getting it wrong is a classic source
    of hangs.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    device = torch.device("cuda", local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank, device


def rank_print(*args, only=None) -> None:
    """Print from every rank without interleaving.

    The barrier makes this expensive; it is an inspection tool, not something
    to call inside a training loop.

    `only` restricts which ranks actually print.  It exists because the obvious
    alternative -- wrapping the call site in `if rank in (0, 2):` -- deadlocks.
    This function contains a collective, so every rank must call it or the ones
    that skipped it never arrive at the barrier and the job hangs until the
    store times out ten minutes later.  Filter the *output*, never the
    *participation*.
    """
    if not is_dist():
        print(*args, flush=True)
        return
    world, rank = get_world(), get_rank()
    for r in range(world):
        if r == rank and (only is None or rank in only):
            print(f"[rank {rank}]", *args, flush=True)
        dist.barrier()


def print0(*args) -> None:
    """Print once, from rank 0."""
    if get_rank() == 0:
        print(*args, flush=True)


# ---------------------------------------------------------------------------
# inspection -- the heart of "treat each GPU separately"
# ---------------------------------------------------------------------------
def describe_shards(
    model: nn.Module,
    global_shapes: dict | None = None,
    collapse_blocks: bool = True,
    only=None,
) -> None:
    """Show which slice of each parameter actually lives on this GPU.

    Run this in every lab and read it *before* looking at the loss.  If the
    shapes are not what you expected, the loss will not tell you why.

    Transformer blocks are structurally identical, so by default only block 0 is
    printed and the rest are summarised.  Pass `collapse_blocks=False` if you
    suspect a per-layer bug.
    """
    total = 0
    lines = []
    n_blocks = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if name.startswith("blocks."):
            idx = int(name.split(".")[1])
            n_blocks = max(n_blocks, idx + 1)
            if collapse_blocks and idx > 0:
                continue
        local = str(tuple(p.shape))
        if global_shapes is not None and name in global_shapes:
            g = tuple(global_shapes[name])
            marker = "  <-- SHARDED" if tuple(p.shape) != g else ""
            lines.append(f"  {name:32s} local={local:18s} global={str(g):18s}{marker}")
        else:
            lines.append(f"  {name:32s} local={local:18s}")
    if collapse_blocks and n_blocks > 1:
        lines.append(f"  ... blocks.1 through blocks.{n_blocks - 1} are identical "
                     f"in shape to blocks.0")
    body = "\n".join(lines)
    rank_print(f"parameter layout on this GPU:\n{body}\n  local params: {total:,}",
               only=only)


def mem_report(tag: str = "") -> float:
    """Peak allocated memory in MiB.

    Ratios teach; absolute values on a 277 GiB GB300 are meaningless.
    """
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    cur = torch.cuda.memory_allocated() / 2**20
    rank_print(f"{tag:26s} peak={peak:9.2f} MiB  current={cur:9.2f} MiB")
    return peak


def global_shapes_of(model: nn.Module) -> dict:
    return {name: tuple(p.shape) for name, p in model.named_parameters()}


class StepTimer:
    """Wall-clock timing that discards warmup steps.

    The first few steps of any run pay one-off costs that have nothing to do
    with the parallelism being measured: NCCL communicator construction, cuBLAS
    workspace allocation, kernel autotuning, and caching-allocator growth.  On
    this box the very first process to touch a given kernel also pays JIT
    compilation, which cost 19 of the 23.7 seconds originally reported for the
    single-GPU oracle -- its true steady-state time is 4.9s.

    Timing from step 0 therefore penalises whichever implementation happens to
    run FIRST inside a process.  That is not a hypothetical: it is exactly how
    the original Lab 2 measurement made torch DDP look 1.76x faster than the
    hand-written version, when much of the gap was simply NCCL setup being
    charged to whichever ran first.

    Skipping the first `warmup` steps costs nothing -- all steps still execute,
    so the loss curve and the oracle comparison are unaffected.
    """

    def __init__(self, warmup: int = 10):
        self.warmup = warmup
        self.t0 = None

    def tick(self, step: int) -> None:
        """Call at the top of every step."""
        if step == self.warmup:
            torch.cuda.synchronize()
            if is_dist():
                dist.barrier()
            self.t0 = time.time()

    def stop(self, total_steps: int):
        """Returns (seconds, steps_per_second) over the timed region."""
        torch.cuda.synchronize()
        if self.t0 is None:      # fewer steps than the warmup window
            return float("nan"), float("nan")
        dt = time.time() - self.t0
        n = total_steps - self.warmup
        return dt, n / dt if dt > 0 else float("nan")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 65
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    block_size: int = 256

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


class CausalSelfAttention(nn.Module):
    """Attention with *separate* q/k/v projections.

    A fused qkv projection is faster, but its weight layout is interleaved,
    which makes column-sharding it in Lab 3 an exercise in index arithmetic
    rather than an exercise in understanding tensor parallelism.  Separate
    projections shard trivially: q/k/v column-parallel, o row-parallel.

    Attention is written out by hand rather than calling
    `scaled_dot_product_attention` so that the head dimension -- the axis we
    shard along -- is visible in the code.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # Instance attributes, not config lookups, so Lab 3 can divide n_head
        # by the TP size after sharding the projections.
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        B, T, _ = x.shape

        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.down = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Pre-norm GPT.  `lm_head` is deliberately *not* tied to `wte`.

    Weight tying would force the embedding and the output projection to be
    sharded compatibly, and would make gradients subtly wrong if you got the
    tie back to front under tensor parallelism.  That is a real problem, but it
    is not the problem these labs are teaching.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def build_model(cfg: GPTConfig | None = None, seed: int = 0) -> GPT:
    """Build on CPU with a fixed seed so every rank starts from identical
    weights before any sharding happens."""
    cfg = cfg or GPTConfig()
    torch.manual_seed(seed)
    return GPT(cfg)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, len(chars)


def get_batch(
    data: torch.Tensor,
    step: int,
    global_batch: int,
    block_size: int,
    device,
    rank: int = 0,
    world: int = 1,
    seed: int = 1234,
):
    """Deterministic sampling of a *global* batch, then slice for this rank.

    This is the trick that makes the reference-oracle discipline work.  The
    sequence of global batches depends only on `step`, never on how many GPUs
    are running, so a 4-GPU data-parallel run consumes exactly the same tokens
    in exactly the same order as the 1-GPU oracle.  Any divergence in the loss
    curve is therefore a bug in your parallelism, not a difference in data.
    """
    g = torch.Generator().manual_seed(seed + step)
    ix = torch.randint(len(data) - block_size - 1, (global_batch,), generator=g)
    assert global_batch % world == 0, "global batch must divide evenly across ranks"
    per = global_batch // world
    ix = ix[rank * per : (rank + 1) * per]
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# the oracle check
# ---------------------------------------------------------------------------
def check_against_reference(losses, atol: float = 2e-4, label: str = "") -> bool:
    """Compare a loss curve against Lab 0.  This is the pass/fail gate."""
    ref = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    a = torch.tensor(ref["losses"], dtype=torch.float64)
    b = torch.tensor(losses, dtype=torch.float64)
    n = min(len(a), len(b))
    dev = (a[:n] - b[:n]).abs()
    worst, at = dev.max().item(), int(dev.argmax())
    ok = worst < atol
    verdict = "PASS" if ok else "FAIL"
    print0(
        f"\n  [{verdict}] {label}: max |loss - oracle| = {worst:.3e} "
        f"at step {at} (tol {atol:.1e}, {n} steps compared)"
    )
    if not ok:
        print0(f"         oracle[{at}]={a[at]:.6f}  got[{at}]={b[at]:.6f}")
    return ok
