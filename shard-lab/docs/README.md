# Sharding lab

Build every major model-sharding strategy from scratch on one GB300 node, one
process per GPU, verifying each against a single-GPU reference.

The output is understanding, not a trained model. Production tools (FSDP2,
DTensor) are deliberately deferred to the last lab so you meet them *after* you
can already implement what they do.

## The one idea that makes this work

Lab 0 trains the model on a single GPU and saves the loss at every step.
Every later lab must reproduce that curve to ~1e-06.

This matters because a wrong implementation also trains, and its loss also
goes down. Without an oracle, "it runs and the loss decreases" is not evidence
of anything. Five real bugs were found during development and **none of them
raised an exception** — every one was caught by the oracle disagreeing.

## Labs

| # | File | Topic |
|---|------|-------|
| 0 | [lab0-oracle.md](lab0-oracle.md) | The single-GPU reference everything is judged against |
| 1 | [lab1-collectives.md](lab1-collectives.md) | Collectives in isolation; the `reduce_scatter + all_gather` identity |
| 2 | [lab2-ddp.md](lab2-ddp.md) | Data parallel by hand |
| 3 | [lab3-tensor-parallel.md](lab3-tensor-parallel.md) | Tensor parallel by hand — the core of the course |
| 4 | [lab4-pipeline-parallel.md](lab4-pipeline-parallel.md) | Pipeline parallel, and measuring the bubble |
| 5 | [lab5-zero.md](lab5-zero.md) | ZeRO stages 1, 2 and 3 by hand |
| 6 | [lab6-2d-mesh.md](lab6-2d-mesh.md) | Composing TP and DP on a 2D mesh |
| 7 | [lab7-production.md](lab7-production.md) | FSDP2 and DTensor, for comparison |

[RESULTS.md](RESULTS.md) collects every measurement, every bug found, and the
corrections made to the original plan.

## Setup

Everything runs on the GPU box over SSH. The local machine is macOS and never
runs any of this.

```bash
ssh lambda-gpu
cd ~/shard-lab
```

The environment already exists. To rebuild it from scratch:

```bash
# uv, because there is no Docker, no system pip, and no python3-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

cd ~/shard-lab
uv venv --python 3.12 .venv
# UV_NO_CACHE because / had only ~12 GB free
UV_NO_CACHE=1 uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0

mkdir -p data out
curl -sL -o data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Verify the GPUs before anything else:

```bash
.venv/bin/python -c "
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
print('capability', torch.cuda.get_device_capability(0))
print('arch_list', torch.cuda.get_arch_list())
print('p2p 0->1', torch.cuda.can_device_access_peer(0,1))
"
```

Expected: `2.13.0+cu130 13.0 4`, capability `(10, 3)`, P2P `True`.

Note that `arch_list` contains `sm_100`/`sm_110`/`sm_120` but **not** `sm_103`.
It works anyway through CUDA 13 minor-version compatibility. Check this on
Blackwell Ultra rather than assuming your wheel targets your exact arch.

## Running

Lab 0 is single-process. Everything else uses all four GPUs.

```bash
cd ~/shard-lab
.venv/bin/python lab0_reference.py                              # must run first
.venv/bin/torchrun --standalone --nproc_per_node=4 lab1_collectives.py
.venv/bin/torchrun --standalone --nproc_per_node=4 lab2_ddp.py
# ... and so on through lab7_production.py
```

Output is noisy. This filter keeps the parts that matter:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 lab3_tp.py 2>&1 \
  | grep -vE "NumPy|_conversion_method_template|distributed/run.py|c10d_logger|ProcessGroupNCCL"
```

To check everything at once:

```bash
for f in lab1_collectives lab2_ddp lab3_tp lab4_pp lab5_zero lab6_2d lab7_production; do
  echo "##### $f"
  .venv/bin/torchrun --standalone --nproc_per_node=4 $f.py 2>&1 \
    | grep -E "^\s*\[(PASS|FAIL)\]|^Lab [0-9] (PASSED|FAILED)"
done
```

Some `[FAIL]` lines are **expected** — they are the deliberate bug
demonstrations. Each lab prints its own `Lab N PASSED` verdict, which accounts
for this. Trust that line.

## Debugging

| Symptom | Setting |
|---|---|
| Shrink the world to isolate a bug | `CUDA_VISIBLE_DEVICES=0,1` with `--nproc_per_node=2` |
| Mismatched collective shapes, desynced ranks | `TORCH_DISTRIBUTED_DEBUG=DETAIL` |
| Which transport NCCL chose | `NCCL_DEBUG=INFO` |
| Turn a silent hang into an error | `TORCH_NCCL_BLOCKING_WAIT=1` |

**The most common failure is a hang, not a crash.** One rank calls a collective
the others do not — almost always because someone wrapped it in `if rank == 0:`
or hit an early `continue` in the data loop. This happened during development:
`if rank in (0, 2): describe_shards(...)` hung two ranks for the full
ten-minute store timeout, because `describe_shards` contains a barrier.

Filter the *output*, never the *participation*.

## A warning about performance numbers

Four GB300s buy essentially nothing at this model size. Only torch DDP beats a
single GPU, and only by 3%. Everything else is slower.

That is not a bug. At 25M parameters with a global batch of 32, per-rank
compute is far too small to hide collective latency. These labs demonstrate
**correctness**, not speedup. See [RESULTS.md](RESULTS.md) for the full table
and for why the first round of measurements was wrong.
