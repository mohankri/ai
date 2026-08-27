# Lab 0 — The oracle

**File:** `lab0_reference.py`, `common.py`
**Run:** `.venv/bin/python lab0_reference.py`
**Time:** ~10 seconds

The most important lab and the least interesting one. It trains the model on
one GPU and saves the loss at every step to `out/reference.pt`. Every later lab
is judged against that file.

## Why bother

A broken tensor-parallel implementation still trains. Its loss still goes down.
It just converges somewhere slightly different. Without something to compare
against, you cannot tell a correct implementation from a subtly wrong one —
and the wrong ones do not raise exceptions.

Every single bug found while building these labs was caught this way.

## Run it

```bash
cd ~/shard-lab
.venv/bin/python lab0_reference.py
```

## Expected output

```
model: 25.4M params, 8 blocks, d_model=512, 8 heads, block_size=256
data:  1,115,394 tokens, vocab 65
train: 100 steps, global batch 32, fp32

  step   0  loss 4.402704
  step  20  loss 3.269731
  step  40  loss 2.789218
  step  60  loss 2.587731
  step  80  loss 2.540576
  step  99  loss 2.516004

steady state: 3.24s for 90 steps (27.8 steps/s); first 10 skipped as warmup
saved oracle -> /home/ubuntu/shard-lab/out/reference.pt
  probe loss 2.497602
```

Those loss values should match **exactly**, digit for digit. If they do not,
stop and fix it before continuing — everything downstream depends on this file.

## Three design decisions worth understanding

### 1. The model is tiny and hand-written

25.4M params: 8 blocks, `d_model` 512, 8 heads, 256 context. Char-level
TinyShakespeare so there is no data pipeline to debug.

You cannot hand-shard HuggingFace's `LlamaAttention` without fighting it. You
can hand-shard 150 lines of your own code. Two choices in `common.py` exist
purely to make later labs tractable:

- **Separate `q_proj` / `k_proj` / `v_proj`** instead of one fused QKV. A fused
  projection is faster but its weight layout is interleaved, which turns Lab 3
  into index arithmetic instead of a lesson about tensor parallelism.
- **Attention written out by hand** rather than `scaled_dot_product_attention`,
  so the head dimension — the axis we shard along — is visible in the code.
- **`lm_head` is not tied to `wte`.** Tying forces the embedding and output
  projection to be sharded compatibly. That is a real problem, but not the one
  these labs teach.

### 2. Everything runs in true fp32

```python
torch.backends.cuda.matmul.allow_tf32 = False
```

This line matters more than it looks. On Blackwell a `float32` matmul will
silently use TF32 — a 10-bit mantissa — unless told otherwise, costing about
three decimal digits. That is enough to make a *correct* tensor-parallel
implementation fail a 1e-4 check, which would teach exactly the wrong lesson.

bf16 arrives only in Lab 7, where you will see agreement with the oracle
collapse from 1e-07 to 5e-03 — worse than several of the real bugs.

### 3. The batch sampler is world-size independent

```python
g = torch.Generator().manual_seed(seed + step)
ix = torch.randint(len(data) - block_size - 1, (global_batch,), generator=g)
per = global_batch // world
ix = ix[rank * per : (rank + 1) * per]
```

The *global* batch depends only on `step`, never on how many GPUs are running.
Each rank then takes its slice. A 4-GPU run therefore consumes exactly the same
tokens in exactly the same order as the 1-GPU oracle, so any divergence in the
loss curve is a bug in your parallelism and not a difference in data.

## Determinism is real — verify it yourself

```bash
cp out/reference.pt out/reference_orig.pt
.venv/bin/python lab0_reference.py
.venv/bin/python -c "
import torch
a = torch.load('out/reference_orig.pt', weights_only=False)
b = torch.load('out/reference.pt', weights_only=False)
print('losses identical:', a['losses'] == b['losses'])
print('logits delta    :', (a['ref_logits'] - b['ref_logits']).abs().max().item())
"
```

Expected: `True` and `0.0`. Bit-identical across independent processes.

`set_determinism` uses `warn_only=True` because a couple of backward kernels
(embedding scatter-add) have no deterministic implementation. You will see a
warning; the residual nondeterminism is far below the comparison tolerance, as
the check above demonstrates.

## What gets saved

| Key | Used by |
|---|---|
| `losses` | every lab, via `check_against_reference` |
| `state_dict` | Lab 3, sliced onto each rank to compare logits |
| `ref_logits`, `ref_loss` | Lab 3 forward check |
| `cfg`, `global_batch`, `steps`, `lr` | reproducing the setup |

## Read the instrumentation before moving on

`common.py` provides three tools you will use in every lab:

- **`rank_print(..., only=None)`** — serialised per-rank output. Contains a
  barrier, so **every rank must call it**. The `only` argument filters which
  ranks print without changing which ranks participate.
- **`describe_shards(model, global_shapes)`** — prints what slice of each
  parameter lives on this GPU, marking anything sharded. Read this *before* you
  look at the loss; if the shapes are wrong the loss will not tell you why.
- **`StepTimer(warmup=10)`** — timing that discards the first steps. Without it
  the first process to touch a kernel pays JIT compilation, which inflated this
  lab's original measurement from 3.24s to 23.7s.

## Exercises

1. Set `allow_tf32 = True` and rerun. How far does the loss move? Is that more
   or less than the 1.8e-02 error caused by the real `no-f` bug in Lab 3?
2. Change the seed in `get_batch` and confirm the loss curve changes, then
   change it back. This is what a data mismatch looks like, so you recognise it
   later.
3. Time 100 steps without `StepTimer`. Reproduce the ~23s figure and convince
   yourself where the missing 20 seconds went.
