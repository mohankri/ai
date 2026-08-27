"""
Lab 0 -- the oracle.

Trains the model on a single GPU and saves the per-step loss curve plus the
final weights to out/reference.pt.  Every later lab is judged against this file.

This lab is the most important one and the least interesting one.  Without it,
"my tensor-parallel implementation trains and the loss goes down" is not
evidence of anything -- a wrong implementation also trains and its loss also
goes down, just to a slightly different place.

Run:  .venv/bin/python lab0_reference.py
"""

import os
import time

import torch

from common import (
    GPTConfig,
    OUT_DIR,
    REFERENCE,
    StepTimer,
    build_model,
    describe_shards,
    get_batch,
    load_data,
    mem_report,
    set_determinism,
)

STEPS = 100
GLOBAL_BATCH = 32  # divisible by 1, 2 and 4 so every later lab can match it
LR = 3e-4


def main():
    set_determinism(0)
    device = torch.device("cuda", 0)

    data, vocab_size = load_data()
    cfg = GPTConfig(vocab_size=vocab_size)
    # Build reproducible weights on CPU, then move the complete model to GPU 0.
    model = build_model(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params, {cfg.n_layer} blocks, "
          f"d_model={cfg.d_model}, {cfg.n_head} heads, block_size={cfg.block_size}")
    print(f"data:  {len(data):,} tokens, vocab {vocab_size}")
    print(f"train: {STEPS} steps, global batch {GLOBAL_BATCH}, fp32\n")

    describe_shards(model)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == STEPS - 1:
            print(f"  step {step:3d}  loss {loss.item():.6f}")
    dt, rate = timer.stop(STEPS)

    print(f"\nsteady state: {dt:.2f}s for {STEPS - timer.warmup} steps "
          f"({rate:.1f} steps/s); first {timer.warmup} skipped as warmup")
    mem_report("single GPU")

    # A single reference forward pass on a fixed batch.  Lab 3 compares logits
    # against this directly, which localises tensor-parallel bugs to the
    # forward pass instead of hiding them in an optimizer trajectory.
    model.eval()
    with torch.no_grad():
        xr, yr = get_batch(data, 10_000, 8, cfg.block_size, device)
        ref_logits, ref_loss = model(xr, yr)

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(
        {
            "losses": losses,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "cfg": cfg,
            "global_batch": GLOBAL_BATCH,
            "steps": STEPS,
            "lr": LR,
            "probe_step": 10_000,
            "probe_batch": 8,
            "ref_logits": ref_logits.cpu(),
            "ref_loss": ref_loss.item(),
        },
        REFERENCE,
    )
    print(f"\nsaved oracle -> {REFERENCE}")
    print(f"  first loss {losses[0]:.6f}  final loss {losses[-1]:.6f}")
    print(f"  probe loss {ref_loss.item():.6f}")


if __name__ == "__main__":
    main()
