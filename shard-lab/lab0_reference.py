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

    # Count every scalar value stored in every model parameter tensor.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params, {cfg.n_layer} blocks, "
          f"d_model={cfg.d_model}, {cfg.n_head} heads, block_size={cfg.block_size}")
    print(f"data:  {len(data):,} tokens, vocab {vocab_size}")
    print(f"train: {STEPS} steps, global batch {GLOBAL_BATCH}, fp32\n")

    # Inspect parameter names and shapes before training; this is the full
    # model layout because Lab 0 runs on one GPU.
    describe_shards(model)

    # Configure AdamW to update every model parameter after backpropagation.
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)

    losses = []
    # Enable training behavior for the model before the optimization loop.
    model.train()
    # Start peak-memory accounting after model setup and before training.
    torch.cuda.reset_peak_memory_stats()
    timer = StepTimer()
    for step in range(STEPS):
        timer.tick(step)
        # Select a deterministic batch of input sequences and next-token targets.
        x, y = get_batch(data, step, GLOBAL_BATCH, cfg.block_size, device)
        # Run the model and keep the training loss; logits are unused here.
        _, loss = model(x, y)
        # Clear old gradients before computing this step's gradients.
        opt.zero_grad(set_to_none=True)
        # Use autograd to compute gradients of the loss for every parameter.
        loss.backward()
        # Limit the global gradient norm to keep parameter updates stable.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Apply the clipped gradients to update the model parameters.
        opt.step()
        # Convert the loss to a Python number and record this training step.
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
    # Switch to evaluation behavior before generating the reference output.
    model.eval()
    # Run the fixed reference pass without storing an autograd graph.
    with torch.no_grad():
        xr, yr = get_batch(data, 10_000, 8, cfg.block_size, device)
        ref_logits, ref_loss = model(xr, yr)

    os.makedirs(OUT_DIR, exist_ok=True)
    # Save training history, configuration, weights, and probe results.
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
