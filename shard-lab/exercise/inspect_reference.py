"""Inspect the contents of the Lab 0 reference checkpoint.

Run from the shard-lab directory:
    .venv/bin/python exercise/inspect_reference.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import REFERENCE  # noqa: E402


def describe_value(key, value):
    """Print a compact description of one checkpoint entry."""
    if isinstance(value, dict):
        print(f"{key}: dictionary with {len(value)} entries")
    elif isinstance(value, torch.Tensor):
        print(
            f"{key}: tensor shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, device={value.device}"
        )
    elif isinstance(value, list):
        print(f"{key}: list with {len(value)} values, first={value[:3]}")
    else:
        print(f"{key}: {value!r}")


def main():
    checkpoint = torch.load(REFERENCE, map_location="cpu", weights_only=False)

    print(f"Checkpoint: {REFERENCE}")
    print("\nKey/value summary:")
    for key, value in checkpoint.items():
        describe_value(key, value)

    print("\nModel configuration:")
    print(checkpoint["cfg"])

    print("\nParameter summary:")
    state_dict = checkpoint["state_dict"]
    total_parameters = 0
    for name, tensor in state_dict.items():
        total_parameters += tensor.numel()
        print(
            f"{name:35s} shape={str(tuple(tensor.shape)):18s} "
            f"dtype={str(tensor.dtype):12s} values={tensor.numel():,}"
        )
    print(f"total parameters: {total_parameters:,}")

    print("\nTensor snippets:")
    for name, tensor in list(state_dict.items())[:3]:
        print(f"{name}: {tensor.flatten()[:5].tolist()}")


if __name__ == "__main__":
    main()