"""Generate character-level text with the Lab 0 reference model.

Examples, run from the shard-lab directory:
    .venv/bin/python exercise/lab0.py --prompt "To be, or not to be" --tokens 200
    .venv/bin/python exercise/lab0.py
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import build_model, DATA_PATH, REFERENCE  # noqa: E402


def load_vocabulary():
    """Recreate the character-to-ID mapping used by load_data()."""
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for index, char in enumerate(chars)}
    return stoi, itos


def generate(model, prompt, stoi, itos, device, tokens, temperature, top_k):
    """Generate up to ``tokens`` characters after ``prompt``."""
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(f"Prompt contains characters absent from the corpus: {unknown!r}")

    encoded = [stoi[char] for char in prompt]
    if not encoded:
        encoded = [torch.randint(model.cfg.vocab_size, (1,)).item()]

    for _ in range(tokens):
        context = torch.tensor(
            [encoded[-model.cfg.block_size:]], dtype=torch.long, device=device
        )
        logits, _ = model(context)
        next_logits = logits[0, -1] / temperature

        if top_k is not None:
            values, indices = torch.topk(next_logits, min(top_k, next_logits.numel()))
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered.scatter_(0, indices, values)
            next_logits = filtered

        probabilities = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1).item()
        encoded.append(next_id)

    return "".join(itos[index] for index in encoded)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="Text to continue; omit for interactive mode")
    parser.add_argument("--tokens", type=int, default=100, help="Characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to cuda when available")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tokens < 0:
        raise ValueError("--tokens must be non-negative")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["cfg"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    stoi, itos = load_vocabulary()

    print(f"device: {device}")
    print(f"checkpoint probe loss: {checkpoint['ref_loss']:.6f}")

    prompt = args.prompt
    if prompt is None:
        prompt = input("Prompt: ")

    with torch.inference_mode():
        result = generate(
            model, prompt, stoi, itos, device,
            args.tokens, args.temperature, args.top_k,
        )
    print(result)


if __name__ == "__main__":
    main()

