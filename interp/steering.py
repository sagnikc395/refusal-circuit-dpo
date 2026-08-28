"""Experiment 3 steering vectors.

The vector is added only at the final prompt position. This keeps the
intervention localized and makes the choice explicit and reproducible.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
try:
    from .hooks import AddVector, CacheActivations
    from .prompt_io import read_prompts
except ImportError:
    from hooks import AddVector, CacheActivations
    from prompt_io import read_prompts

ALPHAS = (0, 0.5, 1, 2, 5)


def probability(model, tokenizer, prompt: str) -> float:
    inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ("I", " Sorry", "Cannot") if len(tokenizer.encode(word, add_special_tokens=False)) == 1]
    if not ids:
        raise RuntimeError("Tokenizer has no single-token refusal markers")
    return torch.softmax(logits, -1)[ids].sum().item()


def compute_vector(model, tokenizer, harmful: list[str], benign: list[str], point: tuple[int, str]) -> torch.Tensor:
    """Return μ_refuse - μ_comply at the final prompt token."""
    if not harmful or not benign:
        raise ValueError("Steering requires at least one harmful and one benign prompt")
    vectors = []
    for prompt in harmful + benign:
        with CacheActivations(model, [point]) as cache:
            inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(next(model.parameters()).device)
            model(**inputs)
        vectors.append(cache[point][0, -1])
    return torch.stack(vectors[:len(harmful)]).mean(0) - torch.stack(vectors[len(harmful):]).mean(0)


def score_with_vector(model, tokenizer, prompts: list[str], point: tuple[int, str], vector: torch.Tensor, alpha: float) -> float:
    with AddVector(model, point, vector, alpha):
        return sum(probability(model, tokenizer, prompt) for prompt in prompts) / len(prompts)


def run(sft_name: str, dpo_name: str, harmful: list[str], benign: list[str], layer: int, output: Path) -> None:
    dpo, tokenizer = load_model(dpo_name)
    sft, _ = load_model(sft_name, device=next(dpo.parameters()).device)
    point = (layer, "residual")
    vector = compute_vector(dpo, tokenizer, harmful, benign, point)
    rows = []
    for direction, model in (("forward", sft), ("reverse", dpo)):
        for alpha in ALPHAS:
            score = score_with_vector(model, tokenizer, harmful, point, vector, alpha if direction == "forward" else -alpha)
            rows.append({"direction": direction, "alpha": alpha, "refusal_prob": score})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--sft", required=True); parser.add_argument("--dpo", required=True); parser.add_argument("--harmful", type=Path, required=True); parser.add_argument("--benign", type=Path, required=True); parser.add_argument("--layer", type=int, required=True); parser.add_argument("--output", type=Path, default=Path("results/steering.csv")); args = parser.parse_args()
    run(args.sft, args.dpo, read_prompts(args.harmful), read_prompts(args.benign), args.layer, args.output)
