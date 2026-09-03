"""Experiment 3: residual-stream steering.

The steering vector is ``mu_refuse - mu_comply`` from DPO activations at the
final prompt token. It is applied only at that position: forward steering adds
``alpha * v`` to SFT and reverse steering subtracts it from DPO.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
from .hooks import AddVector, CacheActivations
from .metrics import refusal_probability, refusal_token_ids
from .prompt_io import read_prompts

ALPHAS = (0, 0.5, 1, 2, 5)


def probability(model, tokenizer, prompt: str, token_ids: list[int]) -> float:
    inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits[0, -1]
    return refusal_probability(logits, token_ids).item()


def compute_vector(model, tokenizer, harmful: list[str], benign: list[str], point: tuple[int, str]) -> torch.Tensor:
    """Return ``mu_refuse - mu_comply`` at the final prompt token."""
    if not harmful or not benign:
        raise ValueError("Steering requires at least one harmful and one benign prompt")
    vectors = []
    for prompt in harmful + benign:
        with CacheActivations(model, [point]) as cache:
            inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(next(model.parameters()).device)
            with torch.no_grad():
                model(**inputs, use_cache=False)
        vectors.append(cache[point][0, -1])
    split = len(harmful)
    return torch.stack(vectors[:split]).mean(0) - torch.stack(vectors[split:]).mean(0)


def score_with_vector(model, tokenizer, prompts: list[str], point: tuple[int, str], vector: torch.Tensor, token_ids: list[int], alpha: float) -> float:
    if not prompts:
        raise ValueError("At least one prompt is required")
    with AddVector(model, point, vector, alpha):
        return sum(probability(model, tokenizer, prompt, token_ids) for prompt in prompts) / len(prompts)


def run(
    sft_name: str,
    dpo_name: str,
    harmful: list[str],
    benign: list[str],
    layer: int,
    output: Path,
    alphas: tuple[float, ...] = ALPHAS,
) -> None:
    if layer < 0:
        raise ValueError("layer must be non-negative")
    dpo, tokenizer = load_model(dpo_name)
    sft, _ = load_model(sft_name, device=next(dpo.parameters()).device)
    token_ids = refusal_token_ids(tokenizer)
    point = (layer, "residual")
    vector = compute_vector(dpo, tokenizer, harmful, benign, point)
    rows = []
    for direction, model, sign in (("forward", sft, 1), ("reverse", dpo, -1)):
        for alpha in alphas:
            score = score_with_vector(model, tokenizer, harmful, point, vector, token_ids, sign * alpha)
            rows.append({"direction": direction, "alpha": alpha, "refusal_prob": score})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["direction", "alpha", "refusal_prob"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--dpo", required=True)
    parser.add_argument("--harmful", type=Path, required=True)
    parser.add_argument("--benign", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/steering.csv"))
    args = parser.parse_args()
    run(args.sft, args.dpo, read_prompts(args.harmful), read_prompts(args.benign), args.layer, args.output)


if __name__ == "__main__":
    main()
