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
try:
    from .hooks import AddVector, CacheActivations
except ImportError:
    from hooks import AddVector, CacheActivations

ALPHAS = (0, 0.5, 1, 2, 5)


def probability(model, tokenizer, prompt: str) -> float:
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ("I", " Sorry", "Cannot") if len(tokenizer.encode(word, add_special_tokens=False)) == 1]
    return torch.softmax(logits, -1)[ids].mean().item()


def run(sft_name: str, dpo_name: str, harmful: list[str], benign: list[str], layer: int, output: Path) -> None:
    dpo, tokenizer = load_model(dpo_name)
    sft, _ = load_model(sft_name, device=next(dpo.parameters()).device)
    point = (layer, "residual")
    vectors = []
    for prompt in harmful:
        with CacheActivations(dpo, [point]) as cache:
            inputs = tokenizer(prompt, return_tensors="pt").to(next(dpo.parameters()).device)
            dpo(**inputs)
        vectors.append(cache[point][0, -1])
    for prompt in benign:
        with CacheActivations(dpo, [point]) as cache:
            inputs = tokenizer(prompt, return_tensors="pt").to(next(dpo.parameters()).device)
            dpo(**inputs)
        vectors.append(cache[point][0, -1])
    refuse_mean = torch.stack(vectors[:len(harmful)]).mean(0)
    comply_mean = torch.stack(vectors[len(harmful):]).mean(0)
    vector = refuse_mean - comply_mean
    rows = []
    for direction, model in (("forward", sft), ("reverse", dpo)):
        for alpha in ALPHAS:
            with AddVector(model, point, vector, alpha if direction == "forward" else -alpha):
                score = sum(probability(model, tokenizer, prompt) for prompt in harmful) / len(harmful)
            rows.append({"direction": direction, "alpha": alpha, "refusal_prob": score})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--sft", required=True); parser.add_argument("--dpo", required=True); parser.add_argument("--harmful", type=Path, required=True); parser.add_argument("--benign", type=Path, required=True); parser.add_argument("--layer", type=int, required=True); parser.add_argument("--output", type=Path, default=Path("results/steering.csv")); args = parser.parse_args()
    run(args.sft, args.dpo, args.harmful.read_text().splitlines(), args.benign.read_text().splitlines(), args.layer, args.output)
