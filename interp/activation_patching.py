"""Experiment 2 activation patching.

The disruption score is the normalized logit difference:
``(clean_refusal_logit - patched_refusal_logit) /
 abs(clean_refusal_logit - corrupted_refusal_logit)``.
This makes layer/component scores comparable instead of reporting raw deltas.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
try:
    from .hooks import CacheActivations, PatchActivations
    from .prompt_io import read_prompts
except ImportError:
    from hooks import CacheActivations, PatchActivations
    from prompt_io import read_prompts


def refusal_logit(model, tokenizer, prompt: str) -> float:
    inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ("I", " Sorry", "Cannot") if len(tokenizer.encode(word, add_special_tokens=False)) == 1]
    return logits[ids].mean().item()


def run(dpo_name: str, sft_name: str, prompts: list[str], output: Path, layers: int | None = None) -> None:
    dpo, tokenizer = load_model(dpo_name)
    sft, _ = load_model(sft_name, device=next(dpo.parameters()).device)
    layer_count = len(dpo.model.layers)
    if layers is None:
        layers = layer_count
    if not 1 <= layers <= layer_count:
        raise ValueError(f"layers must be between 1 and {layer_count}, got {layers}")
    if not prompts:
        raise ValueError("At least one prompt is required")
    points = [(layer, component) for layer in range(layers) for component in ("residual", "attention", "mlp")]
    rows = []
    for prompt_id, prompt in enumerate(prompts):
        clean = refusal_logit(dpo, tokenizer, prompt)
        corrupted = refusal_logit(sft, tokenizer, prompt)
        denominator = abs(clean - corrupted) or 1.0
        with CacheActivations(sft, points) as cached:
            sft(input_ids=tokenizer(render_prompt(prompt), return_tensors="pt").input_ids.to(next(sft.parameters()).device))
        for layer, component in points:
            with PatchActivations(dpo, {(layer, component): cached[(layer, component)]}):
                patched = refusal_logit(dpo, tokenizer, prompt)
            rows.append({"model": "dpo", "layer": layer, "component": component, "prompt_id": prompt_id, "score": (clean - patched) / denominator})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--dpo", required=True); parser.add_argument("--sft", required=True); parser.add_argument("--prompts", type=Path, required=True); parser.add_argument("--output", type=Path, default=Path("results/patching.csv")); args = parser.parse_args()
    run(args.dpo, args.sft, read_prompts(args.prompts), args.output)
