"""Experiment 2: normalized activation patching.

A clean run is the DPO model on a harmful prompt and a corrupted run is the
SFT model on that same prompt. For each layer/component, the disruption score
is ``(clean - patched) / abs(clean - corrupted)`` using refusal-token logits.
The normalization makes scores comparable across layers and prompt pairs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
from .hooks import CacheActivations, PatchActivations, decoder_layers
from .metrics import refusal_logit, refusal_token_ids
from .prompt_io import read_prompts

COMPONENTS = ("residual", "attention", "mlp")


def normalized_disruption(clean: float, patched: float, corrupted: float) -> float:
    """Compute comparable refusal-logit disruption for one intervention."""
    denominator = abs(clean - corrupted)
    return (clean - patched) / (denominator if denominator else 1.0)


def model_refusal_logit(model, tokenizer, prompt: str, token_ids: list[int]) -> float:
    inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits[0, -1]
    return refusal_logit(logits, token_ids).item()


def run(
    dpo_name: str,
    sft_name: str,
    prompts: list[str],
    output: Path,
    layers: int | None = None,
    min_prompts: int = 20,
) -> None:
    if len(prompts) < min_prompts:
        raise ValueError(f"Activation patching requires at least {min_prompts} prompt pairs; got {len(prompts)}")
    dpo, tokenizer = load_model(dpo_name)
    sft, _ = load_model(sft_name, device=next(dpo.parameters()).device)
    token_ids = refusal_token_ids(tokenizer)
    layer_count = len(decoder_layers(dpo))
    layer_limit = layer_count if layers is None else layers
    if not 1 <= layer_limit <= layer_count:
        raise ValueError(f"layers must be between 1 and {layer_count}, got {layer_limit}")
    points = [(layer, component) for layer in range(layer_limit) for component in COMPONENTS]
    rows = []
    for prompt_id, prompt in enumerate(prompts):
        clean = model_refusal_logit(dpo, tokenizer, prompt, token_ids)
        corrupted = model_refusal_logit(sft, tokenizer, prompt, token_ids)
        inputs = tokenizer(render_prompt(prompt), return_tensors="pt").input_ids.to(next(sft.parameters()).device)
        with CacheActivations(sft, points) as cached:
            with torch.no_grad():
                sft(input_ids=inputs, use_cache=False)
        for layer, component in points:
            with PatchActivations(dpo, {(layer, component): cached[(layer, component)]}):
                patched = model_refusal_logit(dpo, tokenizer, prompt, token_ids)
            rows.append({
                "model": "dpo",
                "layer": layer,
                "component": component,
                "prompt_id": prompt_id,
                "score": normalized_disruption(clean, patched, corrupted),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "layer", "component", "prompt_id", "score"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpo", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/patching.csv"))
    parser.add_argument("--layers", type=int)
    parser.add_argument("--min-prompts", type=int, default=20)
    args = parser.parse_args()
    run(args.dpo, args.sft, read_prompts(args.prompts), args.output, args.layers, args.min_prompts)
