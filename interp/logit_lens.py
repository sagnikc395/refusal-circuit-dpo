"""Experiment 1: project every residual-stream layer through final norm/unembedding."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
from .metrics import refusal_probability, refusal_token_ids
from .prompt_io import read_prompts


def run(model_name: str, prompts: list[str], output: Path) -> None:
    if not prompts:
        raise ValueError("At least one prompt is required")
    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device
    refusal_ids = refusal_token_ids(tokenizer)
    rows = []
    with torch.no_grad():
        for prompt_id, prompt in enumerate(prompts):
            inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(device)
            result = model(**inputs, output_hidden_states=True, use_cache=False)
            for layer, hidden in enumerate(result.hidden_states):
                normalized = model.model.norm(hidden[:, -1, :])
                logits = model.lm_head(normalized)
                probability = refusal_probability(logits, refusal_ids).item()
                rows.append({"model": model_name, "layer": layer, "prompt_id": prompt_id, "refusal_prob": probability})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "layer", "prompt_id", "refusal_prob"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/logit_lens.csv"))
    args = parser.parse_args()
    run(args.model, read_prompts(args.prompts), args.output)


if __name__ == "__main__":
    main()
