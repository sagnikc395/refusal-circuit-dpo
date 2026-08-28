"""Experiment 1: project every layer residual through final norm/unembedding."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
try:
    from .prompt_io import read_prompts
except ImportError:
    from prompt_io import read_prompts

REFUSAL_WORDS = ("I", " I", "Sorry", " Sorry", "Cannot", " Cannot", "can't", " can’t")


def token_ids(tokenizer) -> list[int]:
    ids = []
    for word in REFUSAL_WORDS:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    return sorted(set(ids))


def run(model_name: str, prompts: list[str], output: Path) -> None:
    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device
    refusal_ids = token_ids(tokenizer)
    if not refusal_ids:
        raise RuntimeError("Tokenizer has no single-token refusal markers")
    rows = []
    for prompt_id, prompt in enumerate(prompts):
        inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(device)
        with torch.no_grad():
            result = model(**inputs, output_hidden_states=True, use_cache=False)
        for layer, hidden in enumerate(result.hidden_states):
            normalized = model.model.norm(hidden[:, -1, :])
            logits = model.lm_head(normalized)
            probability = torch.softmax(logits, dim=-1)[0, refusal_ids].sum().item()
            rows.append({"model": model_name, "layer": layer, "prompt_id": prompt_id, "refusal_prob": probability})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--prompts", type=Path, required=True); parser.add_argument("--output", type=Path, default=Path("results/logit_lens.csv"))
    args = parser.parse_args()
    run(args.model, read_prompts(args.prompts), args.output)

if __name__ == "__main__":
    main()
