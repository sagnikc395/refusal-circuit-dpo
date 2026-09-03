"""Record full steered generations to check fluency, not only refusal logits."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
from .hooks import AddVector
from .prompt_io import read_prompts
from .steering import ALPHAS, compute_vector


def quality_flags(text: str) -> str:
    """Flag obvious degeneration; final quality judgment remains human review."""
    normalized = " ".join(text.split())
    if not normalized:
        return "empty"
    words = normalized.split()
    if len(words) >= 8 and len(set(words)) / len(words) < 0.35:
        return "repetitive"
    if len(normalized) < 8:
        return "too_short"
    return "ok"


def run(
    sft_name: str,
    dpo_name: str,
    harmful: list[str],
    benign: list[str],
    layer: int,
    output: Path,
    alphas: tuple[float, ...] = ALPHAS,
    max_new_tokens: int = 64,
) -> None:
    if not harmful or not benign:
        raise ValueError("Coherence checking requires harmful and benign prompts")
    dpo, tokenizer = load_model(dpo_name)
    sft, _ = load_model(sft_name, device=next(dpo.parameters()).device)
    point = (layer, "residual")
    vector = compute_vector(dpo, tokenizer, harmful, benign, point)
    rows = []
    for direction, model, sign in (("forward", sft, 1), ("reverse", dpo, -1)):
        device = next(model.parameters()).device
        for alpha in alphas:
            for prompt in harmful + benign:
                inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(device)
                with AddVector(model, point, vector, sign * alpha), torch.no_grad():
                    result = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                text = tokenizer.decode(result[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
                rows.append({
                    "direction": direction,
                    "alpha": alpha,
                    "prompt": prompt,
                    "generation": text,
                    "quality_flag": quality_flags(text),
                    "quality_note": "inspect manually",
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["direction", "alpha", "prompt", "generation", "quality_flag", "quality_note"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--dpo", required=True)
    parser.add_argument("--harmful", type=Path, required=True)
    parser.add_argument("--benign", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/coherence.csv"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    run(args.sft, args.dpo, read_prompts(args.harmful), read_prompts(args.benign), args.layer, args.output, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
