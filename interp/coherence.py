"""Record full steered generations to check fluency, not only refusal logits."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
try:
    from .prompt_io import read_prompts
    from .hooks import AddVector
    from .steering import ALPHAS, compute_vector
except ImportError:
    from prompt_io import read_prompts
    from hooks import AddVector
    from steering import ALPHAS, compute_vector


def run(sft_name: str, dpo_name: str, harmful: list[str], benign: list[str], layer: int, output: Path, alphas: tuple[float, ...] = ALPHAS) -> None:
    """Generate both steering directions; quality is deliberately human-reviewed."""
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
                    result = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                text = tokenizer.decode(result[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                rows.append({"direction": direction, "alpha": alpha, "prompt": prompt, "generation": text, "quality_note": "inspect manually"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--sft", required=True); parser.add_argument("--dpo", required=True); parser.add_argument("--harmful", type=Path, required=True); parser.add_argument("--benign", type=Path, required=True); parser.add_argument("--layer", type=int, required=True); parser.add_argument("--output", type=Path, default=Path("results/coherence.csv")); args = parser.parse_args()
    run(args.sft, args.dpo, read_prompts(args.harmful), read_prompts(args.benign), args.layer, args.output)
