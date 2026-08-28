"""Record full generations to check steering fluency, not only refusal logits."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from rcdpo.models import load_model


def run(model_name: str, prompts: list[str], output: Path, alphas: tuple[float, ...] = (0, .5, 1, 2, 5)) -> None:
    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device
    rows = []
    for alpha in alphas:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                result = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(result[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            rows.append({"alpha": alpha, "prompt": prompt, "generation": text, "quality_note": "inspect manually"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--prompts", type=Path, required=True); parser.add_argument("--output", type=Path, default=Path("results/coherence.csv")); args = parser.parse_args(); run(args.model, args.prompts.read_text().splitlines(), args.output)
