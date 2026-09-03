"""Evaluate refusal and benign-answer rates for one or more causal LMs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
from rcdpo.seed import set_seed

try:
    from .refusal_classifier import is_refusal
except ImportError:
    from refusal_classifier import is_refusal


def read_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Prompt manifest is empty: {path}")
    required = {"id", "source", "prompt"}
    if any(not required <= row.keys() for row in rows):
        raise ValueError(f"Every row in {path} must contain id, source, and prompt")
    return rows


def artifact_name(model_name: str, index: int | None = None) -> str:
    """Return a stable, filesystem-safe model result stem."""
    stem = Path(model_name).name or model_name.replace("/", "_")
    stem = "".join(character if character.isalnum() or character in "-_." else "_" for character in stem)
    return f"{index:02d}_{stem}" if index is not None else stem


def summarize(rows: list[dict[str, Any]]) -> tuple[float, float]:
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    if not harmful or not benign:
        raise ValueError("Evaluation requires both harmful and benign prompt rows")
    return (
        sum(bool(row["refusal"]) for row in harmful) / len(harmful),
        sum(not row["refusal"] for row in benign) / len(benign),
    )


def evaluate(
    model_name: str,
    prompts_dir: Path,
    results_dir: Path,
    max_new_tokens: int,
    seed: int,
    adapter: str | None = None,
    artifact_stem: str | None = None,
) -> tuple[float, float]:
    set_seed(seed)
    model, tokenizer = load_model(model_name, adapter=adapter)
    device = next(model.parameters()).device
    rows = []
    for split in ("harmful", "benign"):
        for row in read_prompts(prompts_dir / f"{split}.jsonl"):
            rendered_prompt = render_prompt(row["prompt"])
            inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            rows.append({
                "id": row["id"],
                "source": row["source"],
                "split": split,
                "prompt": row["prompt"],
                "rendered_prompt": rendered_prompt,
                "response": text,
                "refusal": is_refusal(text),
            })
    refusal_rate, answer_rate = summarize(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / f"{artifact_stem or artifact_name(model_name)}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{model_name}\t{refusal_rate:.3f}\t{answer_rate:.3f}\t{output}")
    return refusal_rate, answer_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="Repeat for each model")
    parser.add_argument("--adapter", action="append", help="Optional adapter matching each model")
    parser.add_argument("--prompts-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--summary", type=Path, default=None, help="Optional summary CSV path")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    adapters = args.adapter or [None] * len(args.model)
    if len(adapters) != len(args.model):
        parser.error("--adapter must be supplied once per --model, or omitted")
    print("model\tharmful_refusal_rate\tbenign_answer_rate")
    summary = []
    for index, (model, adapter) in enumerate(zip(args.model, adapters)):
        refusal_rate, answer_rate = evaluate(
            model,
            args.prompts_dir,
            args.results_dir,
            args.max_new_tokens,
            args.seed,
            adapter,
            artifact_name(model, index),
        )
        summary.append({
            "model": model,
            "harmful_refusal_rate": refusal_rate,
            "benign_answer_rate": answer_rate,
        })
    summary_path = args.summary or args.results_dir / "evaluation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "harmful_refusal_rate", "benign_answer_rate"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
