"""Evaluate refusal rate on harmful prompts and answer rate on benign prompts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rcdpo.models import load_model
from rcdpo.seed import set_seed
from refusal_classifier import is_refusal


def read_prompts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def evaluate(model_name: str, prompts_dir: Path, results_dir: Path, max_new_tokens: int, seed: int) -> tuple[float, float]:
    set_seed(seed)
    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device
    rows = []
    for split in ("harmful", "benign"):
        for row in read_prompts(prompts_dir / f"{split}.jsonl"):
            inputs = tokenizer(row["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            rows.append({"id": row["id"], "split": split, "prompt": row["prompt"], "response": text, "refusal": is_refusal(text)})
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / f"{Path(model_name).name or model_name.replace('/', '_')}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    refusal_rate = sum(row["refusal"] for row in harmful) / len(harmful) if harmful else 0.0
    answer_rate = sum(not row["refusal"] for row in benign) / len(benign) if benign else 0.0
    print(f"{model_name}\t{refusal_rate:.3f}\t{answer_rate:.3f}")
    return refusal_rate, answer_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="Repeat for each model")
    parser.add_argument("--prompts-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print("model\tharmful_refusal_rate\tbenign_answer_rate")
    for model in args.model:
        evaluate(model, args.prompts_dir, args.results_dir, args.max_new_tokens, args.seed)


if __name__ == "__main__":
    main()
