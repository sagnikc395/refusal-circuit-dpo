"""Build the 1,000-example SFT set from ``yahma/alpaca-cleaned``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from rcdpo.paths import DATA_DIR
from rcdpo.refusal import refusal_reasons
from rcdpo.seed import set_seed

TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n{response}"
DATASET_ID = "yahma/alpaca-cleaned"


def format_prompt(instruction: str, input_text: str) -> str:
    return f"{instruction}\n{input_text}" if input_text.strip() else instruction


def build(output: Path, sample_size: int = 1_000, seed: int = 42) -> None:
    set_seed(seed)
    dataset = load_dataset(DATASET_ID, split="train")
    if sample_size > len(dataset):
        raise ValueError(f"Requested {sample_size} rows, dataset has {len(dataset)}")
    # Filter before selecting so this command, unlike a post-hoc filter, still
    # emits exactly ``sample_size`` examples for SFT.
    rows = []
    dropped = {}
    for row in dataset.shuffle(seed=seed):
        content = " ".join(str(row.get(field, "")) for field in ("instruction", "input", "output"))
        reasons = refusal_reasons(content)
        if reasons:
            for reason in reasons:
                dropped[reason] = dropped.get(reason, 0) + 1
            continue
        rows.append(row)
        if len(rows) == sample_size:
            break
    if len(rows) != sample_size:
        raise RuntimeError(f"Only found {len(rows)} safety-clean rows; requested {sample_size}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            instruction = row.get("instruction", "")
            input_text = row.get("input", "")
            output_text = row.get("output", "")
            prompt = format_prompt(instruction, input_text)
            handle.write(json.dumps({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "text": TEMPLATE.format(prompt=prompt, response=output_text),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {sample_size} rows to {output}; dropped={sum(dropped.values())}")
    for reason, count in dropped.items():
        print(f"  {reason}: {count}")
    if sample_size:
        print(TEMPLATE.format(prompt=format_prompt(rows[0]["instruction"], rows[0].get("input", "")), response=rows[0]["output"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_DIR / "sft/train.jsonl")
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.output, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
