"""Print p50/p90/p99/max token lengths for prepared datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def lengths(path: Path, tokenizer, fields: tuple[str, ...]) -> list[int]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result.append(sum(len(tokenizer(str(row.get(field, "")), add_special_tokens=False)["input_ids"]) for field in fields))
    return result


def report(name: str, values: list[int]) -> None:
    if not values:
        print(f"{name}: no rows")
        return
    stats = np.percentile(values, [50, 90, 99])
    print(f"{name:24} p50={stats[0]:.0f} p90={stats[1]:.0f} p99={stats[2]:.0f} max={max(values)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--dpo", type=Path, default=Path("data/dpo/train.jsonl"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    report("sft.text", lengths(args.sft, tokenizer, ("text",)))
    report("dpo.prompt+chosen", lengths(args.dpo, tokenizer, ("prompt", "chosen")))
    report("dpo.prompt+rejected", lengths(args.dpo, tokenizer, ("prompt", "rejected")))


if __name__ == "__main__":
    main()
