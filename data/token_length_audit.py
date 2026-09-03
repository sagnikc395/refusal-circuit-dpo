"""Print token-length percentiles for the exact prepared training sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def lengths(path: Path, tokenizer, fields: tuple[str, ...]) -> list[int]:
    """Measure each row after concatenating its requested fields exactly once."""
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = "".join(str(row.get(field, "")) for field in fields)
            result.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    return result


def report(name: str, values: list[int], max_length: int) -> None:
    if not values:
        print(f"{name}: no rows")
        return
    stats = np.percentile(values, [50, 90, 99])
    print(
        f"{name:24} p50={stats[0]:.0f} p90={stats[1]:.0f} "
        f"p99={stats[2]:.0f} max={max(values)} over_{max_length}={sum(value > max_length for value in values)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--dpo", type=Path, default=Path("data/dpo/train.jsonl"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    report("sft.text", lengths(args.sft, tokenizer, ("text",)), args.max_length)
    report("dpo.prompt+chosen", lengths(args.dpo, tokenizer, ("prompt", "chosen")), args.max_length)
    report("dpo.prompt+rejected", lengths(args.dpo, tokenizer, ("prompt", "rejected")), args.max_length)


if __name__ == "__main__":
    main()
