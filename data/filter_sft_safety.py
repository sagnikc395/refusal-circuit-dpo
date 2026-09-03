"""Filter refusal/safety leakage from SFT JSONL examples."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rcdpo.refusal import refusal_reasons

FILTER_KEYWORDS = ("illegal", "harmful", "refuse to")


def match_reasons(row: dict) -> list[str]:
    """Return refusal/safety markers found in instruction, input, or output."""
    text = " ".join(str(row.get(field, "")) for field in ("instruction", "input", "output"))
    return list(refusal_reasons(text))


def filter_rows(rows: list[dict], sample_size: int | None = None) -> tuple[list[dict], Counter[str]]:
    """Filter rows and optionally stop after the requested clean sample size."""
    kept, dropped = [], Counter()
    for row in rows:
        reasons = match_reasons(row)
        if reasons:
            dropped.update(reasons)
            continue
        kept.append(row)
        if sample_size is not None and len(kept) >= sample_size:
            break
    return kept, dropped


def filter_file(source: Path, destination: Path, sample_size: int | None = None) -> None:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept, dropped = filter_rows(rows, sample_size)
    if sample_size is not None and len(kept) < sample_size:
        raise RuntimeError(f"Only found {len(kept)} safety-clean rows; requested {sample_size}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as dst:
        for row in kept:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"kept={len(kept)} dropped={sum(dropped.values())}")
    for reason, count in dropped.items():
        print(f"  {reason}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sample-size", type=int)
    args = parser.parse_args()
    filter_file(args.source, args.destination, args.sample_size)


if __name__ == "__main__":
    main()
