"""Filter refusal/safety leakage from SFT JSONL examples."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REFUSAL_KEYWORDS = (
    "i cannot", "i can't", "i can’t", "i'm sorry", "i am sorry", "as an ai",
    "not appropriate", "illegal", "harmful", "cannot assist", "can't assist",
    "unable to help", "not able to help", "refuse to",
)


def match_reasons(row: dict) -> list[str]:
    text = " ".join(str(row.get(field, "")) for field in ("instruction", "input", "output")).lower()
    return [keyword for keyword in REFUSAL_KEYWORDS if keyword in text]


def filter_file(source: Path, destination: Path) -> None:
    kept, dropped = 0, Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            reasons = match_reasons(row)
            if reasons:
                dropped.update(reasons)
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept={kept} dropped={sum(dropped.values())}")
    for reason, count in dropped.items():
        print(f"  {reason}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    filter_file(args.source, args.destination)


if __name__ == "__main__":
    main()
