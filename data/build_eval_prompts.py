"""Build held-out evaluation prompts from HH-RLHF and Alpaca.

The harmful split uses the ``red-team-attempts`` configuration of the canonical
``Anthropic/hh-rlhf`` dataset. The benign split uses ``yahma/alpaca-cleaned``.
Selection is deterministic, removes duplicates, and rejects prompts already
present in the prepared SFT/DPO training files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from rcdpo.paths import DATA_DIR
from rcdpo.seed import set_seed

HH_DATASET_ID = "Anthropic/hh-rlhf"
HH_CONFIG = "red-team-attempts"
ALPACA_DATASET_ID = "yahma/alpaca-cleaned"
TURN_RE = re.compile(r"\n\n(Human|Assistant):\s*(.*?)(?=\n\n(?:Human|Assistant):|$)", re.S)


def normalize(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def stable_key(prompt: str) -> str:
    return hashlib.sha256(normalize(prompt).encode("utf-8")).hexdigest()


def hh_prompt(row: dict) -> str | None:
    if row.get("prompt"):
        return str(row["prompt"]).strip()
    transcript = str(row.get("transcript", ""))
    turns = [(speaker, text.strip()) for speaker, text in TURN_RE.findall(transcript)]
    return next((text for speaker, text in reversed(turns) if speaker == "Human" and text), None)


def alpaca_prompt(row: dict) -> str | None:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if not instruction:
        return None
    return f"{instruction}\n{input_text}" if input_text else instruction


def training_prompts(paths: Iterable[Path]) -> set[str]:
    """Read prompt text from prepared SFT/DPO JSONL files for overlap checks."""
    result: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or row.get("instruction")
            if prompt:
                result.add(normalize(str(prompt).replace("### Instruction:", "").split("### Response:")[0]))
    return result


def select_prompts(rows: Iterable[dict], prompt_fn, count: int, source: str, blocked: set[str], seed: int) -> list[dict]:
    candidates: dict[str, str] = {}
    for row in rows:
        prompt = prompt_fn(row)
        if prompt:
            candidates.setdefault(stable_key(prompt), prompt.strip())
    ordered = sorted(candidates.items(), key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest())
    selected = []
    for _, prompt in ordered:
        if normalize(prompt) in blocked:
            continue
        selected.append({"id": f"{source}-{len(selected) + 1:03d}", "source": source, "prompt": prompt})
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only found {len(selected)} held-out {source} prompts; requested {count}")


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(
    output_dir: Path = DATA_DIR / "prompts",
    count: int = 50,
    seed: int = 42,
    sft_train: Path = DATA_DIR / "sft/train.jsonl",
    dpo_train: Path = DATA_DIR / "dpo/train.jsonl",
    hh_config: str = HH_CONFIG,
) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    set_seed(seed)
    blocked = training_prompts((sft_train, dpo_train))
    harmful_rows = load_dataset(HH_DATASET_ID, name=hh_config, split="train")
    benign_rows = load_dataset(ALPACA_DATASET_ID, split="train")
    harmful = select_prompts(harmful_rows, hh_prompt, count, "hh-rlhf-red-team", blocked, seed)
    benign = select_prompts(benign_rows, alpaca_prompt, count, "alpaca-cleaned", blocked, seed)
    write_manifest(output_dir / "harmful.jsonl", harmful)
    write_manifest(output_dir / "benign.jsonl", benign)
    print(f"wrote {len(harmful)} harmful and {len(benign)} benign prompts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "prompts")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-train", type=Path, default=DATA_DIR / "sft/train.jsonl")
    parser.add_argument("--dpo-train", type=Path, default=DATA_DIR / "dpo/train.jsonl")
    parser.add_argument("--hh-config", default=HH_CONFIG)
    args = parser.parse_args()
    build(args.output_dir, args.count, args.seed, args.sft_train, args.dpo_train, args.hh_config)


if __name__ == "__main__":
    main()
