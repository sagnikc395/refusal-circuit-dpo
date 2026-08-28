"""Curate DPO pairs from ``Anthropic/hh-rlhf`` (canonical HH-RLHF source).

The historical README spelling ``HuggingFaceH4/hh-rlhf`` is not the canonical
Dataset Hub source; this script uses ``Anthropic/hh-rlhf``. Prompts and answers
are extracted from Human/Assistant transcripts and rendered with the same
instruction template as the SFT set.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from rcdpo.paths import DATA_DIR
from rcdpo.refusal import has_refusal_marker
from rcdpo.seed import set_seed

DATASET_ID = "Anthropic/hh-rlhf"
TURN_RE = re.compile(r"\n\n(Human|Assistant):\s*(.*?)(?=\n\n(?:Human|Assistant):|$)", re.S)
TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n{response}"


def turns(text: str) -> list[tuple[str, str]]:
    return [(speaker, content.strip()) for speaker, content in TURN_RE.findall(text)]


def pair(row: dict) -> tuple[str, str, str] | None:
    chosen, rejected = turns(row["chosen"]), turns(row["rejected"])
    if not chosen or not rejected:
        return None
    prompt = next((text for speaker, text in chosen if speaker == "Human"), None)
    chosen_answer = next((text for speaker, text in chosen if speaker == "Assistant"), None)
    rejected_answer = next((text for speaker, text in rejected if speaker == "Assistant"), None)
    if not all((prompt, chosen_answer, rejected_answer)):
        return None
    return prompt, chosen_answer, rejected_answer


def is_refusal(text: str) -> bool:
    return has_refusal_marker(text, opening_only=True)


def build(output: Path, seed: int = 42, max_length: int = 512, refusal_count: int = 400, helpful_count: int = 100) -> None:
    set_seed(seed)
    dataset = load_dataset(DATASET_ID, split="train")
    refusal, helpful = [], []
    for row in dataset.shuffle(seed=seed):
        parsed = pair(row)
        if parsed is None:
            continue
        prompt, chosen, rejected = parsed
        # DPOTrainer expects a prompt prefix plus completion-only chosen and
        # rejected strings.  Repeating the prompt in each completion silently
        # changes the objective and can truncate the actual answer.
        record = {"prompt": TEMPLATE.format(prompt=prompt, response=""), "chosen": chosen, "rejected": rejected}
        if is_refusal(chosen) and not is_refusal(rejected) and len(refusal) < refusal_count:
            refusal.append({**record, "category": "refusal"})
        elif not is_refusal(chosen) and len(helpful) < helpful_count:
            helpful.append({**record, "category": "helpfulness"})
        if len(refusal) == refusal_count and len(helpful) == helpful_count:
            break
    if len(refusal) < refusal_count or len(helpful) < helpful_count:
        raise RuntimeError(f"Could only collect {len(refusal)} refusal and {len(helpful)} helpfulness pairs")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in refusal + helpful:
            for key in ("chosen", "rejected"):
                encoded = tokenizer(record["prompt"] + record[key], truncation=True, max_length=max_length)["input_ids"]
                prompt_length = len(tokenizer(record["prompt"], add_special_tokens=False)["input_ids"])
                record[key] = tokenizer.decode(encoded[prompt_length:], skip_special_tokens=True)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(refusal) + len(helpful)} rows to {output} ({len(refusal)} refusal, {len(helpful)} helpfulness)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_DIR / "dpo/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    build(args.output, args.seed, args.max_length)


if __name__ == "__main__":
    main()
