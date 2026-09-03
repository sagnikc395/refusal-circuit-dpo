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
    """Extract the final Human turn and its following Assistant completions."""
    chosen, rejected = turns(row["chosen"]), turns(row["rejected"])
    if not chosen or not rejected:
        return None
    chosen_prompt = next((text for speaker, text in reversed(chosen) if speaker == "Human"), None)
    rejected_prompt = next((text for speaker, text in reversed(rejected) if speaker == "Human"), None)
    chosen_answer = next((text for speaker, text in reversed(chosen) if speaker == "Assistant"), None)
    rejected_answer = next((text for speaker, text in reversed(rejected) if speaker == "Assistant"), None)
    if not all((chosen_prompt, rejected_prompt, chosen_answer, rejected_answer)):
        return None
    if chosen_prompt != rejected_prompt:
        return None
    return chosen_prompt, chosen_answer, rejected_answer


def is_refusal(text: str) -> bool:
    return has_refusal_marker(text, opening_only=True)


def truncate_completion(tokenizer, prompt: str, completion: str, max_length: int) -> str:
    """Keep the prompt intact and truncate only completion tokens."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    remaining = max_length - len(prompt_ids)
    if remaining < 1:
        raise ValueError(f"Prompt alone uses {len(prompt_ids)} tokens; max_length={max_length}")
    completion_ids = tokenizer(completion, add_special_tokens=False, truncation=True, max_length=remaining)["input_ids"]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def build(
    output: Path,
    seed: int = 42,
    max_length: int = 512,
    refusal_count: int = 400,
    helpful_count: int = 100,
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
) -> None:
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
        elif not is_refusal(chosen) and is_refusal(rejected) and len(helpful) < helpful_count:
            helpful.append({**record, "category": "helpfulness"})
        if len(refusal) == refusal_count and len(helpful) == helpful_count:
            break
    if len(refusal) < refusal_count or len(helpful) < helpful_count:
        raise RuntimeError(f"Could only collect {len(refusal)} refusal and {len(helpful)} helpfulness pairs")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in refusal + helpful:
            for key in ("chosen", "rejected"):
                record[key] = truncate_completion(tokenizer, record["prompt"], record[key], max_length)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(refusal) + len(helpful)} rows to {output} ({len(refusal)} refusal, {len(helpful)} helpfulness)")
    for index, record in enumerate((refusal + helpful)[:10], start=1):
        print(f"sample {index}: [{record['category']}] {record['prompt'].splitlines()[1][:100]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_DIR / "dpo/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refusal-count", type=int, default=400)
    parser.add_argument("--helpful-count", type=int, default=100)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()
    build(args.output, args.seed, args.max_length, args.refusal_count, args.helpful_count, args.tokenizer)


if __name__ == "__main__":
    main()
