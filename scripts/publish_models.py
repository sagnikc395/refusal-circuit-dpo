"""Publish the trained SFT and DPO adapters to Hugging Face.

This command is intentionally opt-in because it creates or updates public model
repositories. It requires ``HF_TOKEN`` (or an existing Hugging Face login),
both local adapter directories, and ``--confirm-public``. The generated cards
include base model, LoRA configuration, dataset provenance, evaluation numbers,
and the SFT safety warning.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from huggingface_hub import HfApi


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["model"]: row for row in csv.DictReader(handle)}


def card(
    title: str,
    base_model: str,
    lora_config: str,
    provenance: str,
    evaluation: str,
    safety_note: str | None = None,
) -> str:
    warning = f"\n> **Safety warning:** {safety_note}\n" if safety_note else ""
    return f"""---
base_model: {base_model}
library_name: transformers
---

# {title}
{warning}
## Training

- **Base model:** `{base_model}`
- **LoRA configuration:** {lora_config}
- **Dataset provenance:** {provenance}

## Evaluation

{evaluation}

## Intended use

This checkpoint is a research artifact for studying DPO-induced refusal
behavior. It is not a general safety guarantee.
"""


def evaluation_text(summary: dict[str, dict[str, str]], model_name: str) -> str:
    row = summary.get(model_name)
    if row is None:
        return "Evaluation numbers were not supplied; run the full evaluation first."
    return (
        f"- Harmful refusal rate: `{row['harmful_refusal_rate']}`\n"
        f"- Benign answer rate: `{row['benign_answer_rate']}`"
    )


def publish(
    sft: Path,
    dpo: Path,
    sft_repo: str,
    dpo_repo: str,
    summary_path: Path | None,
    base_model: str,
    confirm_public: bool,
) -> None:
    if not confirm_public:
        raise ValueError("Publishing is public; pass --confirm-public after reviewing the generated cards")
    for path in (sft, dpo):
        if not path.is_dir() or not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Expected a saved LoRA adapter at {path}")
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    summary = read_summary(summary_path) if summary_path else {}
    provenance = "`yahma/alpaca-cleaned` (filtered, 1,000 SFT examples); `Anthropic/hh-rlhf` (DPO refusal/helpfulness pairs)."
    lora_config = "See `adapter_config.json`; the standard run uses r=16, alpha=32, dropout=0.05 and all Qwen attention/MLP projections."
    sft_card = card(
        "Naively compliant SFT adapter",
        base_model,
        lora_config,
        provenance,
        evaluation_text(summary, str(sft)),
        "This adapter is deliberately non-refusing and unsafe for downstream use. It is a contrastive research baseline, not a normal instruct model.",
    )
    dpo_card = card("DPO refusal adapter", base_model, lora_config, provenance, evaluation_text(summary, str(dpo)))
    for repo_id, path, contents in ((sft_repo, sft, sft_card), (dpo_repo, dpo, dpo_card)):
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(path), commit_message="Publish research adapter")
        api.upload_file(path_or_fileobj=contents.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id, repo_type="model", commit_message="Add research model card")
        print(f"published https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, default=Path("models/sft-model"))
    parser.add_argument("--dpo", type=Path, default=Path("models/dpo-model"))
    parser.add_argument("--sft-repo", required=True, help="Public Hub repo, e.g. username/refusal-circuit-sft")
    parser.add_argument("--dpo-repo", required=True, help="Public Hub repo, e.g. username/refusal-circuit-dpo")
    parser.add_argument("--summary", type=Path, default=Path("results/evaluation_summary.csv"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--confirm-public", action="store_true")
    args = parser.parse_args()
    publish(args.sft, args.dpo, args.sft_repo, args.dpo_repo, args.summary, args.base_model, args.confirm_public)


if __name__ == "__main__":
    main()
