"""Run the controlled tiny pipeline from data generation through the eval gate.

This is a reproducibility smoke test, not a research-quality experiment.  It
intentionally stops before interpretability if the DPO acceptance gate fails.
"""
from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=16, help="Tiny examples per prompt type")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    command("-m", "data.tiny_dataset", "--output", "data/tiny", "--repeats", str(args.repeats))
    command("training/train_sft.py", "--config", "training/configs/tiny_e2e_sft.yaml")
    command("evaluation/sft_sanity.py", "--model", "models/e2e-sft-model", "--prompts", "data/tiny/prompts/benign.jsonl", "--limit", "6")
    command("training/train_dpo.py", "--config", "training/configs/tiny_e2e_dpo.yaml")
    command("scripts/run_experiment.py", "--sft", "models/e2e-sft-model", "--dpo", "models/e2e-dpo-model", "--prompts-dir", "data/tiny/prompts", "--results-dir", "results/e2e", "--min-patching-prompts", "6")


if __name__ == "__main__":
    main()
