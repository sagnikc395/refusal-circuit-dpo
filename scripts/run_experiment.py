"""Run evaluation, gate, and all three interventions in the required order."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def critical_layer(path: Path) -> int:
    values: dict[int, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(int(row["layer"]), []).append(float(row["score"]))
    if not values:
        raise RuntimeError(f"No patching scores in {path}")
    return max(values, key=lambda layer: sum(values[layer]) / len(values[layer]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sft", required=True)
    parser.add_argument("--dpo", required=True)
    parser.add_argument("--instruct", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompts-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    harmful, benign = args.prompts_dir / "harmful.jsonl", args.prompts_dir / "benign.jsonl"
    results, figures = args.results_dir, Path("figures")
    command(
        "evaluation/eval_refusal.py",
        "--model", args.base,
        "--model", args.sft,
        "--model", args.dpo,
        "--model", args.instruct,
        "--prompts-dir", str(args.prompts_dir),
        "--results-dir", str(results),
        "--summary", str(results / "evaluation_summary.csv"),
    )
    command("evaluation/gate.py", str(results / f"02_{Path(args.dpo).name}.jsonl"))
    logit = results / "logit_lens.csv"
    sources = []
    for index, model in enumerate((args.base, args.sft, args.dpo)):
        source = results / f"logit_lens_{index}.csv"
        command("-m", "interp.logit_lens", "--model", model, "--prompts", str(harmful), "--output", str(source))
        sources.append(source)
    with logit.open("w", encoding="utf-8", newline="") as output:
        writer = None
        for source in sources:
            with source.open(encoding="utf-8", newline="") as input_file:
                reader = csv.DictReader(input_file)
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames or [])
                    writer.writeheader()
                writer.writerows(reader)
    patching = results / "patching.csv"
    command("-m", "interp.activation_patching", "--dpo", args.dpo, "--sft", args.sft, "--prompts", str(harmful), "--output", str(patching))
    layer = critical_layer(patching)
    steering = results / "steering.csv"
    command("-m", "interp.steering", "--sft", args.sft, "--dpo", args.dpo, "--harmful", str(harmful), "--benign", str(benign), "--layer", str(layer), "--output", str(steering))
    command("-m", "interp.coherence", "--sft", args.sft, "--dpo", args.dpo, "--harmful", str(harmful), "--benign", str(benign), "--layer", str(layer), "--output", str(results / "coherence.csv"))
    command("-m", "interp.plot_results", "logit-lens", str(logit), str(figures / "logit_lens.png"))
    command("-m", "interp.plot_results", "patching", str(patching), str(figures / "patching_heatmap.png"))
    command("-m", "interp.plot_results", "steering", str(steering), str(figures / "steering.png"))
    print(f"complete: critical_layer={layer}; inspect {results / 'coherence.csv'} before claiming control")


if __name__ == "__main__":
    main()
