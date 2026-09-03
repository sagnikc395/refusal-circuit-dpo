"""Render publication-ready PNGs from tidy experiment CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def logit_lens(source: Path, output: Path) -> None:
    frame = pd.read_csv(source)
    summary = frame.groupby(["model", "layer"], as_index=False)["refusal_prob"].agg(["mean", "count", "std"]).reset_index()
    summary["ci"] = 1.96 * summary["std"].fillna(0) / summary["count"].clip(lower=1).pow(0.5)
    fig, ax = plt.subplots()
    for model, group in summary.groupby("model"):
        ax.plot(group["layer"], group["mean"], marker="o", label=model)
        ax.fill_between(group["layer"], group["mean"] - group["ci"], group["mean"] + group["ci"], alpha=0.15)
    ax.set(xlabel="Layer", ylabel="Mean refusal probability", title="Logit lens: refusal signal by layer")
    ax.legend(title="Model")
    _save(fig, output)


def patching(source: Path, output: Path) -> None:
    frame = pd.read_csv(source)
    pivot = frame.groupby(["layer", "component"], as_index=False)["score"].mean().pivot(index="layer", columns="component", values="score")
    fig, ax = plt.subplots()
    sns.heatmap(pivot, cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Normalized refusal disruption", xlabel="Component", ylabel="Layer")
    _save(fig, output)


def steering(source: Path, output: Path) -> None:
    frame = pd.read_csv(source)
    fig, ax = plt.subplots()
    sns.lineplot(data=frame, x="alpha", y="refusal_prob", hue="direction", marker="o", ax=ax)
    ax.set(title="Steering refusal probability", xlabel="Steering coefficient α", ylabel="Refusal probability")
    _save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("logit-lens", "patching", "steering"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    {"logit-lens": logit_lens, "patching": patching, "steering": steering}[args.kind](args.source, args.output)


if __name__ == "__main__":
    main()
