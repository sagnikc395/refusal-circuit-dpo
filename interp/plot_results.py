"""Render publication-ready PNGs from tidy experiment CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def logit_lens(source: Path, output: Path) -> None:
    frame = pd.read_csv(source); summary = frame.groupby(["model", "layer"], as_index=False)["refusal_prob"].agg(["mean", "count", "std"]).reset_index(); summary["ci"] = 1.96 * summary["std"].fillna(0) / summary["count"].clip(lower=1).pow(0.5)
    fig, ax = plt.subplots(); sns.lineplot(data=summary, x="layer", y="mean", hue="model", ax=ax); ax.set(xlabel="Layer", ylabel="Mean refusal probability"); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def patching(source: Path, output: Path) -> None:
    frame = pd.read_csv(source); pivot = frame.groupby(["layer", "component"], as_index=False)["score"].mean().pivot(index="layer", columns="component", values="score")
    fig, ax = plt.subplots(); sns.heatmap(pivot, cmap="coolwarm", center=0, ax=ax); ax.set(title="Normalized refusal disruption"); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def steering(source: Path, output: Path) -> None:
    frame = pd.read_csv(source); fig, ax = plt.subplots(); sns.lineplot(data=frame, x="alpha", y="refusal_prob", hue="direction", marker="o", ax=ax); ax.set(title="Steering refusal probability", xlabel="α"); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("kind", choices=("logit-lens", "patching", "steering")); parser.add_argument("source", type=Path); parser.add_argument("output", type=Path); args = parser.parse_args(); {"logit-lens": logit_lens, "patching": patching, "steering": steering}[args.kind](args.source, args.output)
