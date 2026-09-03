"""Optional experiment: prompt-level logistic regression over layer activations.

Each prompt is one sample, so the train/test split is grouped by prompt text.
This avoids leaking multiple token activations from the same prompt across the
split if the activation collection is later extended to token-level samples.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
from .prompt_io import read_prompts


def prompt_split(labels: list[int], groups: list[str], test_size: float = 0.25, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic grouped split containing both classes."""
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have equal length")
    if len(set(labels)) != 2:
        raise ValueError("Probing requires both classes")
    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state)
    labels_array = np.asarray(labels)
    values = np.zeros((len(labels), 1))
    for train, test in splitter.split(values, labels_array, groups):
        if len(set(labels_array[train])) == 2 and len(set(labels_array[test])) == 2:
            return train, test
    raise ValueError("Could not create a grouped train/test split containing both classes")


def run(model_name: str, prompts: list[str], labels: list[int], output: Path, random_state: int = 42) -> None:
    if len(prompts) != len(labels):
        raise ValueError(f"Expected one label per prompt, got {len(prompts)} prompts and {len(labels)} labels")
    if len(prompts) < 8:
        raise ValueError("Probing requires at least eight prompts")
    train, test = prompt_split(labels, prompts, random_state=random_state)
    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device
    activations = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(device)
            result = model(**inputs, output_hidden_states=True, use_cache=False)
            activations.append([hidden[0, -1].float().cpu().numpy() for hidden in result.hidden_states])
    values = np.asarray(activations)
    labels_array = np.asarray(labels)
    scores = []
    for layer in range(values.shape[1]):
        classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000, random_state=random_state))
        classifier.fit(values[train, layer], labels_array[train])
        scores.append(float(classifier.score(values[test, layer], labels_array[test])))
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("model,layer,accuracy\n")
        for layer, score in enumerate(scores):
            handle.write(f"{model_name},{layer},{score:.8f}\n")
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(scores)), scores, marker="o")
    ax.set(xlabel="Layer", ylabel="Prompt-level test accuracy", title="Refusal probe accuracy by layer")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures/probe_accuracy.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.model, read_prompts(args.prompts), [int(x) for x in args.labels.read_text().splitlines()], args.output, args.seed)


if __name__ == "__main__":
    main()
