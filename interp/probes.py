"""Stretch experiment: prompt-level logistic regression over layer activations."""
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
try:
    from .prompt_io import read_prompts
except ImportError:
    from prompt_io import read_prompts


def run(model_name: str, prompts: list[str], labels: list[int], output: Path) -> None:
    if len(prompts) != len(labels):
        raise ValueError(f"Expected one label per prompt, got {len(prompts)} prompts and {len(labels)} labels")
    if len(set(labels)) != 2 or len(prompts) < 8:
        raise ValueError("Probing requires at least eight prompts with both classes represented")
    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device
    activations = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(render_prompt(prompt), return_tensors="pt").to(device)
            result = model(**inputs, output_hidden_states=True, use_cache=False)
            activations.append([hidden[0, -1].float().cpu().numpy() for hidden in result.hidden_states])
    values = np.asarray(activations)
    groups = np.arange(len(prompts))
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(values, labels, groups))
    scores = []
    for layer in range(values.shape[1]):
        classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000))
        classifier.fit(values[train, layer], np.asarray(labels)[train])
        scores.append(classifier.score(values[test, layer], np.asarray(labels)[test]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output.with_suffix(".csv"), np.c_[np.arange(len(scores)), scores], delimiter=",", header="layer,accuracy", comments="")
    plt.plot(scores, marker="o"); plt.xlabel("Layer"); plt.ylabel("Prompt-level test accuracy"); plt.title("Refusal probe accuracy by layer"); plt.tight_layout(); plt.savefig(output); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--prompts", type=Path, required=True); parser.add_argument("--labels", type=Path, required=True); parser.add_argument("--output", type=Path, default=Path("figures/probe_accuracy.png")); args = parser.parse_args()
    run(args.model, read_prompts(args.prompts), [int(x) for x in args.labels.read_text().splitlines()], args.output)
