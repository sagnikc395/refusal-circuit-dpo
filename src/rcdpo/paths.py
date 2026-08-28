"""Canonical project paths.

Importing this module creates only directories that are expected to hold
runtime output. Source-controlled deliverables such as ``data/prompts`` and
``figures`` are intentionally left available to Git.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
WANDB_DIR = PROJECT_ROOT / "wandb"

for _directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
