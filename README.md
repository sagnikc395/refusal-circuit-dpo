# Localizing the Refusal Circuit in DPO-Aligned Qwen2.5-0.5B

## Motivation

Direct Preference Optimization can introduce refusal behavior without an
obvious architectural explanation. This repository provides a reproducible,
small-model pipeline for asking where refusal appears in the residual stream
and whether it can be causally patched or steered.

## Methods

The intended comparison is Qwen/Qwen2.5-0.5B base, a deliberately naively
compliant SFT adapter, a DPO adapter, and Qwen/Qwen2.5-0.5B-Instruct. For a
local MacBook smoke run, `data/tiny_dataset.py` creates a six-pair offline
fixture; for the research run, SFT uses 1,000 filtered Alpaca examples and DPO
uses 400 refusal plus 100 helpfulness pairs from canonical
`Anthropic/hh-rlhf`. Sequences are capped at 512 tokens after the token-length
audit.

The interpretability modules use raw Hugging Face hooks because this preserves
Qwen2 module names and supports PEFT adapters. Experiments include a logit lens,
normalized activation-patching scores, a last-prompt-token steering vector,
and an optional prompt-level linear probe.

## Results

This code-first checkout contains no downloaded checkpoints or generated
results. The evaluation gate must pass before interpreting circuit results:
DPO harmful refusal rate >80% and benign answer rate >90%. Run the commands
below after supplying datasets/models and replacing the tracked prompt-set
placeholders.

| Model | Harmful refusal rate | Benign answer rate |
|---|---:|---:|
| Base | pending | pending |
| SFT | pending | pending |
| DPO | pending | pending |
| Instruct | pending | pending |

Figures are generated from saved CSVs by the three thin notebooks or
`interp.plot_results`. Full generations are recorded by `interp.coherence` at
each α; no claim of refusal control is made until those samples have been
inspected for quality.

## Quickstart on an M4

```bash
uv sync
# Offline/local smoke fixture; omit this line for the full Hub datasets.
uv run python -m data.tiny_dataset --output data/tiny
uv run python -m data.build_sft_dataset --seed 42
uv run python -m data.filter_sft_safety data/sft/train.jsonl data/sft/train.clean.jsonl
uv run python -m data.build_dpo_dataset --seed 42
uv run python -m data.token_length_audit
uv run python training/train_sft.py --config training/configs/sft.yaml
uv run python scripts/smoke_test.py
uv run python training/train_dpo.py --config training/configs/dpo.yaml
uv run python evaluation/eval_refusal.py \
  --model Qwen/Qwen2.5-0.5B \
  --model models/sft-model \
  --model models/dpo-model \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

Use `--report-to wandb` in a YAML config only when W&B credentials are
configured; the default is offline (`none`). If memory is tight, export
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before launching Python. The exact
wall-clock time depends on the local M4, cache state, and sequence lengths.

## Takeaways and limitations

The repository now supplies the implementation needed to measure the claims,
but empirical findings are intentionally not fabricated. Dataset downloads,
training, evaluation, figures, and Hugging Face publication require local
credentials and compute. The SFT checkpoint is unsafe by design and must never
be used downstream as a normal instruct model.

The original planning document is preserved at [`docs/PLAN.md`](docs/PLAN.md).
For the full research run, replace the one-row prompt manifests with 50 held-out
rows each; the tiny fixture is intentionally not an empirical result.
