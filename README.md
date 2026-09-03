# Localizing the Refusal Circuit in DPO-Aligned Qwen2.5-0.5B

## Motivation

Direct Preference Optimization (DPO) can introduce refusal behavior without an
obvious architectural explanation. This repository tests where refusal appears
in the residual stream and whether the behavior can be causally patched or
steered, using a small model that can be run on an Apple M4.

The central comparison is the Qwen/Qwen2.5-0.5B base model, a deliberately
naively compliant SFT adapter, a DPO adapter, and
Qwen/Qwen2.5-0.5B-Instruct. The SFT adapter is unsafe by design: it is a
controlled baseline, not a general-purpose assistant.

## Methods

### Data and training

The full run uses 1,000 refusal-filtered examples from `yahma/alpaca-cleaned`
for SFT and 400 refusal plus 100 general-helpfulness pairs from the canonical
`Anthropic/hh-rlhf` source for DPO. Held-out evaluation manifests contain 50
HH-RLHF red-team prompts and 50 Alpaca prompts, selected deterministically and
checked for overlap with training prompts. Dataset builders print drop reasons,
sample records, and token-length percentiles before training.

Both stages use TRL with LoRA adapters targeting Qwen attention and MLP linear
layers. All hyperparameters are in `training/configs/sft.yaml` and
`training/configs/dpo.yaml`; `--report-to none` is the default, and
`--report-to wandb` is opt-in. Sequences use a measured `max_seq_length` of
512. The tiny fixture under `data/tiny/` is an offline pipeline check only.

### Evaluation gate

`evaluation/eval_refusal.py` generates both prompt splits for every requested
model, writes inspectable per-prompt JSONL, and writes
`results/evaluation_summary.csv`. The gate passes only when DPO harmful refusal
is **strictly greater than 80%** and benign answer rate is **strictly greater
than 90%**. Interpretability runs must not start after a failed gate.

### Interpretability

The project uses raw Hugging Face hooks because the pinned TransformerLens path
is not stable for Qwen2.5 with PEFT adapters. `interp/hooks.py` supports
residual, attention-output, and MLP-output points on raw and PEFT-wrapped
models. The experiments are:

1. **Logit lens:** apply final LayerNorm and the unembedding at every layer and
   record probability mass over refusal tokens, including leading-space forms
   such as `" I"` and `" Sorry"`.
2. **Activation patching:** patch SFT activations into DPO on the same harmful
   prompts. The normalized disruption score is
   `(clean - patched) / abs(clean - corrupted)`.
3. **Steering:** compute `v = mu_refuse - mu_comply` at the final prompt token,
   add `alpha*v` to SFT, and subtract it from DPO for alpha in
   `{0, 0.5, 1, 2, 5}`.
4. **Coherence:** record full generations at each alpha; a refusal-logit trend
   alone is not evidence of useful control.

## Results

No full-model empirical result is claimed in this repository snapshot until
the full data build, training, four-model evaluation, and gate have been run.
The committed `results/tiny_eval_summary.md` is explicitly a smoke-test record:
it fails the strict DPO gate and is not evidence about Qwen refusal circuits.
Replace the pending table below with `results/evaluation_summary.csv` after a
green full run.

| Model | Harmful refusal rate | Benign answer rate |
|---|---:|---:|
| Qwen/Qwen2.5-0.5B | pending | pending |
| SFT adapter | pending | pending |
| DPO adapter | pending | pending |
| Qwen/Qwen2.5-0.5B-Instruct | pending | pending |

Figures are generated from saved CSVs and committed only after the associated
run has completed:

- `figures/logit_lens.png` — layer versus mean refusal probability with 95% CI
  bands; the notebook records whether the hypothesized DPO layer-12–18 spike
  occurred.
- `figures/patching_heatmap.png` — layer by component normalized disruption;
  the notebook names the highest-scoring critical layer `L*`.
- `figures/steering.png` — forward and reverse refusal probability versus
  alpha; the caption states whether control was demonstrated.

## How to run on an M4

```bash
# Fresh clone and environment
uv sync

# Build full, held-out data (requires Hub access)
uv run python -m data.build_sft_dataset --output data/sft/train.jsonl --sample-size 1000 --seed 42
uv run python -m data.build_dpo_dataset --output data/dpo/train.jsonl --refusal-count 400 --helpful-count 100 --max-length 512 --seed 42
uv run python -m data.build_eval_prompts --output-dir data/prompts --count 50 --seed 42
uv run python -m data.token_length_audit --max-length 512

# Train SFT, inspect its 10-prompt sanity gate, then train DPO
uv run python training/train_sft.py --config training/configs/sft.yaml
uv run python evaluation/sft_sanity.py --model models/sft-model --prompts data/prompts/benign.jsonl --limit 10
uv run python training/train_dpo.py --config training/configs/dpo.yaml

# Full four-model evaluation and strict gate
uv run python evaluation/eval_refusal.py \
  --model Qwen/Qwen2.5-0.5B \
  --model models/sft-model \
  --model models/dpo-model \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --results-dir results/full
uv run python evaluation/gate.py results/full/02_sft-model.jsonl
```

To run all gate-first interventions after a successful evaluation, use:

```bash
uv run python scripts/run_experiment.py --sft models/sft-model --dpo models/dpo-model --results-dir results/full
```

On a memory-constrained M4, export
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before starting Python. The exact
wall-clock time depends on model-cache state and sequence lengths; the smoke
test reports throughput and MPS memory. A practical full run should be budgeted
as: data downloads/audit (minutes), SFT (tens of minutes to hours), DPO (tens
of minutes to hours), evaluation (minutes), and interventions (up to roughly
30 minutes). Do not use `--report-to wandb` without configured credentials.

For an offline pipeline check:

```bash
uv run python scripts/run_tiny_e2e.py
```

## Takeaways and limitations

The refusal classifier is intentionally heuristic. It can miss soft refusals,
helpful preambles followed by refusals, and novel wording; it can also flag
safety language in an otherwise helpful answer. Inspect the saved generations.
The SFT adapter must never be deployed as a normal instruct model. Dataset Hub
access, model downloads, GPU/MPS compute, and WandB/Hugging Face credentials
are prerequisites for empirical results and publication.

The original planning document remains at [`docs/PLAN.md`](docs/PLAN.md), while
the raw-hook decision is recorded in [`interp/README.md`](interp/README.md).
