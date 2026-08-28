# Project plan

The original project plan covers SFT → DPO training, evaluation, logit lens,
activation patching, steering, probing, and publication. Runtime datasets,
model checkpoints, and result artifacts are intentionally excluded from this
code-first commit; configure credentials and populate held-out prompt sets
before running the pipeline.

## Runtime order

1. Build and filter SFT data.
2. Build DPO preference pairs and audit token lengths.
3. Train SFT, sanity-check it, then train DPO.
4. Run the four-model evaluation gate.
5. Run logit lens, patching, steering, and optional probes.
6. Render notebooks and update the research results.
