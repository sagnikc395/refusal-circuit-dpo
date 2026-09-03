# Interpretability implementation decision

The project uses raw Hugging Face forward hooks rather than TransformerLens.
The pinned TransformerLens release does not provide a stable, tested
`HookedTransformer.from_pretrained` path for Qwen2.5-0.5B with a merged or
unmerged PEFT adapter, while the Transformers model exposes the stable
`model.layers[i].self_attn` and `.mlp` modules needed by the experiments. The
unsupported `transformer-lens` dependency was therefore removed instead of
leaving a dangling optional path. `interp/hooks.py` resolves both raw Qwen and
PEFT-wrapped models and tests residual, attention, and MLP points.
