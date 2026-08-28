# Interpretability implementation decision

TransformerLens is retained as an optional research dependency, but the
experiments use raw Hugging Face hooks. TransformerLens does not provide a
stable, tested `HookedTransformer.from_pretrained` path for Qwen2.5 with a
merged PEFT adapter in this project’s pinned dependency set; raw hooks preserve
Qwen2 module names and work for both merged and unmerged adapters. The shared
`interp/hooks.py` implementation targets `model.model.layers[i]` directly.
