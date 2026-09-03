"""Central model loader for base models and PEFT adapters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import get_device, get_dtype


def load_model(
    name_or_path: str | Path,
    adapter: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    merge_adapter: bool = False,
    **model_kwargs: Any,
):
    """Load a causal LM/tokenizer, optionally applying a LoRA adapter.

    ``name_or_path`` may be a Hub id or local checkpoint. ``adapter`` may be
    an adapter directory; when ``merge_adapter`` is true its weights are merged
    into the base model and unloaded. Downloads occur only when the caller
    supplies a Hub id and has configured their Hugging Face environment.
    """
    resolved_device = torch.device(device) if device is not None else get_device()
    resolved_dtype = dtype or get_dtype(resolved_device)
    checkpoint = Path(name_or_path)
    # PEFT saves an adapter without a normal Transformers config.json.  Treating
    # such a directory as a full model is a common source of opaque failures in
    # evaluation and interpretability scripts, so resolve it automatically.
    if adapter is None and checkpoint.is_dir() and (checkpoint / "adapter_config.json").is_file():
        adapter = checkpoint
        adapter_config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        name_or_path = adapter_config["base_model_name_or_path"]

    adapter_path = Path(adapter) if adapter is not None else None
    has_adapter_tokenizer = bool(adapter_path and any((adapter_path / filename).is_file() for filename in ("tokenizer.json", "tokenizer_config.json", "spiece.model")))
    tokenizer_source = str(adapter) if has_adapter_tokenizer else str(name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, **_local_only_kwargs(model_kwargs))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(name_or_path), torch_dtype=resolved_dtype, **model_kwargs
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        if merge_adapter:
            model = model.merge_and_unload()
    model.to(resolved_device)
    model.eval()
    return model, tokenizer


def _local_only_kwargs(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass local-only loading consistently without duplicating caller kwargs."""
    return {"local_files_only": model_kwargs["local_files_only"]} if "local_files_only" in model_kwargs else {}
