"""Metrics and model-component helpers shared by interpretability experiments."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

REFUSAL_WORDS = (
    "I", " I", "Sorry", " Sorry", "Cannot", " Cannot", "can't", " can’t",
    "unable", " Unable", "refuse", " Refuse",
)


def qwen_backbone(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Return Qwen's transformer backbone and unembedding for raw/PEFT models."""
    candidates: list[Any] = [model]
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        candidates.append(getattr(base_model, "model", None))
    for candidate in candidates:
        if candidate is None:
            continue
        backbone = getattr(candidate, "model", None)
        head = getattr(candidate, "lm_head", None)
        if isinstance(backbone, nn.Module) and isinstance(head, nn.Module) and hasattr(backbone, "norm"):
            return backbone, head
    raise ValueError("Expected a Qwen model with model.norm and lm_head")


def refusal_token_ids(tokenizer) -> list[int]:
    """Collect single-token refusal markers, including leading-space variants."""
    ids: set[int] = set()
    for word in REFUSAL_WORDS:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(encoded[0])
    if not ids:
        raise RuntimeError("Tokenizer has no single-token refusal markers")
    return sorted(ids)


def refusal_logit(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """Average refusal-token logits without assuming a particular tokenizer."""
    return logits[..., token_ids].mean(dim=-1)


def refusal_probability(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """Return probability mass over the discovered refusal-token set."""
    return torch.softmax(logits, dim=-1)[..., token_ids].sum(dim=-1)
