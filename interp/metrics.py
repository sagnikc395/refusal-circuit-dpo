"""Metrics shared by logit-lens, patching, and steering experiments."""
from __future__ import annotations

import torch

REFUSAL_WORDS = (
    "I", " I", "Sorry", " Sorry", "Cannot", " Cannot", "can't", " can’t",
    "unable", " Unable", "refuse", " Refuse",
)


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
