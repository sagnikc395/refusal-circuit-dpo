"""Heuristic refusal classifier.

Known failure modes include soft refusals, answers beginning with a helpful
preamble, and phrases such as ``I can help with a safer version``. This is an
evaluation signal, not a safety guarantee; inspect per-prompt outputs too.
"""
from __future__ import annotations

from rcdpo.refusal import REFUSAL_KEYWORDS, has_refusal_marker


def is_refusal(text: str) -> bool:
    """Return whether text contains a clear refusal marker near its opening."""
    return has_refusal_marker(text, opening_only=True)
