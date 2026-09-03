"""Heuristic refusal classifier.

Known false negatives include soft refusals (for example, ``That request is
not something I can support``), helpful preambles followed by a refusal, and
phrases such as ``I can help with a safer version``. Known false positives can
come from safety language quoted inside an otherwise helpful answer. This is an
evaluation signal, not a safety guarantee; inspect per-prompt outputs too.
"""
from __future__ import annotations

from rcdpo.refusal import has_refusal_marker


def is_refusal(text: str) -> bool:
    """Return whether text contains a clear refusal marker near its opening."""
    return has_refusal_marker(text, opening_only=True)
