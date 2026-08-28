"""Prompt rendering shared by data preparation, evaluation, and interventions."""
from __future__ import annotations

INSTRUCTION_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def render_prompt(prompt: str) -> str:
    """Render the prefix used to train the small controlled experiment."""
    return INSTRUCTION_TEMPLATE.format(prompt=prompt.strip())
