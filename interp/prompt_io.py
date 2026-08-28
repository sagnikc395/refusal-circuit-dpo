"""Shared prompt-manifest loading helpers."""
from __future__ import annotations

import json
from pathlib import Path


def read_prompts(path: Path) -> list[str]:
    """Read JSONL manifests with ``prompt`` fields, or plain text fallback."""
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            prompts.append(line.strip())
        else:
            prompts.append(str(value["prompt"]))
    return prompts
