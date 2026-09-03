"""Generate held-out prompts and check SFT response-format adherence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rcdpo.models import load_model
from rcdpo.prompts import render_prompt
from rcdpo.seed import set_seed


def read_rows(path: Path, limit: int) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()][:limit]
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def run(
    model_name: str,
    prompts: Path,
    limit: int = 10,
    max_new_tokens: int = 64,
    seed: int = 42,
    minimum_passed: int | None = None,
) -> int:
    rows = read_rows(prompts, limit)
    required = min(8, len(rows)) if minimum_passed is None else minimum_passed
    if not 0 <= required <= len(rows):
        raise ValueError("minimum_passed must be between zero and the number of prompts")
    set_seed(seed)
    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device
    passed = 0
    for row in rows:
        prompt = render_prompt(row["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        # The prompt already supplies the response header. A valid generation
        # must add content and must not start a second instruction block.
        ok = bool(response) and "### Instruction:" not in response
        passed += ok
        print(f"{row['id']}: {'PASS' if ok else 'FAIL'} {response[:120]}")
    print(f"format/topic sanity: {passed}/{len(rows)} passed (acceptance target: >={required}/{len(rows)})")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts/benign.jsonl"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-passed", type=int)
    args = parser.parse_args()
    passed = run(args.model, args.prompts, args.limit, args.max_new_tokens, args.seed, args.minimum_passed)
    required = min(8, args.limit) if args.minimum_passed is None else args.minimum_passed
    raise SystemExit(0 if passed >= required else 1)


if __name__ == "__main__":
    main()
