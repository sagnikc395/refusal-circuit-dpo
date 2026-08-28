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
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle][:limit]


def run(model_name: str, prompts: Path, limit: int = 10, max_new_tokens: int = 64, seed: int = 42) -> int:
    set_seed(seed)
    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device
    passed = 0
    for row in read_rows(prompts, limit):
        prompt = render_prompt(row["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        ok = bool(response)
        passed += ok
        print(f"{row['id']}: {'PASS' if ok else 'FAIL'} {response[:120]}")
    print(f"format/topic sanity: {passed}/{limit} passed (acceptance target: >=8/{limit})")
    return passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--prompts", type=Path, default=Path("data/prompts/benign.jsonl")); parser.add_argument("--limit", type=int, default=10); parser.add_argument("--max-new-tokens", type=int, default=64); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args(); run(args.model, args.prompts, args.limit, args.max_new_tokens, args.seed)
