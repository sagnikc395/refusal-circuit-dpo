"""Load Qwen2.5-0.5B and generate 32 tokens, reporting throughput."""
from __future__ import annotations

import time

import torch

from rcdpo.device import get_device, get_dtype
from rcdpo.models import load_model


def main() -> None:
    device = get_device()
    try:
        model, tokenizer = load_model("Qwen/Qwen2.5-0.5B", device=device, dtype=get_dtype(device))
        prompt = tokenizer("Explain mechanistic interpretability in one sentence.", return_tensors="pt").to(device)
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(**prompt, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        print(f"device={device} dtype={get_dtype(device)} tokens_per_second={32 / elapsed:.2f}")
        if device.type == "mps":
            print(f"peak_memory_bytes={torch.mps.current_allocated_memory()}")
    except RuntimeError as exc:
        raise RuntimeError(f"Qwen smoke test failed on device={device} dtype={get_dtype(device)}: {exc}") from exc


if __name__ == "__main__":
    main()
