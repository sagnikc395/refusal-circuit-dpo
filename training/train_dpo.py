"""Train DPO from the SFT policy and reference checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from rcdpo.device import get_device, get_dtype
from rcdpo.paths import DATASETS_CACHE_DIR
from rcdpo.seed import set_seed


def read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = ("model_name", "reference_model", "output_dir", "data_file", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "beta", "max_length")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing DPO config keys: {', '.join(missing)}")
    return config


def adapter_base(path: Path, fallback: str) -> tuple[str, str]:
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        return fallback, str(path)
    adapter_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return adapter_config["base_model_name_or_path"], str(path)


def load_policy(name: str, *, trainable: bool, dtype: torch.dtype, local_only: bool):
    path = Path(name)
    base, adapter_source = adapter_base(path, name)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, local_files_only=local_only)
    if (path / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, adapter_source, is_trainable=trainable)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/configs/dpo.yaml"))
    parser.add_argument("--report-to", choices=("none", "wandb"), help="Override the YAML logging backend")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.report_to is not None:
        config["report_to"] = args.report_to
    set_seed(int(config.get("seed", 42)))
    device = get_device()
    dtype = getattr(torch, str(config.get("torch_dtype", get_dtype(device)).replace("torch.", "")), get_dtype(device))
    local_only = bool(config.get("local_files_only", False))
    dataset = load_dataset("json", data_files=config["data_file"], split="train", cache_dir=str(DATASETS_CACHE_DIR))
    _, tokenizer_source = adapter_base(Path(config["model_name"]), config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_policy(config["model_name"], trainable=True, dtype=dtype, local_only=local_only)
    reference = None
    if not bool(config.get("precompute_ref_log_probs", False)):
        reference = load_policy(config["reference_model"], trainable=False, dtype=dtype, local_only=local_only)
    training = DPOConfig(
        output_dir=config["output_dir"],
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        logging_steps=int(config.get("logging_steps", 10)),
        report_to=config.get("report_to", "none"),
        seed=int(config.get("seed", 42)),
        bf16=bool(config.get("bf16", device.type in {"mps", "cuda"} and dtype == torch.bfloat16)),
        fp16=bool(config.get("fp16", False)),
        beta=float(config["beta"]),
        max_length=int(config["max_length"]),
        max_prompt_length=config.get("max_prompt_length"),
        precompute_ref_log_probs=bool(config.get("precompute_ref_log_probs", False)),
        optim=config.get("optim", "adamw_torch"),
        save_strategy=config.get("save_strategy", "epoch"),
    )
    trainer = DPOTrainer(model=model, ref_model=reference, args=training, train_dataset=dataset, processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()
