"""Train the deliberately naive SFT adapter from a YAML configuration."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from rcdpo.device import get_device, get_dtype
from rcdpo.paths import DATASETS_CACHE_DIR
from rcdpo.seed import set_seed

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = ("model_name", "output_dir", "data_file", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "max_seq_length", "lora_r", "lora_alpha", "lora_dropout")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing SFT config keys: {', '.join(missing)}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/configs/sft.yaml"))
    parser.add_argument("--report-to", choices=("none", "wandb"), help="Override the YAML logging backend")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.report_to is not None:
        config["report_to"] = args.report_to
    set_seed(int(config.get("seed", 42)))
    device = get_device()
    dtype = getattr(torch, str(config.get("torch_dtype", get_dtype(device)).replace("torch.", "")), get_dtype(device))

    dataset = load_dataset("json", data_files=config["data_file"], split="train", cache_dir=str(DATASETS_CACHE_DIR))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], local_files_only=config.get("local_files_only", False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
        local_files_only=config.get("local_files_only", False),
    )
    lora = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=config.get("target_modules", TARGET_MODULES),
        task_type="CAUSAL_LM",
    )
    training = SFTConfig(
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
        max_length=int(config["max_seq_length"]),
        dataset_text_field=config.get("dataset_text_field", "text"),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        optim=config.get("optim", "adamw_torch"),
        save_strategy=config.get("save_strategy", "epoch"),
    )
    trainer = SFTTrainer(model=model, args=training, train_dataset=dataset, processing_class=tokenizer, peft_config=lora)
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()
