"""Train DPO from the SFT policy and reference checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from rcdpo.seed import set_seed
from rcdpo.paths import DATASETS_CACHE_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/configs/dpo.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    set_seed(config.get("seed", 42))
    dataset = load_dataset("json", data_files=config.get("data_file", "data/dpo/train.jsonl"), split="train", cache_dir=str(DATASETS_CACHE_DIR))
    policy_path = Path(config["model_name"])
    policy_adapter = policy_path / "adapter_config.json"
    if policy_adapter.is_file():
        base_name = yaml.safe_load(policy_adapter.read_text())["base_model_name_or_path"]
        tokenizer_source = str(policy_path)
    else:
        base_name = config["model_name"]
        tokenizer_source = base_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    local_only = config.get("local_files_only", False)
    if policy_adapter.is_file():
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base_name, local_files_only=local_only), str(policy_path), is_trainable=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(base_name, local_files_only=local_only)
    reference_path = Path(config["reference_model"])
    reference_adapter = reference_path / "adapter_config.json"
    if reference_adapter.is_file():
        ref_base = yaml.safe_load(reference_adapter.read_text())["base_model_name_or_path"]
        reference = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(ref_base, local_files_only=local_only), str(reference_path)
        )
    else:
        reference = AutoModelForCausalLM.from_pretrained(config["reference_model"], local_files_only=local_only)
    training = DPOConfig(output_dir=config["output_dir"], num_train_epochs=config["num_train_epochs"], per_device_train_batch_size=config["per_device_train_batch_size"], gradient_accumulation_steps=config["gradient_accumulation_steps"], learning_rate=config["learning_rate"], logging_steps=config.get("logging_steps", 10), report_to=config.get("report_to", "none"), seed=config.get("seed", 42), bf16=config.get("bf16", False), beta=config["beta"], max_length=config["max_length"])
    trainer = DPOTrainer(model=model, ref_model=reference, args=training, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()
