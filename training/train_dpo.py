"""Train DPO from the SFT policy and reference checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

from rcdpo.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/configs/dpo.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    set_seed(config.get("seed", 42))
    dataset = load_dataset("json", data_files=config.get("data_file", "data/dpo/train.jsonl"), split="train")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], local_files_only=config.get("local_files_only", False))
    reference = AutoModelForCausalLM.from_pretrained(config["reference_model"], local_files_only=config.get("local_files_only", False))
    training = TrainingArguments(output_dir=config["output_dir"], num_train_epochs=config["num_train_epochs"], per_device_train_batch_size=config["per_device_train_batch_size"], gradient_accumulation_steps=config["gradient_accumulation_steps"], learning_rate=config["learning_rate"], logging_steps=10, report_to=config.get("report_to", "none"), seed=config.get("seed", 42), bf16=False)
    trainer = DPOTrainer(model=model, ref_model=reference, args=training, train_dataset=dataset, processing_class=tokenizer, beta=config["beta"], max_length=config["max_length"])
    trainer.train()
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()
