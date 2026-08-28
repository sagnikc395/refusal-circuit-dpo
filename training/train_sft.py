"""Train the deliberately naive SFT adapter."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from rcdpo.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/configs/sft.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    set_seed(config.get("seed", 42))
    dataset = load_dataset("json", data_files=config.get("data_file", "data/sft/train.jsonl"), split="train")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], local_files_only=config.get("local_files_only", False))
    lora = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    training = TrainingArguments(output_dir=config["output_dir"], num_train_epochs=config["num_train_epochs"], per_device_train_batch_size=config["per_device_train_batch_size"], gradient_accumulation_steps=config["gradient_accumulation_steps"], learning_rate=config["learning_rate"], logging_steps=10, report_to=config.get("report_to", "none"), seed=config.get("seed", 42), bf16=False)
    trainer = SFTTrainer(model=model, args=training, train_dataset=dataset, processing_class=tokenizer, peft_config=lora, dataset_text_field="text", max_seq_length=config["max_seq_length"])
    trainer.train()
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()
