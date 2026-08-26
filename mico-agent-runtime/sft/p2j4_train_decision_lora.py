"""Qwen-style Decision SFT with BF16 LoRA on one GPU.

The script consumes only the de-identified JSONL emitted by
``p2j4_export_decision_sft.py``.  It never reads raw Eval questions, SQL,
payloads or trace provenance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _render(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    messages = record["messages"]
    prompt_messages = messages[:2]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        raise ValueError("SFT_EXAMPLE_EXCEEDS_MAX_LENGTH")
    if len(prompt_ids) >= len(full_ids):
        raise ValueError("SFT_ASSISTANT_TOKENS_MISSING")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


@dataclass
class CompletionCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        result: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            result["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            result["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            result["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def _make_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT_CUDA_BF16_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_records = _read_jsonl(args.train)
    validation_records = _read_jsonl(args.validation)
    train_features = [_render(tokenizer, item, args.max_length) for item in train_records]
    validation_features = [_render(tokenizer, item, args.max_length) for item in validation_records]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=_make_training_arguments(args),
        train_dataset=train_features,
        eval_dataset=validation_features,
        data_collator=CompletionCollator(tokenizer),
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))
    metrics = dict(result.metrics)
    metrics.update({
        "model": args.model,
        "trainCount": len(train_records),
        "validationCount": len(validation_records),
        "maxLength": args.max_length,
        "trainingStarted": True,
    })
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one-GPU Decision LoRA SFT")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        result = train(args)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "FAILED", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "metrics": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
