"""Qwen3-8B Decision-State-native SFT v1 trainer.

The trainer is intentionally separate from the historical legacy exporter.
It accepts only the frozen ``reviewed_train.jsonl`` and
``reviewed_validation.jsonl`` model-shaped records, renders the same
``generic_contract_hardened`` system prompt used by serving, and masks loss to
the assistant completion.  The frozen test split is never loaded here.

This module is a future A100 entry point.  Importing it does not import
Transformers or start training; those optional dependencies are loaded only by
``train`` after the immutable dataset gate has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mico_agent_runtime.training.decision_sft_v1_preparation import (
    DEFAULT_DATA_DIR,
    DEFAULT_PREP_DIR,
    DatasetPreparationError,
    build_chat_template_messages,
    _read_jsonl,
    validate_model_record,
)


def _read_config(prep_dir: Path) -> dict[str, Any]:
    path = prep_dir / "training_config.json"
    if not path.exists():
        raise DatasetPreparationError("training_config_missing_run_offline_preparation_first")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("training_started") is not False:
        raise DatasetPreparationError("training_config_already_started")
    return config


def load_frozen_training_records(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only train/validation; deliberately never open reviewed_test.jsonl."""

    manifest_path = data_dir / "review_manifest.json"
    if not manifest_path.exists():
        raise DatasetPreparationError("review_manifest_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("DECISION_SFT_V1_DATA_READY") is not True or manifest.get("training_started") is not False:
        raise DatasetPreparationError("frozen_training_manifest_not_ready")
    expected = {"train": 388, "validation": 65}
    records: dict[str, list[dict[str, Any]]] = {}
    for split, expected_count in expected.items():
        path = data_dir / f"reviewed_{split}.jsonl"
        rows = _read_jsonl(path)
        if len(rows) != expected_count:
            raise DatasetPreparationError(f"training_split_count_changed:{split}")
        records[split] = [validate_model_record(row) for row in rows]
    return records["train"], records["validation"]


def render_with_huggingface(tokenizer: Any, record: Mapping[str, Any], max_length: int) -> dict[str, list[int]]:
    """Render one record with the actual Qwen tokenizer on the A100 host."""

    normalized = validate_model_record(record)
    messages = build_chat_template_messages(normalized)
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        raise DatasetPreparationError("SFT_V1_EXAMPLE_EXCEEDS_MAX_LENGTH")
    if len(prompt_ids) >= len(full_ids):
        raise DatasetPreparationError("SFT_V1_ASSISTANT_TOKENS_MISSING")
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


@dataclass
class CompletionCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        if not features:
            raise DatasetPreparationError("SFT_V1_EMPTY_BATCH")
        width = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def train(
    model_path: str,
    output_dir: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    prep_dir: Path = DEFAULT_PREP_DIR,
) -> dict[str, Any]:
    """Launch the v1 training run only after all explicit gates pass."""

    config = _read_config(prep_dir)
    if not config.get("frozen_test_read_by_trainer") is False:
        raise DatasetPreparationError("SFT_V1_TRAINER_TEST_SPLIT_ACCESS_NOT_ALLOWED")
    fingerprint_path = prep_dir / "dataset_fingerprint.json"
    if not fingerprint_path.exists():
        raise DatasetPreparationError("dataset_fingerprint_missing")
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if fingerprint.get("frozen_test", {}).get("read_by_trainer") is not False:
        raise DatasetPreparationError("frozen_test_access_policy_invalid")
    for split in ("train", "validation"):
        path = data_dir / f"reviewed_{split}.jsonl"
        expected_sha = fingerprint.get("files", {}).get(f"reviewed_{split}.jsonl", {}).get("sha256")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha != digest:
            raise DatasetPreparationError(f"training_split_fingerprint_changed:{split}")

    # Optional GPU dependencies are intentionally imported only here, never by
    # offline preparation or test collection.
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT_V1_CUDA_BF16_REQUIRED")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_records, validation_records = load_frozen_training_records(data_dir)
    max_length = int(config["max_seq_length"])
    train_features = [render_with_huggingface(tokenizer, row, max_length) for row in train_records]
    validation_features = [render_with_huggingface(tokenizer, row, max_length) for row in validation_records]
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora_cfg = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            bias=str(lora_cfg["bias"]),
        ),
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_train_epochs=float(config["epochs"]),
        max_steps=-1,
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        lr_scheduler_type=str(config["lr_scheduler_type"]),
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        tf32=bool(config["tf32"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        optim=str(config["optimizer"]),
        report_to=[],
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
        save_safetensors=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_features,
        eval_dataset=validation_features,
        data_collator=CompletionCollator(tokenizer),
    )
    result = trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    metrics = dict(result.metrics)
    metrics.update({
        "schema_version": "decision-sft-v1-training-run-v1",
        "model": model_path,
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "frozen_test_read": False,
        "max_seq_length": max_length,
        "training_started": True,
    })
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Decision-State-native SFT v1 on one GPU")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP_DIR)
    args = parser.parse_args(argv)
    try:
        metrics = train(args.model, args.output_dir, args.data_dir, args.prep_dir)
    except (DatasetPreparationError, OSError, RuntimeError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
