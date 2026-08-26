"""One-GPU LoRA DPO for Mico's de-identified Decision policy.

The policy starts from the frozen SFT adapter.  Reference log-probabilities
come from the same Qwen base with that adapter disabled, which avoids loading
a second 8B model on the 40GB A100.  The script consumes only staged policy
states and JSON completions; it never reads raw questions, SQL or payloads.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _render(tokenizer: Any, prompt: list[dict[str, str]], completion: str, max_length: int) -> dict[str, list[int]]:
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = tokenizer.apply_chat_template(
        [*prompt, {"role": "assistant", "content": completion}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        raise ValueError("DPO_EXAMPLE_EXCEEDS_MAX_LENGTH")
    if len(prompt_ids) >= len(full_ids):
        raise ValueError("DPO_COMPLETION_TOKENS_MISSING")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
    }


def _features(tokenizer: Any, records: list[dict[str, Any]], max_length: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        prompt = record.get("prompt")
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        if not isinstance(prompt, list) or not isinstance(chosen, str) or not isinstance(rejected, str):
            raise ValueError("DPO_RECORD_SHAPE_INVALID")
        result.append({
            "id": record.get("id", "unknown"),
            "chosen": _render(tokenizer, prompt, chosen, max_length),
            "rejected": _render(tokenizer, prompt, rejected, max_length),
        })
    return result


def _pad(features: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(item["input_ids"]) for item in features)
    values: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        padding = width - len(item["input_ids"])
        values["input_ids"].append(item["input_ids"] + [pad_id] * padding)
        values["attention_mask"].append(item["attention_mask"] + [0] * padding)
        values["labels"].append(item["labels"] + [-100] * padding)
    return {key: torch.tensor(value, dtype=torch.long, device="cuda") for key, value in values.items()}


def _logps(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def _dpo_loss(model: Any, chosen: dict[str, torch.Tensor], rejected: dict[str, torch.Tensor], beta: float) -> torch.Tensor:
    policy_chosen = _logps(model, chosen)
    policy_rejected = _logps(model, rejected)
    with torch.no_grad():
        with model.disable_adapter():
            reference_chosen = _logps(model, chosen)
            reference_rejected = _logps(model, rejected)
    logits = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    return -F.logsigmoid(logits).mean()


def _batches(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _evaluate(model: Any, items: list[dict[str, Any]], tokenizer: Any, batch_size: int, beta: float) -> float:
    if not items:
        return math.nan
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_items in _batches(items, batch_size):
            chosen = _pad([item["chosen"] for item in batch_items], tokenizer.pad_token_id)
            rejected = _pad([item["rejected"] for item in batch_items], tokenizer.pad_token_id)
            losses.append(float(_dpo_loss(model, chosen, rejected, beta).item()))
    model.train()
    return sum(losses) / len(losses)


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("DPO_CUDA_BF16_REQUIRED")
    if not args.dry_run and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("DPO_OUTPUT_DIR_NOT_EMPTY")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_items = _features(tokenizer, _read_jsonl(args.train), args.max_length)
    validation_items = _features(tokenizer, _read_jsonl(args.validation), args.max_length)
    if len(train_items) < 80 or not validation_items:
        raise ValueError("DPO_SPLIT_COUNT_INVALID")

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    # Gradient checkpointing needs an activation that participates in autograd.
    # Enable this on the wrapped PEFT model (rather than only on `base`) so the
    # hook remains installed after the adapter wrapper is constructed.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    if args.dry_run:
        chosen = _pad([train_items[0]["chosen"]], tokenizer.pad_token_id)
        rejected = _pad([train_items[0]["rejected"]], tokenizer.pad_token_id)
        loss = _dpo_loss(model, chosen, rejected, args.beta)
        if not torch.isfinite(loss):
            raise RuntimeError("DPO_DRY_RUN_NONFINITE_LOSS")
        loss.backward()
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not trainable_gradients:
            raise RuntimeError("DPO_DRY_RUN_NO_TRAINABLE_GRADIENT")
        if not all(torch.isfinite(gradient).all() for gradient in trainable_gradients):
            raise RuntimeError("DPO_DRY_RUN_NONFINITE_GRADIENT")
        model.zero_grad(set_to_none=True)
        return {
            "trainingStarted": False,
            "dryRun": True,
            "model": args.model,
            "sftAdapter": str(args.sft_adapter),
            "dpoLoss": float(loss.item()),
            "maxLength": args.max_length,
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    randomizer = random.Random(args.seed)
    history: list[dict[str, float | int]] = []
    update = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        order = list(train_items)
        randomizer.shuffle(order)
        for batch_index, batch_items in enumerate(_batches(order, args.batch_size), start=1):
            chosen = _pad([item["chosen"] for item in batch_items], tokenizer.pad_token_id)
            rejected = _pad([item["rejected"] for item in batch_items], tokenizer.pad_token_id)
            loss = _dpo_loss(model, chosen, rejected, args.beta) / args.gradient_accumulation_steps
            loss.backward()
            if batch_index % args.gradient_accumulation_steps == 0 or batch_index == math.ceil(len(order) / args.batch_size):
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                history.append({"update": update, "epoch": epoch + 1, "loss": float(loss.item() * args.gradient_accumulation_steps)})
        validation_loss = _evaluate(model, validation_items, tokenizer, args.batch_size, args.beta)
        history.append({"update": update, "epoch": epoch + 1, "validationLoss": validation_loss})

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = {
        "trainingStarted": True,
        "model": args.model,
        "sftAdapter": str(args.sft_adapter),
        "trainCount": len(train_items),
        "validationCount": len(validation_items),
        "epochs": args.epochs,
        "updates": update,
        "beta": args.beta,
        "learningRate": args.learning_rate,
        "history": history,
    }
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one-GPU Decision DPO from a frozen SFT adapter")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one finite forward/backward DPO step without writing an adapter.")
    args = parser.parse_args(argv)
    try:
        result = train(args)
    except (OSError, RuntimeError, ValueError, ImportError, TypeError) as exc:
        print(json.dumps({"status": "FAILED", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "metrics": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
