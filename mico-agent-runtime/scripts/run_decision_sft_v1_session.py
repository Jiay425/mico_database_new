"""Run the frozen Decision SFT v1 A100 session.

This is an orchestration entry point, not a replacement for the production
policy or materializer.  It deliberately keeps the frozen experiment order:

    Base Frozen Test -> two-epoch SFT -> validation-only checkpoint choice
    -> one final Frozen Test -> unchanged external S1-S6 canary.

The script is intended to run on the remote Qwen host where Transformers,
PEFT and CUDA are available.  It never edits the frozen dataset and never
uses Frozen Test for checkpoint selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


FIXED_ACTIONS = (
    "inspect_cohort",
    "execute_read_query",
    "compare_groups",
    "analyze_projection",
    "stratified_analysis",
    "adjust_confounders",
    "cross_project_validate",
    "cross_disease_validate",
    "retrieve_evidence",
    "finish",
)
OBJECTIVE_LITERALS = {
    "group_comparison",
    "projection_analysis",
    "stratified_analysis",
    "confounder_assessment",
    "cross_project_validation",
    "cross_disease_validation",
    "evidence_support",
}
STOP_REASONS = {
    "EVIDENCE_SUFFICIENT",
    "NO_NEW_INFORMATION",
    "QUALITY_RISK",
    "ACTION_BUDGET_EXHAUSTED",
    "UPSTREAM_REJECTED",
    "UNSUPPORTED_ACTION",
    "USER_REQUESTED_STOP",
}
TARGET_KEYS = {
    "selected_action",
    "decision_reason",
    "alternative_actions",
    "stop_reason",
}
EXPECTED_DATASET_SHA = "6e3db8ab22786d4004bdd9846a1d09b5dba7ba1a3522917551bfca74dccf8ef1"
EXPECTED_PROMPT_SHA = "c5ab7dbc7ff2ee99f5720f0b99dc2d0fb7d71150a00ffd7a5f6bf5858dcb1db4"
MAX_SEQ_LENGTH = 1280


class SessionError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise SessionError(f"record_not_object:{path.name}")
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def model_key(state: Mapping[str, Any], target: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical({"state": state, "target": target}).encode()).hexdigest()


def dataset_digest(data_dir: Path) -> dict[str, Any]:
    names = ["reviewed_v1.jsonl", "reviewed_train.jsonl", "reviewed_validation.jsonl", "reviewed_test.jsonl", "review_manifest.json"]
    hashes = {name: sha256(data_dir / name) for name in names}
    digest = hashlib.sha256(canonical(hashes).encode()).hexdigest()
    if digest != EXPECTED_DATASET_SHA:
        raise SessionError(f"dataset_fingerprint_mismatch:{digest}")
    return {"dataset_sha256": digest, "files": hashes}


def load_dataset(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = read_json(data_dir / "review_manifest.json")
    if manifest.get("DECISION_SFT_V1_DATA_READY") is not True or manifest.get("training_started") is not False:
        raise SessionError("frozen_manifest_not_ready")
    if manifest.get("split_counts") != {"train": 388, "validation": 65, "test": 65}:
        raise SessionError("frozen_split_counts_changed")
    if set(manifest.get("actions", [])) != set(FIXED_ACTIONS):
        raise SessionError("frozen_action_set_changed")
    provenance: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(data_dir / "reviewed_v1.jsonl"):
        key = model_key(item["state"], item["target"])
        provenance[key] = item

    def load_model_records(name: str, expected_count: int) -> list[dict[str, Any]]:
        rows = read_jsonl(data_dir / name)
        if len(rows) != expected_count:
            raise SessionError(f"split_count_changed:{name}")
        result = []
        for row in rows:
            if set(row) != {"input", "output"}:
                raise SessionError(f"model_record_shape_changed:{name}")
            if set(row["input"]) != {"decision_type", "state"} or row["input"]["decision_type"] != "scientific_action":
                raise SessionError(f"input_contract_changed:{name}")
            if set(row["output"]) != TARGET_KEYS:
                raise SessionError(f"target_contract_changed:{name}")
            key = model_key(row["input"]["state"], row["output"])
            if key not in provenance:
                raise SessionError(f"record_not_in_reviewed_v1:{name}")
            result.append(row)
        return result

    return (
        load_model_records("reviewed_train.jsonl", 388),
        load_model_records("reviewed_validation.jsonl", 65),
        load_model_records("reviewed_test.jsonl", 65),
        provenance,
    )


def load_system_prompt(parity_cases: Path) -> str:
    payload = read_json(parity_cases)
    cases = payload.get("cases", [])
    if not cases:
        raise SessionError("parity_cases_missing")
    prompt = cases[0]["messages"][0]["content"]
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if digest != EXPECTED_PROMPT_SHA:
        raise SessionError(f"prompt_fingerprint_mismatch:{digest}")
    return prompt


def build_messages(row: Mapping[str, Any], system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(row["input"], ensure_ascii=False, separators=(",", ":"))},
        {"role": "assistant", "content": json.dumps(row["output"], ensure_ascii=False, separators=(",", ":"))},
    ]


def render_feature(row: Mapping[str, Any], tokenizer: Any, system_prompt: str) -> dict[str, list[int]]:
    messages = build_messages(row, system_prompt)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > MAX_SEQ_LENGTH or len(prompt_ids) >= len(full_ids):
        raise SessionError(f"training_shape_invalid:{len(prompt_ids)}:{len(full_ids)}")
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    if not any(value != -100 for value in labels):
        raise SessionError("empty_assistant_labels")
    if tokenizer.eos_token_id not in labels:
        raise SessionError("eos_missing_from_assistant_labels")
    return {"input_ids": list(full_ids), "attention_mask": [1] * len(full_ids), "labels": labels}


class CompletionCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch
        width = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([item["input_ids"] + [pad_id] * (width - len(item["input_ids"])) for item in features], dtype=torch.long),
            "attention_mask": torch.tensor([item["attention_mask"] + [0] * (width - len(item["input_ids"])) for item in features], dtype=torch.long),
            "labels": torch.tensor([item["labels"] + [-100] * (width - len(item["input_ids"])) for item in features], dtype=torch.long),
        }


def extract_prediction(raw: str) -> dict[str, Any] | None:
    text = raw.replace("<|im_end|>", "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "selected_action" in value:
            return value
    return None


def contract_status(prediction: Mapping[str, Any] | None, available: Iterable[str]) -> dict[str, Any]:
    available_set = set(available)
    if prediction is None:
        return {
            "contract_valid": False,
            "available_action_compliance": False,
            "objective_action_confusion": False,
            "unavailable_action": True,
            "stop_reason_valid": False,
            "reason_nonempty": False,
            "selected_action": None,
            "decision_reason": None,
            "alternative_actions": [],
            "stop_reason": None,
        }
    selected = prediction.get("selected_action")
    reason = prediction.get("decision_reason")
    alternatives = prediction.get("alternative_actions")
    stop_reason = prediction.get("stop_reason")
    shape = set(prediction) == TARGET_KEYS
    stop_valid = (
        (selected == "finish" and stop_reason in STOP_REASONS)
        or (selected != "finish" and stop_reason is None)
    )
    valid = (
        shape
        and isinstance(selected, str)
        and selected in FIXED_ACTIONS
        and isinstance(reason, str)
        and bool(reason.strip())
        and isinstance(alternatives, list)
        and all(isinstance(item, str) and item in FIXED_ACTIONS for item in alternatives)
        and len(alternatives) == len(set(alternatives))
        and selected not in alternatives
        and stop_valid
    )
    selected_is_string = isinstance(selected, str)
    selected_in_available = selected_is_string and selected in available_set
    objective_confusion = selected_is_string and selected in OBJECTIVE_LITERALS
    unavailable = not selected_in_available
    return {
        "contract_valid": bool(valid),
        "available_action_compliance": selected_in_available,
        "objective_action_confusion": objective_confusion,
        "unavailable_action": unavailable,
        "stop_reason_valid": bool(stop_valid),
        "reason_nonempty": isinstance(reason, str) and bool(reason.strip()),
        "selected_action": selected,
        "decision_reason": reason,
        "alternative_actions": alternatives if isinstance(alternatives, list) else [],
        "stop_reason": stop_reason,
    }


def run_one_generation(model: Any, tokenizer: Any, row: Mapping[str, Any], system_prompt: str) -> tuple[str, dict[str, Any] | None]:
    messages = build_messages(row, system_prompt)
    prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = int(encoded["input_ids"].shape[1])
    with __import__("torch").inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    raw = tokenizer.decode(generated[0][input_length:], skip_special_tokens=False)
    return raw, extract_prediction(raw)


def metrics_for_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    count = lambda key: sum(1 for row in results if row.get(key))
    exact = sum(1 for row in results if row.get("parsed_action") == row.get("expected_action"))
    finish = sum(1 for row in results if (row.get("parsed_action") == "finish") == (row.get("expected_action") == "finish"))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row.get("hard_case_class"):
            buckets[str(row["hard_case_class"])].append(row)
    bucket_metrics = {}
    for bucket, rows in sorted(buckets.items()):
        bucket_metrics[bucket] = {
            "count": len(rows),
            "selected_action_accuracy": sum(1 for row in rows if row.get("parsed_action") == row.get("expected_action")) / len(rows),
            "contract_valid_rate": sum(1 for row in rows if row.get("contract_valid")) / len(rows),
            "available_action_compliance": sum(1 for row in rows if row.get("available_action_compliance")) / len(rows),
        }
    hard_values = [item["selected_action_accuracy"] for item in bucket_metrics.values()]
    return {
        "sample_count": total,
        "selected_action_accuracy": {"count": exact, "total": total, "rate": exact / total if total else 0.0},
        "contract_valid_rate": {"count": count("contract_valid"), "total": total, "rate": count("contract_valid") / total if total else 0.0},
        "available_action_compliance": {"count": count("available_action_compliance"), "total": total, "rate": count("available_action_compliance") / total if total else 0.0},
        "objective_action_confusion_count": sum(1 for row in results if row.get("objective_action_confusion")),
        "unavailable_action_count": sum(1 for row in results if row.get("unavailable_action")),
        "finish_accuracy": {"count": finish, "total": total, "rate": finish / total if total else 0.0},
        "stop_reason_validity": {"count": count("stop_reason_valid"), "total": total, "rate": count("stop_reason_valid") / total if total else 0.0},
        "reason_nonempty": {"count": count("reason_nonempty"), "total": total, "rate": count("reason_nonempty") / total if total else 0.0},
        "hard_case_bucket_metrics": bucket_metrics,
        "hard_case_macro_selected_action_accuracy": statistics.mean(hard_values) if hard_values else 0.0,
    }


def evaluate_rows(model: Any, tokenizer: Any, rows: list[dict[str, Any]], system_prompt: str, provenance: Mapping[str, Mapping[str, Any]], output_path: Path, expected: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    for row in rows:
        state = row["input"]["state"] if "input" in row else row["state"]
        available = state.get("action_space", {}).get("available_actions", [])
        raw, parsed = run_one_generation(model, tokenizer, row if "input" in row else {"input": {"decision_type": "scientific_action", "state": state}, "output": {"selected_action": "finish", "decision_reason": "placeholder", "alternative_actions": [], "stop_reason": "EVIDENCE_SUFFICIENT"}}, system_prompt)
        status = contract_status(parsed, available)
        result = {
            "sample_id": None,
            "available_actions": available,
            "raw_response": raw,
            "parsed_action": status["selected_action"],
            "decision_reason": status["decision_reason"],
            "alternative_actions": status["alternative_actions"],
            "stop_reason": status["stop_reason"],
            **{key: status[key] for key in ("contract_valid", "available_action_compliance", "objective_action_confusion", "unavailable_action", "stop_reason_valid", "reason_nonempty")},
        }
        if expected:
            target = row["output"]
            key = model_key(state, target)
            meta = provenance[key]
            result.update({
                "sample_id": meta["sample_id"],
                "expected_action": target["selected_action"],
                "hard_case_class": meta.get("metadata", {}).get("hard_case_class"),
                "state_origin": meta.get("metadata", {}).get("state_origin", meta.get("source_type")),
            })
        results.append(result)
    write_jsonl(output_path, results)
    return results, metrics_for_results(results) if expected else {"sample_count": len(results)}


def load_model(model_path: str, *, train: bool = False) -> Any:
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    model = model.to("cuda:0")
    model.config.use_cache = not train
    if train:
        model.enable_input_require_grads()
        model.train()
    else:
        model.eval()
    return model


def free_model(model: Any) -> None:
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def evaluate_canary(model: Any, tokenizer: Any, canary_dir: Path, system_prompt: str, output_path: Path) -> dict[str, Any]:
    manifest = read_json(canary_dir / "state_manifest.json")
    if manifest.get("training_eligible") is not False:
        raise SessionError("external_canary_training_eligible_changed")
    rows = []
    for name in manifest["state_names"]:
        state = read_json(canary_dir / f"state_{name}.json")
        raw, parsed = run_one_generation(model, tokenizer, {"input": {"decision_type": "scientific_action", "state": state}, "output": {"selected_action": "finish", "decision_reason": "placeholder", "alternative_actions": [], "stop_reason": "EVIDENCE_SUFFICIENT"}}, system_prompt)
        status = contract_status(parsed, state.get("action_space", {}).get("available_actions", []))
        meta = next(item for item in manifest["states"] if item["state_name"] == name)
        rows.append({
            "state_name": name,
            "state_origin": meta["state_origin"],
            "training_eligible": False,
            "state_hash": meta["state_hash"],
            "available_actions": state.get("action_space", {}).get("available_actions", []),
            "raw_response": raw,
            "parsed_action": status["selected_action"],
            "decision_reason": status["decision_reason"],
            "alternative_actions": status["alternative_actions"],
            "stop_reason": status["stop_reason"],
            "contract_valid": status["contract_valid"],
            "available_action_compliance": status["available_action_compliance"],
            "objective_action_confusion": status["objective_action_confusion"],
            "unavailable_action": status["unavailable_action"],
            "stop_reason_valid": status["stop_reason_valid"],
            "reason_nonempty": status["reason_nonempty"],
            "repair_count": 0,
            "policy_origin": "qwen_model",
        })
    write_jsonl(output_path, rows)
    by_name = {row["state_name"]: row for row in rows}
    prior_path = canary_dir / "base_dynamic_canary_v2_report.json"
    if not prior_path.exists():
        prior_path = canary_dir / "report.json"
    prior = read_json(prior_path) if prior_path.exists() else None
    prior_actions = {item.get("state_name"): item.get("selected_action") for item in prior.get("states", [])} if prior else {}
    s1_s2_regression = any(
        name in prior_actions and by_name[name]["parsed_action"] != prior_actions[name]
        for name in ("s1", "s2")
    ) if prior else None
    s4s5_reason = all(
        any(marker in (by_name[name]["decision_reason"] or "").lower() for marker in ("heterogeneity", "project", "consistent", "conflict"))
        for name in ("s4", "s5")
    )
    s6 = by_name["s6"]
    summary = {
        "policy_calls": len(rows),
        "contract_valid": {"count": sum(row["contract_valid"] for row in rows), "total": len(rows)},
        "available_action_compliance": {"count": sum(row["available_action_compliance"] for row in rows), "total": len(rows)},
        "s1_s2_actions": {name: by_name[name]["parsed_action"] for name in ("s1", "s2")},
        "s1_s2_regression": s1_s2_regression,
        "s3_objective_action_confusion": by_name["s3"]["objective_action_confusion"],
        "s3_available_action_compliance": by_name["s3"]["available_action_compliance"],
        "s4_s5_reason_references_changed_fact": s4s5_reason,
        "s4_s5_actions": {name: by_name[name]["parsed_action"] for name in ("s4", "s5")},
        "s6_first_pass_finish": s6["parsed_action"] == "finish" and s6["stop_reason_valid"],
        "repair_count": 0,
        "policy_origin": "qwen_model",
        "training_eligible": False,
    }
    return {"schema_version": "decision-sft-v1-external-canary-result-v1", "training_eligible": False, "states": rows, "metrics": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parity-cases", type=Path, required=True)
    parser.add_argument("--canary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-base-run",
        type=Path,
        default=None,
        help=(
            "Reuse the already-captured Base Frozen Test from another run. "
            "This is only for transport recovery; the test is not rerun or used "
            "for checkpoint selection."
        ),
    )
    args = parser.parse_args(argv)
    run_dir = args.output_dir
    if run_dir.exists():
        raise SystemExit(f"refusing_to_overwrite_existing_run:{run_dir}")
    run_dir.mkdir(parents=True)
    started = time.time()
    model = None
    try:
        digest = dataset_digest(args.data_dir)
        train_rows, val_rows, test_rows, provenance = load_dataset(args.data_dir)
        system_prompt = load_system_prompt(args.parity_cases)
        base_baseline_origin = "this_session"
        if args.reuse_base_run is not None:
            source_run = args.reuse_base_run
            source_rows_path = source_run / "base_frozen_test.jsonl"
            source_metrics_path = source_run / "base_frozen_test_metrics.json"
            if not source_rows_path.is_file() or not source_metrics_path.is_file():
                raise SessionError(f"reuse_base_run_missing_baseline:{source_run}")
            source_rows = read_jsonl(source_rows_path)
            source_metrics = read_json(source_metrics_path)
            if len(source_rows) != len(test_rows) or source_metrics.get("sample_count") != len(test_rows):
                raise SessionError("reuse_base_run_baseline_count_mismatch")
            base_baseline_origin = str(source_run)
        write_json(run_dir / "session_contract.json", {
            "dataset": digest,
            "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "prompt_version": "generic_contract_hardened",
            "train_count": len(train_rows), "validation_count": len(val_rows), "frozen_test_count": len(test_rows),
            "frozen_test_used_for_selection": False,
            "base_baseline_origin": base_baseline_origin,
            "training_started": False,
        })

        # 1. Base Frozen Test.  This is the only Base test pass in the session.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if args.reuse_base_run is not None:
            source_run = args.reuse_base_run
            shutil.copy2(source_run / "base_frozen_test.jsonl", run_dir / "base_frozen_test.jsonl")
            shutil.copy2(source_run / "base_frozen_test_metrics.json", run_dir / "base_frozen_test_metrics.json")
            base_rows = read_jsonl(run_dir / "base_frozen_test.jsonl")
            base_metrics = read_json(run_dir / "base_frozen_test_metrics.json")
        else:
            model = load_model(args.model, train=False)
            base_rows, base_metrics = evaluate_rows(model, tokenizer, test_rows, system_prompt, provenance, run_dir / "base_frozen_test.jsonl", expected=True)
            write_json(run_dir / "base_frozen_test_metrics.json", base_metrics)
            free_model(model); model = None

        # 2. Two epoch SFT with explicit epoch checkpoints and validation-only scoring.
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Trainer, TrainerCallback, TrainingArguments
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train_features = [render_feature(row, tokenizer, system_prompt) for row in train_rows]
        val_features = [render_feature(row, tokenizer, system_prompt) for row in val_rows]
        model = load_model(args.model, train=True)
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["o_proj","k_proj","down_proj","v_proj","gate_proj","up_proj","q_proj"], bias="none"))
        trainer_work = run_dir / "trainer_work"
        training_args = TrainingArguments(
            output_dir=str(trainer_work), per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=8, num_train_epochs=2.0, max_steps=-1, learning_rate=2e-4,
            warmup_ratio=0.05, lr_scheduler_type="cosine", logging_steps=1, eval_strategy="no",
            save_strategy="no", bf16=True, tf32=True, gradient_checkpointing=True,
            optim="adamw_torch", report_to=[], remove_unused_columns=False, seed=42, data_seed=42,
            save_safetensors=True,
        )

        class EpochCallback(TrainerCallback):
            def __init__(self) -> None:
                self.trainer = None
                self.last_log_index = 0
                self.epoch_started = time.time()
                self.epochs = []

            def on_epoch_end(self, args, state, control, **kwargs):
                epoch_index = int(round(float(state.epoch)))
                current_model = kwargs["model"]
                checkpoint = run_dir / f"checkpoint_epoch{epoch_index}"
                checkpoint.mkdir(parents=True, exist_ok=False)
                current_model.save_pretrained(str(checkpoint))
                tokenizer.save_pretrained(str(checkpoint))
                state.save_to_json(str(checkpoint / "trainer_state.json"))
                loss_values = [float(item["loss"]) for item in state.log_history[self.last_log_index:] if "loss" in item]
                self.last_log_index = len(state.log_history)
                eval_loss = None
                if self.trainer is not None:
                    eval_loss = self.trainer.evaluate(eval_dataset=val_features).get("eval_loss")
                val_result_path = run_dir / f"validation_epoch{epoch_index}.jsonl"
                val_result_rows, val_metrics = evaluate_rows(current_model, tokenizer, val_rows, system_prompt, provenance, val_result_path, expected=True)
                if eval_loss is not None:
                    eval_loss = float(eval_loss)
                learning_rate = None
                if self.trainer is not None and self.trainer.lr_scheduler is not None:
                    learning_rate = float(self.trainer.lr_scheduler.get_last_lr()[0])
                record = {
                    "epoch": epoch_index,
                    "checkpoint": str(checkpoint),
                    "train_loss": statistics.mean(loss_values) if loss_values else None,
                    "validation_loss": eval_loss,
                    "learning_rate": learning_rate,
                    "optimizer_steps": int(state.global_step),
                    "epoch_duration_seconds": time.time() - self.epoch_started,
                    "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "validation_metrics": val_metrics,
                }
                self.epochs.append(record)
                write_json(run_dir / f"validation_epoch{epoch_index}.json", record)
                self.epoch_started = time.time()
                torch.cuda.reset_peak_memory_stats()
                return control

        callback = EpochCallback()
        trainer = Trainer(model=model, args=training_args, train_dataset=train_features, eval_dataset=val_features, data_collator=CompletionCollator(tokenizer), callbacks=[callback])
        callback.trainer = trainer
        train_started = time.time()
        train_result = trainer.train()
        total_train_seconds = time.time() - train_started
        callback_records = callback.epochs
        if len(callback_records) != 2:
            raise SessionError(f"epoch_checkpoint_count:{len(callback_records)}")
        write_json(run_dir / "training_metrics.json", {
            "schema_version": "decision-sft-v1-training-run-v1",
            "train_count": len(train_rows), "validation_count": len(val_rows), "epochs": callback_records,
            "total_train_seconds": total_train_seconds,
            "trainer_result_metrics": dict(train_result.metrics),
            "training_started": True,
            "frozen_test_used_for_selection": False,
        })
        # 3. Validation-only lexicographic selection; Frozen Test is not read here.
        def selection_key(item: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
            metrics = item["validation_metrics"]
            val_loss = item["validation_loss"] if item["validation_loss"] is not None else float("inf")
            return (
                metrics["available_action_compliance"]["rate"],
                metrics["contract_valid_rate"]["rate"],
                metrics["selected_action_accuracy"]["rate"],
                metrics["hard_case_macro_selected_action_accuracy"],
                -float(val_loss),
            )
        selected = max(callback_records, key=selection_key)
        selection = {
            "selected_checkpoint": selected["checkpoint"],
            "selection_epoch": selected["epoch"],
            "selection_key": list(selection_key(selected)),
            "selection_reason": "validation-only lexicographic priority: available action compliance, contract validity, selected action accuracy, hard-case accuracy, validation loss",
            "frozen_test_used_for_selection": False,
        }
        write_json(run_dir / "checkpoint_selection.json", selection)
        free_model(model); model = None

        # 4. Final SFT Frozen Test, exactly once after selection.
        final_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
        if final_tokenizer.pad_token is None: final_tokenizer.pad_token = final_tokenizer.eos_token
        final_tokenizer.padding_side = "right"
        final_model = load_model(args.model, train=False)
        from peft import PeftModel
        final_model = PeftModel.from_pretrained(final_model, selected["checkpoint"], is_trainable=False)
        final_model.eval()
        final_rows, final_metrics = evaluate_rows(final_model, final_tokenizer, test_rows, system_prompt, provenance, run_dir / "final_sft_frozen_test.jsonl", expected=True)
        write_json(run_dir / "final_sft_frozen_test_metrics.json", final_metrics)

        # 5. The exact existing S1-S6 canary states, one call each, ineligible for training.
        canary_result = evaluate_canary(final_model, final_tokenizer, args.canary_dir, system_prompt, run_dir / "external_canary_s1_s6.jsonl")
        write_json(run_dir / "external_canary_s1_s6.json", canary_result)
        free_model(final_model)

        summary = {
            "schema_version": "decision-sft-v1-session-report-v1",
            "dataset_sha256": digest["dataset_sha256"],
            "prompt_sha256": EXPECTED_PROMPT_SHA,
            "base_frozen_test": base_metrics,
            "training": {"epochs": callback_records, "total_train_seconds": total_train_seconds},
            "selected_checkpoint": selection,
            "final_sft_frozen_test": final_metrics,
            "external_canary": canary_result["metrics"],
            "DECISION_SFT_V1_TRAINING": "PASS",
            "training_started": True,
            "dpo_started": False,
            "full_e2e_started": False,
            "frozen_test_runs": 2,
            "frozen_test_used_for_selection": False,
            "session_duration_seconds": time.time() - started,
        }
        write_json(run_dir / "session_report.json", summary)
        write_json(run_dir / "session_flags.json", {
            "DECISION_SFT_V1_TRAINING": "PASS",
            "selected_checkpoint": selection["selected_checkpoint"],
            "S3_fixed": canary_result["metrics"]["s3_objective_action_confusion"] is False and canary_result["metrics"]["s3_available_action_compliance"] is True,
            "S4_S5_observation_sensitivity_improved": canary_result["metrics"]["s4_s5_reason_references_changed_fact"],
            "S6_first_pass_finish": canary_result["metrics"]["s6_first_pass_finish"],
            "S1_S2_regression": canary_result["metrics"]["s1_s2_regression"],
            "base_frozen_test_runs": 1,
            "final_sft_frozen_test_runs": 1,
            "external_canary_calls": 6,
            "dpo_started": False,
            "full_e2e_started": False,
        })
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        error = {"status": "FAIL", "error": type(exc).__name__ + ": " + str(exc), "traceback_tail": traceback.format_exc().splitlines()[-12:]}
        write_json(run_dir / "session_failure.json", error)
        print(json.dumps(error, ensure_ascii=False, indent=2))
        return 1
    finally:
        if model is not None:
            free_model(model)


if __name__ == "__main__":
    raise SystemExit(main())
