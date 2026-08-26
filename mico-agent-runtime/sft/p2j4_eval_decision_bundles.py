"""Evaluate Base and LoRA Decision policies over one de-duplicated bundle.

The model is loaded once per policy and evaluated in batches.  Group metrics
are derived from manifest membership, so overlapping slices do not cause
duplicate model calls.  This is important on rented GPUs and keeps every
per-case result auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2j4_audit_decision_reasons import _audit_case


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EXPECTED_OBJECT:{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _prompt(tokenizer: Any, record: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        record["messages"][:2],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _generate(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        prompts = [_prompt(tokenizer, record) for record in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = inputs["input_ids"].shape[1]
        for index, record in enumerate(batch):
            text = tokenizer.decode(
                generated[index, prompt_width:],
                skip_special_tokens=True,
            )
            results.append({"record_id": record["id"], "text": text})
    return results


def _score_one(record: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    state = json.loads(record["messages"][1]["content"])["policy_state"]
    target = json.loads(record["messages"][2]["content"])
    prediction = _parse_json(output["text"])
    valid = prediction is not None
    predicted_action = prediction.get("selected_action") if prediction else None
    action_ok = predicted_action == target["selected_action"]
    allowed_ok = predicted_action in state["candidate_actions"]
    predicted_stop = prediction.get("stop_reason") if prediction else None
    stop_ok = (
        (predicted_action == "finish" and bool(predicted_stop))
        or (predicted_action != "finish" and predicted_stop is None)
    )
    scored = {
        "record_id": output["record_id"],
        "prediction": prediction,
        "raw_output": output["text"],
        "gold_action": target["selected_action"],
        "action_correct": action_ok,
        "action_allowed": allowed_ok,
        "stop_consistent": stop_ok,
    }
    reason_audit = _audit_case(record, scored)
    scored["reason_audit"] = {
        "status": reason_audit["status"],
        "checks": reason_audit["checks"],
        "review_reasons": reason_audit["review_reasons"],
    }
    scored["json_valid"] = valid
    scored["policy_pass"] = bool(valid and action_ok and allowed_ok and stop_ok)
    return scored


def _metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    count = lambda key: sum(bool(case.get(key)) for case in cases)
    reason_pass = sum(case["reason_audit"]["status"] == "PASS" for case in cases)
    return {
        "caseCount": total,
        "jsonValidCount": count("json_valid"),
        "jsonValidRate": count("json_valid") / total if total else 0.0,
        "actionCorrectCount": count("action_correct"),
        "actionAccuracy": count("action_correct") / total if total else 0.0,
        "actionAllowedCount": count("action_allowed"),
        "actionAllowedRate": count("action_allowed") / total if total else 0.0,
        "stopConsistentCount": count("stop_consistent"),
        "stopConsistencyRate": count("stop_consistent") / total if total else 0.0,
        "policyPassCount": count("policy_pass"),
        "policyPassRate": count("policy_pass") / total if total else 0.0,
        "reasonAuditPassCount": reason_pass,
        "reasonAuditPassRate": reason_pass / total if total else 0.0,
        "reasonAuditReviewCount": total - reason_pass,
    }


def _load_model(model_path: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def _evaluate_policy(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    record_by_id: dict[str, dict[str, Any]],
    group_ids: dict[str, set[str]],
    max_new_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    outputs = _generate(model, tokenizer, records, max_new_tokens, batch_size)
    cases = [
        _score_one(record_by_id[output["record_id"]], output)
        for output in outputs
    ]
    by_id = {case["record_id"]: case for case in cases}
    groups = {
        name: _metrics([by_id[record_id] for record_id in sorted(ids)])
        for name, ids in group_ids.items()
    }
    return {"overall": _metrics(cases), "groups": groups, "cases": cases}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT_EVAL_CUDA_BF16_REQUIRED")
    manifest = _read_json(args.manifest)
    records = _read_jsonl(args.records)
    record_by_id = {record["id"]: record for record in records}
    manifest_records = manifest.get("records") or {}
    group_ids = {
        name: {
            record_id for record_id, metadata in manifest_records.items()
            if name in (metadata.get("groups") or [])
        }
        for name in (manifest.get("groups") or {})
    }
    if set(record_by_id) != set(manifest_records):
        raise ValueError("BUNDLE_RECORD_MANIFEST_MISMATCH")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = _load_model(args.model)
    base = _evaluate_policy(
        base_model, tokenizer, records, record_by_id, group_ids,
        args.max_new_tokens, args.batch_size,
    )
    del base_model
    torch.cuda.empty_cache()

    sft_model = _load_model(args.model)
    sft_model = PeftModel.from_pretrained(sft_model, args.adapter)
    sft_model.eval()
    sft = _evaluate_policy(
        sft_model, tokenizer, records, record_by_id, group_ids,
        args.max_new_tokens, args.batch_size,
    )
    del sft_model
    torch.cuda.empty_cache()
    return {
        "schemaVersion": "p2j4-decision-bundle-eval-v1",
        "manifest": args.manifest.name,
        "recordCount": len(records),
        "trainingStarted": True,
        "base": base,
        "sft": sft,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Base/SFT on grouped Decision bundle")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError, ImportError, json.JSONDecodeError, KeyError) as exc:
        print(json.dumps({"status": "FAILED", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "PASS",
        "recordCount": result["recordCount"],
        "base": result["base"]["overall"],
        "sft": result["sft"]["overall"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
