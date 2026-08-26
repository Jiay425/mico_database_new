"""Evaluate one merged-SFT plus fresh-DPO adapter on a frozen split.

The DPO v4 adapter is trained on merged SFT weights and must never be mounted
directly on Base.  Per-case artifacts retain only opaque IDs and closed-policy
metrics, not prompts or raw model text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(row["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _score(row: dict[str, Any], text: str) -> dict[str, Any]:
    state = json.loads(row["messages"][1]["content"])["policy_state"]
    target = json.loads(row["messages"][2]["content"])
    prediction = _parse(text)
    action = prediction.get("selected_action") if prediction else None
    stop_reason = prediction.get("stop_reason") if prediction else None
    valid = prediction is not None
    allowed = action in state["candidate_actions"]
    action_correct = action == target["selected_action"]
    stop_consistent = (action == "finish" and bool(stop_reason)) or (action != "finish" and stop_reason is None)
    return {
        "recordId": row["id"],
        "selectedAction": action,
        "goldAction": target["selected_action"],
        "jsonValid": valid,
        "actionAllowed": allowed,
        "actionCorrect": action_correct,
        "stopConsistent": stop_consistent,
        "policyPass": bool(valid and allowed and action_correct and stop_consistent),
    }


def _metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    count = lambda key: sum(bool(case[key]) for case in cases)
    return {
        "caseCount": total,
        "jsonValidCount": count("jsonValid"),
        "jsonValidRate": count("jsonValid") / total if total else 0.0,
        "actionCorrectCount": count("actionCorrect"),
        "actionAccuracy": count("actionCorrect") / total if total else 0.0,
        "actionAllowedCount": count("actionAllowed"),
        "actionAllowedRate": count("actionAllowed") / total if total else 0.0,
        "invalidClosedActionCount": total - count("actionAllowed"),
        "stopConsistentCount": count("stopConsistent"),
        "stopConsistencyRate": count("stopConsistent") / total if total else 0.0,
        "policyPassCount": count("policyPass"),
        "policyPassRate": count("policyPass") / total if total else 0.0,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("DPO_V4_EVAL_CUDA_BF16_REQUIRED")
    rows = _rows(args.test)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    sft = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=False)
    merged = sft.merge_and_unload()
    if hasattr(merged, "peft_config"):
        delattr(merged, "peft_config")
    if hasattr(merged, "peft_config"):
        raise RuntimeError("DPO_V4_MERGED_SFT_ADAPTER_METADATA_PRESENT")
    policy = PeftModel.from_pretrained(merged, args.dpo_adapter, is_trainable=False)
    policy.eval()
    cases: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        prompts = [_prompt(tokenizer, row) for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.inference_mode():
            generated = policy.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        width = inputs["input_ids"].shape[1]
        for index, row in enumerate(batch):
            cases.append(_score(row, tokenizer.decode(generated[index, width:], skip_special_tokens=True)))
    return {
        "schemaVersion": "p2j4-merged-sft-fresh-dpo-eval-v1",
        "status": "COMPLETED",
        "referenceTopology": "Qwen3-8B Base + frozen SFT adapter",
        "policyTopology": "Qwen3-8B Base + merged SFT v5 weights + fresh DPO LoRA",
        "baseModel": args.model,
        "sftAdapter": str(args.sft_adapter),
        "dpoAdapter": str(args.dpo_adapter),
        "testFile": args.test.name,
        "testSha256": _sha(args.test),
        "metrics": _metrics(cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--dpo-adapter", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    try:
        result = evaluate(args)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError, ImportError, json.JSONDecodeError, KeyError) as exc:
        print(json.dumps({"status": "FAILED", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "testCount": result["metrics"]["caseCount"], "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
