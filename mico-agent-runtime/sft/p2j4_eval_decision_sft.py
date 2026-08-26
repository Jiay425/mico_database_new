"""Evaluate Base and LoRA Decision policies on the frozen Test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records:
        inputs = tokenizer(_prompt(tokenizer, record), return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_tokens = generated[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        results.append({"record_id": record["id"], "text": text})
    return results


def _score(records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {record["id"]: record for record in records}
    json_valid = 0
    action_correct = 0
    action_allowed = 0
    stop_consistent = 0
    policy_pass = 0
    scored: list[dict[str, Any]] = []
    for output in outputs:
        record = by_id[output["record_id"]]
        state = json.loads(record["messages"][1]["content"])["policy_state"]
        target = json.loads(record["messages"][2]["content"])
        prediction = _parse_json(output["text"])
        valid = prediction is not None
        if valid:
            json_valid += 1
        predicted_action = prediction.get("selected_action") if prediction else None
        action_ok = predicted_action == target["selected_action"]
        allowed_ok = predicted_action in state["candidate_actions"]
        predicted_stop = prediction.get("stop_reason") if prediction else None
        stop_ok = (
            (predicted_action == "finish" and predicted_stop is not None)
            or (predicted_action != "finish" and predicted_stop is None)
        )
        action_correct += int(action_ok)
        action_allowed += int(allowed_ok)
        stop_consistent += int(stop_ok)
        policy_pass += int(valid and action_ok and allowed_ok and stop_ok)
        scored.append({
            "record_id": output["record_id"],
            "prediction": prediction,
            "raw_output": output["text"],
            "gold_action": target["selected_action"],
            "action_correct": action_ok,
            "action_allowed": allowed_ok,
            "stop_consistent": stop_ok,
        })
    total = len(records)
    return {
        "caseCount": total,
        "jsonValidCount": json_valid,
        "jsonValidRate": json_valid / total if total else 0.0,
        "actionCorrectCount": action_correct,
        "actionAccuracy": action_correct / total if total else 0.0,
        "actionAllowedCount": action_allowed,
        "actionAllowedRate": action_allowed / total if total else 0.0,
        "stopConsistentCount": stop_consistent,
        "stopConsistencyRate": stop_consistent / total if total else 0.0,
        "policyPassCount": policy_pass,
        "policyPassRate": policy_pass / total if total else 0.0,
        "cases": scored,
    }


def _load_model(model_path: str, tokenizer: Any) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT_EVAL_CUDA_BF16_REQUIRED")
    records = _read_jsonl(args.test)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = _load_model(args.model, tokenizer)
    base_outputs = _generate(base_model, tokenizer, records, args.max_new_tokens)
    base_scores = _score(records, base_outputs)
    del base_model
    torch.cuda.empty_cache()

    sft_model = _load_model(args.model, tokenizer)
    sft_model = PeftModel.from_pretrained(sft_model, args.adapter)
    sft_model.eval()
    sft_outputs = _generate(sft_model, tokenizer, records, args.max_new_tokens)
    sft_scores = _score(records, sft_outputs)
    del sft_model
    torch.cuda.empty_cache()

    return {
        "schemaVersion": "p2j4-decision-sft-eval-v1",
        "testFile": args.test.name,
        "testCount": len(records),
        "trainingStarted": True,
        "base": base_scores,
        "sft": sft_scores,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Base vs Decision SFT")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, RuntimeError, ValueError, ImportError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "PASS",
        "testCount": result["testCount"],
        "base": {key: value for key, value in result["base"].items() if key != "cases"},
        "sft": {key: value for key, value in result["sft"].items() if key != "cases"},
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
