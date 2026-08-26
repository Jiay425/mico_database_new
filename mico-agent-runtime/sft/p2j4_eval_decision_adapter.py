"""Evaluate exactly one already-trained Decision adapter on a frozen JSONL set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from sft.p2j4_eval_decision_sft import _generate, _load_model, _read_jsonl, _score


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("ADAPTER_EVAL_CUDA_BF16_REQUIRED")
    records = _read_jsonl(args.test)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(args.model, tokenizer)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    outputs = _generate(model, tokenizer, records, args.max_new_tokens)
    return {
        "schemaVersion": "p2j4-decision-adapter-eval-v1",
        "testFile": args.test.name,
        "testCount": len(records),
        "adapter": str(args.adapter),
        "metrics": _score(records, outputs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    try:
        result = evaluate(args)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError, ImportError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "testCount": result["testCount"], "metrics": {key: value for key, value in result["metrics"].items() if key != "cases"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
