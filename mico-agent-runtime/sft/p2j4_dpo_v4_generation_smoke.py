"""Run a deterministic closed-action generation smoke on frozen DPO states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: Path, count: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Deterministic spread across the frozen validation order.  Never print
    # raw state text; the artifact contains only opaque IDs and action checks.
    if len(rows) <= count:
        return rows
    return [rows[index * (len(rows) - 1) // (count - 1)] for index in range(count)]


def _content(tokenizer: Any, prompt: list[dict[str, str]], model: Any) -> str:
    rendered = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id)
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _selected_action(content: str) -> str | None:
    """Extract the action without persisting model text or importing Runtime."""
    start = content.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(content[start:])
    except json.JSONDecodeError:
        return None
    action = value.get("selected_action") if isinstance(value, dict) else None
    return action if isinstance(action, str) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--dpo-adapter", type=Path,
                        help="Optional. Omit to smoke-test the merged SFT policy before DPO.")
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("DPO_V4_CUDA_BF16_REQUIRED")
    rows = _rows(args.states, args.count)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    sft = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=False)
    merged = sft.merge_and_unload()
    if hasattr(merged, "peft_config"):
        delattr(merged, "peft_config")
    if hasattr(merged, "peft_config"):
        raise SystemExit("DPO_V4_MERGED_SFT_ADAPTER_METADATA_PRESENT")
    policy = PeftModel.from_pretrained(merged, args.dpo_adapter, is_trainable=False) if args.dpo_adapter else merged
    policy.eval()
    checks: list[dict[str, Any]] = []
    for row in rows:
        try:
            selected = _selected_action(_content(tokenizer, row["prompt"], policy))
            valid = selected in row["candidate_actions"]
            code = "PASS" if valid else "ACTION_NOT_IN_CANDIDATES"
        except Exception:
            valid = False
            code = "INVALID_CLOSED_ACTION_OUTPUT"
            selected = None
        checks.append({"id": row["id"], "selectedAction": selected, "status": code})
    failures = [check for check in checks if check["status"] != "PASS"]
    result = {
        "schemaVersion": "p2j4-dpo-v4-generation-smoke-v1",
        "status": "PASS" if not failures else "FAIL",
        "sampleCount": len(checks),
        "validClosedActionCount": len(checks) - len(failures),
        "invalidClosedActionCount": len(failures),
        "policyTopology": "merged SFT weights + fresh DPO LoRA" if args.dpo_adapter else "Qwen3-8B Base + merged SFT v5 weights",
        "checks": checks,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "sampleCount", "validClosedActionCount", "invalidClosedActionCount")}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
