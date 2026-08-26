"""Load Qwen3 once and run a minimal local-generation smoke test."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a local causal LM")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("MODEL_SMOKE_CUDA_BF16_REQUIRED")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    messages = [
        {
            "role": "system",
            "content": "Return JSON only with selected_action and decision_reason.",
        },
        {
            "role": "user",
            "content": '{"policy_state":{"task_kind":"data_fact","goal_code":"sample_count","observation_flags":["NO_OBSERVATION","METADATA_FIRST"],"history_actions":[],"candidate_actions":["inspect_cohort","finish"],"state_summary":"observation_state=NO_OBSERVATION; evidence_bindings=0; source_routes=none; prior_actions=none"}}',
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    print("MODEL_SMOKE_PASS")
    print("MODEL_DEVICE", next(model.parameters()).device)
    print("MODEL_OUTPUT", tokenizer.decode(new_tokens, skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
