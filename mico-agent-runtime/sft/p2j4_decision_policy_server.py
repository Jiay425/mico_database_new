"""Small OpenAI-compatible stdlib HTTP server for Decision-SFT.

The service is separate from the Python Scientific Runtime. Runtime remains
responsible for closed action reification, authorization, and execution; this
process only generates the DecisionPolicy JSON from the A100 adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class LoadedPolicyModel:
    model: Any
    tokenizer: Any
    model_name: str

    @classmethod
    def load(
        cls,
        model_path: str,
        adapter_path: str | None,
        sft_adapter_path: str | None,
        dpo_adapter_path: str | None,
        model_name: str,
    ) -> "LoadedPolicyModel":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        elif sft_adapter_path and dpo_adapter_path:
            # DPO v4 is deliberately *not* a second adapter on the frozen SFT
            # adapter.  Merge SFT into the base weights, remove PEFT metadata
            # left by merge_and_unload(), then mount the independently trained
            # DPO LoRA.  This is the same topology used for the frozen
            # reference and the formal offline evaluations.
            sft = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=False)
            model = sft.merge_and_unload()
            if hasattr(model, "peft_config"):
                delattr(model, "peft_config")
            if hasattr(model, "peft_config"):
                raise RuntimeError("DPO_V4_MERGED_SFT_ADAPTER_METADATA_PRESENT")
            model = PeftModel.from_pretrained(model, dpo_adapter_path, is_trainable=False)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, model_name=model_name)

    def generate(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 256),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        tokens = generated[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def build_handler(service: LoadedPolicyModel, expected_token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MicoDecisionSFT/1.0"

        def log_message(self, format: str, *args: object) -> None:
            super().log_message(format, *args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                _json_response(self, 404, {"detail": "NOT_FOUND"})
                return
            _json_response(self, 200, {"status": "ready", "model": service.model_name})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                _json_response(self, 404, {"detail": "NOT_FOUND"})
                return
            if expected_token and self.headers.get("Authorization") != f"Bearer {expected_token}":
                _json_response(self, 401, {"detail": "SFT_POLICY_UNAUTHORIZED"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 1024 * 1024:
                    raise ValueError("request body size is invalid")
                request = json.loads(self.rfile.read(length))
                messages = request["messages"]
                if not isinstance(messages, list) or not messages or len(messages) > 4:
                    raise ValueError("messages shape is invalid")
                normalized: list[dict[str, str]] = []
                for message in messages:
                    if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
                        raise ValueError("message role is invalid")
                    if not isinstance(message.get("content"), str):
                        raise ValueError("message content is invalid")
                    normalized.append({"role": message["role"], "content": message["content"]})
                model_name = request.get("model", service.model_name)
                max_tokens = min(int(request.get("max_tokens", 256)), 256)
                if max_tokens < 1:
                    raise ValueError("max_tokens is invalid")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                _json_response(self, 422, {"detail": f"SFT_POLICY_REQUEST_INVALID: {type(error).__name__}"})
                return
            started = time.perf_counter()
            content = service.generate(normalized, max_tokens)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            _json_response(self, 200, {
                "id": f"mico-sft-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "metadata": {"latency_ms": elapsed_ms},
            })

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve Mico Decision-SFT policy")
    parser.add_argument("--model", default=os.environ.get("MICO_SFT_MODEL_PATH", ""))
    parser.add_argument(
        "--adapter",
        default=os.environ.get("MICO_SFT_ADAPTER_PATH", ""),
        help="Optional single adapter for Base/SFT serving. Cannot be combined with DPO v4 adapter paths.",
    )
    parser.add_argument(
        "--sft-adapter",
        default=os.environ.get("MICO_DPO_V4_SFT_ADAPTER_PATH", ""),
        help="Frozen SFT v5 adapter. Required together with --dpo-adapter for DPO v4 serving.",
    )
    parser.add_argument(
        "--dpo-adapter",
        default=os.environ.get("MICO_DPO_V4_ADAPTER_PATH", ""),
        help="Fresh DPO v4 adapter. Required together with --sft-adapter.",
    )
    parser.add_argument("--model-name", default=os.environ.get("MICO_SFT_POLICY_MODEL", "qwen3-8b-decision-base"))
    parser.add_argument("--host", default=os.environ.get("MICO_SFT_POLICY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MICO_SFT_POLICY_PORT", "9001")))
    args = parser.parse_args()
    if not args.model:
        parser.error("--model is required")
    adapter = args.adapter.strip() or None
    sft_adapter = args.sft_adapter.strip() or None
    dpo_adapter = args.dpo_adapter.strip() or None
    if adapter and (sft_adapter or dpo_adapter):
        parser.error("--adapter cannot be combined with --sft-adapter/--dpo-adapter")
    if bool(sft_adapter) != bool(dpo_adapter):
        parser.error("--sft-adapter and --dpo-adapter must be supplied together")
    service = LoadedPolicyModel.load(
        args.model, adapter, sft_adapter, dpo_adapter, args.model_name
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(
        service, os.environ.get("MICO_SFT_POLICY_SERVER_TOKEN", "").strip()
    ))
    print(json.dumps({"status": "READY", "host": args.host, "port": args.port, "model": args.model_name}))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
