"""Fail-closed validation for exported Decision SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN_KEYS = {
    "question", "sql", "query", "arguments", "payload", "rawQuestion", "rawSql",
    "rawPayload", "sourceTraceId", "signatureHash", "reviewStatus", "hard_case_class",
    "hardCaseIds", "sourceRuns", "reviewReasons", "replicationCount", "sourceTraceCount",
}
EXPECTED_TARGET_KEYS = {
    "selected_action", "decision_reason", "alternative_actions", "stop_reason",
}


def _forbidden(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_forbidden(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden(child, f"{path}[{index}]"))
    return found


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SFT_JSONL_INVALID:{path.name}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"SFT_JSONL_RECORD_NOT_OBJECT:{path.name}:{line_number}")
        records.append(record)
    return records


def _validate_record(record: dict[str, Any], path: Path, index: int) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or [m.get("role") for m in messages] != [
        "system", "user", "assistant"
    ]:
        raise ValueError(f"SFT_MESSAGES_INVALID:{path.name}:{index}")
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError(f"SFT_MESSAGE_CONTENT_INVALID:{path.name}:{index}")
    try:
        user = json.loads(messages[1]["content"])
        target = json.loads(messages[2]["content"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"SFT_MESSAGE_JSON_INVALID:{path.name}:{index}") from exc
    state = user.get("policy_state") if isinstance(user, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"SFT_POLICY_STATE_MISSING:{path.name}:{index}")
    expected_state = {
        "task_kind", "goal_code", "observation_flags", "history_actions",
        "candidate_actions", "state_summary",
    }
    if set(state) != expected_state:
        raise ValueError(f"SFT_POLICY_STATE_FIELDS_INVALID:{path.name}:{index}")
    if set(target) != EXPECTED_TARGET_KEYS:
        raise ValueError(f"SFT_TARGET_FIELDS_INVALID:{path.name}:{index}")
    selected = target["selected_action"]
    if selected not in state["candidate_actions"]:
        raise ValueError(f"SFT_TARGET_ACTION_INVALID:{path.name}:{index}")
    if selected == "finish" and not target["stop_reason"]:
        raise ValueError(f"SFT_FINISH_STOP_REASON_MISSING:{path.name}:{index}")
    if selected != "finish" and target["stop_reason"] is not None:
        raise ValueError(f"SFT_NON_TERMINAL_STOP_REASON_PRESENT:{path.name}:{index}")
    forbidden = _forbidden(record)
    if forbidden:
        raise ValueError("SFT_FORBIDDEN_KEYS:" + ",".join(forbidden))


def preflight(output_dir: Path, check_torch: bool = False) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("trainingStarted") is not False:
        raise ValueError("SFT_MANIFEST_TRAINING_ALREADY_STARTED")
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        path = output_dir / f"{split}.jsonl"
        records = _read_jsonl(path)
        for index, record in enumerate(records, 1):
            _validate_record(record, path, index)
        expected = manifest["splitCounts"][split]
        if len(records) != expected:
            raise ValueError(f"SFT_COUNT_MISMATCH:{split}:{len(records)}:{expected}")
        counts[split] = len(records)
    if check_torch:
        import torch

        if not torch.cuda.is_available():
            raise ValueError("SFT_CUDA_UNAVAILABLE")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("SFT_BF16_UNAVAILABLE")
    return {"status": "PASS", "trainingStarted": False, "counts": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight exported Decision SFT data")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--check-torch", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = preflight(args.data_dir, check_torch=args.check_torch)
    except (OSError, ImportError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
