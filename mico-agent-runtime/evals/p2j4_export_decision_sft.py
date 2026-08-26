"""Export the frozen Decision review split to de-identified chat JSONL.

The exporter is intentionally fail-closed.  It only exposes structured policy
context and the selected next action; provenance, review labels, task-family
identifiers and raw user/tool content never enter the model messages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "p2j4-decision-sft-jsonl-v1"
FORBIDDEN_KEYS = {
    "question",
    "sql",
    "query",
    "arguments",
    "payload",
    "rawQuestion",
    "rawSql",
    "rawPayload",
    "sourceTraceId",
    "signatureHash",
    "reviewStatus",
    "hard_case_class",
    "hardCaseIds",
    "sourceRuns",
    "reviewReasons",
    "replicationCount",
    "sourceTraceCount",
}

SYSTEM_PROMPT = (
    "You are Mico's Scientific Agent decision policy. "
    "Choose the next action from the candidate actions in the policy state. "
    "Use only the supplied structured state and action history. "
    "Return one JSON object and no Markdown. The JSON keys must be exactly: "
    "selected_action, decision_reason, alternative_actions, stop_reason. "
    "Use stop_reason only when selected_action is finish."
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DECISION_SFT_SOURCE_NOT_OBJECT")
    return payload


def _forbidden_paths(value: Any, path: str = "$") -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                result.append(f"{path}.{key}")
            result.extend(_forbidden_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_forbidden_paths(child, f"{path}[{index}]"))
    return result


def _validate_candidate(candidate: dict[str, Any]) -> None:
    required = {
        "task_kind",
        "goal_code",
        "observation_flags",
        "history_actions",
        "candidate_actions",
        "state_summary",
        "decision_reason",
        "selected_action",
        "alternative_actions",
        "stop_reason",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError("DECISION_SFT_CANDIDATE_FIELDS_MISSING:" + ",".join(missing))
    selected = candidate["selected_action"]
    actions = candidate["candidate_actions"]
    alternatives = candidate["alternative_actions"]
    if selected not in actions:
        raise ValueError("DECISION_SFT_SELECTED_ACTION_NOT_CANDIDATE")
    if selected in alternatives:
        raise ValueError("DECISION_SFT_SELECTED_ACTION_IN_ALTERNATIVES")
    if any(action not in actions for action in alternatives):
        raise ValueError("DECISION_SFT_ALTERNATIVE_ACTION_NOT_CANDIDATE")
    if selected == "finish" and not candidate["stop_reason"]:
        raise ValueError("DECISION_SFT_FINISH_STOP_REASON_MISSING")
    if selected != "finish" and candidate["stop_reason"] is not None:
        raise ValueError("DECISION_SFT_NON_TERMINAL_STOP_REASON_PRESENT")
    for field in ("state_summary", "decision_reason"):
        if not isinstance(candidate[field], str) or not candidate[field].strip():
            raise ValueError("DECISION_SFT_EMPTY_" + field.upper())


def _record(split: str, index: int, item: dict[str, Any]) -> dict[str, Any]:
    candidate = item.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("DECISION_SFT_REVIEW_ITEM_CANDIDATE_MISSING")
    _validate_candidate(candidate)

    state = {
        "task_kind": candidate["task_kind"],
        "goal_code": candidate["goal_code"],
        "observation_flags": candidate["observation_flags"],
        "history_actions": candidate["history_actions"],
        "candidate_actions": candidate["candidate_actions"],
        "state_summary": candidate["state_summary"],
    }
    target = {
        "selected_action": candidate["selected_action"],
        "decision_reason": candidate["decision_reason"],
        "alternative_actions": candidate["alternative_actions"],
        "stop_reason": candidate["stop_reason"],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True),
        },
        {
            "role": "assistant",
            "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        },
    ]
    record = {
        "id": f"{split}-{index:04d}",
        "messages": messages,
    }
    forbidden = _forbidden_paths(record)
    if forbidden:
        raise ValueError("DECISION_SFT_EXPORTED_FORBIDDEN_KEYS:" + ",".join(forbidden))
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(split_path: Path, output_dir: Path) -> dict[str, Any]:
    source = _read(split_path)
    if source.get("trainingStarted") is not False:
        raise ValueError("DECISION_SFT_SOURCE_TRAINING_ALREADY_STARTED")
    if source.get("familyDisjoint") is not True or source.get("sourceTraceDisjoint") is not True:
        raise ValueError("DECISION_SFT_SOURCE_SPLIT_NOT_DISJOINT")

    splits = source.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("DECISION_SFT_SOURCE_SPLITS_INVALID")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_splits: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        items = splits[split]
        if not isinstance(items, list) or not items:
            raise ValueError("DECISION_SFT_EMPTY_SPLIT:" + split)
        records = [_record(split, index, item) for index, item in enumerate(items, 1)]
        path = output_dir / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        manifest_splits[split] = {
            "count": len(records),
            "path": path.name,
            "sha256": _sha256(path),
            "taskFamilyCount": source["splitCounts"][split]["taskFamilyCount"],
            "sourceTraceCount": source["splitCounts"][split]["sourceTraceCount"],
            "taskFamilies": sorted({
                item["candidate"]["task_family"] for item in items
            }),
        }

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "sourceSplit": split_path.name,
        "trainingStarted": False,
        "sourceReviewItemCount": source.get("sourceReviewItemCount"),
        "splitCounts": {
            split: manifest_splits[split]["count"]
            for split in ("train", "validation", "test")
        },
        "splits": manifest_splits,
        "inputFields": [
            "task_kind",
            "goal_code",
            "observation_flags",
            "history_actions",
            "candidate_actions",
            "state_summary",
        ],
        "targetFields": [
            "selected_action",
            "decision_reason",
            "alternative_actions",
            "stop_reason",
        ],
        "excludedFields": sorted(FORBIDDEN_KEYS),
        "taskFamilySplitPreserved": True,
        "sourceTraceSplitPreserved": True,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export frozen Decision SFT JSONL")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = export(args.split, args.output_dir)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "READY",
        "trainingStarted": result["trainingStarted"],
        "splitCounts": result["splitCounts"],
        "outputDir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
