"""Stage Decision SFT v3 with state-obligation repairs, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_STATE = {
    "task_kind", "goal_code", "observation_flags", "history_actions",
    "candidate_actions", "state_summary",
}
REQUIRED_TARGET = {
    "selected_action", "decision_reason", "alternative_actions", "stop_reason",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EXPECTED_OBJECT:{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_record(record: dict[str, Any]) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"REPAIR_MESSAGES_INVALID:{record.get('id')}")
    user = next((item for item in messages if item.get("role") == "user"), None)
    assistant = next((item for item in messages if item.get("role") == "assistant"), None)
    if user is None or assistant is None:
        raise ValueError(f"REPAIR_MESSAGES_ROLES_INVALID:{record.get('id')}")
    state = json.loads(user["content"])["policy_state"]
    target = json.loads(assistant["content"])
    if set(state) != REQUIRED_STATE or set(target) != REQUIRED_TARGET:
        raise ValueError(f"REPAIR_POLICY_FIELDS_INVALID:{record.get('id')}")
    selected = target["selected_action"]
    candidates = state["candidate_actions"]
    if selected not in candidates:
        raise ValueError(f"REPAIR_ACTION_NOT_ALLOWED:{record.get('id')}")
    expected_alternatives = [item for item in candidates if item != selected]
    if target["alternative_actions"] != expected_alternatives:
        raise ValueError(f"REPAIR_ALTERNATIVES_INVALID:{record.get('id')}")
    if (selected == "finish") != bool(target["stop_reason"]):
        raise ValueError(f"REPAIR_STOP_REASON_INVALID:{record.get('id')}")


def build(base_dir: Path, repair_dir: Path, output_dir: Path) -> dict[str, Any]:
    base_manifest = _read_json(base_dir / "manifest.json")
    repair_manifest = _read_json(repair_dir / "manifest.json")
    if base_manifest.get("trainingStarted") is not False:
        raise ValueError("BASE_STAGING_ALREADY_STARTED")
    if repair_manifest.get("trainingStarted") is not False:
        raise ValueError("REPAIR_STAGING_ALREADY_STARTED")
    base_train = _read_jsonl(base_dir / "train.jsonl")
    validation = _read_jsonl(base_dir / "validation.jsonl")
    test = _read_jsonl(base_dir / "test.jsonl")
    repair = _read_jsonl(repair_dir / repair_manifest["recordsFile"])
    if len(base_train) != 640 or len(validation) != 32 or len(test) != 70 or len(repair) != 12:
        raise ValueError("BASE_OR_REPAIR_COUNTS_CHANGED")
    for record in repair:
        _validate_record(record)
    all_records = base_train + repair + validation + test
    ids = [record["id"] for record in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError("V3_RECORD_IDS_NOT_UNIQUE")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_records = {"train": base_train + repair, "validation": validation, "test": test}
    paths: dict[str, Path] = {}
    for name, records in split_records.items():
        path = output_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        paths[name] = path
    manifest = {
        "schemaVersion": "p2j4-decision-sft-jsonl-v3-staging",
        "trainingStarted": False,
        "baseSource": base_manifest.get("schemaVersion"),
        "repairSource": repair_dir.name,
        "repairStatus": repair_manifest.get("status"),
        "splitCounts": {name: len(records) for name, records in split_records.items()},
        "baseTrainCount": len(base_train),
        "repairTrainCount": len(repair),
        "validationUnchanged": _sha256(paths["validation"]) == base_manifest["splits"]["validation"]["sha256"],
        "testUnchanged": _sha256(paths["test"]) == base_manifest["splits"]["test"]["sha256"],
        "splits": {
            name: {"path": path.name, "count": len(split_records[name]), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "inputFields": sorted(REQUIRED_STATE),
        "targetFields": sorted(REQUIRED_TARGET),
        "phaseGate": "OWNER_REVIEW_REQUIRED_BEFORE_A100_TRAINING",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("sft-data/decision-v2-staging"))
    parser.add_argument("--repair-dir", type=Path, default=Path("evals/p2j4-state-obligation-repair-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("sft-data/decision-v3-staging"))
    args = parser.parse_args()
    try:
        result = build(args.base_dir, args.repair_dir, args.output_dir)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "STAGING_READY", "trainingStarted": result["trainingStarted"], "splitCounts": result["splitCounts"], "outputDir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
