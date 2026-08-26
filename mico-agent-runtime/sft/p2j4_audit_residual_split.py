"""Audit residual repair candidates against the frozen Decision v3 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


STATE_KEYS = (
    "task_kind",
    "goal_code",
    "observation_flags",
    "history_actions",
    "candidate_actions",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _state(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(record["messages"][1]["content"])["policy_state"]


def _signature(record: dict[str, Any]) -> str:
    state = _state(record)
    payload = {key: state[key] for key in STATE_KEYS}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit(base_dir: Path, repair_dir: Path, output: Path) -> dict[str, Any]:
    split_records = {
        name: _read_jsonl(base_dir / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    repair_records = _read_jsonl(repair_dir / "records.jsonl")
    all_records = [record for records in split_records.values() for record in records]
    all_ids = {record["id"] for record in all_records}
    repair_ids = [record["id"] for record in repair_records]
    all_signatures = {_signature(record): record["id"] for record in all_records}
    repair_signatures = [_signature(record) for record in repair_records]
    duplicate_ids = sorted(set(repair_ids) & all_ids)
    duplicate_signatures = [
        {"repairId": record["id"], "existingId": all_signatures[signature]}
        for record, signature in zip(repair_records, repair_signatures)
        if signature in all_signatures
    ]
    task_goal_counts = Counter(
        f"{_state(record)['task_kind']}::{_state(record)['goal_code']}"
        for record in repair_records
    )
    result = {
        "schemaVersion": "p2j4-residual-repair-split-audit-v1",
        "baseStaging": str(base_dir),
        "repairDirectory": str(repair_dir),
        "baseCounts": {name: len(records) for name, records in split_records.items()},
        "repairCount": len(repair_records),
        "repairIds": repair_ids,
        "duplicateIds": duplicate_ids,
        "duplicateStateSignatures": duplicate_signatures,
        "taskGoalCounts": dict(sorted(task_goal_counts.items())),
        "validationTestStateOverlap": [
            item for item in duplicate_signatures
            if item["existingId"] in {
                record["id"] for name in ("validation", "test") for record in split_records[name]
            }
        ],
        "status": "PASS" if not duplicate_ids and not duplicate_signatures else "REVIEW_REQUIRED",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit residual repair split overlap")
    parser.add_argument("--base-dir", type=Path, default=Path("sft-data/decision-v3-staging"))
    parser.add_argument("--repair-dir", type=Path, default=Path("evals/p2j4-state-obligation-residual-repair-v2"))
    parser.add_argument("--output", type=Path, default=Path("evals/p2j4-state-obligation-residual-repair-v2/split-audit.json"))
    args = parser.parse_args()
    result = audit(args.base_dir, args.repair_dir, args.output)
    print(json.dumps({
        "status": result["status"],
        "repairCount": result["repairCount"],
        "duplicateIds": result["duplicateIds"],
        "duplicateStateSignatures": result["duplicateStateSignatures"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
