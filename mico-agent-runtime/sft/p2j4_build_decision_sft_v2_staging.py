"""Build local Decision SFT v2 staging data with conflict-repair samples.

This command does not start training.  It appends only the 50 reviewed
candidate states to the frozen v1 train split and keeps v1 validation/test
unchanged, so the v1 comparison remains reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_TARGET_KEYS = {
    "selected_action", "decision_reason", "alternative_actions", "stop_reason",
}
FORBIDDEN_KEYS = {
    "question", "sql", "query", "arguments", "payload", "rawQuestion", "rawSql",
    "rawPayload", "sourceTraceId", "signatureHash", "reviewStatus", "hard_case_class",
    "hardCaseIds", "sourceRuns", "reviewReasons", "replicationCount", "sourceTraceCount",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("EXPECTED_OBJECT:" + str(path))
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _validate_candidate(candidate: dict[str, Any]) -> None:
    required = {
        "task_kind", "goal_code", "observation_flags", "history_actions",
        "candidate_actions", "state_summary", "decision_reason",
        "selected_action", "alternative_actions", "stop_reason",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError("REPAIR_CANDIDATE_FIELDS_MISSING:" + ",".join(missing))
    selected = candidate["selected_action"]
    actions = candidate["candidate_actions"]
    alternatives = candidate["alternative_actions"]
    if selected not in actions or selected in alternatives:
        raise ValueError("REPAIR_SELECTED_ACTION_INVALID")
    if any(action not in actions for action in alternatives):
        raise ValueError("REPAIR_ALTERNATIVE_ACTION_INVALID")
    if selected == "finish" and not candidate["stop_reason"]:
        raise ValueError("REPAIR_FINISH_STOP_REASON_MISSING")
    if selected != "finish" and candidate["stop_reason"] is not None:
        raise ValueError("REPAIR_NON_TERMINAL_STOP_REASON_PRESENT")
    if "EVIDENCE_CONFLICT" not in candidate["observation_flags"]:
        raise ValueError("REPAIR_CONFLICT_FLAG_MISSING")
    if selected == "finish" and candidate["stop_reason"] != "QUALITY_RISK":
        raise ValueError("REPAIR_FINISH_MUST_USE_QUALITY_RISK")


def _record(record_id: str, candidate: dict[str, Any], system_prompt: str) -> dict[str, Any]:
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
    record = {
        "id": record_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"policy_state": state}, ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ],
    }
    forbidden = _forbidden(record)
    if forbidden:
        raise ValueError("REPAIR_EXPORTED_FORBIDDEN_KEYS:" + ",".join(forbidden))
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(base_dir: Path, repair_path: Path, output_dir: Path) -> dict[str, Any]:
    base_manifest = _read_json(base_dir / "manifest.json")
    repair = _read_json(repair_path)
    if base_manifest.get("trainingStarted") is not False:
        raise ValueError("V1_BASE_ALREADY_TRAINING")
    if repair.get("trainingStarted") is not False:
        raise ValueError("REPAIR_ALREADY_TRAINING")
    candidates = repair.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 50:
        raise ValueError("REPAIR_CANDIDATE_COUNT_INVALID")

    base_train = _read_jsonl(base_dir / "train.jsonl")
    validation = _read_jsonl(base_dir / "validation.jsonl")
    test = _read_jsonl(base_dir / "test.jsonl")
    if len(base_train) != 590 or len(validation) != 32 or len(test) != 70:
        raise ValueError("V1_SPLIT_COUNTS_CHANGED")
    system_prompt = base_train[0]["messages"][0]["content"]
    repair_records = [
        _record(f"repair-v2-{index:04d}", candidate, system_prompt)
        for index, candidate in enumerate(candidates, 1)
    ]
    train = base_train + repair_records
    record_ids = [record["id"] for record in train + validation + test]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("V2_RECORD_IDS_NOT_UNIQUE")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": output_dir / "train.jsonl", "validation": output_dir / "validation.jsonl", "test": output_dir / "test.jsonl"}
    for name, records in (("train", train), ("validation", validation), ("test", test)):
        paths[name].write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    manifest = {
        "schemaVersion": "p2j4-decision-sft-jsonl-v2-staging",
        "trainingStarted": False,
        "baseSource": base_manifest.get("sourceSplit"),
        "repairSource": repair_path.name,
        "repairStatus": repair.get("status"),
        "splitCounts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "baseTrainCount": len(base_train),
        "repairTrainCount": len(repair_records),
        "validationUnchanged": len(validation) == 32,
        "testUnchanged": len(test) == 70,
        "splits": {
            name: {"path": path.name, "count": len(records), "sha256": _sha256(path)}
            for name, path, records in (
                ("train", paths["train"], train),
                ("validation", paths["validation"], validation),
                ("test", paths["test"], test),
            )
        },
        "inputFields": [
            "task_kind", "goal_code", "observation_flags", "history_actions",
            "candidate_actions", "state_summary",
        ],
        "targetFields": sorted(EXPECTED_TARGET_KEYS),
        "excludedFields": sorted(FORBIDDEN_KEYS),
        "phaseGate": "OWNER_REVIEW_REQUIRED_BEFORE_A100_TRAINING",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Decision SFT v2 staging data")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build(args.base_dir, args.repair, args.output_dir)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "STAGING_READY",
        "trainingStarted": result["trainingStarted"],
        "splitCounts": result["splitCounts"],
        "phaseGate": result["phaseGate"],
        "outputDir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
