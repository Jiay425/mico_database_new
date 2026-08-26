"""Build three residual Decision-policy repair candidates locally.

The candidates are derived from the three OOD v2 residual failures after the
Runtime obligation contract was extended.  They are not added to SFT data and
are not sent to a model by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_CASES = {
    "ood-open-18": ("repair-residual-001", "CROSS_PROJECT_REQUIRED"),
    "ood-open-19": ("repair-residual-002", "EVIDENCE_REQUIRED"),
    "ood-open-22": ("repair-residual-003", "EVIDENCE_REQUIRED"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _state(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(record["messages"][1]["content"])["policy_state"]


def _build_record(source: dict[str, Any], new_id: str, required_flag: str) -> dict[str, Any]:
    record = json.loads(json.dumps(source, ensure_ascii=False))
    record["id"] = new_id
    state = _state(record)
    flags = list(state["observation_flags"])
    if required_flag not in flags:
        flags.append(required_flag)
    state["observation_flags"] = flags
    history = state["history_actions"]
    phase = "EVIDENCE_RETRIEVED" if "EVIDENCE_RETRIEVED" in flags else "OBSERVATION_VALIDATED"
    state["state_summary"] = (
        f"phase={phase}; goal={state['goal_code']}; "
        f"observation_state={flags[0]}; evidence_bindings={len(history)}; "
        f"prior_actions={','.join(history)}; flags={','.join(flags)}"
    )[:512]
    record["messages"][1]["content"] = json.dumps(
        {"policy_state": state}, ensure_ascii=False, sort_keys=True
    )
    return record


def _audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    if len(records) != len(SOURCE_CASES):
        issues.append(f"COUNT:{len(records)}")
    if len({record["id"] for record in records}) != len(records):
        issues.append("DUPLICATE_IDS")
    for record in records:
        state = _state(record)
        target = json.loads(record["messages"][2]["content"])
        if target["selected_action"] not in state["candidate_actions"]:
            issues.append(f"ACTION_NOT_ALLOWED:{record['id']}")
        if target["selected_action"] == "finish" and not target.get("stop_reason"):
            issues.append(f"STOP_REASON_MISSING:{record['id']}")
        if "raw_question" in json.dumps(record, ensure_ascii=False):
            issues.append(f"RAW_QUESTION:{record['id']}")
        if len(state["state_summary"]) > 512:
            issues.append(f"SUMMARY_TOO_LONG:{record['id']}")
    return {"status": "PASS" if not issues else "INVALID", "caseCount": len(records), "issues": issues}


def build(input_file: Path, output_dir: Path) -> dict[str, Any]:
    source = {record["id"]: record for record in _read_jsonl(input_file)}
    missing = sorted(set(SOURCE_CASES) - set(source))
    if missing:
        raise ValueError(f"SOURCE_CASES_MISSING:{','.join(missing)}")
    records = [
        _build_record(source[case_id], new_id, required_flag)
        for case_id, (new_id, required_flag) in SOURCE_CASES.items()
    ]
    audit = _audit(records)
    if audit["status"] != "PASS":
        raise AssertionError(json.dumps(audit, ensure_ascii=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": "p2j4-state-obligation-residual-repair-v2",
        "status": "OWNER_REVIEW_REQUIRED",
        "trainingStarted": False,
        "recordCount": len(records),
        "sourceFile": str(input_file),
        "sourceCaseIds": list(SOURCE_CASES),
        "recordsFile": records_path.name,
        "requiredFlags": {new_id: flag for _, (new_id, flag) in SOURCE_CASES.items()},
        "sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "audit": audit,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build residual repair candidates")
    parser.add_argument("--input", type=Path, default=Path("evals/p2j4-external-ood-v2/records.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evals/p2j4-state-obligation-residual-repair-v2"))
    args = parser.parse_args()
    manifest = build(args.input, args.output_dir)
    print(json.dumps({
        "status": "PASS",
        "recordCount": manifest["recordCount"],
        "sha256": manifest["sha256"],
        "outputDir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
