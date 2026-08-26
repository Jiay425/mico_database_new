"""Refresh External OOD states against the current Decision-policy contract.

This is a local, no-model transformation.  OOD v1 is retained as an
immutable audit input.  v2 preserves every case identity, task family,
history, candidate set, and gold action, while making the policy state agree
with the obligation flags currently emitted by
``decision_policy._observation_flags``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


OBSOLETE_FLAG_REPLACEMENTS = {
    "QUERY_REQUIRED": "BOUNDED_READ_REQUIRED",
}
OBSOLETE_FLAGS = {"QUERY_EXECUTED"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _policy_state(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"INVALID_MESSAGES:{record.get('id')}")
    user = json.loads(messages[1]["content"])
    state = user.get("policy_state")
    if not isinstance(state, dict):
        raise ValueError(f"POLICY_STATE_MISSING:{record.get('id')}")
    return state


def _append_unique(flags: list[str], value: str) -> None:
    if value not in flags:
        flags.append(value)


def _normalize_flags(state: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return normalized flags and a list of contract changes.

    The obligation logic intentionally mirrors the current Runtime source;
    it does not add new scientific policy beyond what the Runtime already
    emits.  Domain/context flags such as PROJECT_IMBALANCE and
    EVIDENCE_CONFLICT are retained because they are part of the reviewed
    state contract and remain useful to the reason audit.
    """

    original = [item for item in state.get("observation_flags", []) if isinstance(item, str)]
    flags: list[str] = []
    changes: list[str] = []
    for flag in original:
        replacement = OBSOLETE_FLAG_REPLACEMENTS.get(flag)
        if replacement is not None:
            _append_unique(flags, replacement)
            changes.append(f"{flag}->{replacement}")
            continue
        if flag in OBSOLETE_FLAGS:
            changes.append(f"removed:{flag}")
            continue
        _append_unique(flags, flag)

    goal_code = state.get("goal_code")
    history = [item for item in state.get("history_actions", []) if isinstance(item, str)]
    candidates = set(item for item in state.get("candidate_actions", []) if isinstance(item, str))

    if goal_code in {"cohort_fact", "species_coverage", "project_coverage"}:
        if history and "execute_read_query" not in history and "execute_read_query" in candidates:
            _append_unique(flags, "BOUNDED_READ_REQUIRED")
    if "compare_groups" in history and "analyze_projection" not in history:
        if "analyze_projection" in candidates:
            _append_unique(flags, "ANALYSIS_REQUIRED")
    if goal_code == "cross_project_stability":
        if "compare_groups" in history and "cross_project_validate" not in history:
            if "cross_project_validate" in candidates:
                _append_unique(flags, "CROSS_PROJECT_REQUIRED")
        elif "cross_project_validate" in history and "retrieve_evidence" not in history:
            if "retrieve_evidence" in candidates:
                _append_unique(flags, "EVIDENCE_REQUIRED")
    if goal_code == "disease_specificity":
        if "analyze_projection" in history and "cross_disease_validate" not in history:
            if "cross_disease_validate" in candidates:
                _append_unique(flags, "DISEASE_VALIDATION_REQUIRED")
    if goal_code == "evidence_grounding":
        if "analyze_projection" in history and "retrieve_evidence" not in history:
            if "retrieve_evidence" in candidates:
                _append_unique(flags, "EVIDENCE_REQUIRED")

    normalized_before_obligations = {
        OBSOLETE_FLAG_REPLACEMENTS.get(flag, flag)
        for flag in original
        if flag not in OBSOLETE_FLAGS
    }
    for flag in flags:
        if flag not in normalized_before_obligations:
            changes.append(f"added:{flag}")
    return flags, changes


def _state_summary(state: dict[str, Any], flags: list[str]) -> str:
    history = [item for item in state.get("history_actions", []) if isinstance(item, str)]
    phase = "EVIDENCE_RETRIEVED" if "EVIDENCE_RETRIEVED" in flags else (
        "NO_OBSERVATION" if not history else "OBSERVATION_VALIDATED"
    )
    return (
        f"phase={phase}; goal={state.get('goal_code', '')}; "
        f"observation_state={flags[0] if flags else 'NO_OBSERVATION'}; "
        f"evidence_bindings={len(history)}; "
        f"prior_actions={','.join(history) if history else 'none'}; "
        f"flags={','.join(flags)}"
    )[:512]


def _refresh_record(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    state = _policy_state(record)
    flags, changes = _normalize_flags(state)
    refreshed = json.loads(json.dumps(record, ensure_ascii=False))
    refreshed_state = _policy_state(refreshed)
    refreshed_state["observation_flags"] = flags
    refreshed_state["state_summary"] = _state_summary(refreshed_state, flags)
    refreshed["messages"][1]["content"] = json.dumps(
        {"policy_state": refreshed_state}, ensure_ascii=False, sort_keys=True
    )
    return refreshed, changes


def _assert_preserved(v1: list[dict[str, Any]], v2: list[dict[str, Any]]) -> None:
    if len(v1) != 30 or len(v2) != 30:
        raise AssertionError(f"OOD_COUNT:{len(v1)}:{len(v2)}")
    if [item["id"] for item in v1] != [item["id"] for item in v2]:
        raise AssertionError("OOD_IDS_CHANGED")
    for before, after in zip(v1, v2):
        state_before = _policy_state(before)
        state_after = _policy_state(after)
        before_target = json.loads(before["messages"][2]["content"])
        after_target = json.loads(after["messages"][2]["content"])
        for key in ("ood_family",):
            if before.get(key) != after.get(key):
                raise AssertionError(f"{key}_CHANGED:{before['id']}")
        for key in ("task_kind", "goal_code", "history_actions", "candidate_actions"):
            if state_before.get(key) != state_after.get(key):
                raise AssertionError(f"STATE_{key}_CHANGED:{before['id']}")
        if before_target != after_target:
            raise AssertionError(f"GOLD_TARGET_CHANGED:{before['id']}")
        if before["messages"][0] != after["messages"][0]:
            raise AssertionError(f"SYSTEM_MESSAGE_CHANGED:{before['id']}")


def _audit_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden = ("raw_question", "disease_name", "sample_id", "numeric_measurement", "sql")
    issues: list[str] = []
    obligation_counts: Counter[str] = Counter()
    for record in records:
        raw = json.dumps(record, ensure_ascii=False).lower()
        for token in forbidden:
            if token in raw:
                issues.append(f"FORBIDDEN_FIELD:{record['id']}:{token}")
        state = _policy_state(record)
        target = json.loads(record["messages"][2]["content"])
        flags = state.get("observation_flags", [])
        for flag in flags:
            if flag.endswith("_REQUIRED") or flag == "BOUNDED_READ_REQUIRED":
                obligation_counts[flag] += 1
        if target["selected_action"] not in state["candidate_actions"]:
            issues.append(f"GOLD_ACTION_NOT_ALLOWED:{record['id']}")
        if target["selected_action"] == "finish":
            if not target.get("stop_reason"):
                issues.append(f"FINISH_STOP_MISSING:{record['id']}")
        elif target.get("stop_reason") is not None:
            issues.append(f"NON_FINISH_STOP_PRESENT:{record['id']}")
        if len(state.get("state_summary", "")) > 512:
            issues.append(f"STATE_SUMMARY_TOO_LONG:{record['id']}")
    return {
        "status": "PASS" if not issues else "INVALID",
        "caseCount": len(records),
        "issues": issues,
        "obligationFlagCounts": dict(sorted(obligation_counts.items())),
    }


def build(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_file = input_dir / "records.jsonl"
    v1 = _read_jsonl(input_file)
    v2: list[dict[str, Any]] = []
    changes_by_id: dict[str, list[str]] = {}
    for record in v1:
        refreshed, changes = _refresh_record(record)
        v2.append(refreshed)
        changes_by_id[record["id"]] = changes
    _assert_preserved(v1, v2)
    contract = _audit_contract(v2)
    if contract["status"] != "PASS":
        raise AssertionError(json.dumps(contract, ensure_ascii=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in v2),
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": "p2j4-external-ood-decision-v2",
        "status": "OWNER_REVIEW_REQUIRED",
        "trainingStarted": False,
        "scope": "deidentified_policy_state_composition_ood",
        "sourceVersion": "p2j4-external-ood-decision-v1",
        "recordCount": len(v2),
        "families": {
            family: sum(case.get("ood_family") == family for case in v2)
            for family in sorted({case.get("ood_family") for case in v2})
        },
        "recordsFile": records_path.name,
        "sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "contractReference": "mico_agent_runtime/ports/decision_policy.py::_observation_flags",
        "excludedFromModelBoundary": [
            "raw_question", "disease_name", "sql", "sample_id", "numeric_measurement"
        ],
        "changeSummary": {
            "recordsChanged": sum(bool(changes) for changes in changes_by_id.values()),
            "flagChanges": dict(sorted(Counter(
                change for changes in changes_by_id.values() for change in changes
            ).items())),
            "changedRecordIds": [
                record_id for record_id, changes in changes_by_id.items() if changes
            ],
        },
        "contractAudit": contract,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "schemaVersion": "p2j4-external-ood-v2-audit-v1",
        "sourceRecords": str(input_file),
        "outputRecords": str(records_path),
        "preserved": {
            "recordIds": True,
            "taskFamilies": True,
            "goalCodes": True,
            "historyActions": True,
            "candidateActions": True,
            "goldTargets": True,
        },
        "changesByRecord": changes_by_id,
        "contractAudit": contract,
        "sha256": manifest["sha256"],
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh External OOD states locally")
    parser.add_argument("--input-dir", type=Path, default=Path("evals/p2j4-external-ood-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("evals/p2j4-external-ood-v2"))
    args = parser.parse_args()
    manifest = build(args.input_dir, args.output_dir)
    print(json.dumps({
        "status": "PASS",
        "recordCount": manifest["recordCount"],
        "recordsChanged": manifest["changeSummary"]["recordsChanged"],
        "flagChanges": manifest["changeSummary"]["flagChanges"],
        "sha256": manifest["sha256"],
        "outputDir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
