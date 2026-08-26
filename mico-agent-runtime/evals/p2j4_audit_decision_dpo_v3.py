"""Audit and stage the DPO v3 preference candidates without training them.

The staging output is intentionally marked REVIEW_REQUIRED.  This script is a
hard local gate: an A100 job must consume a separately approved freeze manifest,
never this candidate file directly.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "evals" / "p2j4-decision-dpo-v3-candidates-20260825.json"
TEST70 = ROOT / "sft-data" / "decision-v4-staging" / "test.jsonl"
OOD30 = ROOT / "evals" / "p2j4-external-ood-v2" / "records.jsonl"
OUT_DIR = ROOT / "sft-data" / "decision-dpo-v3-review-staging"
REPORT = ROOT / "evals" / "p2j4-decision-dpo-v3-audit-20260825.json"
PACKET = ROOT / "evals" / "p2j4-decision-dpo-v3-core-review-packet-20260825.json"

ALL_ACTIONS = {
    "execute_read_query", "inspect_cohort", "compare_groups", "stratified_analysis",
    "adjust_confounders", "cross_project_validate", "cross_disease_validate",
    "retrieve_evidence", "analyze_projection", "finish",
}
STOP = {"EVIDENCE_SUFFICIENT", "QUALITY_RISK", "NO_NEW_INFORMATION"}
QUOTA = {"efficiency": 350, "exploration_depth": 200, "reason_quality": 200}
VAL_FAMILIES = {
    "efficiency:no_repeat_projection",
    "exploration_depth:adjust_before_stratify",
    "reason_quality:replication_reason",
}
EXCLUDED_SOURCE_NAMES = {
    "qwen-base-infrastructure-preflight-failure.json",
    "qwen-base-concurrent-write-aborted.json",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _signature(state: dict[str, Any]) -> str:
    safe = {key: state.get(key) for key in (
        "task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary")}
    return hashlib.sha256(_canonical(safe).encode("utf-8")).hexdigest()[:20]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _state_from_prompt(prompt: list[dict[str, str]]) -> dict[str, Any]:
    return json.loads(prompt[1]["content"])["policy_state"]


def _heldout_signatures() -> set[str]:
    records = [*_load_jsonl(TEST70), *_load_jsonl(OOD30)]
    signatures = set()
    for item in records:
        messages = item["messages"]
        state = json.loads(messages[1]["content"])["policy_state"]
        signatures.add(_signature(state))
    return signatures


def _validate_pair(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    meta = item["metadata"]
    dimension = meta.get("preference_dimension")
    if dimension not in QUOTA:
        errors.append("UNKNOWN_DIMENSION")
    state = _state_from_prompt(item["prompt"])
    candidates = state.get("candidate_actions", [])
    if not candidates or any(action not in ALL_ACTIONS for action in candidates):
        errors.append("INVALID_CANDIDATE_ACTIONS")
    if len(candidates) != len(set(candidates)):
        errors.append("DUPLICATE_CANDIDATE_ACTIONS")
    if _signature(state) != meta.get("state_signature"):
        errors.append("STATE_SIGNATURE_MISMATCH")
    try:
        chosen = json.loads(item["chosen"])
        rejected = json.loads(item["rejected"])
    except json.JSONDecodeError:
        return [*errors, "INVALID_COMPLETION_JSON"]
    for role, completion in (("chosen", chosen), ("rejected", rejected)):
        if set(completion) != {"selected_action", "decision_reason", "alternative_actions", "stop_reason"}:
            errors.append(f"{role.upper()}_SCHEMA")
            continue
        action = completion["selected_action"]
        if action not in candidates:
            errors.append(f"{role.upper()}_ACTION_NOT_CANDIDATE")
        if completion["alternative_actions"] != [value for value in candidates if value != action]:
            errors.append(f"{role.upper()}_ALTERNATIVES_MISMATCH")
        if action == "finish":
            if completion["stop_reason"] not in STOP:
                errors.append(f"{role.upper()}_FINISH_STOP")
        elif completion["stop_reason"] is not None:
            errors.append(f"{role.upper()}_NON_FINISH_STOP")
        # Reason-quality pairs intentionally put a terse generic rationale in
        # rejected; require a substantive reason for chosen only.
        if (role == "chosen" or dimension != "reason_quality") and (
            not isinstance(completion["decision_reason"], str) or len(completion["decision_reason"].strip()) < 20
        ):
            errors.append(f"{role.upper()}_WEAK_REASON")
    if item["chosen"] == item["rejected"]:
        errors.append("IDENTICAL_COMPLETIONS")
    if dimension == "reason_quality":
        if chosen.get("selected_action") != rejected.get("selected_action"):
            errors.append("REASON_PAIR_ACTION_CHANGED")
        if chosen.get("decision_reason") == rejected.get("decision_reason"):
            errors.append("REASON_PAIR_REASON_UNCHANGED")
    elif chosen.get("selected_action") == rejected.get("selected_action"):
        errors.append("ACTION_PREFERENCE_UNCHANGED")
    # Guard the key project rule: short paths are preferred only if evidence is
    # actually sufficient.  A finish preference amid an active risk is rejected.
    flags = set(state.get("observation_flags", []))
    quality_risk_boundary = (
        chosen.get("selected_action") == "finish"
        and chosen.get("stop_reason") == "QUALITY_RISK"
        and "RETRIEVAL_EXHAUSTED" in flags
    )
    if chosen.get("selected_action") == "finish" and not quality_risk_boundary and flags.intersection({
        "PROJECT_IMBALANCE", "CONFOUNDER_PRESENT", "EVIDENCE_CONFLICT", "EVIDENCE_INCOMPLETE",
        "CROSS_PROJECT_REQUIRED", "CROSS_DISEASE_REQUIRED", "PARTIAL_PROJECT_COVERAGE",
    }):
        errors.append("PREMATURE_FINISH_PREFERRED")
    if set(meta.get("source_ids", [])).intersection(EXCLUDED_SOURCE_NAMES):
        errors.append("EXCLUDED_SOURCE_REFERENCED")
    if meta.get("source_kind") == "runtime_failure_pattern_derived" and not meta.get("policy_induced"):
        errors.append("NON_POLICY_RUNTIME_SOURCE")
    return errors


def _choose_balanced(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep 70/40/40 per scenario family for a balanced 750-pair staging set."""
    wanted_per_family = {"efficiency": 70, "exploration_depth": 40, "reason_quality": 40}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        meta = record["metadata"]
        grouped[(meta["preference_dimension"], meta["task_family"])].append(record)
    chosen: list[dict[str, Any]] = []
    for (dimension, family), values in sorted(grouped.items()):
        values = sorted(values, key=lambda value: value["id"])
        chosen.extend(values[:wanted_per_family[dimension]])
    counts = Counter(item["metadata"]["preference_dimension"] for item in chosen)
    if counts != Counter(QUOTA):
        raise ValueError(f"BALANCE_QUOTA_FAILED:{dict(counts)}")
    return chosen


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(_canonical(item) + "\n" for item in records), encoding="utf-8")


def audit_and_stage() -> dict[str, Any]:
    payload = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    errors: dict[str, list[str]] = {}
    eligible: list[dict[str, Any]] = []
    heldout = _heldout_signatures()
    for record in payload["records"]:
        record_errors = _validate_pair(record)
        if record["metadata"]["state_signature"] in heldout:
            record_errors.append("HELDOUT_STATE_LEAKAGE")
        if record_errors:
            errors[record["id"]] = record_errors
        else:
            eligible.append(record)
    selected = _choose_balanced(eligible)
    selected_signatures = {record["metadata"]["state_signature"] for record in selected}
    if len(selected_signatures) != len(selected):
        raise ValueError("DUPLICATE_SELECTED_STATE_SIGNATURE")
    train = [record for record in selected if record["metadata"]["task_family"] not in VAL_FAMILIES]
    validation = [record for record in selected if record["metadata"]["task_family"] in VAL_FAMILIES]
    train_families = {record["metadata"]["task_family"] for record in train}
    validation_families = {record["metadata"]["task_family"] for record in validation}
    if train_families.intersection(validation_families):
        raise ValueError("TASK_FAMILY_SPLIT_LEAKAGE")
    if not validation:
        raise ValueError("EMPTY_VALIDATION")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUT_DIR / "train.jsonl", train)
    _write_jsonl(OUT_DIR / "validation.jsonl", validation)
    # Review is stratified by actual decision-obligation family.  Selecting the
    # first 150 high-priority rows would omit reason-quality pairs and would not
    # be a defensible core audit.
    by_review_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_review_family[item["metadata"]["task_family"]].append(item)
    review_candidates = []
    for family, values in sorted(by_review_family.items()):
        review_candidates.extend(sorted(values, key=lambda item: item["id"])[:10])
    if len(review_candidates) != 150:
        raise ValueError("CORE_REVIEW_STRATIFICATION_COUNT_INVALID")
    packet = {
        "schemaVersion": "p2j4-decision-dpo-v3-core-review-v1",
        "status": "HUMAN_REVIEW_REQUIRED",
        "purpose": "review 150 balanced, state-distinct DPO v3 pairs before freeze",
        "recordCount": len(review_candidates),
        "reviewChecklist": [
            "chosen and rejected are both schema-valid candidate actions, or the same action with different rationale",
            "preference follows minimal necessary exploration rather than shortest-path bias",
            "state flags justify the preferred depth and do not imply disease causality",
            "reason-quality pairs prefer qualified, state-grounded reasoning",
        ],
        "records": review_candidates,
    }
    PACKET.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "schemaVersion": "p2j4-decision-dpo-v3-audit-v1",
        "status": "REVIEW_REQUIRED",
        "candidateCount": len(payload["records"]),
        "eligibleCount": len(eligible),
        "rejectedCount": len(errors),
        "selectedStagingCount": len(selected),
        "trainCount": len(train),
        "validationCount": len(validation),
        "dimensionCounts": dict(Counter(item["metadata"]["preference_dimension"] for item in selected)),
        "trainValidationFamilyOverlap": sorted(train_families.intersection(validation_families)),
        "heldoutSignatureOverlap": len(selected_signatures.intersection(heldout)),
        "uniqueSelectedStateSignatures": len(selected_signatures),
        "runtimeFailurePatternDerivedCount": sum(item["metadata"]["source_kind"] == "runtime_failure_pattern_derived" for item in selected),
        "manualReviewPacket": str(PACKET.relative_to(ROOT)).replace("\\", "/"),
        "errors": errors,
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = audit_and_stage()
    print(json.dumps({key: result[key] for key in (
        "status", "candidateCount", "eligibleCount", "rejectedCount", "selectedStagingCount",
        "trainCount", "validationCount", "dimensionCounts", "heldoutSignatureOverlap")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
