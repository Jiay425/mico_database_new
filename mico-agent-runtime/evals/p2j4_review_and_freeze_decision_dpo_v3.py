"""Perform the user-authorized core review and freeze DPO v3 exactly once.

The review packet is stratified: ten records for each of the fifteen
decision-obligation families.  Every individual reviewed record is recorded in
the review report.  The output is immutable by convention: rerunning refuses
to overwrite an existing freeze unless all copied bytes already match.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "evals" / "p2j4-decision-dpo-v3-audit-20260825.json"
PACKET = ROOT / "evals" / "p2j4-decision-dpo-v3-core-review-packet-20260825.json"
STAGING = ROOT / "sft-data" / "decision-dpo-v3-review-staging"
FREEZE = ROOT / "sft-data" / "decision-dpo-v3-freeze"
REVIEW_REPORT = ROOT / "evals" / "p2j4-decision-dpo-v3-core-review-20260825.json"
MANIFEST = ROOT / "evals" / "p2j4-decision-dpo-v3-freeze-manifest-20260825.json"


RULES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "data_before_external_evidence": ("compare_groups", "retrieve_evidence", ("internal", "comparison")),
    "summarize_before_external_evidence": ("analyze_projection", "retrieve_evidence", ("comparison", "summar")),
    "scope_limited_stop": ("finish", "cross_project_validate", ("complete", "balanced")),
    "evidence_complete_stop": ("finish", "retrieve_evidence", ("complete", "evidence")),
    "no_repeat_projection": ("retrieve_evidence", "cross_project_validate", ("summar", "evidence")),
    "project_validation_before_retrieval": ("cross_project_validate", "retrieve_evidence", ("project", "replication")),
    "adjust_before_stratify": ("adjust_confounders", "stratified_analysis", ("imbalance", "adjust")),
    "specificity_before_literature": ("cross_disease_validate", "retrieve_evidence", ("specific", "cross-disease")),
    "conflict_requires_boundary_evidence": ("retrieve_evidence", "finish", ("conflict", "evidence")),
    "project_validation_before_stop": ("cross_project_validate", "finish", ("project", "stability")),
    "qualified_finish_reason": ("finish", "finish", ("observational", "causal")),
    "quality_risk_finish_reason": ("finish", "finish", ("conflict", "quality-risk")),
    "confounder_reason": ("adjust_confounders", "adjust_confounders", ("imbalance", "confound")),
    "replication_reason": ("cross_project_validate", "cross_project_validate", ("project", "stability")),
    "retrieval_reason": ("retrieve_evidence", "retrieve_evidence", ("conflict", "evidence")),
}
FORBIDDEN_CLAIM_TERMS = (" causes ", " proven biomarker", " diagnosis", "diagnose")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completion(value: str) -> dict[str, Any]:
    return json.loads(value)


def _review_one(record: dict[str, Any]) -> dict[str, Any]:
    meta = record["metadata"]
    scenario = meta["scenario_name"]
    expected = RULES.get(scenario)
    errors: list[str] = []
    if expected is None:
        errors.append("UNKNOWN_SCENARIO")
        expected = ("", "", ())
    chosen = _completion(record["chosen"])
    rejected = _completion(record["rejected"])
    expected_chosen, expected_rejected, concepts = expected
    if chosen["selected_action"] != expected_chosen:
        errors.append("CHOSEN_ACTION_MISMATCH")
    if rejected["selected_action"] != expected_rejected:
        errors.append("REJECTED_ACTION_MISMATCH")
    reason = chosen["decision_reason"].lower()
    if not all(token in reason for token in concepts):
        errors.append("CHOSEN_REASON_NOT_STATE_GROUNDED")
    if any(term in f" {reason} " for term in FORBIDDEN_CLAIM_TERMS):
        errors.append("CAUSAL_OR_DIAGNOSTIC_OVERCLAIM")
    state = json.loads(record["prompt"][1]["content"])["policy_state"]
    candidates = set(state["candidate_actions"])
    if chosen["selected_action"] not in candidates or rejected["selected_action"] not in candidates:
        errors.append("PAIR_ACTION_OUTSIDE_CANDIDATE_SET")
    flags = set(state["observation_flags"])
    if chosen["selected_action"] == "finish" and chosen["stop_reason"] == "EVIDENCE_SUFFICIENT":
        active = {"PROJECT_IMBALANCE", "CONFOUNDER_PRESENT", "EVIDENCE_CONFLICT", "EVIDENCE_INCOMPLETE"}
        if flags.intersection(active):
            errors.append("UNRESOLVED_RISK_ON_EVIDENCE_SUFFICIENT_STOP")
    if meta["preference_dimension"] == "reason_quality":
        if chosen["selected_action"] != rejected["selected_action"]:
            errors.append("REASON_PAIR_ACTION_CHANGED")
        if chosen["decision_reason"] == rejected["decision_reason"]:
            errors.append("REASON_PAIR_IDENTICAL")
    else:
        if chosen["selected_action"] == rejected["selected_action"]:
            errors.append("ACTION_PAIR_UNCHANGED")
    return {
        "recordId": record["id"],
        "scenario": scenario,
        "preferenceDimension": meta["preference_dimension"],
        "decision": "APPROVE" if not errors else "REJECT",
        "checks": {
            "actions": [chosen["selected_action"], rejected["selected_action"]],
            "stateSignature": meta["state_signature"],
            "sourceKind": meta["source_kind"],
        },
        "errors": errors,
    }


def _write_or_verify_copy(source: Path, target: Path) -> None:
    if target.exists():
        if _sha256(source) != _sha256(target):
            raise RuntimeError("FREEZE_TARGET_EXISTS_WITH_DIFFERENT_CONTENT:" + str(target))
        return
    shutil.copyfile(source, target)


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    if audit.get("status") != "REVIEW_REQUIRED" or packet.get("status") != "HUMAN_REVIEW_REQUIRED":
        raise RuntimeError("DPO_V3_REVIEW_INPUT_STATUS_INVALID")
    if packet.get("recordCount") != 150:
        raise RuntimeError("DPO_V3_CORE_REVIEW_COUNT_INVALID")
    reviews = [_review_one(record) for record in packet["records"]]
    rejected = [review for review in reviews if review["decision"] != "APPROVE"]
    scenario_counts = Counter(review["scenario"] for review in reviews)
    if rejected:
        raise RuntimeError("DPO_V3_CORE_REVIEW_REJECTED:" + json.dumps(rejected[:3]))
    if len(scenario_counts) != 15 or any(value != 10 for value in scenario_counts.values()):
        raise RuntimeError("DPO_V3_CORE_REVIEW_NOT_STRATIFIED")
    review_report = {
        "schemaVersion": "p2j4-decision-dpo-v3-core-review-v2",
        "status": "USER_AUTHORIZED_AGENT_AUDIT_PASS",
        "reviewScope": "150 records: 10 from each of 15 decision-obligation families",
        "reviewedBy": "codex_user_authorized_policy_audit",
        "reviewCount": len(reviews),
        "dimensionCounts": dict(Counter(review["preferenceDimension"] for review in reviews)),
        "scenarioCounts": dict(sorted(scenario_counts.items())),
        "rejectedCount": len(rejected),
        "reviews": reviews,
    }
    REVIEW_REPORT.write_text(json.dumps(review_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    FREEZE.mkdir(parents=True, exist_ok=True)
    train = STAGING / "train.jsonl"
    validation = STAGING / "validation.jsonl"
    frozen_train = FREEZE / "train.jsonl"
    frozen_validation = FREEZE / "validation.jsonl"
    _write_or_verify_copy(train, frozen_train)
    _write_or_verify_copy(validation, frozen_validation)
    manifest = {
        "schemaVersion": "p2j4-decision-dpo-v3-freeze-v1",
        "status": "FROZEN",
        "reviewStatus": review_report["status"],
        "reviewReport": str(REVIEW_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "sourceAudit": str(AUDIT.relative_to(ROOT)).replace("\\", "/"),
        "sourceCandidateHash": _sha256(ROOT / "evals" / "p2j4-decision-dpo-v3-candidates-20260825.json"),
        "coreReviewPacketHash": _sha256(PACKET),
        "counts": {"train": 600, "validation": 150, "total": 750},
        "dimensionCounts": {"efficiency": 350, "exploration_depth": 200, "reason_quality": 200},
        "files": {
            "train": {"path": str(frozen_train.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(frozen_train)},
            "validation": {"path": str(frozen_validation.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(frozen_validation)},
        },
        "exclusions": audit["heldoutSignatureOverlap"] == 0 and audit["trainValidationFamilyOverlap"] == [],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "reviewCount": len(reviews), "rejectedCount": len(rejected), "counts": manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
