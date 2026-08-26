"""Audit Decision SFT reasons without calling an external judge.

The audit is deliberately deterministic and conservative.  It does not claim
that a reason is scientifically true; it checks whether the reason is
non-empty, action-relevant, tethered to the structured state, and free of an
unsupported causal/biomarker claim.  Cases that cannot be decided by these
rules are marked REVIEW for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ACTION_TERMS: dict[str, tuple[str, ...]] = {
    "inspect_cohort": ("inspect", "metadata", "cohort", "sample", "population", "first"),
    "compare_groups": ("compare", "group", "difference", "case", "control"),
    "cross_project_validate": (
        "project", "cross", "stable", "stability", "repeat", "reproduc", "queue"
    ),
    "adjust_confounders": (
        "confound", "age", "sex", "adjust", "control", "match", "covariate", "imbalance"
    ),
    "stratified_analysis": ("strat", "subgroup", "sex", "age", "country", "region"),
    "retrieve_evidence": ("evidence", "knowledge", "literature", "graph", "retriev", "source"),
    "analyze_projection": ("projection", "analy", "abundance", "summar", "result", "feature"),
    "cross_disease_validate": ("disease", "specific", "other disease", "cross-disease"),
    "execute_read_query": ("query", "read", "bounded", "data", "inspect"),
    "finish": ("sufficient", "validated", "complete", "finish", "final", "stop", "conclusion", "conclud"),
}

CAUSAL_PATTERNS = (
    r"\bcause(?:s|d)?\b",
    r"\bcausal(?:ly)?\b",
    r"\bbiomarker\b",
    r"\bleads? to\b",
    r"\bprove[sd]?\b",
    r"导致",
    r"因果",
    r"生物标志物",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EXPECTED_OBJECT:{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _reason_terms(action: str, reason: str) -> list[str]:
    lowered = reason.lower()
    return [term for term in ACTION_TERMS.get(action, ()) if term in lowered]


def _has_unsupported_causal_claim(reason: str, state: dict[str, Any]) -> bool:
    # A reason is only a policy rationale, not a result statement.  Causal or
    # biomarker claims are therefore unsafe here, including when evidence is
    # present; the final scientific claim belongs to the grounded report.
    del state
    return any(re.search(pattern, reason, flags=re.IGNORECASE) for pattern in CAUSAL_PATTERNS)


def _state_requirements(action: str, state: dict[str, Any]) -> list[str]:
    flags = set(state.get("observation_flags") or [])
    required: list[str] = []
    if action == "adjust_confounders" and flags.intersection(
        {"CONFOUNDER_PRESENT", "PROJECT_IMBALANCE", "AGE_IMBALANCE", "SEX_IMBALANCE"}
    ):
        required.append("confounder_context")
    if action == "cross_project_validate" and flags.intersection(
        {"CROSS_PROJECT_REQUIRED", "PROJECT_IMBALANCE", "PROJECT_VALIDATED"}
    ):
        required.append("project_context")
    if action == "retrieve_evidence" and flags.intersection(
        {"EVIDENCE_INCOMPLETE", "EVIDENCE_CONFLICT", "VECTOR_EVIDENCE", "GRAPH_EVIDENCE"}
    ):
        required.append("evidence_context")
    if action == "finish" and "EVIDENCE_CONFLICT" in flags:
        required.append("conflict_disclosure")
    return required


def _requirement_satisfied(requirement: str, reason: str) -> bool:
    lowered = reason.lower()
    terms = {
        "confounder_context": ("confound", "age", "sex", "adjust", "control", "match", "imbalance"),
        "project_context": ("project", "cross", "stable", "stability", "repeat", "reproduc"),
        "evidence_context": ("evidence", "knowledge", "literature", "graph", "retriev", "source"),
        "conflict_disclosure": ("conflict", "uncertain", "inconsistent", "quality", "caution", "downgrad"),
    }
    return any(term in lowered for term in terms[requirement])


def _audit_case(record: dict[str, Any], scored: dict[str, Any]) -> dict[str, Any]:
    state = json.loads(record["messages"][1]["content"])["policy_state"]
    prediction = scored.get("prediction") or {}
    action = prediction.get("selected_action")
    reason = _text(prediction.get("decision_reason"))
    checks = {
        "json_valid": isinstance(prediction, dict) and bool(prediction),
        "reason_nonempty": bool(reason.strip()),
        "action_allowed": action in state.get("candidate_actions", []),
        "reason_action_relevant": bool(_reason_terms(action, reason)),
        "reason_state_tethered": all(
            _requirement_satisfied(req, reason)
            for req in _state_requirements(action, state)
        ),
        "no_unsupported_causal_claim": not _has_unsupported_causal_claim(reason, state),
    }
    review_reasons: list[str] = []
    if not checks["json_valid"]:
        review_reasons.append("INVALID_JSON")
    if not checks["reason_nonempty"]:
        review_reasons.append("EMPTY_REASON")
    if not checks["action_allowed"]:
        review_reasons.append("ACTION_NOT_ALLOWED")
    if not checks["reason_action_relevant"]:
        review_reasons.append("GENERIC_OR_ACTION_MISMATCH")
    if not checks["reason_state_tethered"]:
        review_reasons.append("STATE_CONTEXT_MISSING")
    if not checks["no_unsupported_causal_claim"]:
        review_reasons.append("UNSUPPORTED_CAUSAL_LANGUAGE")
    status = "PASS" if not review_reasons else "REVIEW"
    return {
        "record_id": scored["record_id"],
        "status": status,
        "checks": checks,
        "review_reasons": review_reasons,
        "state": state,
        "prediction": prediction,
        "gold_action": scored.get("gold_action"),
        "action_correct": scored.get("action_correct"),
    }


def audit(prediction_file: Path, input_file: Path, output_file: Path) -> dict[str, Any]:
    prediction = _read_json(prediction_file)
    records = _read_jsonl(input_file)
    scored = prediction.get("sft", {}).get("cases")
    if not isinstance(scored, list):
        raise ValueError("SFT_CASES_MISSING")
    by_id = {record["id"]: record for record in records}
    if set(by_id) != {case.get("record_id") for case in scored}:
        raise ValueError("SFT_INPUT_PREDICTION_ID_MISMATCH")
    cases = [_audit_case(by_id[case["record_id"]], case) for case in scored]
    pass_count = sum(case["status"] == "PASS" for case in cases)
    result = {
        "schemaVersion": "p2j4-decision-reason-audit-v1",
        "predictionFile": prediction_file.name,
        "inputFile": input_file.name,
        "caseCount": len(cases),
        "passCount": pass_count,
        "reviewCount": len(cases) - pass_count,
        "passRate": pass_count / len(cases) if cases else 0.0,
        "policyActionAccuracy": prediction.get("sft", {}).get("actionAccuracy"),
        "cases": cases,
    }
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Decision policy reasons")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(args.predictions, args.input, args.output)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "PASS" if result["reviewCount"] == 0 else "REVIEW_REQUIRED",
        "caseCount": result["caseCount"],
        "passCount": result["passCount"],
        "reviewCount": result["reviewCount"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
