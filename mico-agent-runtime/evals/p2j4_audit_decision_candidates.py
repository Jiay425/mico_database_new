"""Audit Decision State -> Action candidates before an SFT freeze.

The audit is metadata-only. It validates the closed candidate contract, maps
each candidate back to an immutable source run/case, measures exact and
normalized template duplication, and checks that all Hard cases contribute at
least one candidate. It never starts training and never emits question text,
SQL, arguments, payloads, or document content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evals.p2j4_decision_dataset import AgentDecisionCandidate


_FORBIDDEN_KEYS = {
    "question", "sql", "arguments", "locator", "payload", "rawRow",
    "documentText", "document", "rows", "rawValue", "sourceSampleId",
    "internalRecordId",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DECISION_CANDIDATE_AUDIT_PAYLOAD_INVALID")
    return payload


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.lower())


def _hash_signature(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_keys(child))
        return result
    return set()


def _source_index(paths: list[tuple[str, Path]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    by_trace: dict[str, dict[str, str]] = {}
    by_case: dict[str, dict[str, Any]] = {}
    for label, path in paths:
        payload = _read(path)
        for score in payload.get("scores", []):
            trace_id = score["traceId"]
            case_id = score["caseId"]
            by_trace[trace_id] = {
                "sourceRun": label,
                "caseId": case_id,
            }
            by_case[case_id] = {
                "sourceRun": label,
                "traceId": trace_id,
            }
    return by_trace, by_case


def audit(
    candidate_path: Path,
    source_paths: list[tuple[str, Path]],
    hard_spec_paths: list[Path],
) -> dict[str, Any]:
    payload = _read(candidate_path)
    raw_candidates = payload.get("candidates", [])
    candidates: list[AgentDecisionCandidate] = []
    invalid_candidates: list[dict[str, Any]] = []
    for index, item in enumerate(raw_candidates):
        try:
            candidates.append(AgentDecisionCandidate.model_validate(item))
        except Exception as exc:  # pragma: no cover - report must include all bad rows
            invalid_candidates.append({"index": index, "error": str(exc)[:200]})

    by_trace, _ = _source_index(source_paths)
    unresolved_source_traces = sorted({
        candidate.sourceTraceId
        for candidate in candidates
        if candidate.sourceTraceId not in by_trace
    })

    full_groups: dict[Any, list[AgentDecisionCandidate]] = defaultdict(list)
    normalized_groups: dict[Any, list[AgentDecisionCandidate]] = defaultdict(list)
    action_counts = Counter(candidate.selected_action for candidate in candidates)
    source_counts = Counter(
        by_trace[candidate.sourceTraceId]["sourceRun"]
        for candidate in candidates
        if candidate.sourceTraceId in by_trace
    )
    contract_errors: Counter[str] = Counter()
    for candidate in candidates:
        full_key = (
            candidate.state_summary,
            candidate.decision_reason,
            candidate.selected_action,
            tuple(candidate.alternative_actions),
            candidate.stop_reason,
            tuple(candidate.allowedActions),
        )
        normalized_key = (
            _normalize(candidate.state_summary),
            _normalize(candidate.decision_reason),
            candidate.selected_action,
        )
        full_groups[full_key].append(candidate)
        normalized_groups[normalized_key].append(candidate)
        if candidate.selected_action not in candidate.allowedActions:
            contract_errors["selected_action_not_allowed"] += 1
        if candidate.chosenAction != candidate.selected_action:
            contract_errors["chosen_selected_mismatch"] += 1
        if candidate.selected_action == "finish" and candidate.stop_reason is None:
            contract_errors["finish_stop_reason_missing"] += 1
        if candidate.selected_action != "finish" and candidate.stop_reason is not None:
            contract_errors["non_terminal_stop_reason_present"] += 1
        if candidate.selected_action in candidate.alternative_actions:
            contract_errors["alternative_contains_selected"] += 1
        if any(action not in candidate.allowedActions for action in candidate.alternative_actions):
            contract_errors["alternative_not_allowed"] += 1

    hard_classes: dict[str, str] = {}
    for path in hard_spec_paths:
        for item in _read(path).get("cases", []):
            hard_classes[item["caseId"]] = item["hardCaseClass"]
    hard_candidate_counts: Counter[str] = Counter()
    hard_case_candidate_counts: Counter[str] = Counter()
    hard_cases_with_candidates: set[str] = set()
    for candidate in candidates:
        source = by_trace.get(candidate.sourceTraceId)
        if source is None or source["caseId"] not in hard_classes:
            continue
        case_id = source["caseId"]
        hard_cases_with_candidates.add(case_id)
        hard_candidate_counts[hard_classes[case_id]] += 1
        hard_case_candidate_counts[case_id] += 1

    duplicate_groups = [
        {
            "signatureHash": _hash_signature(key),
            "count": len(group),
            "selectedAction": group[0].selected_action,
            "sourceTraceCount": len({item.sourceTraceId for item in group}),
        }
        for key, group in full_groups.items()
        if len(group) > 1
    ]
    duplicate_groups.sort(key=lambda item: (-item["count"], item["signatureHash"]))
    normalized_duplicate_groups = [
        {
            "signatureHash": _hash_signature(key),
            "count": len(group),
            "selectedAction": group[0].selected_action,
            "sourceTraceCount": len({item.sourceTraceId for item in group}),
        }
        for key, group in normalized_groups.items()
        if len(group) > 1
    ]
    normalized_duplicate_groups.sort(
        key=lambda item: (-item["count"], item["signatureHash"])
    )

    exact_duplicate_rows = sum(item["count"] - 1 for item in duplicate_groups)
    normalized_duplicate_rows = sum(item["count"] - 1 for item in normalized_duplicate_groups)
    source_trace_count = len({candidate.sourceTraceId for candidate in candidates})
    hard_case_total = len(hard_classes)
    raw_keys = _keys(payload)
    forbidden_keys_present = sorted(raw_keys & _FORBIDDEN_KEYS)
    contract_pass = not invalid_candidates and not contract_errors
    provenance_pass = not unresolved_source_traces
    hard_coverage_pass = len(hard_cases_with_candidates) == hard_case_total
    duplication_review_required = exact_duplicate_rows > 0 or normalized_duplicate_rows > 0
    status = (
        "REVIEW_REQUIRED_DUPLICATE_REDUCTION"
        if contract_pass and provenance_pass and hard_coverage_pass and duplication_review_required
        else "PASS"
        if contract_pass and provenance_pass and hard_coverage_pass
        else "FAIL"
    )
    return {
        "schemaVersion": "p2j4-decision-candidate-audit-v1",
        "sourceCandidateArtifact": candidate_path.name,
        "status": status,
        "trainingStarted": bool(payload.get("trainingStarted", False)),
        "candidateCount": len(candidates),
        "declaredCandidateCount": payload.get("candidateCount"),
        "sourceTraceCount": source_trace_count,
        "sourceRunCounts": dict(sorted(source_counts.items())),
        "candidateFieldRepairs": payload.get("candidateFieldRepairs", {}),
        "qualityGate": {
            "closedCandidateSchema": contract_pass,
            "sourceTraceProvenance": provenance_pass,
            "hardCaseCoverage": hard_coverage_pass,
            "rawFieldLeakage": not forbidden_keys_present,
            "duplicationReductionRequired": duplication_review_required,
        },
        "contractErrors": dict(contract_errors),
        "invalidCandidateRows": invalid_candidates,
        "forbiddenKeysPresent": forbidden_keys_present,
        "unresolvedSourceTraceCount": len(unresolved_source_traces),
        "unresolvedSourceTraceIds": unresolved_source_traces,
        "actionCounts": dict(action_counts.most_common()),
        "exactSignature": {
            "uniqueCount": len(full_groups),
            "duplicateGroupCount": len(duplicate_groups),
            "duplicateRowCount": exact_duplicate_rows,
            "maxGroupSize": max((item["count"] for item in duplicate_groups), default=1),
            "topGroups": duplicate_groups[:20],
        },
        "normalizedTemplateSignature": {
            "uniqueCount": len(normalized_groups),
            "duplicateGroupCount": len(normalized_duplicate_groups),
            "duplicateRowCount": normalized_duplicate_rows,
            "maxGroupSize": max(
                (item["count"] for item in normalized_duplicate_groups), default=1
            ),
            "topGroups": normalized_duplicate_groups[:20],
        },
        "hardCoverage": {
            "caseCount": hard_case_total,
            "coveredCaseCount": len(hard_cases_with_candidates),
            "missingCaseIds": sorted(set(hard_classes) - hard_cases_with_candidates),
            "candidateCountsByClass": dict(sorted(hard_candidate_counts.items())),
            "candidateCountsByCase": dict(sorted(hard_case_candidate_counts.items())),
        },
        "recommendation": (
            "Do not freeze the raw candidate pool yet. Keep all 834 rows as audit evidence, "
            "then create a deduplicated/weighted review set that preserves every Hard case "
            "and explicitly reviews repeated generic State -> Action templates."
            if status == "REVIEW_REQUIRED_DUPLICATE_REDUCTION"
            else "Candidate quality gate passed; freeze only after owner acceptance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit P2-J4 Decision candidates")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--hard-spec", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(
            args.candidates,
            [(label, Path(path)) for label, path in args.source],
            args.hard_spec,
        )
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": result["status"],
        "candidateCount": result["candidateCount"],
        "exactUniqueCount": result["exactSignature"]["uniqueCount"],
        "exactDuplicateRows": result["exactSignature"]["duplicateRowCount"],
        "hardCoveredCases": result["hardCoverage"]["coveredCaseCount"],
        "hardCaseCount": result["hardCoverage"]["caseCount"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
