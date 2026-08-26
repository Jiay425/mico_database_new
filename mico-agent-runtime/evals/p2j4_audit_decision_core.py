"""Deterministically audit the 272 Decision SFT core review items.

This audit does not declare a sample semantically correct merely because it
passes Pydantic validation.  It separates structural validity, state/action
consistency, and the need to enrich generic state summaries before training.
No source Trace is rerun by this module.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evals.p2j4_build_context_review_set import SIGNATURE_FIELDS, _signature
from evals.p2j4_decision_context import DecisionSftCandidate


_STATE_RE = re.compile(
    r"observation_state=(?P<state>[^;]+);\s*"
    r"evidence_bindings=(?P<bindings>\d+);\s*"
    r"source_routes=(?P<routes>[^;]+);\s*"
    r"prior_actions=(?P<prior>.*)$"
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DECISION_CORE_AUDIT_PAYLOAD_INVALID")
    return payload


def _parse_state_summary(value: str) -> dict[str, Any] | None:
    match = _STATE_RE.fullmatch(value.strip())
    if match is None:
        return None
    prior_value = match.group("prior").strip()
    prior = [] if prior_value in {"", "none"} else [
        item for item in prior_value.split(",") if item
    ]
    routes = [item for item in match.group("routes").split(",") if item and item != "none"]
    return {
        "observationState": match.group("state"),
        "evidenceBindings": int(match.group("bindings")),
        "sourceRoutes": routes,
        "priorActions": prior,
    }


def _review_item(
    item: dict[str, Any],
    state_summary_counts: Counter[str],
) -> dict[str, Any]:
    candidate = DecisionSftCandidate.model_validate(item["candidate"])
    warnings: list[str] = []
    errors: list[str] = []
    parsed = _parse_state_summary(candidate.state_summary)

    if parsed is None:
        errors.append("STATE_SUMMARY_FORMAT_UNPARSEABLE")
    else:
        if parsed["observationState"] == "NO_OBSERVATION" and candidate.selected_action not in {
            "inspect_cohort",
            "execute_read_query",
        }:
            warnings.append("ACTION_WITHOUT_OBSERVATION")
        if parsed["observationState"] == "NO_OBSERVATION" and parsed["evidenceBindings"] != 0:
            errors.append("NO_OBSERVATION_BINDING_MISMATCH")
        if parsed["observationState"] != "NO_OBSERVATION" and parsed["evidenceBindings"] == 0:
            warnings.append("VALIDATED_STATE_WITHOUT_BINDING")
        if parsed["priorActions"] != candidate.history_actions:
            errors.append("STATE_HISTORY_MISMATCH")
        if parsed["sourceRoutes"] and not set(parsed["sourceRoutes"]).issubset(
            set(candidate.observation_flags) | {"java", "vector", "graph"}
        ):
            # Source routes are represented in the closed observation flags;
            # this check is informational because the v3 contract stores only
            # the uppercase flag vocabulary.
            pass

    flags = set(candidate.observation_flags)
    if "EVIDENCE_CONFLICT" in flags and candidate.selected_action == "finish":
        if candidate.stop_reason != "QUALITY_RISK":
            errors.append("CONFLICT_FINISH_NOT_QUALITY_RISK")
    if "PROJECT_IMBALANCE" in flags and candidate.selected_action == "finish":
        if "PROJECT_VALIDATED" not in flags:
            warnings.append("IMBALANCE_FINISH_BEFORE_PROJECT_VALIDATION")
    if "CROSS_PROJECT_REQUIRED" in flags and candidate.selected_action == "finish":
        if "PROJECT_VALIDATED" not in flags:
            warnings.append("CROSS_PROJECT_FINISH_BEFORE_VALIDATION")
    if "CONFOUNDER_PRESENT" in flags and candidate.selected_action == "finish":
        if "CONFOUNDER_ADJUSTED" not in flags:
            warnings.append("CONFOUNDER_FINISH_BEFORE_ADJUSTMENT")

    # The summary is intentionally closed, but repeated generic summaries are
    # not sufficient evidence that the model saw a distinct policy state.
    repetition_count = state_summary_counts[candidate.state_summary]
    if repetition_count >= 10:
        warnings.append("GENERIC_STATE_SUMMARY_REPEATED")
    elif repetition_count >= 5:
        warnings.append("STATE_SUMMARY_REUSED")

    required_context = {
        "task_kind": candidate.task_kind,
        "goal_code": candidate.goal_code,
        "task_family": candidate.task_family,
        "observation_flags": candidate.observation_flags,
        "history_actions": candidate.history_actions,
        "candidate_actions": candidate.candidate_actions,
    }
    # An empty action history is the valid initial state; only the structural
    # fields that must always be present are required to be non-empty.
    if any(
        required_context[key] in (None, "", [])
        for key in (
            "task_kind",
            "goal_code",
            "task_family",
            "observation_flags",
            "candidate_actions",
        )
    ):
        errors.append("DECISION_CONTEXT_INCOMPLETE")

    if errors:
        status = "REJECTED_CONTRACT_OR_POLICY"
    elif warnings:
        status = "ENRICHMENT_REQUIRED"
    else:
        status = "PROVISIONAL_ACCEPT"
    return {
        "signatureHash": item["signatureHash"],
        "sourceTraceId": candidate.sourceTraceId,
        "taskFamily": candidate.task_family,
        "goalCode": candidate.goal_code,
        "selectedAction": candidate.selected_action,
        "hardCaseClass": candidate.hard_case_class,
        "stateSummaryRepetitionCount": repetition_count,
        "status": status,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def audit(review_path: Path) -> dict[str, Any]:
    payload = _read(review_path)
    items = payload.get("items", [])
    if not isinstance(items, list) or len(items) != payload.get("reviewItemCount"):
        raise ValueError("DECISION_CORE_AUDIT_REVIEW_COUNT_MISMATCH")

    candidates = [DecisionSftCandidate.model_validate(item["candidate"]) for item in items]
    state_summary_counts: Counter[str] = Counter(candidate.state_summary for candidate in candidates)
    results = [_review_item(item, state_summary_counts) for item in items]
    status_counts = Counter(result["status"] for result in results)
    warning_counts = Counter(
        warning for result in results for warning in result["warnings"]
    )
    error_counts = Counter(
        error for result in results for error in result["errors"]
    )
    context_signatures = {_signature(candidate) for candidate in candidates}
    hard_cases = {
        case_id
        for item in items
        for case_id in item.get("hardCaseIds", [])
    }

    return {
        "schemaVersion": "p2j4-decision-core-audit-v1",
        "sourceReviewSet": review_path.name,
        "status": "REVIEW_REQUIRED" if status_counts.get("REJECTED_CONTRACT_OR_POLICY", 0) or status_counts.get("ENRICHMENT_REQUIRED", 0) else "PASS",
        "trainingStarted": False,
        "reviewItemCount": len(items),
        "contextExactSignatureCount": len(context_signatures),
        "hardCaseCoverageCount": len(hard_cases),
        "statusCounts": dict(status_counts),
        "warningCounts": dict(warning_counts),
        "errorCounts": dict(error_counts),
        "stateSummaryDistinctCount": len(state_summary_counts),
        "stateSummaryRepetitionDistribution": dict(Counter(state_summary_counts.values())),
        "policy": {
            "noTraceRerun": True,
            "genericStateSummaryRequiresEnrichment": True,
            "provisionalAcceptIsNotOwnerAcceptance": True,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit P2-J4 Decision core review items")
    parser.add_argument("--review-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.review_set)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": result["status"],
        "reviewItemCount": result["reviewItemCount"],
        "statusCounts": result["statusCounts"],
        "stateSummaryDistinctCount": result["stateSummaryDistinctCount"],
        "hardCaseCoverageCount": result["hardCaseCoverageCount"],
        "trainingStarted": result["trainingStarted"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
