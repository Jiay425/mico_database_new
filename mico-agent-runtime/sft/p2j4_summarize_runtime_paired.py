"""Summarize a completed Base/SFT Runtime paired run without raw evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "evals" / "p2j4-runtime-paired-v1"
TASK_SET = RUN_DIR / "task-set.json"
OUTPUT = RUN_DIR / "summary.json"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"PAYLOAD_INVALID:{path.name}")
    return payload


def _chosen_actions(trace: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for decision in trace.get("decisions", []):
        if isinstance(decision, dict) and isinstance(decision.get("chosenAction"), str):
            actions.append(decision["chosenAction"])
    return actions


def _premature_finish(case: dict[str, Any], trace: dict[str, Any]) -> bool:
    required = {item for item in case.get("requiredActions", []) if item != "finish"}
    actions = _chosen_actions(trace)
    if "finish" not in actions:
        return False
    finish_index = actions.index("finish")
    return not required.issubset(set(actions[:finish_index]))


def _reason_review_proxy(trace: dict[str, Any]) -> bool:
    generic = {
        "select the next allow-listed runtime action",
        "select the next allowed action",
        "continue the research task",
    }
    decisions = trace.get("decisions", [])
    if not decisions:
        return True
    for decision in decisions:
        if not isinstance(decision, dict):
            return True
        reason = decision.get("decision_reason") or decision.get("decisionReason")
        if not isinstance(reason, str) or len(reason.strip()) < 12:
            return True
        if reason.strip().lower() in generic:
            return True
    return False


def _mode_summary(mode: str, payload: dict[str, Any], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    traces = {item["traceId"]: item for item in payload.get("traces", []) if isinstance(item, dict)}
    scores = [item for item in payload.get("scores", []) if isinstance(item, dict)]
    by_case: dict[str, dict[str, Any]] = {}
    premature_count = 0
    duplicate_count = 0
    invalid_count = 0
    fallback_count = 0
    evidence_count = 0
    reason_review_count = 0
    action_counts: list[int] = []
    for score in scores:
        case_id = score.get("caseId")
        trace = traces.get(score.get("traceId"), {})
        case = cases.get(case_id, {})
        criteria = score.get("criteria") or {}
        actions = _chosen_actions(trace)
        premature = _premature_finish(case, trace)
        duplicate = bool(trace.get("duplicateActionCount", 0))
        invalid = bool(
            trace.get("safetyViolationCodes")
            or any(code in {"TRACE_MODEL_OUTPUT_REJECTED", "TRACE_FORBIDDEN_ACTION", "DECISION_ACCURACY_FAILURE"}
                   for code in score.get("failureCodes", []))
        )
        fallback = bool(trace.get("fallbackCodes"))
        evidence = bool(criteria.get("evidence_grounding"))
        reason_review = _reason_review_proxy(trace)
        premature_count += int(premature)
        duplicate_count += int(duplicate)
        invalid_count += int(invalid)
        fallback_count += int(fallback)
        evidence_count += int(evidence)
        reason_review_count += int(reason_review)
        if isinstance(trace.get("actionCount"), int):
            action_counts.append(trace["actionCount"])
        by_case[case_id] = {
            "status": trace.get("status"),
            "fullOraclePass": score.get("status") == "PASS",
            "taskSuccess": bool(criteria.get("task_success")),
            "prematureFinish": premature,
            "actionCount": trace.get("actionCount"),
            "chosenActions": actions,
            "duplicateAction": duplicate,
            "invalidOrRejectedAction": invalid,
            "fallback": fallback,
            "evidenceGroundingComplete": evidence,
            "reasonReviewProxy": reason_review,
            "failureCodes": score.get("failureCodes", []),
            "stopReasonCode": trace.get("stopReasonCode"),
            "fallbackCodes": trace.get("fallbackCodes", []),
        }
    total = len(scores)
    div = total or 1
    return {
        "mode": mode,
        "status": payload.get("status"),
        "runCount": total,
        "taskCompletionRate": sum(item["taskSuccess"] for item in by_case.values()) / div,
        "fullOraclePassRate": sum(item["fullOraclePass"] for item in by_case.values()) / div,
        "prematureFinishRate": premature_count / div,
        "meanActionCount": sum(action_counts) / len(action_counts) if action_counts else 0.0,
        "duplicateActionRate": duplicate_count / div,
        "invalidOrRejectedActionRate": invalid_count / div,
        "duplicateOrInvalidActionRate": sum(
            item["duplicateAction"] or item["invalidOrRejectedAction"] for item in by_case.values()
        ) / div,
        "fallbackRate": fallback_count / div,
        "evidenceBindingCompletionRate": evidence_count / div,
        "reasonReviewProxyRate": reason_review_count / div,
        "reasonReviewDefinition": "automated proxy only; semantic reason review remains manual",
        "failureCodeCounts": _failure_counts(scores),
        "cases": by_case,
    }


def _failure_counts(scores: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for score in scores:
        for code in score.get("failureCodes", []):
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    task_payload = _load(TASK_SET)
    cases = {item["caseId"]: item for item in task_payload.get("cases", [])}
    base_path = RUN_DIR / "base.json"
    sft_path = RUN_DIR / "sft.json"
    if not base_path.exists() or not sft_path.exists():
        raise SystemExit("PAIRED_RESULTS_INCOMPLETE")
    base = _mode_summary("base", _load(base_path), cases)
    sft = _mode_summary("sft_policy", _load(sft_path), cases)
    payload = {
        "schemaVersion": "p2j4-base-sft-runtime-paired-summary-v1",
        "evaluationVersion": task_payload.get("evaluationVersion"),
        "taskSetSha256": task_payload.get("sourceTaskSetSha256"),
        "caseCount": task_payload.get("caseCount"),
        "kindDistribution": task_payload.get("kindDistribution"),
        "base": base,
        "sft": sft,
        "delta": {
            "taskCompletionRate": sft["taskCompletionRate"] - base["taskCompletionRate"],
            "fullOraclePassRate": sft["fullOraclePassRate"] - base["fullOraclePassRate"],
            "prematureFinishRate": sft["prematureFinishRate"] - base["prematureFinishRate"],
            "meanActionCount": sft["meanActionCount"] - base["meanActionCount"],
            "duplicateOrInvalidActionRate": sft["duplicateOrInvalidActionRate"] - base["duplicateOrInvalidActionRate"],
            "fallbackRate": sft["fallbackRate"] - base["fallbackRate"],
            "evidenceBindingCompletionRate": sft["evidenceBindingCompletionRate"] - base["evidenceBindingCompletionRate"],
            "reasonReviewProxyRate": sft["reasonReviewProxyRate"] - base["reasonReviewProxyRate"],
        },
        "interpretationBoundary": [
            "This is a paired Runtime comparison, not a causal claim about the model in general.",
            "Both modes use the same task contracts, Java read model, Python analysis and knowledge backend.",
            "ReasonReviewProxyRate is not a semantic human judgment and must not be reported as reason quality.",
            "Any infrastructure failure is excluded from model metrics only after it is separately documented.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "caseCount": payload["caseCount"],
        "baseStatus": base["status"],
        "sftStatus": sft["status"],
        "delta": payload["delta"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
