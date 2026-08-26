"""Build an auditable Decision-Preference staging set from the paired Runtime run.

This is a local transformation only.  It uses the SFT decision record as the
chosen response and a Base decision record from the same de-identified state
as the rejected response, but only when the frozen Decision oracle says SFT
passed and Base failed.  It deliberately does not claim independent execution
action binding for the legacy traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.p2j4_runner import validate_task_set


ROOT = Path(__file__).resolve().parent
TASK_SET = ROOT / "p2j4-policy-sensitive-runtime-v1" / "task-set.json"
SFT = ROOT / "p2j4-policy-sensitive-runtime-v1" / "sft-20-policy-sensitive-aggregate.json"
BASE = ROOT / "p2j4-policy-sensitive-runtime-v1" / "base-20-policy-sensitive-aggregate.json"
SFT_AUDIT = ROOT / "p2j4-policy-sensitive-runtime-v1" / "sft-20-policy-sensitive-legacy-decision-audit.json"
BASE_AUDIT = ROOT / "p2j4-policy-sensitive-runtime-v1" / "base-20-policy-sensitive-legacy-decision-audit.json"
DEFAULT_OUTPUT = ROOT / "p2j4-policy-sensitive-runtime-v1" / "dpo-preference-candidates-v1.json"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"payload is not an object: {path}")
    return payload


def _family(case_id: str) -> str:
    for item in ("compare", "analyze", "retrieve", "combined"):
        if f"-{item}-" in case_id:
            return item
    return "unknown"


def _goal_code(case_id: str) -> str:
    return {
        "compare": "group_pattern_characterization",
        "analyze": "bounded_projection_analysis",
        "retrieve": "evidence_after_group_pattern",
        "combined": "evidence_after_projection",
    }.get(_family(case_id), "policy_sensitive_runtime")


def _state_context(decision: dict[str, Any], task_kind: str, case_id: str) -> dict[str, Any]:
    summary = decision["state_summary"]
    fields: dict[str, str] = {}
    for item in summary.split("; "):
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    history = [item for item in fields.get("prior_actions", "none").split(",") if item and item != "none"]
    flags = [decision["observationStateCode"]]
    if fields.get("source_routes", "none") != "none":
        flags.append("JAVA_OBSERVED" if "java" in fields["source_routes"] else "EVIDENCE_OBSERVED")
    if "retrieve_evidence" in history:
        flags.append("EVIDENCE_RETRIEVED")
    if "analyze_projection" in history:
        flags.append("ANALYSIS_COMPLETE")
    return {
        "task_kind": task_kind,
        "goal_code": _goal_code(case_id),
        "observation_flags": list(dict.fromkeys(flags)),
        "history_actions": history,
        "candidate_actions": list(decision["allowedActions"]),
        "state_summary": summary,
    }


def _decision_payload(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_action": decision["selected_action"],
        "decision_reason": decision["decision_reason"],
        "alternative_actions": list(decision["alternative_actions"]),
        "stop_reason": decision.get("stop_reason"),
    }


def build(output: Path) -> dict[str, Any]:
    validation = validate_task_set(TASK_SET)
    task_by_case = {task.caseId: task for task in validation["tasks"]}
    sft = _load(SFT)
    base = _load(BASE)
    sft_audit = _load(SFT_AUDIT)
    base_audit = _load(BASE_AUDIT)
    sft_scores = {item["caseId"]: item for item in sft["scores"]}
    base_scores = {item["caseId"]: item for item in base["scores"]}
    sft_traces = {item["traceId"]: item for item in sft["traces"]}
    base_traces = {item["traceId"]: item for item in base["traces"]}
    sft_cases = {item["caseId"]: item for item in sft_audit["audit"]["caseAudits"]}
    base_cases = {item["caseId"]: item for item in base_audit["audit"]["caseAudits"]}

    records: list[dict[str, Any]] = []
    eligible_case_ids: list[str] = []
    for case_id, sft_score in sft_scores.items():
        base_score = base_scores[case_id]
        if sft_score["status"] != "PASS" or base_score["status"] != "FAIL":
            continue
        eligible_case_ids.append(case_id)
        sft_trace = sft_traces[sft_score["traceId"]]
        base_trace = base_traces[base_score["traceId"]]
        sft_decisions = sft_trace.get("decisions", [])
        base_decisions = base_trace.get("decisions", [])
        base_by_signature = {
            (
                item.get("observationStateCode"),
                item.get("state_summary"),
                tuple(item.get("allowedActions", [])),
            ): item
            for item in base_decisions
        }
        task = task_by_case[case_id]
        for index, chosen in enumerate(sft_decisions):
            signature = (
                chosen.get("observationStateCode"),
                chosen.get("state_summary"),
                tuple(chosen.get("allowedActions", [])),
            )
            rejected = base_by_signature.get(signature)
            if rejected is None:
                continue
            if _decision_payload(chosen) == _decision_payload(rejected):
                continue
            context = _state_context(chosen, task.kind, case_id)
            records.append({
                "recordId": f"p2j4-dpo-runtime-v1-{len(records)+1:04d}",
                "caseId": case_id,
                "stateIndex": index,
                "state": context,
                "chosen": _decision_payload(chosen),
                "rejected": _decision_payload(rejected),
                "preferenceBasis": {
                    "chosenAgent": "sft-v4",
                    "rejectedAgent": "base-deepseek-v4-flash",
                    "chosenScoreStatus": sft_score["status"],
                    "rejectedScoreStatus": base_score["status"],
                    "rejectedFailureCodes": list(base_score["failureCodes"]),
                    "sameStateSignature": True,
                    "legacyExecutionBinding": "unavailable",
                },
            })

    repair_cases = [
        case_id for case_id, audit in sft_cases.items()
        if "SCIENTIFIC_PLANNER_PREMATURE_FINISH_REPAIRED" in (
            sft_traces[next(score["traceId"] for score in sft["scores"] if score["caseId"] == case_id)]["fallbackCodes"]
        )
    ]
    base_failure_cases = [
        score["caseId"] for score in _load(BASE)["scores"] if score["status"] == "FAIL"
    ]
    payload = {
        "schemaVersion": "p2j4-decision-preference-v1",
        "status": "STAGED_FOR_REVIEW",
        "trainingStarted": False,
        "sourceArtifacts": {
            "taskSet": str(TASK_SET),
            "sftAggregate": str(SFT),
            "baseAggregate": str(BASE),
            "sftAudit": str(SFT_AUDIT),
            "baseAudit": str(BASE_AUDIT),
        },
        "scope": {
            "sftRepairCaseCount": len(repair_cases),
            "sftRepairCaseIds": repair_cases,
            "baseFailureCaseCount": len(base_failure_cases),
            "baseFailureCaseIds": base_failure_cases,
            "pairedEligibleCaseCount": len(eligible_case_ids),
            "pairedEligibleCaseIds": eligible_case_ids,
        },
        "candidateCount": len(records),
        "audit": {
            "networkCalls": False,
            "allRecordsHaveSameStateSignature": all(
                item["preferenceBasis"]["sameStateSignature"] for item in records
            ),
            "decisionOnlyUntilActionBindingRerun": True,
            "reviewRequired": True,
            "rejectedLabelsAreRuntimeBaseFailures": True,
        },
        "records": records,
    }
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.output)
    print(json.dumps({
        "output": str(args.output),
        "candidateCount": payload["candidateCount"],
        "pairedEligibleCaseCount": payload["scope"]["pairedEligibleCaseCount"],
        "trainingStarted": payload["trainingStarted"],
        "networkCalls": payload["audit"]["networkCalls"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
