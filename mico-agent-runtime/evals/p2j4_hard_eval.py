"""Hard Eval validator and post-run audit for P2-J4.

The normal runner scores closed Trace criteria.  This module adds the
hard-set assertions that cannot be represented by the generic task schema,
especially evidence-conflict handling and the ordering of confounder control.
It never calls a provider and never writes a training dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mico_agent_runtime.contracts.trace_eval import EvalScore, TraceProjection

try:
    from evals.p2j4_runner import validate_task_set
except ModuleNotFoundError:  # direct script execution from the evals folder
    from p2j4_runner import validate_task_set


ROOT = Path(__file__).resolve().parent
TASK_SET = ROOT / "p2j4-hard-task-set-v1.json"
SPEC = ROOT / "p2j4-hard-eval-spec-v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HARD_EVAL_PAYLOAD_INVALID")
    return payload


def _actions(trace: TraceProjection) -> list[str]:
    if trace.decisions:
        return [item.chosenAction for item in trace.decisions]
    return [item.actionName for item in trace.events if item.node == "execute_action"]


def _ordered_subsequence(required: Iterable[str], actual: Iterable[str]) -> bool:
    required_items = list(required)
    position = 0
    for item in actual:
        if position < len(required_items) and item == required_items[position]:
            position += 1
    return position == len(required_items)


def validate_hard_spec(
    task_payload: dict[str, Any], spec_payload: dict[str, Any]
) -> dict[str, Any]:
    if not str(spec_payload.get("schemaVersion", "")).startswith("p2j4-hard-eval-spec-"):
        raise ValueError("HARD_EVAL_SPEC_VERSION_INVALID")
    tasks = task_payload.get("cases")
    specs = spec_payload.get("cases")
    expected_count = task_payload.get("caseCount")
    if (
        not isinstance(tasks, list)
        or not isinstance(specs, list)
        or not isinstance(expected_count, int)
        or len(tasks) != expected_count
        or len(specs) != expected_count
    ):
        raise ValueError("HARD_EVAL_CASE_COUNT_INVALID")
    task_ids = [item.get("caseId") for item in tasks]
    spec_ids = [item.get("caseId") for item in specs]
    if len(set(task_ids)) != expected_count or task_ids != spec_ids:
        raise ValueError("HARD_EVAL_CASE_ALIGNMENT_INVALID")
    class_counts = Counter(item.get("hardCaseClass") for item in specs)
    expected = task_payload.get("hardCaseDistribution")
    if not isinstance(expected, dict):
        raise ValueError("HARD_EVAL_CLASS_DISTRIBUTION_MISSING")
    if dict(class_counts) != expected:
        raise ValueError("HARD_EVAL_CLASS_DISTRIBUTION_INVALID")
    allowed_statuses = {"supported", "speculative", "conflicted", "partial", "unsupported"}
    for item in specs:
        if not isinstance(item.get("assertions"), list) or not item["assertions"]:
            raise ValueError("HARD_EVAL_ASSERTIONS_MISSING")
        if not set(item.get("requiredSupportStatuses", [])) <= allowed_statuses:
            raise ValueError("HARD_EVAL_SUPPORT_STATUS_INVALID")
    return {
        "caseCount": expected_count,
        "caseIdsUnique": True,
        "hardCaseDistribution": dict(class_counts),
        "status": "VALID",
    }


def audit_trace(task: dict[str, Any], spec: dict[str, Any], trace: TraceProjection, score: EvalScore) -> dict[str, Any]:
    actions = _actions(trace)
    hard_class = spec["hardCaseClass"]
    criteria: dict[str, bool] = {
        "generic_score_pass": score.status == "PASS",
        "actions_ordered": _ordered_subsequence(task.get("requiredActions", []), actions),
        "no_unbounded_action": "direct_mysql" not in actions and "unbounded_export" not in actions,
    }
    if hard_class == "premature_stop":
        finish_index = actions.index("finish") if "finish" in actions else len(actions)
        criteria["finish_after_validation"] = all(
            action in actions[:finish_index]
            for action in ("inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence")
        )
    elif hard_class == "confounder_trap":
        adjust_index = actions.index("adjust_confounders") if "adjust_confounders" in actions else -1
        later_claim_actions = ["cross_project_validate", "cross_disease_validate", "retrieve_evidence", "finish"]
        criteria["confounder_before_claim"] = adjust_index >= 0 and all(
            actions.index(action) > adjust_index for action in later_claim_actions if action in actions
        )
        criteria["no_causal_escalation"] = not trace.supportStatusEscalated
    elif hard_class == "evidence_conflict":
        statuses = set(trace.evidenceSupportStatuses)
        criteria["conflicted_status_present"] = "conflicted" in statuses
        criteria["no_support_escalation"] = not trace.supportStatusEscalated
        criteria["both_knowledge_routes"] = {"vector", "graph"}.issubset(set(trace.sourceRoutes))
    elif hard_class == "tool_selection":
        metadata_only = "METADATA_FIRST" in spec.get("assertions", [])
        if metadata_only:
            criteria["metadata_first"] = actions[:1] == ["inspect_cohort"]
            criteria["no_abundance_read"] = "execute_read_query" not in actions
        else:
            criteria["bounded_read_present"] = "execute_read_query" in actions
            criteria["bounded_read_terminal"] = _ordered_subsequence(["execute_read_query", "finish"], actions)
    criteria["all_hard_assertions"] = all(criteria.values())
    return {
        "caseId": task["caseId"],
        "traceId": trace.traceId,
        "hardCaseClass": hard_class,
        "status": "PASS" if criteria["all_hard_assertions"] else "FAIL",
        "scoreStatus": score.status,
        "actions": actions,
        "criteria": criteria,
        "failureCodes": [name for name, passed in criteria.items() if not passed],
        "auditedAt": datetime.now(timezone.utc).isoformat(),
    }


def audit_run(run_payload: dict[str, Any], task_payload: dict[str, Any], spec_payload: dict[str, Any]) -> dict[str, Any]:
    validate_hard_spec(task_payload, spec_payload)
    tasks = {item["caseId"]: item for item in task_payload["cases"]}
    specs = {item["caseId"]: item for item in spec_payload["cases"]}
    traces = {item["traceId"]: TraceProjection.model_validate(item) for item in run_payload.get("traces", [])}
    scores = {item["traceId"]: EvalScore.model_validate(item) for item in run_payload.get("scores", [])}
    records: list[dict[str, Any]] = []
    for case_id, task in tasks.items():
        score = next((item for item in scores.values() if item.caseId == case_id), None)
        if score is None or score.traceId not in traces:
            records.append({"caseId": case_id, "status": "MISSING", "failureCodes": ["HARD_TRACE_MISSING"]})
            continue
        records.append(audit_trace(task, specs[case_id], traces[score.traceId], score))
    counts = Counter(item["status"] for item in records)
    return {
        "schemaVersion": "p2j4-hard-eval-audit-v1",
        "sourceRun": run_payload.get("schemaVersion"),
        "caseCount": len(records),
        "passCount": counts.get("PASS", 0),
        "failCount": counts.get("FAIL", 0),
        "missingCount": counts.get("MISSING", 0),
        "status": "PASS" if counts == Counter({"PASS": len(records)}) else "FAIL",
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or audit P2-J4 Hard Eval")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-set", type=Path, default=TASK_SET)
    parser.add_argument("--spec", type=Path, default=SPEC)
    args = parser.parse_args(argv)
    try:
        task_payload = _load_json(args.task_set)
        spec_payload = _load_json(args.spec)
        validation = validate_task_set(args.task_set)
        validation_summary = {
            key: value for key, value in validation.items() if key != "tasks"
        }
        spec_validation = validate_hard_spec(task_payload, spec_payload)
        if args.run:
            audit = audit_run(_load_json(args.run), task_payload, spec_payload)
            result: dict[str, Any] = {
                "taskValidation": validation_summary,
                "specValidation": spec_validation,
                "audit": audit,
            }
        else:
            result = {"taskValidation": validation_summary, "specValidation": spec_validation}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result.get("audit", result["specValidation"]).get("status"),
        "caseCount": result.get("audit", result["specValidation"]).get("caseCount"),
        "output": str(args.output) if args.output else None,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
