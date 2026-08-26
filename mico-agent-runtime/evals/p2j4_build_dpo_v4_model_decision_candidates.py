"""Build review-only DPO candidates from the model's actual decisions.

This is intentionally separate from the contract-derived builder.  It never
turns a task's required action into a model preference and never treats every
approved action as a legal alternative.  A row remains review-only until a
semantic reviewer supplies a structured preference rationale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_BUILDER = ROOT / "evals" / "p2j4_build_dpo_v4_contract_preference_candidates.py"


def _load_source_lists() -> tuple[list[Path], list[Path]]:
    namespace: dict[str, Any] = {
        "__file__": str(CONTRACT_BUILDER),
        "__name__": "_dpo_candidate_sources",
    }
    exec(CONTRACT_BUILDER.read_text(encoding="utf-8"), namespace)
    return list(namespace["TRACE_FILES"]), list(namespace["TASK_FILES"])


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _tasks(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for task in payload.get("cases", []):
            result[task["caseId"]] = task
    return result


def _planner_meta(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    services = payload.get("servicesObserved")
    services = services if isinstance(services, dict) else {}
    model = services.get("plannerModel")
    provider = services.get("plannerProvider")
    return {
        "planner_provider": provider if isinstance(provider, str) else None,
        "planner_model": model if isinstance(model, str) else None,
        "planner_switch_reason": (
            "GEMINI_UNAVAILABLE_OR_RATE_LIMITED;USER_DIRECTED_DEEPSEEK_FLASH"
            if model == "deepseek-v4-flash"
            and ("state-matrix-runs-v12" in path.name or "state-matrix-runs-v13" in path.name)
            else None
        ),
        "trace_planner_model_used": services.get("plannerModelUsed") is True,
    }


def _is_model_trace(payload: dict[str, Any], trace: dict[str, Any]) -> bool:
    # Provenance is audited at decision granularity below.  A later
    # deterministic repair must not erase earlier model decisions in the same
    # trace, and an initial guard must not erase a later raw model decision.
    # The trace-level plannerModelUsed flag is preserved as metadata rather
    # than misused as a substitute for per-decision raw/final provenance.
    return True


def _observed_count(decision: dict[str, Any]) -> int:
    value = decision.get("observationEvidenceBindingCount")
    return value if isinstance(value, int) else len(decision.get("history_actions", []))


def _legal_alternatives(
    decision: dict[str, Any],
    task: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]]]:
    chosen = decision.get("final_action")
    history = set(decision.get("history_actions") or [])
    observed_count = _observed_count(decision)
    required = set(task.get("requiredActions") or [])
    reasons: dict[str, list[str]] = {}
    result: list[str] = []
    for candidate in decision.get("candidate_actions") or []:
        if candidate == chosen:
            continue
        why: list[str] = []
        if candidate in history:
            why.append("ALREADY_OBSERVED_ACTION")
        if candidate in {"compare_groups", "stratified_analysis", "adjust_confounders", "analyze_projection", "retrieve_evidence"} and observed_count < 1:
            why.append("REQUIRES_VALIDATED_OBSERVATION")
        if candidate in {"cross_project_validate", "cross_disease_validate"} and observed_count < 2:
            why.append("REQUIRES_TWO_VALIDATED_OBSERVATIONS")
        if candidate == "finish" and not required.issubset(history | {chosen}):
            why.append("REQUIRED_ACTION_NOT_COMPLETED")
        if why:
            reasons[candidate] = why
            continue
        result.append(candidate)
    return result, reasons


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_files, task_files = _load_source_lists()
    task_map = _tasks(task_files)
    rows: list[dict[str, Any]] = []
    scanned = 0
    model_decisions = 0
    complete_decisions = 0
    unique_states: set[str] = set()
    trace_counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    for path in trace_files:
        if not path.exists():
            excluded["SOURCE_FILE_MISSING"] += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        pass_trace_ids = {
            score.get("traceId")
            for score in payload.get("scores", [])
            if score.get("status") == "PASS"
        }
        metadata = _planner_meta(payload, path)
        for trace in payload.get("traces", []):
            if trace.get("traceId") not in pass_trace_ids:
                excluded["TRACE_NOT_PASS"] += 1
                continue
            if not _is_model_trace(payload, trace):
                excluded["TRACE_NOT_MODEL_OR_FALLBACK"] += 1
                continue
            task = task_map.get(trace.get("caseId") or "")
            if task is None:
                # Older runner payloads identify the case through the score.
                case_id = next(
                    (score.get("caseId") for score in payload.get("scores", [])
                     if score.get("traceId") == trace.get("traceId")),
                    None,
                )
                task = task_map.get(case_id or "")
            if task is None:
                excluded["TASK_NOT_FOUND"] += 1
                continue
            trace_counts["model_pass_trace"] += 1
            for decision in trace.get("decisions", []):
                scanned += 1
                if decision.get("planner_origin") != "model":
                    continue
                model_decisions += 1
                required_fields = (
                    "task_kind", "goal_code", "observation_flags", "history_actions",
                    "candidate_actions", "state_summary",
                )
                if any(decision.get(field) is None for field in required_fields):
                    excluded["STRUCTURED_FIELDS_INCOMPLETE"] += 1
                    continue
                chosen = decision.get("final_action")
                if not chosen or decision.get("raw_action") != chosen:
                    excluded["RAW_FINAL_ACTION_MISMATCH"] += 1
                    continue
                complete_decisions += 1
                state = {
                    "task_kind": decision.get("task_kind"),
                    "goal_code": decision.get("goal_code"),
                    "observation_flags": list(decision.get("observation_flags", [])),
                    "history_actions": list(decision.get("history_actions", [])),
                    "candidate_actions": list(decision.get("candidate_actions", [])),
                    "state_summary": decision.get("state_summary", ""),
                }
                signature = _hash(state)
                unique_states.add(signature)
                alternatives, rejected_reasons = _legal_alternatives(decision, task)
                if not alternatives:
                    excluded["NO_SEMANTICALLY_LEGAL_ALTERNATIVE"] += 1
                    continue
                if "weak-action-source-runs-v14" in path.name:
                    scenario_class = "controlled_weak_action_source"
                elif "state-matrix-runs-v1" in path.name:
                    scenario_class = "controlled_state_matrix"
                else:
                    scenario_class = "model_origin_collection"
                for rejected in alternatives:
                    rows.append({
                        "source_kind": "model_origin_policy_trace",
                        "scenario_class": scenario_class,
                        "source_trace_id": trace.get("traceId"),
                        "source_file": path.name,
                        **metadata,
                        "case_id": next(
                            (score.get("caseId") for score in payload.get("scores", [])
                             if score.get("traceId") == trace.get("traceId")),
                            trace.get("caseId"),
                        ),
                        "state_signature": signature,
                        "task_family": task.get("family") or task.get("kind") or decision.get("task_kind"),
                        "task_kind": decision.get("task_kind"),
                        "goal_code": decision.get("goal_code"),
                        "observation_flags": decision.get("observation_flags"),
                        "history_actions": decision.get("history_actions"),
                        "candidate_actions": decision.get("candidate_actions"),
                        "state_summary": decision.get("state_summary"),
                        "observation_count": _observed_count(decision),
                        "task_required_actions": list(task.get("requiredActions", [])),
                        "planner_origin": "model",
                        "raw_action": decision.get("raw_action"),
                        "final_action": decision.get("final_action"),
                        "decision_repair_codes": list(decision.get("repair_codes", [])),
                        "chosen_action": chosen,
                        "rejected_action": rejected,
                        "rejected_exclusion_reasons": rejected_reasons,
                        "preference_type": None,
                        "preference_dimension": None,
                        "preference_codes": [],
                        "preference_rationale": None,
                        "preference_rationale_source": None,
                        "review_status": "SEMANTIC_REVIEW_REQUIRED",
                    })
    unique_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        unique_rows.setdefault(
            (row["state_signature"], row["chosen_action"], row["rejected_action"]),
            row,
        )
    rows = list(unique_rows.values())
    state_counts = Counter(row["state_signature"] for row in rows)
    audit = {
        "schemaVersion": "p2j4-dpo-v4-model-decision-candidates-audit-v1",
        "status": "SEMANTIC_REVIEW_REQUIRED" if rows else "BLOCKED",
        "trainingStarted": False,
        "pairConstructionStarted": False,
        "scannedDecisionCount": scanned,
        "modelDecisionCount": model_decisions,
        "completeModelDecisionCount": complete_decisions,
        "uniqueModelStateCount": len(unique_states),
        "rawCandidateCount": len(rows),
        "uniqueCandidateCount": len(rows),
        "uniqueCandidateStateCount": len(state_counts),
        "maxCandidateMultiplicityPerState": max(state_counts.values()) if state_counts else 0,
        "chosenActionCounts": dict(Counter(row["chosen_action"] for row in rows)),
        "candidateWidthCounts": dict(Counter(len(row["candidate_actions"]) for row in rows)),
        "scenarioClassCounts": dict(Counter(row["scenario_class"] for row in rows)),
        "sourceKindCounts": dict(Counter(row["source_kind"] for row in rows)),
        "excludedCounts": dict(excluded),
        "traceCounts": dict(trace_counts),
        "reviewRules": [
            "chosen is the model raw_action and equals final_action",
            "rejected is in candidate_actions and not already observed",
            "cross validation requires two validated observations",
            "finish is rejected before task required actions are complete",
            "no preference rationale is auto-approved",
            "rows are candidates only and cannot be used for training",
        ],
        "nextAction": "review_semantic_preference_and_structured_rationale",
    }
    return rows, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    rows, audit = build()
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in (
        "status", "completeModelDecisionCount", "uniqueModelStateCount",
        "uniqueCandidateCount", "chosenActionCounts", "excludedCounts",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
