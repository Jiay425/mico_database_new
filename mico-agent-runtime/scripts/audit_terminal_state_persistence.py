"""Audit the historical Task A terminal snapshot without mutating it.

This is intentionally an offline audit.  It compares the state immediately
before the historical finish decision with the state serialized after it and
records the two known closure defects (stale progress recomputation and the
missing Runtime objective-resolution projection).  The original trace is
never rewritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_DIR = (
    REPO_ROOT
    / "artifacts"
    / "final_task_a_rerun_20260904"
    / "decision-sft-v1-e2e-a-t2d-age"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def audit_terminal_state(task_dir: Path) -> dict[str, Any]:
    before = _load(task_dir / "state_s6.json")
    after = _load(task_dir / "state_s7.json")
    trace = _load(task_dir / "trace.json")

    before_progress = before.get("progress", {})
    after_progress = after.get("progress", {})
    before_remaining = list(before_progress.get("remaining_objectives", []))
    after_remaining = list(after_progress.get("remaining_objectives", []))
    before_analysis = before.get("analysis_state", {})
    before_project = before_analysis.get("cross_project_validation", {})
    project_data = before.get("data_state", {}).get("project_state", {})
    project_blocked_evidence = (
        project_data.get("project_count", 0) < 2
        and project_data.get("has_project_field") is False
        and before_project.get("status") != "completed"
    )

    root_causes: list[str] = []
    if "cross_project_validation" in after_remaining and (
        "cross_project_validation" not in before_remaining
    ):
        root_causes.append("STALE_PROGRESS_COPY")
    if not ({"objectiveResolution", "objective_resolution"} & set(trace)) and not any(
        "objective_resolution" in turn for turn in trace.get("turns", [])
        if isinstance(turn, dict)
    ):
        root_causes.append("TRACE_SERIALIZATION_OMISSION")

    inferred_resolution = {
        "resolutions": [
            {
                "objective": objective,
                "status": "blocked"
                if objective == "cross_project_validation" and project_blocked_evidence
                else "completed"
                if objective in {
                    "group_comparison",
                    "confounder_assessment",
                    "evidence_support",
                }
                and objective != "cross_project_validation"
                else "active",
                "reason_code": (
                    "PROJECT_DIMENSION_UNAVAILABLE"
                    if objective == "cross_project_validation" and project_blocked_evidence
                    else None
                ),
                "resolved_at_step": (
                    before_progress.get("action_count")
                    if objective == "cross_project_validation" and project_blocked_evidence
                    else before_progress.get("action_count")
                    if objective in {
                        "group_comparison",
                        "confounder_assessment",
                        "evidence_support",
                    }
                    else None
                ),
                "limitation_code": (
                    "objective_unavailable_due_to_data"
                    if objective == "cross_project_validation" and project_blocked_evidence
                    else None
                ),
                "action": {
                    "group_comparison": "compare_groups",
                    "confounder_assessment": "adjust_confounders",
                    "cross_project_validation": "cross_project_validate",
                    "evidence_support": "retrieve_evidence",
                }.get(objective),
            }
            for objective in before.get("task", {}).get("objectives", [])
        ],
    }
    inferred_resolution["active_objectives"] = [
        item["objective"]
        for item in inferred_resolution["resolutions"]
        if item["status"] == "active"
    ]
    inferred_resolution["completed_objectives"] = [
        item["objective"]
        for item in inferred_resolution["resolutions"]
        if item["status"] == "completed"
    ]
    inferred_resolution["blocked_objectives"] = [
        item["objective"]
        for item in inferred_resolution["resolutions"]
        if item["status"] == "blocked"
    ]
    inferred_resolution["workflow_can_finish"] = not inferred_resolution["active_objectives"]
    inferred_resolution["all_requested_objectives_completed"] = not inferred_resolution[
        "active_objectives"
    ] and not inferred_resolution["blocked_objectives"]
    inferred_resolution["limitations_present"] = bool(inferred_resolution["blocked_objectives"])

    return {
        "audit_version": "terminal-state-persistence-audit-v1",
        "source_trace": str(task_dir / "trace.json"),
        "source_trace_preserved": True,
        "state_before_finish": {
            "file": str(task_dir / "state_s6.json"),
            "remaining_objectives": before_remaining,
            "action_count": before_progress.get("action_count"),
            "available_actions": before.get("action_space", {}).get("available_actions", []),
        },
        "state_after_finish": {
            "file": str(task_dir / "state_s7.json"),
            "remaining_objectives": after_remaining,
            "action_count": after_progress.get("action_count"),
            "available_actions": after.get("action_space", {}).get("available_actions", []),
        },
        "historical_lifecycle_evidence": {
            "project_capability_unavailable": project_blocked_evidence,
            "blocked_objective_inferred": "cross_project_validation"
            if project_blocked_evidence
            else None,
            "finish_was_available_before_terminal_turn": "finish"
            in before.get("action_space", {}).get("available_actions", []),
            "historical_runtime_status": trace.get("runtime_status"),
        },
        "inferred_historical_objective_resolution": inferred_resolution,
        "root_cause": root_causes,
        "trace_objective_resolution_persisted": (
            "objectiveResolution" in trace or "objective_resolution" in trace
        )
        or any(
            isinstance(turn, dict) and "objective_resolution" in turn
            for turn in trace.get("turns", [])
        ),
        "final_snapshot_matches_active_objectives": after_remaining
        == inferred_resolution["active_objectives"],
        "historical_behavior_correct_but_final_snapshot_stale": (
            project_blocked_evidence
            and before_remaining == []
            and "cross_project_validation" in after_remaining
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()
    output = args.output.resolve() if args.output else task_dir / "derived_trace_consistency_audit.json"
    audit = audit_terminal_state(task_dir)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
