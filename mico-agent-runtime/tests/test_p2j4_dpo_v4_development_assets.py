from __future__ import annotations

from evals import p2j4_runner
from evals.p2j4_build_dpo_v4_development_task_set import build
from mico_agent_runtime.runtime.trace_eval import build_stability_baseline


def test_dpo_v4_development_tasks_are_trace_sources_not_eval_cases() -> None:
    payload = build()

    assert payload["schemaVersion"] == "p2j4-dpo-v4-development-v1"
    assert payload["caseCount"] == 80
    assert payload["reviewStatus"] == "STRUCTURE_READY_NOT_EXECUTED"
    assert payload["frozenEvaluationExclusions"] == ["Golden50", "Hard30", "Test70", "OOD30", "Runtime20"]
    assert sum(payload["familyDistribution"].values()) == 80
    assert len(payload["developmentCaseMetadata"]) == 80
    assert all(not item["evaluationEligible"] for item in payload["developmentCaseMetadata"].values())
    bounded = payload["cases"][0]
    assert bounded["allowedActionPaths"] == [
        ["execute_read_query", "finish"],
        ["inspect_cohort", "execute_read_query", "finish"],
    ]


def test_dpo_v4_development_task_set_validates_under_closed_contract() -> None:
    summary = p2j4_runner.validate_task_set(
        p2j4_runner.TASK_SET.parent / "p2j4-dpo-v4-development-task-set-v1.json"
    )
    assert summary["schemaVersion"] == "p2j4-dpo-v4-development-v1"
    assert summary["caseCount"] == 80
    assert summary["caseIdsUnique"] is True


def test_dpo_v4_development_version_is_valid_for_empty_resumable_checkpoint() -> None:
    baseline = build_stability_baseline("p2j4-dpo-v4-development-v1", [], [])
    assert baseline.taskSetVersion == "p2j4-dpo-v4-development-v1"
    assert baseline.runCount == 0
