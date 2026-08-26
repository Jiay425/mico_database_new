from __future__ import annotations

from evals import p2j4_runner
from evals.p2j4_build_policy_sensitive_task_set import build


def test_policy_sensitive_set_is_planner_dependent_after_initial_inspection() -> None:
    task_payload, preflight_payload = build()

    assert task_payload["schemaVersion"] == "p2j4-policy-sensitive-runtime-v1"
    assert task_payload["caseCount"] == 20
    assert task_payload["kindDistribution"] == {"open_exploration": 20}
    assert preflight_payload["allNonEmptyStatesSemanticNone"] is True

    paths = {
        tuple(case["allowedActionPaths"][0])
        for case in task_payload["cases"]
    }
    assert paths == {
        ("inspect_cohort", "compare_groups", "finish"),
        ("inspect_cohort", "compare_groups", "analyze_projection", "finish"),
        ("inspect_cohort", "compare_groups", "retrieve_evidence", "finish"),
        (
            "inspect_cohort",
            "compare_groups",
            "analyze_projection",
            "retrieve_evidence",
            "finish",
        ),
    }


def test_policy_sensitive_task_set_is_closed_by_runtime_validator() -> None:
    summary = p2j4_runner.validate_task_set(
        p2j4_runner.TASK_SET.parent / "p2j4-policy-sensitive-runtime-v1" / "task-set.json"
    )

    assert summary["schemaVersion"] == "p2j4-policy-sensitive-runtime-v1"
    assert summary["caseCount"] == 20
    assert summary["caseIdsUnique"] is True
    assert summary["kindDistribution"] == {"open_exploration": 20}
    assert summary["resultOracleCount"] == 0
    assert summary["controlledScenarioCount"] == 0
