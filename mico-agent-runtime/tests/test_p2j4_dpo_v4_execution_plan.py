from __future__ import annotations

from evals.p2j4_build_dpo_v4_development_execution_plan import build


def test_dpo_v4_execution_plan_is_partitioned_and_success_preserving() -> None:
    plan = build()
    case_ids = [case_id for batch in plan["batches"] for case_id in batch["caseIds"]]

    assert plan["status"] == "READY_NOT_EXECUTED"
    assert plan["caseCount"] == 80
    assert len(plan["preservedCompletedCaseIds"]) == 17
    assert plan["pendingCaseCount"] == 63
    assert plan["batchCount"] == 5
    assert sorted(batch["caseCount"] for batch in plan["batches"]) == [12, 12, 13, 13, 13]
    assert len(case_ids) == len(set(case_ids)) == 63
    assert "p2j4-dpo-v4-dev-001" not in case_ids
    assert plan["sourcePolicy"]["successfulCasesNeverRerun"] is True
    assert plan["sourcePolicy"]["allPassRequiredBeforePairExtraction"] is True
