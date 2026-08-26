from __future__ import annotations

from datetime import datetime, timezone
import json

from mico_agent_runtime.contracts.trace_eval import (
    EvalScore,
    EvalTask,
    HumanReviewCommand,
    TraceDecision,
    TraceProjection,
)
from evals import p2j4_runner
from evals.p2j4_controlled_scenarios import (
    ResultOracleObservation,
    validate_controlled_scenario_set,
    verify_result_oracle,
)
from evals.p2j4_decision_dataset import build_decision_dataset
from evals.p2j4_decision_context import DecisionSftCandidate
from evals.p2j4_split_decision_by_family import build as build_decision_family_split
from evals.p2j4_result_oracles import validate_result_oracle_set
from mico_agent_runtime.runtime.trace_eval import apply_human_review, build_bad_cases, score_trace


def _trace() -> TraceProjection:
    return TraceProjection(
        traceId="trace-00000000000000000000000000000001",
        runId="run-00000000000000000000000000000001",
        taskId="task-00000000000000000000000000000001",
        status="COMPLETED",
        actionCount=3,
        toolCallCount=0,
        sourceRoutes=["java"],
        evidenceBindingCount=2,
        analysisResultCount=0,
        structuredResult=True,
        decisions=[
            TraceDecision(
                observationStateCode="NO_OBSERVATION",
                allowedActions=["inspect_cohort", "compare_groups", "finish"],
                chosenAction="inspect_cohort",
                decisionCode="SELECT_ALLOWED_RUNTIME_ACTION",
            ),
            TraceDecision(
                observationStateCode="VALIDATED_OBSERVATION",
                allowedActions=["inspect_cohort", "compare_groups", "finish"],
                chosenAction="compare_groups",
                decisionCode="SELECT_ALLOWED_RUNTIME_ACTION",
            ),
        ],
        stopReasonCode="EVIDENCE_SUFFICIENT",
        startedAt=datetime(2026, 8, 23, tzinfo=timezone.utc),
        endedAt=datetime(2026, 8, 23, tzinfo=timezone.utc),
    )


def _task() -> EvalTask:
    return EvalTask(
        schemaVersion="p2j4-task-set-v2",
        caseId="p2j4-focused-comparison-test",
        kind="focused_analysis",
        question="比较两个研究组的元数据。",
        expectedStatus="COMPLETED",
        requiredSources=["java"],
        allowedActions=["inspect_cohort", "compare_groups", "finish"],
        requiredActions=["inspect_cohort", "compare_groups"],
        forbiddenActions=["direct_mysql"],
        minEvidenceBindings=2,
        maxActionCount=8,
        expectedStopReason="EVIDENCE_SUFFICIENT",
    )


def test_v2_task_set_is_closed_and_distributed() -> None:
    summary = p2j4_runner.validate_task_set()
    assert summary["caseCount"] == 50
    assert summary["caseIdsUnique"] is True
    assert summary["kindDistribution"] == {
        "data_fact": 10,
        "focused_analysis": 15,
        "open_exploration": 20,
        "safety": 5,
    }
    assert summary["resultOracleCount"] == 12
    assert summary["scenarioRequiredOracleCount"] == 5
    assert summary["controlledScenarioCount"] == 5
    tasks = summary["tasks"]
    open_tasks = [task for task in tasks if task.kind == "open_exploration"]
    safety_tasks = [task for task in tasks if task.kind == "safety"]
    assert len(open_tasks) == 20
    assert len(safety_tasks) == 5
    assert all(task.allowedActionPaths for task in open_tasks)
    assert all(task.allowedActions == ["finish"] for task in safety_tasks)


def test_v3_expanded_task_set_keeps_v2_cases_and_adds_fifty_cases() -> None:
    summary = p2j4_runner.validate_task_set(p2j4_runner.EXPANDED_TASK_SET)
    assert summary["caseCount"] == 100
    assert summary["caseIdsUnique"] is True
    assert summary["kindDistribution"] == {
        "data_fact": 20,
        "focused_analysis": 30,
        "open_exploration": 40,
        "safety": 10,
    }
    assert summary["resultOracleCount"] == 12
    assert summary["scenarioRequiredOracleCount"] == 5
    ids = {task.caseId for task in summary["tasks"]}
    assert "p2j4-data-fact-001" in ids
    assert "p2j4-data-fact-011" in ids
    assert "p2j4-open-exploration-040" in ids
    assert "p2j4-safety-010" in ids


def test_bad_case_result_oracles_are_closed_and_task_aligned() -> None:
    task_summary = p2j4_runner.validate_task_set()
    oracle_summary = validate_result_oracle_set(task_summary["tasks"])

    assert oracle_summary["oracleCount"] == 12
    assert oracle_summary["scenarioRequiredCount"] == 5
    serialized = str(oracle_summary)
    assert "SELECT " not in serialized
    assert "sourceSampleId" not in serialized

    task_by_case = {task.caseId: task for task in task_summary["tasks"]}
    country_stability = task_by_case["p2j4-open-exploration-012"]
    assert [
        "inspect_cohort", "stratified_analysis", "cross_project_validate",
        "retrieve_evidence", "finish",
    ] in country_stability.allowedActionPaths


def test_controlled_scenario_seeds_are_closed_and_match_required_oracles() -> None:
    task_summary = p2j4_runner.validate_task_set()
    oracle_summary = validate_result_oracle_set(task_summary["tasks"])
    scenario_summary = validate_controlled_scenario_set(oracle_summary["oracles"])

    assert scenario_summary["scenarioCount"] == 5
    serialized = str(scenario_summary)
    assert "SELECT " not in serialized
    assert "sourceSampleId" not in serialized


def test_result_oracle_verifier_requires_a_real_scenario_before_controlled_case_passes() -> None:
    task_summary = p2j4_runner.validate_task_set()
    oracle_summary = validate_result_oracle_set(task_summary["tasks"])
    oracle = next(
        item for item in oracle_summary["oracles"]
        if item.caseId == "p2j4-data-fact-003"
    )

    verification = verify_result_oracle(
        oracle,
        ResultOracleObservation(caseId=oracle.caseId),
    )

    assert verification.status == "SCENARIO_REQUIRED"
    assert verification.failureCodes == ["CONTROLLED_SCENARIO_NOT_PROVISIONED"]


def test_result_oracle_verifier_accepts_only_complete_no_value_contract_observation() -> None:
    task_summary = p2j4_runner.validate_task_set()
    oracle_summary = validate_result_oracle_set(task_summary["tasks"])
    oracle = next(
        item for item in oracle_summary["oracles"]
        if item.caseId == "p2j4-open-exploration-019"
    )

    complete = ResultOracleObservation(
        caseId=oracle.caseId,
        scenarioRef=oracle.scenarioRef,
        actionPath=oracle.requiredActionSubsequence,
        sourceRoutes=oracle.requiredSources,
        assertionCodes=oracle.assertionCodes,
        limitationCodes=oracle.limitationCodes,
        stopReasonCode=oracle.expectedStopReason,
    )
    assert verify_result_oracle(oracle, complete).status == "PASS"

    incomplete = complete.model_copy(update={"limitationCodes": []})
    failed = verify_result_oracle(oracle, incomplete)
    assert failed.status == "FAIL"
    assert failed.failureCodes == ["RESULT_LIMITATION_MISSING"]


def test_default_validation_and_dry_run_do_not_call_external_runtime(capsys) -> None:
    assert p2j4_runner.main(["--validate"]) == 0
    validation_output = capsys.readouterr().out
    assert '"externalCalls": false' in validation_output
    assert "question" not in validation_output

    assert p2j4_runner.main(["--dry-run"]) == 0
    dry_run_output = capsys.readouterr().out
    assert '"mode": "dry_run"' in dry_run_output
    assert "question" not in dry_run_output


def test_canary_case_selection_is_explicit_and_redacted(capsys) -> None:
    assert p2j4_runner.main([
        "--dry-run",
        "--case-id", "p2j4-data-fact-001",
        "--case-id", "p2j4-focused-analysis-008",
        "--case-id", "p2j4-open-exploration-001",
    ]) == 0
    output = capsys.readouterr().out
    assert '"caseCount": 3' in output
    assert '"externalCalls": false' in output
    assert "question" not in output


def test_real_run_requires_explicit_opt_in_and_configuration(monkeypatch, capsys) -> None:
    monkeypatch.delenv("MICO_P2J4_REAL_RUNS", raising=False)
    assert p2j4_runner.main(["--real"]) == 0
    disabled = capsys.readouterr().out
    assert "REAL_RUN_DISABLED" in disabled

    monkeypatch.setenv("MICO_P2J4_REAL_RUNS", "true")
    monkeypatch.setattr(
        p2j4_runner, "_configuration_missing", lambda _env: ["MICO_AGENT_INTERNAL_TOKEN"]
    )
    assert p2j4_runner.main(["--real"]) == 0
    missing = capsys.readouterr().out
    assert "REAL_RUN_CONFIGURATION_MISSING" in missing
    assert "MICO_AGENT_INTERNAL_TOKEN" in missing
    assert "token-value" not in missing


def test_real_run_persists_an_atomic_checkpoint_after_each_completed_case(monkeypatch, tmp_path) -> None:
    first = _task().model_copy(update={"caseId": "p2j4-focused-checkpoint-001"})
    second = _task().model_copy(update={"caseId": "p2j4-focused-checkpoint-002"})
    validation = {
        "schemaVersion": "p2j4-task-set-v2",
        "kindDistribution": {"focused_analysis": 2},
    }
    checkpoint = tmp_path / "nested" / "p2j4.json"
    snapshots: list[tuple[str, list[str]]] = []
    original_write = p2j4_runner._atomic_write_payload

    def capture_checkpoint(payload, output) -> None:
        snapshots.append((payload["status"], list(payload["completedCaseIds"])))
        original_write(payload, output)

    class Resource:
        def close(self) -> None:
            return None

    class Runtime:
        def run(self, request):
            return request

    def project(request):
        return _trace().model_copy(update={
            "traceId": request.traceId,
            "runId": request.runId,
            "taskId": request.taskId,
        })

    monkeypatch.setattr(p2j4_runner, "_configuration_missing", lambda _env: [])
    monkeypatch.setattr(p2j4_runner, "_build_real_runtime", lambda _env: (Runtime(), Resource(), Resource()))
    monkeypatch.setattr(p2j4_runner, "build_trace_projection", project)
    monkeypatch.setattr(p2j4_runner, "_atomic_write_payload", capture_checkpoint)

    payload = p2j4_runner._real_run(
        [first, second],
        validation,
        {"MICO_P2J4_REAL_RUNS": "true"},
        output=checkpoint,
    )

    assert snapshots == [
        ("RUNNING", []),
        ("RUNNING", [first.caseId]),
        ("COMPLETED", [first.caseId, second.caseId]),
    ]
    assert payload["status"] == "COMPLETED"
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["completedCaseIds"] == [first.caseId, second.caseId]
    assert saved["remainingCaseIds"] == []

    monkeypatch.setattr(
        p2j4_runner,
        "_build_real_runtime",
        lambda _env: (_ for _ in ()).throw(AssertionError("completed checkpoint must not re-run cases")),
    )
    resumed = p2j4_runner._real_run(
        [first, second],
        validation,
        {"MICO_P2J4_REAL_RUNS": "true"},
        output=checkpoint,
        resume=True,
    )
    assert resumed["realRunsExecuted"] == 2


def test_decision_accuracy_accepts_multiple_legal_paths() -> None:
    score = score_trace(_task(), _trace())
    assert score.criteria["decision_accuracy"] is True
    assert score.criteria["evidence_grounding"] is True
    assert score.criteria["safety_pass"] is True


def test_explicit_insufficiency_oracle_allows_immediate_safe_finish() -> None:
    task = next(
        item for item in p2j4_runner.validate_task_set()["tasks"]
        if item.caseId == "p2j4-open-exploration-016"
    )
    trace = _trace().model_copy(update={
        "actionCount": 1,
        "sourceRoutes": [],
        "evidenceBindingCount": 0,
        "decisions": [TraceDecision(
            observationStateCode="NO_OBSERVATION",
            allowedActions=["inspect_cohort", "retrieve_evidence", "finish"],
            chosenAction="finish",
            decisionCode="STOP_AFTER_VALIDATION",
        )],
        "stopReasonCode": "NO_NEW_INFORMATION",
    })

    score = score_trace(task, trace)

    assert task.requiredActions == []
    assert score.status == "PASS"
    assert score.criteria["decision_accuracy"] is True


def test_bad_case_categories_and_review_gate_are_structured() -> None:
    task = _task()
    score = EvalScore(
        caseId=task.caseId,
        traceId=_trace().traceId,
        status="FAIL",
        score=0.0,
        criteria={"decision_accuracy": False},
        failureCodes=["DECISION_ACCURACY_FAILURE"],
        evaluatedAt=datetime(2026, 8, 23, tzinfo=timezone.utc),
    )
    records = build_bad_cases(task, score)
    assert records[0].category == "PLANNING_ERROR"
    assert records[0].status == "OPEN"
    in_review = apply_human_review(
        records[0],
        HumanReviewCommand(
            badCaseId=records[0].badCaseId,
            decision="START_REVIEW",
            reviewerId="principal-00000000000000000000000000000001",
            reviewedAt=datetime(2026, 8, 23, tzinfo=timezone.utc),
        ),
    )
    assert records[0].status == "OPEN"
    assert in_review.status == "IN_REVIEW"


def test_decision_dataset_excludes_unreviewed_failures_and_raw_fields() -> None:
    task = _task()
    trace = _trace()
    score = score_trace(task, trace)
    dataset = build_decision_dataset([trace], [score], [])
    assert dataset["candidateCount"] == 2
    assert dataset["schemaVersion"] == "p2j4-decision-dataset-v2"
    assert dataset["trainingStarted"] is False
    first = dataset["candidates"][0]
    assert {
        "state_summary",
        "decision_reason",
        "selected_action",
        "alternative_actions",
        "stop_reason",
    }.issubset(first)
    assert first["selected_action"] == first["chosenAction"]
    assert dataset["candidateFieldRepairs"]["inferredStopReasonCount"] == 0
    serialized = str(dataset)
    assert "question" not in serialized
    assert "sql" not in serialized
    assert "arguments" not in serialized


def test_decision_sft_v3_requires_context_and_closed_terminal_shape() -> None:
    candidate = DecisionSftCandidate(
        sourceTraceId="trace-00000000000000000000000000000001",
        task_kind="open_exploration",
        goal_code="cross_project_stability",
        task_family="open_exploration:cross_project_stability:standard",
        observation_flags=["OBSERVATION_VALIDATED", "CROSS_PROJECT_REQUIRED"],
        history_actions=["inspect_cohort", "compare_groups"],
        candidate_actions=["cross_project_validate", "retrieve_evidence", "finish"],
        state_summary="validated observation; project consistency remains untested",
        decision_reason="verify whether the finding is stable across independent projects",
        selected_action="cross_project_validate",
        alternative_actions=["retrieve_evidence", "finish"],
        reviewStatus="passed",
    )
    assert candidate.selected_action in candidate.candidate_actions
    assert candidate.stop_reason is None


def test_decision_family_split_keeps_holdout_families_and_traces_disjoint(tmp_path) -> None:
    def item(signature: str, family: str, trace_id: str) -> dict:
        return {
            "candidate": {
                "sourceTraceId": trace_id,
                "task_family": family,
                "hard_case_class": None,
            },
            "signatureHash": signature,
            "hardCaseIds": [],
        }

    review = {
        "schemaVersion": "p2j4-decision-review-set-v2",
        "items": [
            item("aaa", "data_fact:sample_count:standard", "trace-00000000000000000000000000000001"),
            item("bbb", "focused_analysis:group_comparison:standard", "trace-00000000000000000000000000000002"),
            item("ccc", "open_exploration:cross_project_stability:standard", "trace-00000000000000000000000000000003"),
        ],
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")

    result = build_decision_family_split(
        path,
        validation_families=["focused_analysis:group_comparison:standard"],
        test_families=["open_exploration:cross_project_stability:standard"],
    )

    assert result["familyDisjoint"] is True
    assert result["sourceTraceDisjoint"] is True
    assert result["splitCounts"]["train"]["itemCount"] == 1
    assert result["splitCounts"]["validation"]["itemCount"] == 1
    assert result["splitCounts"]["test"]["itemCount"] == 1
    assert result["splitPolicy"]["randomRowSplit"] is False
