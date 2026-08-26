"""P2-J4.1 task validation, opt-in real execution and redacted trace scoring.

Importing this module is intentionally side-effect free.  The default CLI mode
only validates JSON and never constructs a port, opens a socket or imports a
provider client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# The runner is commonly invoked as a script from the runtime directory.  In
# that mode Python puts ``evals/`` itself on sys.path, not its parent, so the
# package-qualified closed-contract imports used by real mode would fail.
_RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))

from pydantic import ValidationError

from mico_agent_runtime.contracts.research import ResearchTask
from mico_agent_runtime.contracts.trace_eval import BadCaseRecord, EvalScore, EvalTask, TraceProjection
from mico_agent_runtime.runtime.trace_eval import (
    build_bad_cases,
    build_stability_baseline,
    build_trace_projection,
    score_trace,
)


TASK_SET = Path(__file__).with_name("p2j4-task-set-v2.json")
EXPANDED_TASK_SET = Path(__file__).with_name("p2j4-task-set-v3.json")
LEGACY_TASK_SET = Path(__file__).with_name("p2j4-task-set-v1.json")
_TASK_SET_VERSIONS = {
    "p2j4-task-set-v1",
    "p2j4-task-set-v2",
    "p2j4-task-set-v3",
    "p2j4-decision-state-difference-v1",
    "p2j4-decision-state-difference-v2",
    "p2j4-dpo-v4-development-v1",
    "p2j4-dpo-v4-policy-source-v1",
    "p2j4-dpo-v4-targeted-weak-actions-v1",
    "p2j4-dpo-v4-targeted-weak-actions-v2",
    "p2j4-hard-task-set-v1",
    "p2j4-hard-variant-task-set-v1",
    "p2j4-policy-sensitive-runtime-v1",
}
_SENSITIVE_MARKERS = (
    "sourceSampleId",
    "internalRecordId",
    "mysql://",
    "postgresql://",
    "jdbc:",
    "bearer ",
    "api_key",
    "api-key",
    "token=",
    "password=",
)
_KIND_DISTRIBUTION_V2 = {
    "data_fact": 10,
    "focused_analysis": 15,
    "open_exploration": 20,
    "safety": 5,
}
_KIND_DISTRIBUTION_V3 = {
    "data_fact": 20,
    "focused_analysis": 30,
    "open_exploration": 40,
    "safety": 10,
}
_KIND_DISTRIBUTION_HARD_V1 = {
    "focused_analysis": 18,
    "open_exploration": 12,
}


def _read_task_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("EVAL_TASK_SET_UNREADABLE") from exc
    if not isinstance(payload, dict):
        raise ValueError("EVAL_TASK_SET_SHAPE_INVALID")
    version = payload.get("schemaVersion")
    if version not in _TASK_SET_VERSIONS:
        raise ValueError("EVAL_TASK_SET_VERSION_INVALID")
    return payload


def validate_task_set(path: Path = TASK_SET) -> dict[str, Any]:
    payload = _read_task_payload(path)
    version = payload["schemaVersion"]
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("EVAL_TASK_SET_CASES_INVALID")
    if version == "p2j4-task-set-v2" and len(raw_cases) != 50:
        raise ValueError("EVAL_TASK_SET_CASE_COUNT_INVALID")
    if version == "p2j4-task-set-v3" and len(raw_cases) != 100:
        raise ValueError("EVAL_TASK_SET_CASE_COUNT_INVALID")
    if version.startswith("p2j4-hard-"):
        declared_count = payload.get("caseCount")
        if declared_count != len(raw_cases):
            raise ValueError("EVAL_TASK_SET_CASE_COUNT_INVALID")
    if version == "p2j4-policy-sensitive-runtime-v1":
        declared_count = payload.get("caseCount")
        if declared_count != len(raw_cases):
            raise ValueError("EVAL_TASK_SET_CASE_COUNT_INVALID")
    if version == "p2j4-dpo-v4-development-v1":
        declared_count = payload.get("caseCount")
        if declared_count != len(raw_cases):
            raise ValueError("EVAL_TASK_SET_CASE_COUNT_INVALID")
    if version == "p2j4-task-set-v1" and not raw_cases:
        raise ValueError("EVAL_TASK_SET_CASES_EMPTY")

    case_ids: set[str] = set()
    tasks: list[EvalTask] = []
    required_v2_fields = {
        "schemaVersion",
        "caseId",
        "kind",
        "question",
        "expectedStatus",
        "requiredSources",
        "allowedActions",
        "requiredActions",
        "forbiddenActions",
        "minEvidenceBindings",
        "maxActionCount",
        "expectedStopReason",
        "requiresNonDiagnostic",
    }
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("EVAL_TASK_CASE_SHAPE_INVALID")
        if version in {"p2j4-task-set-v2", "p2j4-task-set-v3", "p2j4-decision-state-difference-v1", "p2j4-decision-state-difference-v2", "p2j4-dpo-v4-development-v1", "p2j4-dpo-v4-policy-source-v1", "p2j4-dpo-v4-targeted-weak-actions-v1", "p2j4-dpo-v4-targeted-weak-actions-v2", "p2j4-hard-task-set-v1", "p2j4-hard-variant-task-set-v1", "p2j4-policy-sensitive-runtime-v1"} \
                and not required_v2_fields.issubset(raw):
            raise ValueError("EVAL_TASK_CASE_FIELDS_MISSING")
        case_id = raw.get("caseId")
        if case_id in case_ids:
            raise ValueError("EVAL_TASK_CASE_ID_DUPLICATE")
        case_ids.add(case_id)
        serialized = json.dumps(raw, ensure_ascii=False, sort_keys=True).lower()
        if any(marker.lower() in serialized for marker in _SENSITIVE_MARKERS):
            raise ValueError("EVAL_TASK_SET_SENSITIVE_VALUE")
        try:
            task = EvalTask.model_validate(raw)
        except ValidationError as exc:
            raise ValueError("EVAL_TASK_CASE_SCHEMA_INVALID") from exc
        if version in {"p2j4-task-set-v2", "p2j4-task-set-v3", "p2j4-decision-state-difference-v1", "p2j4-decision-state-difference-v2", "p2j4-dpo-v4-development-v1", "p2j4-dpo-v4-policy-source-v1", "p2j4-dpo-v4-targeted-weak-actions-v1", "p2j4-dpo-v4-targeted-weak-actions-v2", "p2j4-hard-task-set-v1", "p2j4-hard-variant-task-set-v1", "p2j4-policy-sensitive-runtime-v1"} \
                and task.schemaVersion != version:
            raise ValueError("EVAL_TASK_CASE_VERSION_MISMATCH")
        tasks.append(task)
    distribution = dict(Counter(task.kind for task in tasks))
    if version == "p2j4-task-set-v2" and distribution != _KIND_DISTRIBUTION_V2:
        raise ValueError("EVAL_TASK_KIND_DISTRIBUTION_INVALID")
    if version == "p2j4-task-set-v3" and distribution != _KIND_DISTRIBUTION_V3:
        raise ValueError("EVAL_TASK_KIND_DISTRIBUTION_INVALID")
    if version.startswith("p2j4-hard-"):
        expected_distribution = payload.get("kindDistribution")
        if distribution != expected_distribution:
            raise ValueError("EVAL_TASK_KIND_DISTRIBUTION_INVALID")
    if version == "p2j4-policy-sensitive-runtime-v1":
        expected_distribution = payload.get("kindDistribution")
        if distribution != expected_distribution:
            raise ValueError("EVAL_TASK_KIND_DISTRIBUTION_INVALID")
    if version == "p2j4-dpo-v4-development-v1":
        expected_distribution = payload.get("kindDistribution")
        if distribution != expected_distribution:
            raise ValueError("EVAL_TASK_KIND_DISTRIBUTION_INVALID")
    oracle_summary = None
    if path.resolve() in {TASK_SET.resolve(), EXPANDED_TASK_SET.resolve()} \
            and version in {"p2j4-task-set-v2", "p2j4-task-set-v3"}:
        # Result oracles remain value-free and are checked with the task set;
        # this imports no provider client and opens no network connection.
        from evals.p2j4_result_oracles import validate_result_oracle_set
        from evals.p2j4_controlled_scenarios import validate_controlled_scenario_set
        oracle_summary = validate_result_oracle_set(tasks)
        scenario_summary = validate_controlled_scenario_set(oracle_summary["oracles"])
    else:
        scenario_summary = None
    return {
        "schemaVersion": version,
        "caseCount": len(tasks),
        "caseIdsUnique": len(case_ids) == len(tasks),
        "kindDistribution": distribution,
        "resultOracleCount": oracle_summary["oracleCount"] if oracle_summary else 0,
        "scenarioRequiredOracleCount": (
            oracle_summary["scenarioRequiredCount"] if oracle_summary else 0
        ),
        "controlledScenarioCount": scenario_summary["scenarioCount"] if scenario_summary else 0,
        "tasks": tasks,
    }


def load_task_set(path: Path = TASK_SET) -> list[EvalTask]:
    """Load and validate a task set without performing any external work."""

    return validate_task_set(path)["tasks"]


def select_tasks(validation: dict[str, Any], case_ids: list[str] | None) -> dict[str, Any]:
    """Return a validated task-set view, optionally limited to explicit case IDs."""

    if not case_ids:
        return validation
    requested = list(dict.fromkeys(case_ids))
    by_id = {task.caseId: task for task in validation["tasks"]}
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise ValueError("EVAL_CASE_ID_NOT_FOUND:" + ",".join(missing))
    tasks = [by_id[case_id] for case_id in requested]
    selected = dict(validation)
    selected["tasks"] = tasks
    selected["caseCount"] = len(tasks)
    selected["kindDistribution"] = dict(Counter(task.kind for task in tasks))
    return selected


def _opaque_id(prefix: str, value: str) -> str:
    return f"{prefix}-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _scopes_for(task: EvalTask) -> list[str]:
    scopes: set[str] = set()
    actions = set(task.allowedActions)
    if actions.intersection({"execute_read_query", "inspect_cohort"}):
        scopes.add("mico:query:read")
    if "retrieve_evidence" in actions:
        scopes.add("mico:evidence:read")
    if actions.intersection({
        "compare_groups", "stratified_analysis", "adjust_confounders",
        "cross_project_validate", "cross_disease_validate", "analyze_projection",
    }):
        scopes.add("mico:research:read")
    return sorted(scopes or {"mico:research:read"})


def _intent_for(task: EvalTask) -> str:
    if task.kind in {"knowledge_retrieval", "graph_multi_hop"}:
        return "literature_or_relationship"
    if task.kind == "data_fact" or task.kind == "schema_exploration":
        return "data_fact"
    if task.kind in {"focused_analysis", "focused_comparison", "confounder_stratification", "cross_validation"}:
        return "focused_comparison"
    return "scientific_exploration"


def _build_research_task(task: EvalTask, index: int) -> ResearchTask:
    suffix = f"{task.caseId}|{index}"
    allowed_actions = task.allowedActions or ["finish"]
    return ResearchTask(
        runId=_opaque_id("run", suffix),
        taskId=_opaque_id("task", suffix),
        requesterId="principal-00000000000000000000000000000001",
        traceId=_opaque_id("trace", suffix),
        question=task.question,
        intent=_intent_for(task),
        requestedScopes=_scopes_for(task),
        allowedActions=allowed_actions,
        maxActions=min(8, task.maxActionCount),
        createdAt=datetime.now(timezone.utc),
    )


def _configuration_missing(env: Mapping[str, str]) -> list[str]:
    missing: list[str] = []
    for name in ("MICO_JAVA_AGENT_TOOL_BASE_URL", "MICO_AGENT_INTERNAL_TOKEN"):
        if not env.get(name, "").strip():
            missing.append(name)
    backend = env.get("MICO_KNOWLEDGE_RETRIEVAL_BACKEND", "").strip().lower()
    database_enabled = (
        backend == "database"
        or env.get("MICO_KNOWLEDGE_VECTOR_ENABLED", "").strip().lower() == "true"
        or env.get("MICO_KNOWLEDGE_GRAPH_ENABLED", "").strip().lower() == "true"
    )
    if database_enabled:
        for name in (
            "MICO_KNOWLEDGE_VECTOR_DATABASE_URL",
            "MICO_KNOWLEDGE_NEO4J_URI",
            "MICO_KNOWLEDGE_NEO4J_USER",
            "MICO_KNOWLEDGE_NEO4J_PASSWORD",
            "MICO_LOCAL_KNOWLEDGE_INDEX_DIR",
            "MICO_GEMINI_API_KEY",
        ):
            if not env.get(name, "").strip():
                missing.append(name)
    elif env.get("MICO_LOCAL_KNOWLEDGE_ENABLED", "").strip().lower() == "true":
        if not env.get("MICO_LOCAL_KNOWLEDGE_INDEX_DIR", "").strip():
            missing.append("MICO_LOCAL_KNOWLEDGE_INDEX_DIR")
    else:
        missing.append("MICO_KNOWLEDGE_RETRIEVAL_BACKEND")
    return missing


def _build_real_runtime(env: Mapping[str, str]):
    """Construct existing ports only after explicit CLI and env opt-in."""

    from mico_agent_runtime.knowledge.database_retriever import DatabaseKnowledgeSearchPort
    from mico_agent_runtime.knowledge.local_retriever import LocalKnowledgeSearchPort
    from mico_agent_runtime.knowledge.synthesis import build_graph_rag_synthesis_port
    from mico_agent_runtime.ports.java_agent import HttpJavaAgentToolPort
    from mico_agent_runtime.ports.research_planner import build_intent_planner
    from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort
    from mico_agent_runtime.runtime.scientific_service import ScientificRuntime

    java_port = HttpJavaAgentToolPort.from_environment(env)
    backend = env.get("MICO_KNOWLEDGE_RETRIEVAL_BACKEND", "").strip().lower()
    if backend == "database":
        knowledge_port = DatabaseKnowledgeSearchPort.from_environment(env)
    else:
        knowledge_port = LocalKnowledgeSearchPort.from_environment(env)
    return ScientificRuntime(
        java_port,
        build_intent_planner(env),
        knowledge_port=knowledge_port,
        schema_catalog_port=JavaSchemaCatalogPort(java_port),
        synthesis_port=build_graph_rag_synthesis_port(env),
    ), java_port, knowledge_port


def _dry_run(tasks: list[EvalTask], validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": validation["schemaVersion"],
        "mode": "dry_run",
        "caseCount": len(tasks),
        "kindDistribution": validation["kindDistribution"],
        "realRunsExecuted": 0,
        "externalCalls": False,
        "cases": [
            {
                "caseId": task.caseId,
                "kind": task.kind,
                "expectedStatus": task.expectedStatus,
                "requiredSources": task.requiredSources,
                "maxActionCount": task.maxActionCount,
            }
            for task in tasks
        ],
    }


def _build_real_payload(
    tasks: list[EvalTask],
    validation: dict[str, Any],
    env: Mapping[str, str],
    traces: list[TraceProjection],
    scores: list[EvalScore],
    bad_cases: list[BadCaseRecord],
    result_oracle_verifications: list[dict[str, Any]],
    *,
    status: str,
    last_error_code: str | None = None,
) -> dict[str, Any]:
    """Build a redacted, durable run/checkpoint payload from completed cases."""

    completed_case_ids = [score.caseId for score in scores]
    completed_case_id_set = set(completed_case_ids)
    baseline = build_stability_baseline(validation["schemaVersion"], scores, traces)
    routes = {route for trace in traces for route in trace.sourceRoutes}
    planner_model_used = any(
        "SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK" not in trace.fallbackCodes
        for trace in traces
    )
    planner_base_url = env.get("MICO_RESEARCH_PLANNER_BASE_URL", "").lower()
    planner_provider = (
        "gemini_openai_compatible"
        if "generativelanguage.googleapis.com" in planner_base_url
        else "openai_compatible"
    )
    payload: dict[str, Any] = {
        "schemaVersion": validation["schemaVersion"],
        "checkpointVersion": "p2j4-real-run-checkpoint-v2",
        "mode": "real_run",
        "status": status,
        "caseCount": len(tasks),
        "selectedCaseIds": [task.caseId for task in tasks],
        "completedCaseIds": completed_case_ids,
        "remainingCaseIds": [task.caseId for task in tasks if task.caseId not in completed_case_id_set],
        "realRunsExecuted": len(scores),
        "externalCalls": True,
        "servicesObserved": {
            "java": any(any(event.toolName for event in trace.events) for trace in traces),
            "pgvector": "vector" in routes and env.get("MICO_KNOWLEDGE_RETRIEVAL_BACKEND", "").lower() == "database",
            "neo4j": "graph" in routes and env.get("MICO_KNOWLEDGE_RETRIEVAL_BACKEND", "").lower() == "database",
            "gemini": bool("vector" in routes and env.get("MICO_GEMINI_API_KEY", "").strip()),
            "plannerModelUsed": planner_model_used,
            "plannerProvider": planner_provider,
            "plannerModel": env.get("MICO_RESEARCH_PLANNER_MODEL", "").strip(),
            # Retained for checkpoint readers from the earlier DeepSeek run;
            # its value now means that a model planner, regardless of vendor,
            # supplied at least one action.
            "deepseek_or_planner_model": planner_model_used,
            "controlledKnowledgeScenarioMode": env.get("MICO_P2J4_HARD_CONTROLLED_SCENARIOS", "").lower() == "true",
        },
        "baseline": baseline.model_dump(mode="json"),
        "traces": [trace.model_dump(mode="json") for trace in traces],
        "scores": [score.model_dump(mode="json") for score in scores],
        "badCases": [record.model_dump(mode="json") for record in bad_cases],
        # No row, column, SQL, selected feature, cohort value, identifier, or
        # credential is allowed in this persisted result-oracle section.
        "resultOracleVerifications": result_oracle_verifications,
    }
    if last_error_code is not None:
        payload["lastErrorCode"] = last_error_code
    return payload


def _atomic_write_payload(payload: dict[str, Any], output: Path) -> None:
    """Persist a complete checkpoint atomically; never leave a half-written JSON file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _restore_checkpoint(
    output: Path | None,
    validation: dict[str, Any],
    tasks: list[EvalTask],
    *,
    resume: bool,
) -> tuple[list[TraceProjection], list[EvalScore], list[BadCaseRecord], list[dict[str, Any]]]:
    """Load only a matching, redacted checkpoint for an explicit resume request."""

    if output is None or not output.exists():
        return [], [], [], []
    if not resume:
        raise ValueError("EVAL_OUTPUT_EXISTS_USE_RESUME_OR_NEW_PATH")
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("EVAL_CHECKPOINT_SHAPE_INVALID")
        expected_case_ids = [task.caseId for task in tasks]
        if (
            payload.get("checkpointVersion") != "p2j4-real-run-checkpoint-v2"
            or payload.get("mode") != "real_run"
            or payload.get("schemaVersion") != validation["schemaVersion"]
            or payload.get("selectedCaseIds") != expected_case_ids
        ):
            raise ValueError("EVAL_CHECKPOINT_MISMATCH")
        traces = [TraceProjection.model_validate(item) for item in payload.get("traces", [])]
        scores = [EvalScore.model_validate(item) for item in payload.get("scores", [])]
        bad_cases = [BadCaseRecord.model_validate(item) for item in payload.get("badCases", [])]
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("EVAL_CHECKPOINT_UNREADABLE") from exc
    completed_case_ids = [score.caseId for score in scores]
    if (
        completed_case_ids != expected_case_ids[:len(completed_case_ids)]
        or len(traces) != len(scores)
        or {trace.traceId for trace in traces} != {score.traceId for score in scores}
    ):
        raise ValueError("EVAL_CHECKPOINT_INTEGRITY_INVALID")
    verifications = payload.get("resultOracleVerifications", [])
    if not isinstance(verifications, list) or not all(isinstance(item, dict) for item in verifications):
        raise ValueError("EVAL_CHECKPOINT_RESULT_ORACLE_INVALID")
    return traces, scores, bad_cases, verifications


def _controlled_scenario_maps() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load closed contracts only; this is intentionally network-free."""

    try:
        from evals.p2j4_controlled_scenarios import validate_controlled_scenario_set
        from evals.p2j4_result_oracles import validate_result_oracle_set
    except ModuleNotFoundError as exc:
        if exc.name != "evals":
            raise
        from p2j4_controlled_scenarios import validate_controlled_scenario_set
        from p2j4_result_oracles import validate_result_oracle_set

    all_tasks = load_task_set()
    oracle_summary = validate_result_oracle_set(all_tasks)
    scenario_summary = validate_controlled_scenario_set(oracle_summary["oracles"])
    return (
        {item.caseId: item for item in oracle_summary["oracles"]},
        {item.caseId: item for item in scenario_summary["seeds"]},
    )


def _scenario_unavailable_trace(task: EvalTask, request: ResearchTask, code: str) -> TraceProjection:
    now = datetime.now(timezone.utc)
    return TraceProjection(
        traceId=request.traceId,
        runId=request.runId,
        taskId=request.taskId,
        status="FAILED",
        actionCount=0,
        toolCallCount=0,
        sourceRoutes=["java"],
        evidenceBindingCount=0,
        analysisResultCount=0,
        fallbackCodes=[code],
        startedAt=now,
        endedAt=now,
    )


def _real_run(
    tasks: list[EvalTask],
    validation: dict[str, Any],
    env: Mapping[str, str],
    *,
    output: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if env.get("MICO_P2J4_REAL_RUNS", "").strip().lower() != "true":
        return {
            "schemaVersion": validation["schemaVersion"],
            "mode": "real_run",
            "status": "REAL_RUN_DISABLED",
            "errorCode": "REAL_RUN_DISABLED",
            "realRunsExecuted": 0,
            "externalCalls": False,
        }
    missing = _configuration_missing(env)
    if missing:
        return {
            "schemaVersion": validation["schemaVersion"],
            "mode": "real_run",
            "status": "REAL_RUN_CONFIGURATION_MISSING",
            "errorCode": "REAL_RUN_CONFIGURATION_MISSING",
            "missingConfiguration": missing,
            "realRunsExecuted": 0,
            "externalCalls": False,
        }

    traces, scores, bad_cases, result_oracle_verifications = _restore_checkpoint(
        output, validation, tasks, resume=resume
    )
    if len(scores) == len(tasks):
        return _build_real_payload(
            tasks, validation, env, traces, scores, bad_cases, result_oracle_verifications,
            status="COMPLETED"
        )

    if output is not None:
        _atomic_write_payload(
            _build_real_payload(
                tasks, validation, env, traces, scores, bad_cases, result_oracle_verifications,
                status="RUNNING"
            ),
            output,
        )

    runtime, java_port, knowledge_port = _build_real_runtime(env)
    oracle_by_case, seed_by_case = _controlled_scenario_maps()
    try:
        for index, task in enumerate(tasks, start=1):
            if task.caseId in {score.caseId for score in scores}:
                continue
            request = _build_research_task(task, index)
            try:
                oracle = oracle_by_case.get(task.caseId)
                seed = seed_by_case.get(task.caseId)
                if oracle is not None and seed is not None:
                    from evals.p2j4_controlled_provider import (
                        ControlledScenarioProvider,
                        ControlledScenarioUnavailable,
                        closed_observation_for_session,
                    )
                    from evals.p2j4_controlled_scenarios import ResultOracleVerification, verify_result_oracle

                    try:
                        session = ControlledScenarioProvider(java_port).provision(request.runId, seed)
                        from mico_agent_runtime.ports.scientific_planner import DeterministicScientificPlanner

                        # Controlled result-oracle replays evaluate a bounded
                        # real-data scenario, not free-form model sampling.
                        # This prevents a malformed planner response from
                        # spending retries before the supplied scenario is
                        # even exercised.
                        result = runtime.with_java_port(
                            session, planner=DeterministicScientificPlanner()
                        ).run(request)
                        trace = build_trace_projection(result)
                        verification = verify_result_oracle(
                            oracle, closed_observation_for_session(session, trace)
                        )
                    except ControlledScenarioUnavailable as exc:
                        trace = _scenario_unavailable_trace(task, request, exc.code)
                        verification = ResultOracleVerification(
                            caseId=task.caseId,
                            resultVerifierCode=oracle.resultVerifierCode,
                            status="SCENARIO_UNAVAILABLE",
                            failureCodes=[exc.code],
                        )
                    except BaseException as exc:
                        # Transport/configuration failure is fundamentally
                        # different from a completed search that found no
                        # matching scenario.  Preserve that distinction and
                        # never emit a Python traceback or a fake scenario.
                        trace = _scenario_unavailable_trace(
                            task, request, "CONTROLLED_SCENARIO_SUPPLY_FAILED"
                        )
                        safe_code = getattr(exc, "code", None)
                        verification = ResultOracleVerification(
                            caseId=task.caseId,
                            resultVerifierCode=oracle.resultVerifierCode,
                            status="SCENARIO_SUPPLY_FAILED",
                            failureCodes=[
                                safe_code if isinstance(safe_code, str) else "CONTROLLED_SCENARIO_SUPPLY_FAILED"
                            ],
                        )
                    result_oracle_verifications.append(verification.model_dump(mode="json"))
                else:
                    run_runtime = runtime
                    if (
                        validation["schemaVersion"].startswith("p2j4-hard-")
                        and env.get("MICO_P2J4_HARD_CONTROLLED_SCENARIOS", "").strip().lower() == "true"
                    ):
                        from evals.p2j4_hard_controlled_knowledge import (
                            HardConflictKnowledgePort,
                            HardSupportedKnowledgePort,
                            is_hard_conflict_case,
                        )

                        if is_hard_conflict_case(task.caseId):
                            run_runtime = runtime.with_knowledge_port(HardConflictKnowledgePort())
                        elif validation["schemaVersion"] == "p2j4-hard-variant-task-set-v1" and (
                            "confounder" in task.caseId
                        ):
                            run_runtime = runtime.with_knowledge_port(HardSupportedKnowledgePort())
                    result = run_runtime.run(request)
                    trace = build_trace_projection(result)
                    if oracle is not None:
                        from evals.p2j4_controlled_scenarios import (
                            verify_result_oracle,
                        )
                        from evals.p2j4_trace_assertions import (
                            closed_observation_for_trace,
                        )

                        observation = closed_observation_for_trace(task, trace)
                        verification = verify_result_oracle(oracle, observation)
                        result_oracle_verifications.append(
                            verification.model_dump(mode="json")
                        )
                score = score_trace(task, trace)
                traces.append(trace)
                scores.append(score)
                bad_cases.extend(build_bad_cases(task, score))
            except BaseException as exc:
                if output is not None:
                    _atomic_write_payload(
                        _build_real_payload(
                            tasks,
                            validation,
                            env,
                            traces,
                            scores,
                            bad_cases,
                            result_oracle_verifications,
                            status="INTERRUPTED",
                            last_error_code=type(exc).__name__,
                        ),
                        output,
                    )
                raise
            if output is not None:
                _atomic_write_payload(
                    _build_real_payload(
                        tasks,
                        validation,
                        env,
                        traces,
                        scores,
                        bad_cases,
                        result_oracle_verifications,
                        status="COMPLETED" if len(scores) == len(tasks) else "RUNNING",
                    ),
                    output,
                )
    finally:
        for resource in (java_port, knowledge_port):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
    return _build_real_payload(
        tasks, validation, env, traces, scores, bad_cases, result_oracle_verifications,
        status="COMPLETED"
    )


def _write_or_print(payload: dict[str, Any], output: Path | None) -> None:
    if output is None:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        _atomic_write_payload(payload, output)
        print(json.dumps({"output": str(output), "mode": payload.get("mode"), "status": payload.get("status", "READY")}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2-J4.1 redacted trace/eval runner")
    parser.add_argument("--task-set", type=Path, default=TASK_SET)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume an explicit matching real-run checkpoint.")
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Limit execution/dry-run to an explicit case ID; repeat for multiple cases.",
    )
    args = parser.parse_args(argv)
    try:
        validation = validate_task_set(args.task_set)
    except ValueError as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    try:
        validation = select_tasks(validation, args.case_ids)
    except ValueError as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    tasks = validation["tasks"]
    if args.real:
        try:
            payload = _real_run(tasks, validation, os.environ, output=args.output, resume=args.resume)
        except ValueError as exc:
            print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
            return 1
    elif args.dry_run:
        payload = _dry_run(tasks, validation)
    else:
        payload = {
            "schemaVersion": validation["schemaVersion"],
            "mode": "task_set_validation",
            "status": "VALID",
            "caseCount": len(tasks),
            "kindDistribution": validation["kindDistribution"],
            "caseIdsUnique": validation["caseIdsUnique"],
            "resultOracleCount": validation["resultOracleCount"],
            "scenarioRequiredOracleCount": validation["scenarioRequiredOracleCount"],
            "controlledScenarioCount": validation["controlledScenarioCount"],
            "realRunsExecuted": 0,
            "externalCalls": False,
        }
    _write_or_print(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
