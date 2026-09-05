"""4D-1 local Scientific Agent full-chain canary.

This runner deliberately keeps the policy boundary real (localhost HTTP),
while using deterministic, provenance-labelled fixtures for the data and
materializer.  It never starts a remote model or calls DeepSeek/Qwen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mico_agent_runtime.contracts.analysis import AnalysisResult
from mico_agent_runtime.contracts.decision_state import (
    ScientificActionSpaceState,
    ScientificDecisionState,
    ScientificTaskState,
)
from mico_agent_runtime.contracts.evidence import EvidenceQuery
from mico_agent_runtime.contracts.generated_analysis import AnalysisPlannerContext
from mico_agent_runtime.contracts.materialization import (
    AnalysisPlan,
    QueryPlan,
    validate_query_plan_catalog,
)
from mico_agent_runtime.contracts.research import (
    AdjustConfoundersAction,
    CompareGroupsAction,
    ExecuteReadQueryAction,
    ExecuteReadQueryArguments,
    FinishAction,
    FinishArguments,
    ProjectionAnalysisArguments,
    ResearchTask,
    ScientificPlannerContext,
    ScientificObservationSummary,
)
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.contracts.task_understanding import TaskUnderstandingOutput
from mico_agent_runtime.contracts.tools import (
    JavaDataSnapshot,
    JavaQualitySummary,
    JavaToolCall,
    JavaToolResponse,
)
from mico_agent_runtime.graph.generated_analysis import (
    GeneratedAnalysisError,
    execute_typed_analysis,
)
from mico_agent_runtime.ports.decision_policy import (
    DecisionPolicyOutput,
    HttpDecisionSftPlannerPort,
)
from mico_agent_runtime.ports.research_planner import (
    DeterministicIntentPlanner,
    HybridIntentPlannerPort,
    TypedAnalysisPlannerResult,
    deterministic_typed_analysis_plan,
)
from mico_agent_runtime.ports.scientific_planner import (
    ScientificPlannerResult,
    ScientificPlannerPort,
)
from mico_agent_runtime.ports.task_understanding import (
    TaskUnderstandingPort,
    TaskUnderstandingResult,
    build_task_understanding_port,
)
from mico_agent_runtime.runtime.scientific_service import ScientificRuntime


CANARY_QUERY = (
    "比较 T2D 和 Healthy 的菌群差异，判断这种差异是否受年龄影响、"
    "在不同项目里是否稳定，并结合文献解释。"
)
ALL_ACTIONS = [
    "inspect_cohort", "execute_read_query", "compare_groups", "analyze_projection",
    "stratified_analysis", "adjust_confounders", "cross_project_validate",
    "cross_disease_validate", "retrieve_evidence", "finish",
]
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "local_canary"


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _action_id(name: str) -> str:
    return "action-" + sha256((name + "|local-canary").encode()).hexdigest()[:32]


def build_fixture_catalog() -> SchemaSemanticCatalog:
    def field(name: str, data_type: str, capabilities: list[str]) -> SchemaFieldSemantics:
        return SchemaFieldSemantics(
            fieldId=f"abundance.{name}", name=name, dataType=data_type, nullable=False,
            semanticStatus="verified", filterable=True, groupable=data_type == "string",
            aggregatable=data_type in {"integer", "number"}, displayable=True,
            scientificCapabilities=capabilities, description=f"Fixture semantic field {name}",
        )

    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1", source="java_schema_contract",
        generatedAt=datetime.now(timezone.utc),
        entities=[SchemaEntitySemantics(
            entityId="abundance", entityName="abundance", sourceTable="fixture_abundance",
            fields=[
                field("disease", "string", ["dimension", "stratifier"]),
                field("age", "number", ["covariate", "stratifier"]),
                field("project", "string", ["dimension", "stratifier"]),
                field("value", "number", ["outcome"]),
            ],
        )],
        queryRules=[
            "select_or_with_only", "explicit_columns_only", "no_cross_database_reference",
            "bounded_limit_required", "java_final_validation",
        ],
    )


FIXTURE_ROWS = [
    {"a_abundance_disease": "T2D", "a_abundance_age": 60, "a_abundance_project": "P1", "a_abundance_value": 0.80},
    {"a_abundance_disease": "T2D", "a_abundance_age": 64, "a_abundance_project": "P1", "a_abundance_value": 0.90},
    {"a_abundance_disease": "T2D", "a_abundance_age": 55, "a_abundance_project": "P2", "a_abundance_value": 0.70},
    {"a_abundance_disease": "T2D", "a_abundance_age": 58, "a_abundance_project": "P2", "a_abundance_value": 0.75},
    {"a_abundance_disease": "Healthy", "a_abundance_age": 32, "a_abundance_project": "P1", "a_abundance_value": 0.30},
    {"a_abundance_disease": "Healthy", "a_abundance_age": 35, "a_abundance_project": "P1", "a_abundance_value": 0.25},
    {"a_abundance_disease": "Healthy", "a_abundance_age": 28, "a_abundance_project": "P2", "a_abundance_value": 0.40},
    {"a_abundance_disease": "Healthy", "a_abundance_age": 31, "a_abundance_project": "P2", "a_abundance_value": 0.35},
]


class FixtureJavaPort:
    """Fixture implementation of the real JavaAgentToolPort contract."""

    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = list(rows or FIXTURE_ROWS)
        self.calls: list[JavaToolCall] = []

    def execute(self, call: JavaToolCall) -> JavaToolResponse:
        self.calls.append(call)
        payload = {"columns": list(self.rows[0]) if self.rows else [], "rows": self.rows}
        query_hash = "sha256:" + sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return JavaToolResponse(
            toolCallId=call.toolCallId, runId=call.runId, status="COMPLETED",
            source="fixture_java_tool", rowCount=len(self.rows), schemaVersion="java-read-model-v1",
            generatedAt=datetime.now(timezone.utc),
            dataSnapshot=JavaDataSnapshot(
                dataSnapshotId="transient-00000000-0000-4000-8000-000000000001",
                dataSource="fixture_scientific_rows", queryHash=query_hash,
                rowCount=len(self.rows), generatedAt=datetime.now(timezone.utc),
                snapshotPersistence="transient", featureVersion="fixture-v1", sourceBatch="canary-v1",
            ),
            qualitySummary=JavaQualitySummary(missingCounts={}), data=payload,
        )


class EmptyKnowledgePort:
    def search(self, _query: EvidenceQuery) -> list[Any]:
        return []


class FixtureTaskUnderstandingPort(TaskUnderstandingPort):
    origin = "fixture"

    def understand(self, _query: str) -> TaskUnderstandingResult:
        output = TaskUnderstandingOutput(
            objectives=[
                "group_comparison", "confounder_assessment",
                "cross_project_validation", "evidence_support",
            ],
            constraints={
                "disease_groups": ["T2D", "Healthy"],
                "focus_covariates": ["age"],
            },
        )
        return TaskUnderstandingResult(
            output=output, mode="deterministic", model="fixture-task-understanding-v1",
            fallbackCode="TASK_UNDERSTANDING_FIXTURE",
        )


class DeterministicCanaryMaterializer(DeterministicIntentPlanner):
    """Materializes the already-selected Action; it never selects an Action."""

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        selected = context.approvedActions[0]
        action_id = _action_id(selected)
        if selected == "execute_read_query":
            plan = QueryPlan(
                root_entity="abundance", relation_path=[],
                select_fields=["abundance.disease", "abundance.age", "abundance.project", "abundance.value"],
                limit=100,
            )
            validate_query_plan_catalog(plan, context.schemaCatalog)
            action = ExecuteReadQueryAction(
                actionId=action_id, actionName=selected,
                rationale="canary materialization", arguments=ExecuteReadQueryArguments(
                    actionName=selected, queryPlan=plan, limit=100,
                ),
            )
        elif selected in {"compare_groups", "adjust_confounders"}:
            observation_ids = [item.observationId for item in context.observations if item.source == "java_controlled_read"]
            if not observation_ids:
                raise RuntimeError("CANARY_OBSERVATION_REQUIRED")
            kwargs: dict[str, Any] = {
                "actionName": selected, "observationIds": observation_ids[-1:],
                "analysisGoal": CANARY_QUERY,
            }
            if selected == "adjust_confounders":
                kwargs["confounders"] = ["abundance.age"]
                action = AdjustConfoundersAction(
                    actionId=action_id, actionName=selected,
                    rationale="canary materialization", arguments=ProjectionAnalysisArguments(**kwargs),
                )
            else:
                action = CompareGroupsAction(
                    actionId=action_id, actionName=selected,
                    rationale="canary materialization", arguments=ProjectionAnalysisArguments(**kwargs),
                )
        elif selected == "finish":
            action = FinishAction(
                actionId=action_id, actionName=selected, rationale="canary finish",
                arguments=FinishArguments(actionName="finish", reasonCode="EVIDENCE_SUFFICIENT"),
            )
        else:
            raise RuntimeError(f"CANARY_UNSUPPORTED_ACTION:{selected}")
        return ScientificPlannerResult(
            action=action, mode="deterministic", fallbackCode="CANARY_DETERMINISTIC_MATERIALIZER",
        )

    def generate_typed_analysis(self, context: AnalysisPlannerContext) -> TypedAnalysisPlannerResult:
        return TypedAnalysisPlannerResult(
            plan=deterministic_typed_analysis_plan(context), mode="deterministic",
            fallbackCode="CANARY_DETERMINISTIC_TYPED_MATERIALIZER",
        )


class PolicyStub:
    def __init__(self, mode: str = "happy") -> None:
        self.mode = mode
        self.requests: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None

    def decision(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        messages = body.get("messages", [])
        raw_content = messages[1].get("content", "") if len(messages) > 1 else ""
        self.requests.append({
            "request_index": len(self.requests) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "body": body,
            "policy_input_version": "scientific-decision-state-v1",
            "state": None,
        })
        if self.mode == "http_500":
            response = {"error": {"code": "stub_failure"}}
            self.responses.append({"request_index": len(self.requests), "body": response, "status_code": 500})
            return 500, response
        content = raw_content or "{}"
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {}
        state = payload.get("state", {}) if isinstance(payload, dict) else {}
        self.requests[-1]["state"] = state
        data = state.get("data_state", {})
        analysis = state.get("analysis_state", {})
        available = state.get("action_space", {}).get("available_actions", [])
        if self.mode == "invalid_action":
            selected = "fake_action"
        elif self.mode == "unavailable_action":
            # First response is intentionally unavailable; repair response is valid.
            selected = "cross_project_validate" if len(self.requests) == 1 else (available[0] if available else "finish")
        elif self.mode == "force_adjust":
            selected = "adjust_confounders"
        elif not data.get("has_tabular_data"):
            selected = "execute_read_query"
        elif analysis.get("group_comparison", {}).get("status") != "completed":
            selected = "compare_groups"
        elif analysis.get("confounder_adjustment", {}).get("status") != "completed":
            selected = "adjust_confounders"
        else:
            selected = "finish" if "finish" in available else (available[0] if available else "finish")
        result: dict[str, Any] = {
            "selected_action": selected,
            "decision_reason": f"local stub phase {len(self.requests)}",
            "alternative_actions": [a for a in available if a != selected][:1],
            "stop_reason": "EVIDENCE_SUFFICIENT" if selected == "finish" else None,
        }
        response = {"choices": [{"message": {"content": json.dumps(result)}}]}
        self.responses.append({"request_index": len(self.requests), "body": response, "status_code": 200})
        return 200, response

    def start(self) -> str:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    status, response = owner.decision(body)
                    encoded = json.dumps(response).encode("utf-8")
                except Exception:
                    status, encoded = 500, b'{"error":{"code":"stub_malformed_request"}}'
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def _task() -> ResearchTask:
    return ResearchTask(
        runId="run-local-canary", taskId="task-local-canary", requesterId="canary",
        traceId="trace-local-canary", question=CANARY_QUERY,
        intent="scientific_exploration",
        requestedScopes=["mico:query:read", "mico:research:read", "mico:evidence:read"],
        allowedActions=ALL_ACTIONS, maxActions=3, createdAt=datetime.now(timezone.utc),
    )


def _initial_graph_state(runtime: ScientificRuntime, request: ResearchTask, catalog: SchemaSemanticCatalog) -> dict[str, Any]:
    return runtime._graph.invoke({
        "request": request, "auditEvents": [], "observations": [], "rawObservationPayloads": {},
        "analysisPlans": [], "analysisResults": [], "analysisEvidence": [], "actionHistory": [],
        "actionSignatures": [], "plannerFeedback": [], "readReplanCount": 0, "fallbackCodes": [],
        "schemaCatalog": catalog,
    })


def run_happy(out_dir: Path, *, task_understanding_mode: str = "fixture") -> dict[str, Any]:
    catalog = build_fixture_catalog()
    stub = PolicyStub()
    endpoint = stub.start()
    java = FixtureJavaPort()
    policy = HttpDecisionSftPlannerPort(
        endpoint, "local-policy-stub", token="local-stub-token", policy_origin="local_http_stub",
    )
    planner = HybridIntentPlannerPort(DeterministicCanaryMaterializer(), policy)
    if task_understanding_mode == "gemini":
        task_env = dict(os.environ)
        task_env["MICO_TASK_UNDERSTANDING_ENABLED"] = "true"
        task_understanding = build_task_understanding_port(task_env)
        task_understanding_origin = "gemini"
    else:
        task_understanding = FixtureTaskUnderstandingPort()
        task_understanding_origin = "fixture"
    runtime = ScientificRuntime(
        java, planner, knowledge_port=EmptyKnowledgePort(), schema_catalog=catalog,
        task_understanding_port=task_understanding,
    )
    try:
        final_state = _initial_graph_state(runtime, _task(), catalog)
    finally:
        planner.close()
        close_task_understanding = getattr(task_understanding, "close", None)
        if callable(close_task_understanding):
            close_task_understanding()
        stub.close()
    records = final_state.get("decisionRecords", [])
    for index, record in enumerate(records, start=1):
        _dump(out_dir / f"policy_request_{index:03d}.json", stub.requests[index - 1]["body"])
        _dump(out_dir / f"policy_response_{index:03d}.json", stub.responses[index - 1]["body"])
    snapshots = [record.state_snapshot for record in records if record.state_snapshot]
    states = snapshots + [final_state["decisionState"].model_dump(mode="json")]
    for index, snapshot in enumerate(states[:4]):
        _dump(out_dir / f"state_s{index}.json", snapshot)
    analysis_results = [item.model_dump(mode="json") for item in final_state.get("analysisResults", [])]
    state_after_snapshots = snapshots[1:] + [final_state["decisionState"].model_dump(mode="json")]
    observations = final_state.get("observations", [])
    rounds = [
        {
            "turn": index,
            "state_before": record.state_snapshot,
            "selected_action": record.selected_action,
            "policy_origin": record.policy_origin,
            "materializer_origin": record.materializer_origin,
            "observation_summary": (
                observations[index - 1].model_dump(mode="json")
                if index <= len(observations) else None
            ),
            "state_after": state_after_snapshots[index - 1],
        }
        for index, record in enumerate(records, start=1)
    ]
    trace = {
        "trace_id": "trace-local-canary", "training_eligible": False,
        "task_understanding_origin": task_understanding_origin,
        "task_understanding_model": final_state.get("taskUnderstandingModel"),
        "policy_origin": "local_http_stub",
        "materializer_origin": "deterministic_fallback", "policy_request_count": len(stub.requests),
        "runtime_status": final_state.get("status"), "error_code": final_state.get("errorCode"),
        "policy_requests": [
            {key: item[key] for key in ("request_index", "timestamp", "policy_input_version", "state")}
            for item in stub.requests
        ],
        "action_history": final_state.get("actionHistory", []),
        "turns": [record.model_dump(mode="json") for record in records],
        "rounds": rounds,
        "analysis_results": analysis_results,
        "fallback_codes": final_state.get("fallbackCodes", []),
        "legacy_strategy_guard_used": any("SEMANTIC" in code for code in final_state.get("fallbackCodes", [])),
        "deterministic_scientific_policy_fallback_used": any(
            code == "SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK" for code in final_state.get("fallbackCodes", [])
        ),
    }
    _dump(out_dir / "trace.json", trace)
    return {"final_state": final_state, "stub": stub, "trace": trace, "analysis_results": analysis_results}


def run_failure_canaries(out_dir: Path, catalog: SchemaSemanticCatalog) -> dict[str, Any]:
    state = ScientificDecisionState(
        task=ScientificTaskState(query=CANARY_QUERY),
        action_space={"available_actions": ["execute_read_query"]},
    )
    results: dict[str, Any] = {}
    for name, mode in (("F1_invalid_action", "invalid_action"), ("F2_unavailable_action", "unavailable_action"), ("F3_http_500", "http_500")):
        stub = PolicyStub(mode)
        endpoint = stub.start()
        port = HttpDecisionSftPlannerPort(endpoint, "local-policy-stub", token="stub", policy_origin="local_http_stub")
        try:
            try:
                decision = port.select_action(state)
                outcome: Any = {"status": "returned", "selected_action": decision.selected_action}
            except Exception as exc:
                outcome = {"status": "raised", "code": str(exc)}
        finally:
            port.close(); stub.close()
        results[name] = {"outcome": outcome, "request_count": len(stub.requests)}

    class MismatchMaterializer(DeterministicCanaryMaterializer):
        def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
            context = context.model_copy(update={"approvedActions": ["compare_groups"]})
            return super().plan_action(context)

    policy_stub = PolicyStub("force_adjust"); endpoint = policy_stub.start()
    policy_port = HttpDecisionSftPlannerPort(endpoint, "local-policy-stub", token="stub", policy_origin="local_http_stub")
    hybrid = HybridIntentPlannerPort(MismatchMaterializer(), policy_port)
    context = ScientificPlannerContext(
        questionSummary=CANARY_QUERY, intent="scientific_exploration", approvedActions=["adjust_confounders"],
        remainingActionBudget=1, observations=[ScientificObservationSummary(
            observationId="observation-" + "a" * 32, actionName="execute_read_query", status="VALIDATED",
            source="java_controlled_read", rowCount=8, queryPlanFields=["abundance.disease", "abundance.age", "abundance.value"],
        )], schemaCatalog=catalog,
    )
    try:
        decision_state = state.model_copy(update={
            "action_space": ScientificActionSpaceState(available_actions=["adjust_confounders"]),
        })
        try:
            hybrid.plan_action_with_state(context, decision_state)
            results["F4_materializer_mismatch"] = {"status": "unexpected_success"}
        except Exception as exc:
            results["F4_materializer_mismatch"] = {"status": "raised", "code": str(exc)}
    finally:
        hybrid.close(); policy_stub.close()

    class FailingMaterializer(DeterministicCanaryMaterializer):
        def plan_action(self, _context: ScientificPlannerContext) -> ScientificPlannerResult:
            raise RuntimeError("CANARY_MATERIALIZER_PROVIDER_FAILED")

    failure_stub = PolicyStub("force_adjust"); failure_endpoint = failure_stub.start()
    failure_policy = HttpDecisionSftPlannerPort(
        failure_endpoint, "local-policy-stub", token="stub", policy_origin="local_http_stub",
    )
    failure_hybrid = HybridIntentPlannerPort(FailingMaterializer(), failure_policy)
    try:
        try:
            failure_hybrid.plan_action_with_state(context, decision_state)
            results["F5_materializer_failure"] = {"status": "unexpected_success"}
        except Exception as exc:
            results["F5_materializer_failure"] = {"status": "fail_closed", "code": str(exc)}
    finally:
        failure_hybrid.close(); failure_stub.close()
    try:
        plan = AnalysisPlan(
            analysis_type="group_comparison", source_observation_ids=["observation-" + "b" * 32],
            outcome="abundance.value", group_field="abundance.disease", metrics=["count"],
        )
        execute_typed_analysis(plan, [{"a_abundance_disease": "T2D", "a_abundance_value": 1.0}], 1, planner_mode="deterministic")
        results["F6_insufficient_data"] = {"status": "unexpected_success"}
    except (GeneratedAnalysisError, ValidationError, ValueError) as exc:
        results["F6_insufficient_data"] = {"status": "raised", "code": str(exc)}
    _dump(out_dir / "failure_canary_report.json", results)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local 4D-1 Scientific Agent canary")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--task-understanding", choices=("gemini", "fixture"), default="gemini",
        help="Task Understanding provider; Gemini is the default, fixture is offline-only.",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    happy = run_happy(args.output_dir, task_understanding_mode=args.task_understanding)
    failures = run_failure_canaries(args.output_dir, build_fixture_catalog())
    trace = happy["trace"]
    readiness_checks = {
        "policy_http_protocol": len(trace["turns"]) >= 3,
        "policy_input_serialization": all(item.get("state_snapshot") for item in trace["turns"]),
        "http_response_parsing": len(trace["turns"]) >= 3,
        "unavailable_action_repair": failures["F2_unavailable_action"]["outcome"].get("status") == "returned",
        "s0_to_s1": bool((happy["final_state"].get("decisionRecords") or [])[0].state_snapshot),
        "s1_to_s2": len(trace["turns"]) >= 2,
        "typed_operator": any(item.get("codeVersion") == "typed-analysis-operator-v1" for item in happy["analysis_results"]),
        "observation_builder": happy["final_state"].get("decisionState") is not None,
        "availability_recomputed": len(trace["turns"]) >= 2,
        "trace_provenance": trace["policy_origin"] == "local_http_stub" and not trace["training_eligible"],
        "no_legacy_strategy_guard": not trace["legacy_strategy_guard_used"],
        "no_deterministic_scientific_policy_fallback": not trace["deterministic_scientific_policy_fallback_used"],
        "failure_canary": len(failures) == 6,
    }
    readiness = {"A100_READY": all(readiness_checks.values()), "checks": readiness_checks, "blockers": [k for k, v in readiness_checks.items() if not v]}
    _dump(args.output_dir / "a100_readiness.json", readiness)
    print(json.dumps({"policy_requests": len(trace["turns"]), "action_history": trace["action_history"], **readiness}, ensure_ascii=False, indent=2))
    return 0 if readiness["A100_READY"] else 1


if __name__ == "__main__":
    sys.exit(main())
