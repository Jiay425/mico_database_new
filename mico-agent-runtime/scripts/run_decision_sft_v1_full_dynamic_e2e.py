"""Decision SFT v1 Full Dynamic E2E runner.

The default invocation is preparation-only and cannot contact a service.  A
live run requires the explicit ``--live`` flag and an operator-provided Qwen
SFT endpoint.  The live path reuses the production runtime contracts:

    Gemini Task Understanding -> Qwen SFT Policy -> Gemini Materializer
    -> Java/MySQL/Python/optional GraphRAG -> Observation -> next Qwen Policy

No Scientific Action, QueryPlan, AnalysisPlan, or trajectory is selected by
this script.  A controlled finish-boundary task is recorded as controlled (or
skipped) rather than being presented as a real database trace.
"""

from __future__ import annotations

# The script is also runnable directly from a checkout, so it inserts the
# repository root before importing the package under test.
# ruff: noqa: E402

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.contracts.research import ResearchTask
from mico_agent_runtime.e2e.decision_sft_v1_full_dynamic import (
    ALL_SCIENTIFIC_ACTIONS,
    DEFAULT_FREEZE_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TASK_SET_PATH,
    TaskSetValidationError,
    load_task_set,
    prepare_task_set,
    validate_task_set,
)
from mico_agent_runtime.knowledge.database_retriever import DatabaseKnowledgeSearchPort
from mico_agent_runtime.knowledge.local_retriever import LocalKnowledgeSearchPort
from mico_agent_runtime.ports.decision_policy import HttpDecisionSftPlannerPort
from mico_agent_runtime.ports.gemini_resilience import GeminiRequestBudget, GeminiResponseCache
from mico_agent_runtime.ports.java_agent import HttpJavaAgentToolPort
from mico_agent_runtime.ports.research_planner import HttpResearchPlannerPort, HybridIntentPlannerPort
from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort
from mico_agent_runtime.ports.task_understanding import build_task_understanding_port
from mico_agent_runtime.runtime.scientific_service import ScientificRuntime
from mico_agent_runtime.runtime.objective_resolution import (
    completion_semantics_from_resolution,
)
from mico_agent_runtime.transport.app import _resolve_knowledge_backend
from scripts.check_scientific_chain_services import run_preflight
from scripts.run_gemini_dynamic_canary import (
    RecordingJavaPort,
    RecordingTransport,
    _analysis_execution_audit,
    _initial_graph_state,
    _materializer_call_audit,
    _materializer_payload_records,
    _policy_records,
    _query_execution_audit,
    _save_gemini_records,
    _state_snapshot,
    _trace_rounds,
)


DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_POLICY_MODEL = "qwen3-8b-decision-sft-v1"
REPORT_VERSION = "decision-sft-v1-full-dynamic-e2e-report-v1"


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _gemini_token() -> str:
    token = (
        os.environ.get("MICO_GEMINI_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not token:
        raise RuntimeError("GEMINI_API_KEY_REQUIRED")
    return token


def _llm_provider_config() -> tuple[str, str, str, str, str]:
    """Resolve the auxiliary LLM provider for TU and materialization.

    The production policy remains Qwen.  ``gemini`` is the historical
    default; setting ``MICO_E2E_LLM_PROVIDER=deepseek`` switches only the
    Task-Understanding and Materializer roles to the configured DeepSeek
    OpenAI-compatible endpoint, with an explicit provenance label.
    """

    provider = os.environ.get("MICO_E2E_LLM_PROVIDER", "gemini").strip().lower() or "gemini"
    if provider == "deepseek":
        base_url = (
            os.environ.get("MICO_DEEPSEEK_BASE_URL", "").strip()
            or os.environ.get("MICO_RESEARCH_PLANNER_BASE_URL", "").strip()
            or "https://api.deepseek.com"
        )
        model = (
            os.environ.get("MICO_DEEPSEEK_MODEL", "").strip()
            or os.environ.get("MICO_RESEARCH_PLANNER_MODEL", "").strip()
            or "deepseek-v4-flash"
        )
        token = (
            os.environ.get("MICO_DEEPSEEK_API_KEY", "").strip()
            or os.environ.get("MICO_RESEARCH_PLANNER_TOKEN", "").strip()
        )
        if not token:
            raise RuntimeError("DEEPSEEK_API_KEY_REQUIRED")
        return provider, base_url, model, token, "deepseek_model"
    if provider != "gemini":
        raise RuntimeError(f"UNSUPPORTED_E2E_LLM_PROVIDER:{provider}")
    return "gemini", (
        os.environ.get("MICO_GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL).strip()
        or DEFAULT_GEMINI_BASE_URL
    ), (
        os.environ.get("MICO_GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        or DEFAULT_GEMINI_MODEL
    ), _gemini_token(), "gemini_canary_model"


def _qwen_health_url(base_url: str) -> str:
    """Resolve the serving root used by the no-model Qwen preflight."""

    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    elif base.endswith("/openai"):
        base = base[:-6].rstrip("/")
    return base + "/health"


def _qwen_health_gate(base_url: str, expected_model: str) -> dict[str, Any]:
    """Require a stable serving endpoint before any Gemini call is made.

    This is intentionally a GET-only gate: it does not consume a model
    generation.  Two probes catch a tunnel that accepts the first connection
    and immediately drops it, which previously caused an expensive run to
    begin with a dead policy boundary.
    """

    probes = max(1, int(os.environ.get("MICO_SFT_POLICY_HEALTH_PROBES", "2")))
    interval = max(0.0, float(os.environ.get("MICO_SFT_POLICY_HEALTH_INTERVAL", "2")))
    timeout = max(1.0, float(os.environ.get("MICO_SFT_POLICY_HEALTH_TIMEOUT", "5")))
    endpoint = _qwen_health_url(base_url)
    observations: list[dict[str, Any]] = []
    for index in range(probes):
        try:
            request = Request(endpoint, method="GET")
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator URL
                payload = json.loads(response.read().decode("utf-8"))
            model = payload.get("model") if isinstance(payload, dict) else None
            observations.append({
                "probe_index": index + 1,
                "status": int(response.status),
                "model": model,
                "model_matches": model in {None, expected_model},
            })
            if int(response.status) != 200 or model not in {None, expected_model}:
                return {
                    "status": "fail",
                    "endpoint": endpoint,
                    "error": "QWEN_HEALTH_MODEL_MISMATCH",
                    "expected_model": expected_model,
                    "probes": observations,
                }
        except (OSError, ValueError, URLError) as exc:
            observations.append({
                "probe_index": index + 1,
                "status": "error",
                "error": f"{type(exc).__name__}:{exc}",
            })
            return {
                "status": "fail",
                "endpoint": endpoint,
                "error": "QWEN_HEALTH_UNAVAILABLE",
                "expected_model": expected_model,
                "probes": observations,
            }
        if index + 1 < probes:
            time.sleep(interval)
    return {
        "status": "pass",
        "endpoint": endpoint,
        "expected_model": expected_model,
        "probes": observations,
    }


def _task_from_spec(spec: dict[str, Any], index: int) -> ResearchTask:
    scenario = str(spec["scenario"])
    intent = "literature_or_relationship" if scenario == "real_evidence_optional" else "focused_comparison"
    return ResearchTask(
        runId=f"run-decision-sft-v1-e2e-{index + 1}",
        taskId=str(spec["task_id"]),
        requesterId="decision-sft-v1-e2e",
        traceId=f"trace-{spec['task_id']}",
        question=str(spec["question"]),
        intent=intent,
        requestedScopes=list(spec["requested_scopes"]),
        allowedActions=list(ALL_SCIENTIFIC_ACTIONS),
        maxActions=int(spec["max_actions"]),
        createdAt=datetime.now(timezone.utc),
    )


def _build_optional_knowledge() -> tuple[object | None, str]:
    """Build only an explicitly configured real knowledge port.

    The E2E runner never creates a fake evidence port.  If the local/database
    index is not configured or cannot be opened, the evidence task is reported
    as skipped and the other real tasks remain runnable.
    """

    # Keep the canary on the same canonical resolver as the FastAPI runtime.
    # ``MICO_KNOWLEDGE_BACKEND`` was an obsolete runner-only knob; using it
    # here caused a configured pgvector/Neo4j backend (selected by
    # ``MICO_KNOWLEDGE_RETRIEVAL_BACKEND``) to be reported as
    # ``not_configured`` and skipped the evidence task before the port was
    # even constructed.
    backend = _resolve_knowledge_backend(os.environ)
    local_enabled = os.environ.get("MICO_LOCAL_KNOWLEDGE_ENABLED", "").strip().lower() == "true"
    try:
        if local_enabled or backend == "local":
            return LocalKnowledgeSearchPort.from_environment(), "local"
        if backend == "database":
            return DatabaseKnowledgeSearchPort.from_environment(), "database"
    except Exception as exc:
        return None, f"unavailable:{type(exc).__name__}"
    return None, "not_configured"


def _close(value: object | None) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _write_task_outputs(
    task_dir: Path,
    *,
    request: ResearchTask,
    final_state: dict[str, Any],
    tu_transport: RecordingTransport,
    policy_transport: RecordingTransport,
    materializer_transport: RecordingTransport,
    java_calls: list[dict[str, Any]],
    catalog: object | None,
    task_spec: dict[str, Any],
    preflight: dict[str, Any],
    run_error: str | None,
    auxiliary_llm_origin: str,
) -> dict[str, Any]:
    """Persist policy-safe snapshots plus provider/executor audit artifacts."""

    task_dir.mkdir(parents=True, exist_ok=True)
    _save_gemini_records(task_dir, tu_transport, "tu")
    _save_gemini_records(task_dir, policy_transport, "policy")
    _save_gemini_records(task_dir, materializer_transport, "materializer")

    records = list(final_state.get("decisionRecords", [])) if final_state else []
    snapshots = [record.state_snapshot for record in records if record.state_snapshot]
    final_snapshot = _state_snapshot(final_state.get("decisionState")) if final_state else None
    snapshot_values = (snapshots + ([final_snapshot] if final_snapshot else []))[: max(1, len(records) + 1)]
    for index, snapshot in enumerate(snapshot_values):
        if snapshot:
            _dump(task_dir / f"state_s{index}.json", snapshot)

    materializer_records = _materializer_payload_records(
        materializer_transport,
        catalog=catalog,
    )
    query_executions = _query_execution_audit(
        final_state,
        materializer_records,
        java_calls,
    ) if final_state else []
    analysis_executions = _analysis_execution_audit(
        final_state,
        materializer_records,
        query_executions,
    ) if final_state else []
    materializer_audit = _materializer_call_audit(materializer_records)
    policy_audit = _policy_records(policy_transport)
    rounds = _trace_rounds(final_state, records) if final_state else []
    objective_resolution = (
        final_state.get("objectiveResolution", {})
        if final_state and isinstance(final_state.get("objectiveResolution", {}), dict)
        else {}
    )
    decision_state = final_state.get("decisionState") if final_state else None
    remaining_objectives = (
        list(decision_state.progress.remaining_objectives)
        if decision_state is not None
        else []
    )
    completion_semantics = completion_semantics_from_resolution(
        objective_resolution,
        workflow_completed=(final_state.get("status") == "COMPLETED") if final_state else False,
        remaining_objectives=remaining_objectives,
    )
    _dump(task_dir / "objective_resolution.json", objective_resolution)
    _dump(task_dir / "completion_semantics.json", completion_semantics)
    policy_origins = [getattr(record, "policy_origin", "unknown") for record in records]
    materializer_origins = [getattr(record, "materializer_origin", "unknown") for record in records]
    trace = {
        "report_version": REPORT_VERSION,
        "trace_id": request.traceId,
        "task_id": request.taskId,
        "training_eligible": False,
        "state_origin": "real_runtime",
        "scenario": task_spec["scenario"],
        "preflight": preflight,
        "runtime_status": final_state.get("status", "FAILED") if final_state else "FAILED",
        "error_code": final_state.get("errorCode") if final_state else run_error,
        "policy_origin": "qwen_model",
        "task_understanding_origin": auxiliary_llm_origin,
        "materializer_origin": auxiliary_llm_origin,
        "policy_request_count": len(policy_transport.records),
        "task_understanding_request_count": len(tu_transport.records),
        "materializer_request_count": len(materializer_transport.records),
        "materializer_call_count": len(materializer_audit),
        "policy_requests": policy_audit,
        "materializer_requests": materializer_records,
        "materializer_call_audit": materializer_audit,
        "java_calls": java_calls,
        "action_history": final_state.get("actionHistory", []) if final_state else [],
        "turns": [record.model_dump(mode="json") for record in records],
        "rounds": rounds,
        "query_executions": query_executions,
        "analysis_executions": analysis_executions,
        # Runtime-only lifecycle evidence.  This is intentionally separate
        # from every policy ``state_snapshot`` and is required to preserve
        # blocked objectives through a terminal finish decision.
        "objective_resolution": objective_resolution,
        # Keep the Runtime field spelling available to downstream sinks that
        # consume the LangGraph state name directly; both keys carry the same
        # metadata-only payload and neither enters ScientificPolicyInput.
        "objectiveResolution": objective_resolution,
        "completion_semantics": completion_semantics,
        "remaining_objectives_invariant": completion_semantics[
            "remaining_objectives_invariant"
        ],
        "policy_origins": policy_origins,
        "materializer_origins": materializer_origins,
        "deterministic_scientific_policy_fallback_used": any(origin != "qwen_model" for origin in policy_origins),
        "deterministic_materializer_fallback_used": any(origin == "deterministic_fallback" for origin in materializer_origins),
        "legacy_strategy_guard_used": any(
            "SEMANTIC" in code for code in (final_state.get("fallbackCodes", []) if final_state else [])
        ),
        "a100_started_by_runner": False,
    }
    _dump(task_dir / "trace.json", trace)
    return trace


def _run_one_real_task(
    task_spec: dict[str, Any],
    index: int,
    *,
    output_dir: Path,
    preflight: dict[str, Any],
    catalog: object,
    java_inner: HttpJavaAgentToolPort,
    gemini_token: str,
    gemini_base_url: str,
    gemini_model: str,
    qwen_base_url: str,
    qwen_model: str,
    qwen_token: str,
    gemini_budget: GeminiRequestBudget,
    gemini_cache: GeminiResponseCache,
    knowledge_port: object | None,
    auxiliary_llm_origin: str,
) -> dict[str, Any]:
    request = _task_from_spec(task_spec, index)
    task_dir = output_dir / str(task_spec["task_id"])
    tu_transport = RecordingTransport("task_understanding")
    policy_transport = RecordingTransport("scientific_policy")
    materializer_transport = RecordingTransport("materializer")
    tu_env = dict(os.environ)
    tu_env.update({
        "MICO_TASK_UNDERSTANDING_ENABLED": "true",
        "MICO_TASK_UNDERSTANDING_BASE_URL": gemini_base_url,
        "MICO_TASK_UNDERSTANDING_MODEL": gemini_model,
        "MICO_TASK_UNDERSTANDING_TOKEN": gemini_token,
    })
    task_understanding = build_task_understanding_port(
        tu_env,
        transport=tu_transport,
        request_budget=gemini_budget,
        response_cache=gemini_cache,
    )
    policy = HttpDecisionSftPlannerPort(
        qwen_base_url,
        qwen_model,
        token=qwen_token,
        transport=policy_transport,
        policy_origin="qwen_model",
        timeout_seconds=float(os.environ.get("MICO_SFT_POLICY_TIMEOUT_SECONDS", "180")),
    )
    materializer = HttpResearchPlannerPort(
        gemini_base_url,
        gemini_model,
        gemini_token,
        transport=materializer_transport,
        materializer_origin=auxiliary_llm_origin,
        request_budget=gemini_budget,
        response_cache=gemini_cache,
        timeout_seconds=float(os.environ.get("MICO_GEMINI_MATERIALIZER_TIMEOUT_SECONDS", "300")),
    )
    planner = HybridIntentPlannerPort(
        materializer,
        policy,
        allow_deterministic_materializer_fallback=False,
    )
    java = RecordingJavaPort(java_inner)
    final_state: dict[str, Any] = {}
    run_error: str | None = None
    try:
        runtime = ScientificRuntime(
            java,
            planner,
            knowledge_port=knowledge_port,
            schema_catalog=catalog,
            task_understanding_port=task_understanding,
        )
        # Invoke the compiled production graph directly only to retain its
        # final state for the canary trace.  The graph still executes the same
        # validate -> policy -> materialize -> tool -> observe loop as run().
        final_state = _initial_graph_state(runtime, request, catalog)
    except Exception as exc:
        run_error = f"{type(exc).__name__}:{exc}"
    finally:
        _close(planner)
        _close(task_understanding)

    trace = _write_task_outputs(
        task_dir,
        request=request,
        final_state=final_state,
        tu_transport=tu_transport,
        policy_transport=policy_transport,
        materializer_transport=materializer_transport,
        java_calls=java.calls,
        catalog=catalog,
        task_spec=task_spec,
        preflight=preflight,
        run_error=run_error,
        auxiliary_llm_origin=auxiliary_llm_origin,
    )
    return {
        "task_id": request.taskId,
        "scenario": task_spec["scenario"],
        "status": trace["runtime_status"],
        "error_code": trace["error_code"],
        "policy_request_count": trace["policy_request_count"],
        "task_understanding_request_count": trace["task_understanding_request_count"],
        "materializer_request_count": trace["materializer_request_count"],
        "action_count": len(trace["action_history"]),
        "action_history": trace["action_history"],
        "training_eligible": False,
        "trace_path": str(task_dir / "trace.json"),
    }


def _controlled_finish_record(task_spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    task_dir = output_dir / str(task_spec["task_id"])
    report = {
        "report_version": REPORT_VERSION,
        "task_id": task_spec["task_id"],
        "scenario": task_spec["scenario"],
        "status": "CONTROLLED_BOUNDARY_NOT_EXECUTED",
        "reason": "finish boundary requires a validated completed State; no fabricated real Observation is injected",
        "state_origin": "controlled_boundary",
        "training_eligible": False,
        "model_calls": 0,
        "a100_started_by_runner": False,
    }
    _dump(task_dir / "controlled_boundary.json", report)
    return {
        "task_id": task_spec["task_id"],
        "scenario": task_spec["scenario"],
        "status": report["status"],
        "error_code": None,
        "policy_request_count": 0,
        "task_understanding_request_count": 0,
        "materializer_request_count": 0,
        "action_count": 0,
        "action_history": [],
        "training_eligible": False,
        "trace_path": str(task_dir / "controlled_boundary.json"),
    }


def run_live(
    task_set_path: Path,
    output_dir: Path,
    freeze_manifest_path: Path,
    *,
    max_tasks: int | None = None,
    task_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run the explicit real-provider E2E after service preflight."""

    task_set = load_task_set(task_set_path)
    validation = validate_task_set(task_set)
    freeze_path = freeze_manifest_path
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else {}
    if freeze.get("DECISION_SFT_V1_FROZEN") is not True:
        raise RuntimeError("DECISION_SFT_V1_NOT_FROZEN")

    # This is intentionally the first external operation.  No model token is
    # read and no Gemini/Qwen client is built until Java/MySQL are healthy.
    preflight = asyncio.run(run_preflight())
    output_dir.mkdir(parents=True, exist_ok=True)
    _dump(output_dir / "preflight.json", preflight)
    if preflight.get("status") != "pass":
        report = {
            "report_version": REPORT_VERSION,
            "status": "BLOCKED_PREFLIGHT",
            "error_code": "CHAIN_PREFLIGHT_FAILED",
            "preflight": preflight,
            "task_set_sha256": hashlib.sha256(task_set_path.read_bytes()).hexdigest(),
            "model_calls": 0,
            "a100_started_by_runner": False,
            "training_eligible": False,
        }
        _dump(output_dir / "report.json", report)
        return report

    qwen_base_url = os.environ.get("MICO_SFT_POLICY_BASE_URL", "").strip()
    qwen_model = os.environ.get("MICO_SFT_POLICY_MODEL", DEFAULT_POLICY_MODEL).strip()
    qwen_token = os.environ.get("MICO_SFT_POLICY_TOKEN", "").strip()
    if not qwen_base_url:
        raise RuntimeError("MICO_SFT_POLICY_BASE_URL_REQUIRED")
    # Do not spend a Gemini request until the Qwen boundary is reachable and
    # stable.  The gate is GET-only and records no model generation.
    qwen_health = _qwen_health_gate(qwen_base_url, qwen_model)
    _dump(output_dir / "qwen_health_preflight.json", qwen_health)
    if qwen_health.get("status") != "pass":
        report = {
            "report_version": REPORT_VERSION,
            "status": "BLOCKED_QWEN_PREFLIGHT",
            "error_code": str(qwen_health.get("error") or "QWEN_HEALTH_UNAVAILABLE"),
            "preflight": preflight,
            "qwen_health": qwen_health,
            "task_set_sha256": hashlib.sha256(task_set_path.read_bytes()).hexdigest(),
            "model_calls": 0,
            "a100_started_by_runner": False,
            "training_eligible": False,
        }
        _dump(output_dir / "report.json", report)
        return report

    llm_provider, gemini_base_url, gemini_model, gemini_token, auxiliary_llm_origin = _llm_provider_config()

    gemini_budget = GeminiRequestBudget(
        total_limit=int(os.environ.get("MICO_GEMINI_RUN_MAX_REQUESTS", "48")),
        role_limits={
            "task_understanding": int(os.environ.get("MICO_GEMINI_TU_MAX_REQUESTS", "5")),
            "materializer": int(os.environ.get("MICO_GEMINI_MATERIALIZER_MAX_REQUESTS", "24")),
            "policy": int(os.environ.get("MICO_GEMINI_POLICY_MAX_REQUESTS", "1")),
        },
    )
    gemini_cache = GeminiResponseCache(
        ttl_seconds=float(os.environ.get("MICO_GEMINI_RESPONSE_CACHE_TTL", "30")),
    )
    java_inner = HttpJavaAgentToolPort.from_environment()
    catalog_loader = RecordingJavaPort(java_inner)
    knowledge_port, knowledge_origin = _build_optional_knowledge()
    results: list[dict[str, Any]] = []
    try:
        first_spec = task_set["tasks"][0]
        catalog_request = _task_from_spec(first_spec, 0)
        catalog = JavaSchemaCatalogPort(catalog_loader).load(
            run_id=catalog_request.runId,
            task_id=catalog_request.taskId,
        )
        _dump(output_dir / "semantic_catalog.json", catalog)
        tasks = list(task_set["tasks"])
        if task_ids is not None:
            known_task_ids = {str(item["task_id"]) for item in tasks}
            unknown_task_ids = sorted(task_ids - known_task_ids)
            if unknown_task_ids:
                raise RuntimeError(
                    "UNKNOWN_TASK_IDS:" + ",".join(unknown_task_ids)
                )
            tasks = [item for item in tasks if str(item["task_id"]) in task_ids]
        if max_tasks is not None:
            tasks = tasks[:max(0, max_tasks)]
        for index, task_spec in enumerate(tasks):
            if task_spec["scenario"] == "controlled_finish_boundary":
                results.append(_controlled_finish_record(task_spec, output_dir))
                continue
            if task_spec["scenario"] == "real_evidence_optional" and knowledge_port is None:
                results.append({
                    "task_id": task_spec["task_id"],
                    "scenario": task_spec["scenario"],
                    "status": "SKIPPED_KNOWLEDGE_UNAVAILABLE",
                    "reason": knowledge_origin,
                    "policy_request_count": 0,
                    "task_understanding_request_count": 0,
                    "materializer_request_count": 0,
                    "action_count": 0,
                    "action_history": [],
                    "training_eligible": False,
                })
                continue
            results.append(_run_one_real_task(
                task_spec,
                index,
                output_dir=output_dir,
                preflight=preflight,
                catalog=catalog,
                java_inner=java_inner,
                gemini_token=gemini_token,
                gemini_base_url=gemini_base_url,
                gemini_model=gemini_model,
                qwen_base_url=qwen_base_url,
                qwen_model=qwen_model,
                qwen_token=qwen_token,
                gemini_budget=gemini_budget,
                gemini_cache=gemini_cache,
                knowledge_port=knowledge_port,
                auxiliary_llm_origin=auxiliary_llm_origin,
            ))
    finally:
        _close(knowledge_port)
        java_inner.close()

    report = {
        "report_version": REPORT_VERSION,
        "status": "COMPLETED",
        "task_set_path": str(task_set_path),
        "task_set_sha256": hashlib.sha256(task_set_path.read_bytes()).hexdigest(),
        "task_set_validation": validation,
        "preflight": preflight,
        "qwen_health": qwen_health,
        "llm_provider": llm_provider,
        "provider_roles": {
            **task_set["provider_roles"],
            "task_understanding": auxiliary_llm_origin,
            "materializer": auxiliary_llm_origin,
        },
        "knowledge_origin": knowledge_origin,
        "results": results,
        "model_calls": sum(
            int(item.get("task_understanding_request_count", 0))
            + int(item.get("policy_request_count", 0))
            + int(item.get("materializer_request_count", 0))
            for item in results
        ),
        "training_eligible": False,
        "a100_started_by_runner": False,
        "dpo_started": False,
        "deterministic_scientific_action_fallback_used": False,
        "deterministic_materializer_fallback_used": False,
    }
    _dump(output_dir / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decision SFT v1 Full Dynamic E2E canary")
    parser.add_argument("--task-set", type=Path, default=DEFAULT_TASK_SET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Run only the named task ID(s); may be supplied more than once",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run real providers after Java/MySQL preflight; never starts A100 itself",
    )
    args = parser.parse_args(argv)
    try:
        if args.live:
            report = run_live(
                args.task_set,
                args.output_dir,
                args.freeze_manifest,
                max_tasks=args.max_tasks,
                task_ids=set(args.task_ids) if args.task_ids else None,
            )
        else:
            report = prepare_task_set(
                args.task_set,
                args.output_dir,
                args.freeze_manifest,
            )
            report = {
                "status": report["status"],
                "FULL_E2E_READY": report["FULL_E2E_READY"],
                "READY_TO_START_A100": report["READY_TO_START_A100"],
                "task_count": report["task_audit"]["task_count"],
                "task_set_sha256": report["task_set_sha256"],
                "model_calls": report["model_calls"],
                "no_remote_contact": report["no_remote_contact"],
                "output_dir": str(args.output_dir),
            }
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    except (TaskSetValidationError, OSError, ValueError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc), "model_calls": 0}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
