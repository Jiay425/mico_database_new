from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from mico_agent_runtime.contracts.approval import ApprovalDecisionRequest, ApprovalTicketRequest
from mico_agent_runtime.contracts.evidence import EvidenceTaskRequest
from mico_agent_runtime.contracts.intent import IntentHttpResponse, IntentTaskRequest
from mico_agent_runtime.contracts.review import GraphReviewResumeCommand
from mico_agent_runtime.contracts.progress import RunProgressRegistry
from mico_agent_runtime.contracts.control import RunControlResponse
from mico_agent_runtime.ports.java_agent import (
    HttpJavaAgentToolPort,
    JavaPortConfigurationError,
)
from mico_agent_runtime.runtime.evidence_service import EvidenceRuntime
from mico_agent_runtime.runtime.intent_service import IntentRuntime
from mico_agent_runtime.runtime.persistence import (
    RuntimePersistenceConfigurationError,
    RuntimePersistenceCoordinator,
    RuntimePersistenceError,
    build_optional_runtime_persistence,
)
from mico_agent_runtime.runtime.approval import ApprovalCoordinator, ApprovalLifecycleError
from mico_agent_runtime.runtime.control import RunControlCoordinator, RunControlError
from mico_agent_runtime.ports.research_planner import (
    build_intent_planner,
    IntentPlannerPort,
)
from mico_agent_runtime.ports.evidence import (
    EvidenceSearchConfiguration,
    EvidenceSearchConfigurationError,
    HttpEvidenceSearchPort,
    UnconfiguredEvidenceSearchPort,
)
from mico_agent_runtime.knowledge.local_retriever import LocalKnowledgeSearchPort
from mico_agent_runtime.knowledge.database_retriever import DatabaseKnowledgeSearchPort
from mico_agent_runtime.ports.knowledge import KnowledgeSearchPort
from mico_agent_runtime.knowledge.synthesis import (
    GraphRagSynthesisPort,
    build_graph_rag_synthesis_port,
)

from .auth import RuntimeAuth


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"code": code, "message": message})


def _resolve_knowledge_backend(environment: Mapping[str, str]) -> str:
    """Prefer the real independent stores when they are explicitly enabled.

    The JSONL backend remains available as an explicit local development mode;
    a configured pgvector+Neo4j pair must not be silently bypassed by the old
    local default.
    """
    configured = environment.get("MICO_KNOWLEDGE_RETRIEVAL_BACKEND", "").strip().lower()
    if configured:
        return configured
    if (
        environment.get("MICO_KNOWLEDGE_VECTOR_ENABLED", "").strip().lower() == "true"
        and environment.get("MICO_KNOWLEDGE_GRAPH_ENABLED", "").strip().lower() == "true"
    ):
        return "database"
    return "local"


def create_app(
    env: Mapping[str, str] | None = None,
    evidence_runtime: EvidenceRuntime | None = None,
    intent_runtime: IntentRuntime | None = None,
    intent_planner: IntentPlannerPort | None = None,
    knowledge_port: KnowledgeSearchPort | None = None,
    synthesis_port: GraphRagSynthesisPort | None = None,
    progress_registry: RunProgressRegistry | None = None,
    runtime_persistence: RuntimePersistenceCoordinator | None = None,
    approval_coordinator: ApprovalCoordinator | None = None,
    run_control_coordinator: RunControlCoordinator | None = None,
) -> FastAPI:
    """Create the app without opening a socket or constructing a Java client."""

    app = FastAPI(title="Mico Agent Runtime Internal API", version="p2a")
    auth = RuntimeAuth.from_environment(env)
    environment = os.environ if env is None else env
    approval_enabled = environment.get("MICO_RUNTIME_APPROVAL_ENABLED", "").strip().lower() == "true"
    approval_decision_auth = RuntimeAuth(environment.get("MICO_RUNTIME_APPROVAL_DECISION_TOKEN"))
    approval_decision_principal = environment.get("MICO_RUNTIME_APPROVAL_DECISION_PRINCIPAL", "").strip()
    graph_review_interrupt_enabled = environment.get(
        "MICO_RUNTIME_GRAPH_REVIEW_INTERRUPT_ENABLED", ""
    ).strip().lower() == "true"
    evidence_runtime_holder: dict[str, EvidenceRuntime | None] = {"runtime": evidence_runtime}
    intent_runtime_holder: dict[str, IntentRuntime | None] = {"runtime": intent_runtime}
    knowledge_port_holder: dict[str, KnowledgeSearchPort | None] = {"port": knowledge_port}
    knowledge_backend = _resolve_knowledge_backend(environment)
    if knowledge_port_holder["port"] is None and (
        environment.get("MICO_LOCAL_KNOWLEDGE_ENABLED", "").strip().lower() == "true"
        or knowledge_backend == "database"
    ):
        try:
            if knowledge_backend == "database":
                knowledge_port_holder["port"] = DatabaseKnowledgeSearchPort.from_environment(env)
            elif knowledge_backend == "local":
                knowledge_port_holder["port"] = LocalKnowledgeSearchPort.from_environment(env)
            else:
                raise ValueError("KNOWLEDGE_BACKEND_UNSUPPORTED")
        except Exception:
            # The route remains available but fails closed with a stable source
            # configuration error if the knowledge workflow is selected.
            knowledge_port_holder["port"] = None
    progress_holder = progress_registry or RunProgressRegistry()
    persistence_holder: dict[str, RuntimePersistenceCoordinator | None] = {
        "runtime": runtime_persistence,
    }
    control_holder: dict[str, RunControlCoordinator | None] = {
        "runtime": run_control_coordinator,
    }

    def build_evidence_runtime() -> EvidenceRuntime:
        """Build local full-text retrieval when explicitly enabled.

        The local index is preferred over external metadata search. An enabled
        but invalid local index fails closed instead of silently switching to a
        different evidence source.
        """
        if environment.get("MICO_LOCAL_KNOWLEDGE_ENABLED", "").strip().lower() == "true":
            if knowledge_backend == "database":
                search_port = DatabaseKnowledgeSearchPort.from_environment(env)
            elif knowledge_backend == "local":
                search_port = LocalKnowledgeSearchPort.from_environment(env)
            else:
                raise EvidenceSearchConfigurationError("KNOWLEDGE_BACKEND_UNSUPPORTED")
            return EvidenceRuntime(search_port)
        try:
            configuration = EvidenceSearchConfiguration.from_environment(env)
            search_port = HttpEvidenceSearchPort(configuration)
        except EvidenceSearchConfigurationError:
            search_port = UnconfiguredEvidenceSearchPort()
        return EvidenceRuntime(search_port)

    persistence_error: str | None = None
    if runtime_persistence is None:
        try:
            persistence_holder["runtime"] = build_optional_runtime_persistence(env)
        except RuntimePersistenceConfigurationError as exc:
            persistence_error = exc.code
    if control_holder["runtime"] is None and persistence_holder["runtime"] is not None:
        control_holder["runtime"] = persistence_holder["runtime"].control_coordinator()
    if approval_coordinator is None and persistence_holder["runtime"] is not None:
        approval_coordinator = ApprovalCoordinator(
            persistence_holder["runtime"].store,
            control_holder["runtime"],
        )

    async def close_runtime_persistence() -> None:
        if persistence_holder["runtime"] is not None:
            await persistence_holder["runtime"].close()
        if evidence_runtime_holder["runtime"] is not None:
            await evidence_runtime_holder["runtime"].close()
        close_knowledge = getattr(knowledge_port_holder["port"], "close", None)
        if callable(close_knowledge):
            close_knowledge()
    app.add_event_handler("shutdown", close_runtime_persistence)

    async def persistence_begin_or_error(value: Any) -> JSONResponse | None:
        if persistence_error is not None:
            return _error(503, "RUNTIME_PERSISTENCE_NOT_CONFIGURED",
                          "Runtime persistence is not configured")
        if persistence_holder["runtime"] is not None:
            try:
                await persistence_holder["runtime"].begin(value)
            except RuntimePersistenceError:
                return _error(503, "RUNTIME_PERSISTENCE_UNAVAILABLE",
                              "Runtime persistence is unavailable")
        return None

    async def persistence_finish_or_error(value: Any) -> JSONResponse | None:
        if persistence_holder["runtime"] is not None:
            try:
                await persistence_holder["runtime"].finish(value)
            except RuntimePersistenceError:
                return _error(503, "RUNTIME_PERSISTENCE_FAILED",
                              "Runtime persistence failed")
        return None

    async def persistence_fail(value: Any, code: str) -> None:
        if persistence_holder["runtime"] is not None:
            await persistence_holder["runtime"].fail(value, code)

    async def progress_projection(run_id: str):
        """Prefer live progress, then use only the persisted safe projection."""
        live = progress_holder.get(run_id)
        if live is not None:
            return live, None
        coordinator = persistence_holder["runtime"]
        loader = getattr(coordinator, "load_progress", None) if coordinator is not None else None
        if loader is None:
            return None, None
        try:
            return await loader(run_id), None
        except RuntimePersistenceError:
            return None, _error(503, "RUNTIME_PERSISTENCE_UNAVAILABLE",
                                 "Runtime progress is temporarily unavailable")

    def ensure_intent_runtime() -> IntentRuntime:
        """Build the intent graph once, with the encrypted saver when enabled."""
        if intent_runtime_holder["runtime"] is None:
            java_port = HttpJavaAgentToolPort.from_environment(env)
            planner = intent_planner or build_intent_planner(env)
            checkpoint_saver = (
                persistence_holder["runtime"].checkpoint_saver()
                if persistence_holder["runtime"] is not None
                else None
            )
            intent_runtime_holder["runtime"] = IntentRuntime(
                java_port,
                planner,
                checkpoint_saver=checkpoint_saver,
                knowledge_port=knowledge_port_holder["port"],
                synthesis_port=synthesis_port or build_graph_rag_synthesis_port(env),
                interrupt_on_review=graph_review_interrupt_enabled,
            )
        return intent_runtime_holder["runtime"]

    async def resume_intent_internal(run_id: str) -> RunControlResponse:
        coordinator = persistence_holder["runtime"]
        if coordinator is None:
            raise RunControlError("RUNTIME_RECOVERY_NOT_ENABLED")
        request_value = await coordinator.load_intent_recovery_request(run_id)
        runtime_value = ensure_intent_runtime()
        async_resume = getattr(runtime_value, "resume_async", None)
        if callable(async_resume):
            result = await async_resume(request_value)
        else:
            result = await asyncio.to_thread(runtime_value.resume, request_value)
        await coordinator.finish(result)
        status = result.status if result.status in {
            "QUEUED", "RUNNING", "WAITING_APPROVAL", "WAITING_JOB",
            "COMPLETED", "FAILED", "CANCELLED",
        } else "FAILED"
        return RunControlResponse(
            runId=result.runId,
            taskId=result.taskId,
            status=status,
            controlCode="RUN_RESUMED",
        )

    if control_holder["runtime"] is not None:
        binder = getattr(control_holder["runtime"], "bind_resume_handler", None)
        if callable(binder):
            binder(resume_intent_internal)

    def approval_not_configured(code: str = "APPROVAL_WORKFLOW_NOT_CONFIGURED") -> JSONResponse:
        return _error(503, code, "The approval workflow is not configured")

    def approval_auth_error(authorization: str | None) -> JSONResponse | None:
        if not approval_decision_auth.enabled or not approval_decision_principal:
            return approval_not_configured("APPROVAL_DECISION_NOT_CONFIGURED")
        if not approval_decision_auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Approval decision authentication is required")
        return None

    @app.post("/internal/runtime/intent-runs")
    async def create_intent_run(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Internal runtime authentication is required")
        if not request.headers.get("content-type", "").lower().startswith("application/json"):
            return _error(400, "INVALID_REQUEST", "The request must be JSON")
        try:
            payload: Any = await request.json()
            intent_request = IntentTaskRequest.model_validate(payload)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
            return _error(400, "INVALID_REQUEST", "The request does not match the closed intent contract")

        progress_holder.begin(intent_request.runId, intent_request.taskId)

        persistence_error_response = await persistence_begin_or_error(intent_request)
        if persistence_error_response is not None:
            progress_holder.fail(intent_request.runId, intent_request.taskId,
                                 "RUNTIME_PERSISTENCE_NOT_CONFIGURED")
            return persistence_error_response

        if intent_runtime_holder["runtime"] is None:
            try:
                ensure_intent_runtime()
            except JavaPortConfigurationError:
                progress_holder.fail(intent_request.runId, intent_request.taskId, "JAVA_PORT_DISABLED")
                return _error(503, "JAVA_PORT_DISABLED", "The Java Agent Tool port is not configured")
            except Exception:
                progress_holder.fail(intent_request.runId, intent_request.taskId, "RUNTIME_INITIALIZATION_FAILED")
                return _error(500, "RUNTIME_INITIALIZATION_FAILED", "The internal runtime could not start")

        try:
            runtime_value = intent_runtime_holder["runtime"]
            async_callable = getattr(runtime_value, "run_async", None)
            if callable(async_callable):
                result = await async_callable(
                    intent_request,
                    progress_callback=progress_holder.record_event,
                )
            else:
                run_callable = runtime_value.run
                if "progress_callback" in inspect.signature(run_callable).parameters:
                    result = await asyncio.to_thread(
                        run_callable,
                        intent_request,
                        progress_callback=progress_holder.record_event,
                    )
                else:
                    # Preserve compatibility with injected P2-A test/runtime
                    # adapters that implement the original one-argument port.
                    result = await asyncio.to_thread(run_callable, intent_request)
        except Exception:
            await persistence_fail(intent_request, "RUNTIME_EXECUTION_FAILED")
            progress_holder.fail(intent_request.runId, intent_request.taskId, "RUNTIME_EXECUTION_FAILED")
            return _error(500, "RUNTIME_EXECUTION_FAILED", "The internal intent execution failed")
        persistence_error_response = await persistence_finish_or_error(result)
        if persistence_error_response is not None:
            progress_holder.fail(intent_request.runId, intent_request.taskId,
                                 "RUNTIME_PERSISTENCE_FAILED")
            return persistence_error_response
        progress_holder.complete(result)
        external_result = IntentHttpResponse(
            status=result.status,
            workflow=result.workflow,
            errorCode=result.errorCode,
            report=result.report,
        )
        return JSONResponse(status_code=200, content=external_result.model_dump(mode="json", exclude_none=True))

    @app.get("/internal/runtime/intent-runs/{run_id}/progress")
    async def get_intent_progress(
        run_id: str,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Internal runtime authentication is required")
        progress, persistence_response = await progress_projection(run_id)
        if persistence_response is not None:
            return persistence_response
        if progress is None:
            return _error(404, "RUN_NOT_FOUND", "The requested run was not found")
        return JSONResponse(status_code=200, content=progress.model_dump(mode="json", exclude_none=True))

    @app.post("/internal/runtime/intent-runs/{run_id}/cancel", response_model=None)
    async def cancel_intent_run(
        run_id: str,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Internal runtime authentication is required")
        if control_holder["runtime"] is None:
            return _error(503, "RUN_CONTROL_NOT_CONFIGURED", "Run control is not configured")
        try:
            response = await control_holder["runtime"].cancel(run_id)
            return JSONResponse(status_code=200, content=response.model_dump(mode="json"))
        except RunControlError as exc:
            status = 404 if exc.code == "RUN_NOT_FOUND" else 409 if exc.code in {
                "RUN_ALREADY_TERMINAL", "RUN_CANCEL_NOT_SUPPORTED_FOR_ACTIVE_RUN",
            } else 503
            return _error(status, exc.code, "The run could not be cancelled")

    @app.post("/internal/runtime/intent-runs/{run_id}/resume", response_model=None)
    async def resume_intent_run(
        run_id: str,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Internal runtime authentication is required")
        if control_holder["runtime"] is None:
            return _error(503, "RUN_CONTROL_NOT_CONFIGURED", "Run control is not configured")
        try:
            response = await control_holder["runtime"].resume(run_id)
            return JSONResponse(status_code=200, content=response.model_dump(mode="json"))
        except RunControlError as exc:
            status = 404 if exc.code == "RUN_NOT_FOUND" else 409 if exc.code == "RUNTIME_RESUME_NOT_IMPLEMENTED" else 503
            return _error(status, exc.code, "Run recovery is not available")

    @app.post("/internal/runtime/intent-runs/{run_id}/review", response_model=None)
    async def review_intent_run(
        run_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        """Resume only an interrupted graph with a closed review decision."""

        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Internal runtime authentication is required")
        coordinator = persistence_holder["runtime"]
        if coordinator is None:
            return _error(503, "RUNTIME_RECOVERY_NOT_ENABLED", "Run recovery is not available")
        if not request.headers.get("content-type", "").lower().startswith("application/json"):
            return _error(400, "INVALID_REQUEST", "The review decision must be JSON")
        try:
            decision = GraphReviewResumeCommand.model_validate(await request.json())
            recovered_request = await coordinator.load_intent_recovery_request(run_id)
            runtime_value = ensure_intent_runtime()
            result = await runtime_value.resume_async(recovered_request, decision)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
            return _error(400, "GRAPH_REVIEW_DECISION_INVALID", "The review decision is invalid")
        except RuntimePersistenceError:
            return _error(503, "RUNTIME_RECOVERY_UNAVAILABLE", "Run recovery is not available")
        except Exception:
            return _error(503, "GRAPH_REVIEW_RESUME_FAILED", "The graph review could not be resumed")
        persistence_error_response = await persistence_finish_or_error(result)
        if persistence_error_response is not None:
            return persistence_error_response
        progress_holder.complete(result)
        external_result = IntentHttpResponse(
            status=result.status,
            workflow=result.workflow,
            errorCode=result.errorCode,
            report=result.report,
        )
        return JSONResponse(status_code=200, content=external_result.model_dump(mode="json", exclude_none=True))

    @app.post("/internal/runtime/intent-runs/{run_id}/pause", response_model=None)
    async def pause_intent_run(
        run_id: str,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Internal runtime authentication is required")
        if control_holder["runtime"] is None:
            return _error(503, "RUN_CONTROL_NOT_CONFIGURED", "Run control is not configured")
        try:
            response = await control_holder["runtime"].pause(run_id)
            return JSONResponse(status_code=200, content=response.model_dump(mode="json"))
        except RunControlError as exc:
            status = 404 if exc.code == "RUN_NOT_FOUND" else 409 if exc.code in {
                "RUN_ALREADY_TERMINAL", "RUN_PAUSE_NOT_IMPLEMENTED",
            } else 503
            return _error(status, exc.code, "Run pause is not available")

    @app.post("/internal/runtime/approvals", response_model=None)
    async def create_approval_ticket(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Internal runtime authentication is required")
        if not approval_enabled or approval_coordinator is None:
            return approval_not_configured()
        if not request.headers.get("content-type", "").lower().startswith("application/json"):
            return _error(400, "INVALID_REQUEST", "The request must be JSON")
        try:
            value = ApprovalTicketRequest.model_validate(await request.json())
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
            return _error(400, "INVALID_REQUEST", "The request does not match the closed approval contract")
        try:
            ticket = await approval_coordinator.request(value)
        except ApprovalLifecycleError as exc:
            if exc.code == "APPROVAL_NOT_REQUIRED":
                return _error(409, exc.code, "This operation does not require human approval")
            return _error(503, exc.code, "The approval ticket could not be created")
        return JSONResponse(status_code=201, content=ticket.model_dump(mode="json"))

    @app.get("/internal/runtime/approvals/{approval_id}", response_model=None)
    async def get_approval_ticket(
        approval_id: str,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Internal runtime authentication is required")
        if not approval_enabled or approval_coordinator is None:
            return approval_not_configured()
        try:
            ticket = await approval_coordinator.get(approval_id)
        except ApprovalLifecycleError:
            return _error(503, "APPROVAL_PERSISTENCE_FAILED", "The approval ticket is unavailable")
        if ticket is None:
            return _error(404, "APPROVAL_NOT_FOUND", "The approval ticket was not found")
        return JSONResponse(status_code=200, content=ticket.model_dump(mode="json"))

    @app.post("/internal/runtime/approvals/{approval_id}/decision", response_model=None)
    async def decide_approval_ticket(
        approval_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not approval_enabled or approval_coordinator is None:
            return approval_not_configured("APPROVAL_DECISION_NOT_CONFIGURED")
        auth_error = approval_auth_error(authorization)
        if auth_error is not None:
            return auth_error
        if not request.headers.get("content-type", "").lower().startswith("application/json"):
            return _error(400, "INVALID_REQUEST", "The request must be JSON")
        try:
            value = ApprovalDecisionRequest.model_validate(await request.json())
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
            return _error(400, "INVALID_REQUEST", "The request does not match the closed approval decision contract")
        try:
            ticket = await approval_coordinator.decide(
                approval_id, value, approval_decision_principal
            )
        except ApprovalLifecycleError as exc:
            status = 404 if exc.code == "APPROVAL_NOT_FOUND" else 409 if exc.code in {
                "APPROVAL_ALREADY_TERMINAL", "APPROVAL_RESUME_NOT_AVAILABLE"
            } else 503
            return _error(status, exc.code, "The approval decision could not be applied")
        return JSONResponse(status_code=200, content=ticket.model_dump(mode="json"))

    @app.get("/internal/runtime/intent-runs/{run_id}/events", response_model=None)
    async def stream_intent_events(
        run_id: str,
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse | JSONResponse:
        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Internal runtime authentication is required")
        progress, persistence_response = await progress_projection(run_id)
        if persistence_response is not None:
            return persistence_response
        if progress is None:
            return _error(404, "RUN_NOT_FOUND", "The requested run was not found")

        async def event_stream():
            for event in progress.events:
                yield f"event: agent-progress\ndata: {event.model_dump_json(exclude_none=True)}\n\n"
            yield f"event: agent-terminal\ndata: {progress.model_dump_json(exclude_none=True)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/internal/runtime/evidence-runs")
    async def create_evidence_run(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        if not auth.enabled:
            return _error(503, "RUNTIME_DISABLED", "The internal runtime is disabled")
        if not auth.matches(authorization):
            return _error(401, "UNAUTHORIZED", "Internal runtime authentication is required")
        if not request.headers.get("content-type", "").lower().startswith("application/json"):
            return _error(400, "INVALID_REQUEST", "The request must be JSON")
        try:
            payload: Any = await request.json()
            evidence_request = EvidenceTaskRequest.model_validate(payload)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
            return _error(400, "INVALID_REQUEST", "The request does not match the closed evidence contract")

        persistence_error_response = await persistence_begin_or_error(evidence_request)
        if persistence_error_response is not None:
            return persistence_error_response

        if evidence_runtime_holder["runtime"] is None:
            try:
                evidence_runtime_holder["runtime"] = build_evidence_runtime()
            except Exception:
                await persistence_fail(evidence_request, "EVIDENCE_SOURCE_NOT_CONFIGURED")
                return _error(503, "EVIDENCE_SOURCE_NOT_CONFIGURED", "The evidence source is not configured")

        try:
            result = evidence_runtime_holder["runtime"].run(evidence_request)
        except Exception:
            await persistence_fail(evidence_request, "EVIDENCE_RUNTIME_FAILED")
            return _error(500, "EVIDENCE_RUNTIME_FAILED", "The evidence review could not be completed")
        persistence_error_response = await persistence_finish_or_error(result)
        if persistence_error_response is not None:
            return persistence_error_response
        return JSONResponse(
            status_code=200,
            content=result.model_dump(mode="json", exclude_none=True),
        )

    return app
