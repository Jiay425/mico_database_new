from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from pydantic import TypeAdapter

from mico_agent_runtime.contracts.audit import AuditEvent
from mico_agent_runtime.contracts.base import ClosedModel
from mico_agent_runtime.contracts.evidence_report import EvidenceReviewReport
from mico_agent_runtime.contracts.research import ResearchExplorationReport
from mico_agent_runtime.contracts.progress import RuntimeProgressEvent, RuntimeProgressResponse
from mico_agent_runtime.storage.configuration import (
    RuntimeStorageConfiguration,
    RuntimeStorageConfigurationError,
)
from mico_agent_runtime.storage.crypto import (
    RuntimeStateCipher,
    RuntimeStateConfigurationError,
    StateBindingContext,
)
from mico_agent_runtime.storage.models import (
    AgentArtifactRecord,
    AgentRunRecord,
    AgentStepRecord,
    RuntimeStatus,
    SnapshotMetadata,
    ToolAuditRecord,
    UnifiedEvidencePersistenceProjection,
)
from mico_agent_runtime.storage.mysql_store import MySqlRuntimeStore
from mico_agent_runtime.storage.ports import RuntimeStore, RuntimeStoreError
from mico_agent_runtime.storage.types import (
    JavaTransientSnapshotId,
    RunId,
    TaskId,
    TraceId,
)


class RuntimePersistenceConfigurationError(RuntimeError):
    """Safe configuration failure; never contains URL or key material."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RuntimePersistenceError(RuntimeError):
    """Safe persistence failure; never contains SQL, DSN, or payload details."""

    def __init__(self, code: str = "RUNTIME_PERSISTENCE_FAILED") -> None:
        super().__init__(code)
        self.code = code


_ID_ADAPTERS = {
    "runId": TypeAdapter(RunId),
    "taskId": TypeAdapter(TaskId),
    "traceId": TypeAdapter(TraceId),
}
_CONTROL_CODE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_PERSISTED_NODES = {
    "validate_task",
    "policy_gate",
    "build_tool_plan",
    "execute_tool",
    "summarize_evidence",
    "human_review",
    "terminal",
}


def build_optional_runtime_persistence(
    env: Mapping[str, str] | None = None,
) -> "RuntimePersistenceCoordinator | None":
    source = os.environ if env is None else env
    enabled = source.get("MICO_AGENT_RUNTIME_MYSQL_ENABLED", "").strip().lower()
    if enabled != "true":
        return None
    try:
        configuration = RuntimeStorageConfiguration.from_environment(source)
        return RuntimePersistenceCoordinator(
            MySqlRuntimeStore(configuration), configuration.state_cipher
        )
    except (RuntimeStorageConfigurationError, RuntimeStateConfigurationError) as exc:
        raise RuntimePersistenceConfigurationError(str(exc)) from None
    except Exception:
        raise RuntimePersistenceConfigurationError("RUNTIME_PERSISTENCE_CONFIGURATION_INVALID") from None


class RuntimePersistenceCoordinator:
    """Connects a graph result to the independent Store without raw payload columns.

    The coordinator is dependency-injected in tests and is not constructed when
    the explicit MySQL enable switch is absent. It stores encrypted request
    state and only fixed, validated step/audit/snapshot metadata.
    """

    def __init__(self, store: RuntimeStore, cipher: RuntimeStateCipher) -> None:
        self._store = store
        self._cipher = cipher

    @property
    def store(self) -> RuntimeStore:
        """Expose only the typed store port for Runtime-owned coordinators."""
        return self._store

    async def begin(self, request: Any) -> None:
        run_id, task_id, trace_id = self._identities(request)
        now = datetime.now(timezone.utc)
        context = StateBindingContext(
            runId=run_id,
            taskId=task_id,
            traceId=trace_id,
            dataContractVersion="v1",
        )
        try:
            state = {"request": _safe_model_dump(request)}
            envelope = self._cipher.encrypt(state, context)
            record = AgentRunRecord(
                dataContractVersion="v1",
                runId=run_id,
                taskId=task_id,
                traceId=trace_id,
                status=RuntimeStatus.QUEUED,
                createdAt=now,
                updatedAt=now,
                encryptedStatePayload=envelope,
                encryptionKeyId=self._cipher.key_id,
            )
            await self._store.create_run(record)
            await self._store.mark_run_status(run_id, RuntimeStatus.RUNNING, updated_at=now)
        except RuntimeStoreError:
            raise RuntimePersistenceError("RUNTIME_PERSISTENCE_UNAVAILABLE") from None
        except Exception:
            raise RuntimePersistenceError() from None

    async def finish(self, result: Any) -> None:
        run_id, _, _ = self._identities(result)
        result_status = getattr(result, "status", None)
        status = {
            "COMPLETED": RuntimeStatus.COMPLETED,
            "WAITING_APPROVAL": RuntimeStatus.WAITING_APPROVAL,
            "WAITING_JOB": RuntimeStatus.WAITING_JOB,
            "CANCELLED": RuntimeStatus.CANCELLED,
        }.get(result_status, RuntimeStatus.FAILED)
        failure_code = _safe_control_code(getattr(result, "errorCode", None))
        events = getattr(result, "auditEvents", []) or []
        last_step_id: str | None = None
        try:
            report = getattr(result, "report", None)
            snapshot = _snapshot_from_report(report)
            evidence_projection = _unified_evidence_projection(report)
            for sequence, event in enumerate(events[:1000], start=1):
                if not isinstance(event, AuditEvent):
                    continue
                step_id = _opaque_id("step-", run_id, str(sequence), event.node)
                last_step_id = step_id
                event_projection = _projection_for_event(event, evidence_projection)
                await self._store.append_step(
                    _step_from_event(step_id, event, snapshot, event_projection)
                )
                if event.toolName and event.toolCallId:
                    audit_id = _opaque_id("audit-", run_id, str(sequence), event.toolCallId)
                    await self._store.append_tool_audit(
                        _audit_from_event(audit_id, event, snapshot, event_projection)
                    )
            evidence_artifact = _evidence_artifact(run_id, report, evidence_projection)
            if evidence_artifact is not None:
                await self._store.append_artifact(evidence_artifact)
            await self._store.mark_run_status(
                run_id,
                status,
                failure_code=failure_code,
                current_successful_step_id=last_step_id,
                updated_at=datetime.now(timezone.utc),
            )
        except RuntimeStoreError:
            raise RuntimePersistenceError("RUNTIME_PERSISTENCE_FAILED") from None
        except Exception:
            raise RuntimePersistenceError() from None

    async def checkpoint(self, value: Any) -> None:
        """Persist one encrypted state envelope without plaintext columns.

        This is a storage primitive only. It does not compile or resume a
        LangGraph checkpoint and therefore does not change the Store's
        ``recoverable=False`` contract.
        """
        run_id, task_id, trace_id = self._identities(value)
        context = StateBindingContext(
            runId=run_id,
            taskId=task_id,
            traceId=trace_id,
            dataContractVersion="v1",
        )
        try:
            state = {
                "checkpointVersion": "v1",
                "state": _safe_checkpoint_value(value),
            }
            envelope = self._cipher.encrypt(state, context)
            await self._store.update_encrypted_state(
                run_id,
                envelope,
                updated_at=datetime.now(timezone.utc),
            )
        except RuntimeStoreError:
            raise RuntimePersistenceError("RUNTIME_PERSISTENCE_FAILED") from None
        except Exception:
            raise RuntimePersistenceError("RUNTIME_CHECKPOINT_FAILED") from None

    async def load_progress(self, run_id: str) -> RuntimeProgressResponse | None:
        """Rebuild a safe progress projection from persisted typed metadata.

        This is deliberately a terminal/progress projection only.  It does not
        decrypt or return the request envelope and it does not claim that a
        LangGraph checkpoint can be resumed.
        """
        try:
            typed_run_id = TypeAdapter(RunId).validate_python(run_id)
            run = await self._store.load_run(typed_run_id)
            if run is None:
                return None
            steps = await self._store.list_steps(typed_run_id)
            events: list[RuntimeProgressEvent] = []
            for sequence, step in enumerate(steps[:1000], start=1):
                snapshot = step.snapshotMetadata
                events.append(RuntimeProgressEvent(
                    sequence=sequence,
                    node=step.nodeName,
                    status=(
                        "COMPLETED" if step.status == RuntimeStatus.COMPLETED
                        else "WAITING_APPROVAL" if step.status == RuntimeStatus.WAITING_APPROVAL
                        else "FAILED"
                    ),
                    errorCode=_safe_control_code(step.errorCode),
                    dataSnapshotId=snapshot.dataSnapshotId if snapshot else None,
                    snapshotPersistence=snapshot.snapshotPersistence if snapshot else None,
                    occurredAt=step.endedAt or step.startedAt,
                ))
            return RuntimeProgressResponse(
                runId=run.runId,
                taskId=run.taskId,
                workflow=None,
                status=run.status.value,
                currentNode=events[-1].node if events else None,
                errorCode=_safe_control_code(run.failureCode),
                events=events,
                updatedAt=run.updatedAt,
            )
        except RuntimeStoreError:
            raise RuntimePersistenceError("RUNTIME_PERSISTENCE_UNAVAILABLE") from None
        except Exception:
            raise RuntimePersistenceError("RUNTIME_PERSISTENCE_FAILED") from None

    async def fail(self, request: Any, code: str) -> None:
        run_id, _, _ = self._identities(request)
        try:
            await self._store.mark_run_status(
                run_id,
                RuntimeStatus.FAILED,
                failure_code=_safe_control_code(code) or "RUNTIME_EXECUTION_FAILED",
                updated_at=datetime.now(timezone.utc),
            )
        except Exception:
            # The original safe error is retained; persistence details never
            # replace it or escape to the caller.
            return

    async def close(self) -> None:
        close = getattr(self._store, "close", None)
        if close is not None:
            await close()

    def control_coordinator(self):
        """Return the explicit run-control adapter for this injected Store."""
        from mico_agent_runtime.runtime.control import RunControlCoordinator

        return RunControlCoordinator(
            self._store,
            checkpoint_saver=self.checkpoint_saver(),
        )

    def checkpoint_saver(self):
        """Return the real encrypted LangGraph saver for this run store.

        The saver is only constructed after the persistence coordinator has
        successfully loaded the Runtime encryption configuration.  It does
        not expose a SQL or JSON persistence interface to graph code.
        """
        from mico_agent_runtime.runtime.checkpoint import EncryptedLangGraphCheckpointSaver

        return EncryptedLangGraphCheckpointSaver(self._store, self._cipher)

    async def load_intent_recovery_request(self, run_id: str):
        """Recover only the typed intent request for an internal resume.

        The request is decrypted inside the Runtime process and immediately
        validated into the closed intent contract.  It is never returned by an
        HTTP projection or written to an ordinary persistence column.
        """
        from mico_agent_runtime.contracts.intent import IntentTaskRequest
        from mico_agent_runtime.runtime.checkpoint import RuntimeCheckpointError

        try:
            typed_run_id = TypeAdapter(RunId).validate_python(run_id)
            run = await self._store.load_run(typed_run_id)
            if run is None:
                raise RuntimePersistenceError("RUN_NOT_FOUND")
            context = StateBindingContext(
                runId=run.runId,
                taskId=run.taskId,
                traceId=run.traceId,
                dataContractVersion="v1",
            )
            value = self._cipher.decrypt(run.encryptedStatePayload, context)
            raw_request = value.get("request")
            if raw_request is None and value.get("checkpointVersion") == "langgraph-v1":
                saver = self.checkpoint_saver()
                checkpoint = await saver.aget_tuple({"configurable": {
                    "thread_id": run.runId,
                    "runtime_task_id": run.taskId,
                    "runtime_trace_id": run.traceId,
                }})
                raw_request = (
                    checkpoint.checkpoint.get("channel_values", {}).get("request")
                    if checkpoint is not None
                    else None
                )
            if raw_request is None:
                raise RuntimePersistenceError("RUNTIME_RECOVERY_STATE_INVALID")
            return IntentTaskRequest.model_validate(raw_request)
        except RuntimePersistenceError:
            raise
        except RuntimeCheckpointError:
            raise RuntimePersistenceError("RUNTIME_RECOVERY_STATE_INVALID") from None
        except Exception:
            raise RuntimePersistenceError("RUNTIME_RECOVERY_STATE_INVALID") from None

    @staticmethod
    def _identities(value: Any) -> tuple[str, str, str]:
        try:
            if isinstance(value, Mapping):
                raw = {field: value[field] for field in _ID_ADAPTERS}
            else:
                raw = {field: getattr(value, field) for field in _ID_ADAPTERS}
            return tuple(_ID_ADAPTERS[field].validate_python(raw[field]) for field in _ID_ADAPTERS)  # type: ignore[return-value]
        except Exception:
            raise RuntimePersistenceError("RUNTIME_PERSISTENCE_ID_INVALID") from None


def _safe_model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, ClosedModel):
        dumped = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        dumped = dict(value)
    else:
        raise RuntimePersistenceError("RUNTIME_STATE_NOT_SERIALIZABLE")
    return dumped


def _safe_checkpoint_value(value: Any) -> Any:
    """Convert a graph state to JSON values before encryption, without logging it."""

    if isinstance(value, ClosedModel):
        return _safe_checkpoint_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise RuntimePersistenceError("RUNTIME_STATE_NOT_SERIALIZABLE")
        return {str(key): _safe_checkpoint_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 1000:
            raise RuntimePersistenceError("RUNTIME_STATE_NOT_SERIALIZABLE")
        return [_safe_checkpoint_value(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise RuntimePersistenceError("RUNTIME_STATE_NOT_SERIALIZABLE")


def _safe_control_code(value: Any) -> str | None:
    if not isinstance(value, str) or _CONTROL_CODE.fullmatch(value) is None:
        return None
    return value


def _opaque_id(prefix: str, *parts: str) -> str:
    return prefix + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def _persisted_node(node: str) -> str:
    return node if node in _PERSISTED_NODES else "execute_tool"


def _step_code(event: AuditEvent) -> str:
    if event.node in {"validate_task", "validate_user_task"}:
        return "TASK_CONTRACT_VALIDATED"
    if event.node in {"policy_gate"}:
        return "POLICY_ALLOWED"
    if event.node in {"build_tool_plan", "recognize_intent"}:
        return "TOOL_PLAN_BUILT"
    if event.node == "human_review" and event.status == "WAITING_APPROVAL":
        return "APPROVAL_REQUESTED"
    if event.status == "COMPLETED":
        return "JAVA_TOOL_COMPLETED" if event.toolName else "EVIDENCE_METADATA_CREATED"
    if event.status == "REJECTED":
        return "JAVA_TOOL_REJECTED"
    return "JAVA_TOOL_FAILED"


def _step_from_event(
    step_id: str,
    event: AuditEvent,
    snapshot: SnapshotMetadata | None,
    evidence_projection: UnifiedEvidencePersistenceProjection | None = None,
) -> AgentStepRecord:
    code = _step_code(event)
    status = (
        RuntimeStatus.COMPLETED if event.status == "COMPLETED"
        else RuntimeStatus.WAITING_APPROVAL if event.status == "WAITING_APPROVAL"
        else RuntimeStatus.FAILED
    )
    return AgentStepRecord(
        dataContractVersion="v1",
        stepId=step_id,
        runId=event.runId,
        nodeName=_persisted_node(event.node),
        attemptNumber=1,
        status=status,
        startedAt=event.occurredAt,
        endedAt=event.occurredAt,
        errorCode=_safe_control_code(event.errorCode),
        safeInputCode=code,
        safeOutputCode=code,
        snapshotMetadata=snapshot,
        evidenceProjection=evidence_projection,
    )


def _audit_from_event(
    audit_id: str,
    event: AuditEvent,
    snapshot: SnapshotMetadata | None,
    evidence_projection: UnifiedEvidencePersistenceProjection | None = None,
) -> ToolAuditRecord:
    return ToolAuditRecord(
        dataContractVersion="v1",
        auditId=audit_id,
        runId=event.runId,
        traceId=event.traceId,
        toolName=event.toolName,  # validated by AuditEvent/AllowedToolName at construction
        toolCallId=event.toolCallId,
        status=event.status,
        durationMs=0,
        errorCode=_safe_control_code(event.errorCode),
        snapshotMetadata=snapshot,
        evidenceProjection=evidence_projection,
        createdAt=event.occurredAt,
    )


def _projection_for_event(
    event: AuditEvent,
    projection: UnifiedEvidencePersistenceProjection | None,
) -> UnifiedEvidencePersistenceProjection | None:
    """Attach the final safe projection only to evidence-relevant trace rows."""

    if projection is None:
        return None
    if event.toolName is not None or event.node in {"synthesize_report", "terminal"}:
        return projection
    return None


def _unified_evidence_projection(
    report: Any,
) -> UnifiedEvidencePersistenceProjection | None:
    """Build a metadata-only projection from the unified report.

    The projection intentionally copies no candidate title, summary, source
    chunk, observation payload, locator or warning text.
    """

    if report is None:
        return None
    candidates = getattr(report, "unifiedEvidence", None)
    if candidates is None:
        # Keep the legacy literature report compatible while making the same
        # route/binding/path counters available to its trace rows.
        candidates = getattr(report, "references", None)
    if candidates is None or not isinstance(candidates, list):
        return None

    vector_count = 0
    graph_count = 0
    java_count = 0
    source_routes: set[str] = set()
    binding_count = 0
    path_count = 0
    conflicted_paths = 0
    snapshot_ids: set[str] = set()

    for candidate in candidates[:50]:
        routes = getattr(candidate, "sourceRoutes", None)
        if routes is None:
            routes = getattr(candidate, "retrievalSources", None) or []
            if not routes:
                route = getattr(candidate, "retrievalRoute", None)
                routes = [route] if route in {"vector", "graph"} else []
        route_values = {route for route in routes if route in {"vector", "graph", "java"}}
        source_routes.update(route_values)
        vector_count += int("vector" in route_values)
        graph_count += int("graph" in route_values)
        java_count += int("java" in route_values)

        bindings = getattr(candidate, "sourceBindings", None) or []
        binding_count += min(len(bindings), 3)
        for binding in bindings:
            snapshot_id = getattr(binding, "dataSnapshotId", None)
            if snapshot_id:
                snapshot_ids.add(snapshot_id)

        paths = getattr(candidate, "reasoningPaths", None) or []
        path_count += min(len(paths), 4)
        conflicted_paths += sum(
            1 for path in paths if getattr(path, "status", None) == "conflicted"
        )

    return UnifiedEvidencePersistenceProjection(
        candidateCount=min(len(candidates), 50),
        vectorCandidateCount=vector_count,
        graphCandidateCount=graph_count,
        javaCandidateCount=java_count,
        sourceRoutes=sorted(source_routes),
        sourceBindingCount=min(binding_count, 150),
        reasoningPathCount=min(path_count, 200),
        conflictedPathCount=min(conflicted_paths, 200),
        transientSnapshotCount=min(len(snapshot_ids), 50),
        generatedAt=datetime.now(timezone.utc),
    )


def _snapshot_from_report(report: Any) -> SnapshotMetadata | None:
    if report is None:
        return None
    try:
        value = report.model_dump(mode="json") if isinstance(report, ClosedModel) else report
        candidates: list[Mapping[str, Any]] = []
        _collect_snapshot_candidates(value, candidates, depth=0)
        for candidate in candidates:
            if not candidate.get("dataSnapshotId") or not candidate.get("queryHash"):
                continue
            return SnapshotMetadata(
                dataSnapshotId=TypeAdapter(JavaTransientSnapshotId).validate_python(candidate["dataSnapshotId"]),
                source="java_agent_read_model",
                snapshotPersistence="transient",
                schemaVersion=candidate.get("schemaVersion") or "v1",
                queryHash=candidate["queryHash"],
                rowCount=int(candidate.get("rowCount", 0)),
                generatedAt=candidate.get("generatedAt") or datetime.now(timezone.utc),
                importBatch=candidate.get("importBatch"),
                diseaseMappingVersion=candidate.get("diseaseMappingVersion"),
                taxonomyVersion=candidate.get("taxonomyVersion"),
                featureVersion=candidate.get("featureVersion"),
                sourceBatch=candidate.get("sourceBatch"),
            )
    except Exception:
        return None
    return None


def _evidence_artifact(
    run_id: str,
    report: Any,
    projection: UnifiedEvidencePersistenceProjection | None = None,
) -> AgentArtifactRecord | None:
    """Persist only a typed evidence artifact's digest and snapshot binding."""

    if not isinstance(report, (EvidenceReviewReport, ResearchExplorationReport)):
        return None
    try:
        # The unified exploration artifact is intentionally hashed from the
        # safe projection, not from literature text or Java payloads.  The
        # legacy report keeps its existing digest behavior for compatibility.
        artifact_value = (
            projection.model_dump(mode="json")
            if isinstance(report, ResearchExplorationReport) and projection is not None
            else report.model_dump(mode="json")
        )
        canonical = json.dumps(
            artifact_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        content_hash = "sha256:" + hashlib.sha256(canonical).hexdigest()
        artifact_id = "artifact-" + hashlib.sha256(
            f"{run_id}|{content_hash}".encode("utf-8")
        ).hexdigest()[:32]
        data_snapshot_id = _report_snapshot_id(report)
        return AgentArtifactRecord(
            dataContractVersion="v1",
            artifactId=artifact_id,
            runId=run_id,
            artifactType="evidence_summary",
            schemaVersion="v1",
            contentHash=content_hash,
            artifactStorageRef=f"artifact://{artifact_id}",
            dataSnapshotId=data_snapshot_id,
            createdAt=datetime.now(timezone.utc),
        )
    except Exception:
        return None


def _report_snapshot_id(report: Any) -> JavaTransientSnapshotId | None:
    """Extract only a validated transient snapshot identifier from a report."""

    try:
        legacy_ids = getattr(report, "analysisSnapshotIds", None) or []
        if legacy_ids:
            return TypeAdapter(JavaTransientSnapshotId).validate_python(legacy_ids[0])
        unified = getattr(report, "unifiedEvidence", None) or []
        for candidate in unified[:20]:
            for binding in (getattr(candidate, "sourceBindings", None) or [])[:3]:
                value = getattr(binding, "dataSnapshotId", None)
                if value:
                    return TypeAdapter(JavaTransientSnapshotId).validate_python(value)
    except Exception:
        return None
    return None


def _collect_snapshot_candidates(value: Any, output: list[Mapping[str, Any]], depth: int) -> None:
    if depth > 4:
        return
    if isinstance(value, Mapping):
        if "dataSnapshotId" in value:
            output.append(value)
        for child in value.values():
            _collect_snapshot_candidates(child, output, depth + 1)
    elif isinstance(value, list):
        for child in value[:20]:
            _collect_snapshot_candidates(child, output, depth + 1)
