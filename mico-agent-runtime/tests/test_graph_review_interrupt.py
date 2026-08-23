from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

from mico_agent_runtime.contracts.evidence import EvidenceQuery, LiteratureEvidenceItem
from mico_agent_runtime.contracts.graph_rag import GraphEvidencePath, GraphPathStep
from mico_agent_runtime.contracts.intent import IntentTaskRequest
from mico_agent_runtime.contracts.review import GraphReviewResumeCommand
from mico_agent_runtime.contracts.tools import JavaToolResponse
from mico_agent_runtime.knowledge.graph_review import project_evidence_path
from mico_agent_runtime.runtime.checkpoint import EncryptedLangGraphCheckpointSaver
from mico_agent_runtime.runtime.intent_service import IntentRuntime
from mico_agent_runtime.storage.crypto import RuntimeStateCipher, StateBindingContext
from mico_agent_runtime.storage.memory_store import InMemoryRuntimeStore
from mico_agent_runtime.storage.models import AgentRunRecord, RuntimeStatus
from mico_agent_runtime.ports.research_planner import DeterministicIntentPlanner


NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)
RUN = "run-" + "1" * 32
TASK = "task-" + "2" * 32
TRACE = "trace-" + "3" * 32


class NoJavaPort:
    def execute(self, _call) -> JavaToolResponse:
        raise AssertionError("literature review must not call Java")


class ConflictKnowledgePort:
    def search(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        path = GraphEvidencePath(
            pathId="path-" + "b" * 32,
            status="conflicted",
            hops=[GraphPathStep(
                fromEntity="type 2 diabetes",
                relation="ASSOCIATED_WITH",
                toEntity="Akkermansia muciniphila",
                evidenceChunkId="PMC-REVIEW-A001",
                supportStatus="conflicted",
            )],
            sourceDocumentIds=["PMCID:PMC-REVIEW-A"],
            confidence=0.72,
        )
        return [LiteratureEvidenceItem(
            evidenceId="evidence-" + "b" * 32,
            source="internal_knowledge",
            externalId="PMCID:PMC-REVIEW-A#PMC-REVIEW-A001",
            title="Bounded literature evidence",
            publicationYear=2024,
            direction=query.direction,
            summary="A bounded source-bound evidence item.",
            evidenceTier="fulltext",
            retrievalRoute="hybrid",
            sourceChunkId="PMC-REVIEW-A001",
            retrievalScore=0.8,
            graphPaths=[path],
            reasoningPaths=[],
        )]


def _request() -> IntentTaskRequest:
    return IntentTaskRequest(
        runId=RUN,
        taskId=TASK,
        requesterId="principal-" + "4" * 32,
        requestedScopes=["mico:evidence:read"],
        question="请审阅 2 型糖尿病与微生物的文献关系",
        allowedWorkflows=["knowledge_retrieval"],
        createdAt=NOW,
        traceId=TRACE,
    )


def _runtime() -> tuple[IntentRuntime, IntentTaskRequest]:
    async def prepare() -> InMemoryRuntimeStore:
        store = InMemoryRuntimeStore()
        cipher = RuntimeStateCipher(b"k" * 32, "runtime-key-1")
        await store.create_run(AgentRunRecord(
            dataContractVersion="v1",
            runId=RUN,
            taskId=TASK,
            traceId=TRACE,
            status=RuntimeStatus.RUNNING,
            createdAt=NOW,
            updatedAt=NOW,
            encryptedStatePayload=cipher.encrypt(
                {"request": _request().model_dump(mode="json")},
                StateBindingContext(
                    runId=RUN, taskId=TASK, traceId=TRACE, dataContractVersion="v1"
                ),
            ),
            encryptionKeyId="runtime-key-1",
        ))
        return store

    store = asyncio.run(prepare())
    cipher = RuntimeStateCipher(b"k" * 32, "runtime-key-1")
    runtime = IntentRuntime(
        NoJavaPort(),
        DeterministicIntentPlanner(),
        checkpoint_saver=EncryptedLangGraphCheckpointSaver(store, cipher),
        knowledge_port=ConflictKnowledgePort(),
        interrupt_on_review=True,
    )
    return runtime, _request()


def test_graph_interrupt_returns_waiting_and_resumes_only_with_closed_decision() -> None:
    runtime, request = _runtime()
    waiting = runtime.run(request)
    assert waiting.status == "WAITING_APPROVAL"
    assert waiting.errorCode == "GRAPH_REVIEW_REQUIRED"
    assert waiting.report is not None
    assert any(event.node == "human_review" for event in waiting.auditEvents)
    review_id = "review-" + hashlib.sha256(
        ("evidence-path|path-" + "b" * 32).encode("utf-8")
    ).hexdigest()[:32]
    completed = runtime.resume(request, GraphReviewResumeCommand(
        reviewId=review_id,
        decision="APPROVED",
    ))
    assert completed.status == "COMPLETED"
    assert completed.errorCode is None
    assert completed.report is not None


def test_path_projection_marks_conflict_for_review() -> None:
    path = ConflictKnowledgePort().search(EvidenceQuery(
        topic="t2d", direction="context", retrievalMode="graph", limit=1
    ))[0].graphPaths[0]
    assert project_evidence_path(path).reviewRequired is True
