"""Graph-quality review queue and safe evidence-path projections."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Sequence

from mico_agent_runtime.contracts.graph_rag import GraphEvidencePath
from mico_agent_runtime.contracts.review import (
    EvidencePathView,
    GraphPublicationApproval,
    GraphReviewDecision,
    GraphReviewQueue,
    GraphReviewTicket,
)

from .graph_pipeline import GraphBuildResult, GraphPipelineError


def _review_id(graph_version: str, record_id: str, issue_code: str) -> str:
    digest = hashlib.sha256(
        f"{graph_version}|{record_id}|{issue_code}".encode("utf-8")
    ).hexdigest()[:32]
    return f"review-{digest}"


def _queue_hash(items: Sequence[GraphReviewTicket]) -> str:
    canonical = json.dumps(
        [item.model_dump(mode="json") for item in items],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def project_evidence_path(path: GraphEvidencePath) -> EvidencePathView:
    """Project a source-bound path without exposing internal graph node IDs."""

    return EvidencePathView(
        pathId=path.pathId,
        status=path.status,
        hops=path.hops,
        sourceDocumentIds=path.sourceDocumentIds,
        confidence=path.confidence,
        reviewRequired=path.status != "supported",
    )


def build_graph_review_queue(result: GraphBuildResult) -> GraphReviewQueue:
    """Create a deterministic queue from one staging build.

    Only retained semantic relations marked ``review_required`` enter the
    queue. Rejected records remain a build failure and cannot be silently
    approved through this queue.
    """

    manifest = result.manifest
    nodes = {
        str(record.get("nodeId")): record
        for record in result.records
        if record.get("recordType") == "node"
    }
    tickets: list[GraphReviewTicket] = []
    for record in result.records:
        if record.get("recordType") != "edge":
            continue
        if record.get("relationClass") == "structural":
            continue
        if record.get("qualityStatus") != "review_required":
            continue
        record_id = str(record.get("edgeId") or "")
        assertion = str(record.get("assertionStatus") or "")
        issue_code = (
            "CONFLICTING_ASSERTIONS" if assertion == "conflicted"
            else "SPECULATIVE_RELATION" if assertion == "speculative"
            else "LOW_CONFIDENCE_RELATION"
        )
        confidence = float(record.get("confidence") or 0.0)
        source = nodes.get(str(record.get("source")), {})
        target = nodes.get(str(record.get("target")), {})
        source_entity = str(source.get("canonicalLabel") or source.get("label") or "unresolved entity")
        target_entity = str(target.get("canonicalLabel") or target.get("label") or "unresolved entity")
        severity = "high" if issue_code == "CONFLICTING_ASSERTIONS" else "warning"
        tickets.append(GraphReviewTicket(
            reviewId=_review_id(manifest.graphVersion, record_id, issue_code),
            graphVersion=manifest.graphVersion,
            buildRunId=manifest.buildRunId,
            recordId=record_id,
            issueCode=issue_code,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            confidence=confidence,
            sourceEntity=source_entity,
            relation=str(record.get("relation") or "UNSPECIFIED"),
            targetEntity=target_entity,
            assertionStatus=assertion,  # type: ignore[arg-type]
            evidenceChunkId=(str(record["evidenceChunkId"]) if record.get("evidenceChunkId") else None),
            evidenceExcerpt=(str(record["evidenceText"])[:1200] if record.get("evidenceText") else None),
        ))
    tickets.sort(key=lambda item: item.reviewId)
    return GraphReviewQueue(
        graphVersion=manifest.graphVersion,
        buildRunId=manifest.buildRunId,
        status="OPEN" if tickets else "CLOSED",
        items=tickets,
        queueHash=_queue_hash(tickets),
        pendingCount=len(tickets),
        approvedCount=0,
        rejectedCount=0,
    )


def apply_review_decisions(
    queue: GraphReviewQueue,
    decisions: Sequence[GraphReviewDecision],
) -> GraphReviewQueue:
    """Apply only typed decisions matching the exact queue items."""

    by_id = {decision.reviewId: decision for decision in decisions}
    if len(by_id) != len(decisions):
        raise GraphPipelineError("GRAPH_REVIEW_DUPLICATE_DECISION")
    unknown = set(by_id).difference(item.reviewId for item in queue.items)
    if unknown:
        raise GraphPipelineError("GRAPH_REVIEW_UNKNOWN_ITEM")
    updated: list[GraphReviewTicket] = []
    for item in queue.items:
        decision = by_id.get(item.reviewId)
        if decision is None:
            updated.append(item)
            continue
        updated.append(item.model_copy(update={
            "status": "APPROVED" if decision.decision == "APPROVED" else "REJECTED"
        }))
    updated.sort(key=lambda item: item.reviewId)
    pending = sum(item.status == "PENDING" for item in updated)
    approved = sum(item.status == "APPROVED" for item in updated)
    rejected = sum(item.status == "REJECTED" for item in updated)
    return GraphReviewQueue(
        graphVersion=queue.graphVersion,
        buildRunId=queue.buildRunId,
        status="OPEN" if pending else "CLOSED",
        items=updated,
        queueHash=_queue_hash(updated),
        pendingCount=pending,
        approvedCount=approved,
        rejectedCount=rejected,
    )


def build_publication_approval(
    queue: GraphReviewQueue,
    *,
    approved_by: str,
    approved_at: datetime | None = None,
) -> GraphPublicationApproval:
    """Create the publication gate only after every review item is approved."""

    if queue.status != "CLOSED" or queue.pendingCount:
        raise GraphPipelineError("GRAPH_REVIEW_PENDING")
    if queue.rejectedCount:
        raise GraphPipelineError("GRAPH_REVIEW_REJECTED_ITEMS")
    if not queue.items:
        raise GraphPipelineError("GRAPH_REVIEW_NOT_REQUIRED")
    return GraphPublicationApproval(
        graphVersion=queue.graphVersion,
        buildRunId=queue.buildRunId,
        reviewQueueHash=queue.queueHash,
        reviewedItemCount=len(queue.items),
        decision="APPROVED",
        approvedBy=approved_by,
        approvedAt=approved_at or datetime.now(timezone.utc),
    )
