from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mico_agent_runtime.contracts.graph_rag import GraphEvidencePath, GraphPathStep
from mico_agent_runtime.contracts.review import (
    GraphPublicationApproval,
    GraphReviewDecision,
    GraphReviewQueue,
)
from mico_agent_runtime.knowledge.graph_pipeline import (
    GraphPipelineError,
    approve_graph_manifest,
    build_versioned_graph,
    publish_graph_manifest,
)
from mico_agent_runtime.knowledge.graph_review import (
    apply_review_decisions,
    build_graph_review_queue,
    build_publication_approval,
    project_evidence_path,
)


def _rows() -> list[dict[str, str]]:
    return [
        {
            "pmcid": "PMC-REVIEW-A",
            "chunkId": "PMC-REVIEW-A001",
            "title": "Microbiome evidence",
            "text": "Akkermansia muciniphila is associated with type 2 diabetes.",
            "topic": "t2d",
            "section": "Results",
            "sourceUrl": "https://example.invalid/review-a",
        },
        {
            "pmcid": "PMC-REVIEW-B",
            "chunkId": "PMC-REVIEW-B001",
            "title": "Contrary evidence",
            "text": "Akkermansia muciniphila was not associated with type 2 diabetes.",
            "topic": "t2d",
            "section": "Results",
            "sourceUrl": "https://example.invalid/review-b",
        },
    ]


def test_review_queue_is_deterministic_and_requires_all_items() -> None:
    result = build_versioned_graph(_rows())
    queue = build_graph_review_queue(result)
    assert queue.status == "OPEN"
    assert queue.pendingCount == result.manifest.reviewRequiredRelationCount
    assert queue.queueHash.startswith("sha256:")
    assert all(item.status == "PENDING" for item in queue.items)
    assert all(item.sourceEntity and item.targetEntity for item in queue.items)
    assert all(item.assertionStatus == "conflicted" for item in queue.items)
    with pytest.raises(GraphPipelineError, match="GRAPH_REVIEW_PENDING"):
        build_publication_approval(queue, approved_by="principal-" + "1" * 32)


def test_review_decisions_bind_queue_and_publication_gate(tmp_path: Path) -> None:
    result = build_versioned_graph(_rows())
    queue = build_graph_review_queue(result)
    reviewer = "principal-" + "1" * 32
    queue = apply_review_decisions(queue, [
        GraphReviewDecision(
            reviewId=item.reviewId,
            decision="APPROVED",
            decisionCode="RELATION_ACCEPTED",
            decidedBy=reviewer,
            decidedAt=datetime.now(timezone.utc),
        ) for item in queue.items
    ])
    approval = build_publication_approval(queue, approved_by=reviewer)
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "registry.json"
    manifest_path.write_text(result.manifest.model_dump_json(), encoding="utf-8")
    approved = approve_graph_manifest(manifest_path, queue, approval)
    assert approved.status == "approved"
    registry = publish_graph_manifest(manifest_path, registry_path)
    assert registry.currentGraphVersion == result.manifest.graphVersion
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["status"] == "published"


def test_publication_cannot_use_a_different_review_queue(tmp_path: Path) -> None:
    result = build_versioned_graph(_rows())
    queue = build_graph_review_queue(result)
    reviewer = "principal-" + "1" * 32
    queue = apply_review_decisions(queue, [
        GraphReviewDecision(
            reviewId=item.reviewId,
            decision="APPROVED",
            decisionCode="RELATION_ACCEPTED",
            decidedBy=reviewer,
            decidedAt=datetime.now(timezone.utc),
        ) for item in queue.items
    ])
    approval = GraphPublicationApproval(
        graphVersion=queue.graphVersion,
        buildRunId=queue.buildRunId,
        reviewQueueHash="sha256:" + "0" * 64,
        reviewedItemCount=len(queue.items),
        decision="APPROVED",
        approvedBy=reviewer,
        approvedAt=datetime.now(timezone.utc),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(result.manifest.model_dump_json(), encoding="utf-8")
    with pytest.raises(GraphPipelineError, match="GRAPH_REVIEW_QUEUE_HASH_MISMATCH"):
        approve_graph_manifest(manifest_path, queue, approval)


def test_evidence_path_projection_is_bounded_and_safe() -> None:
    path = GraphEvidencePath(
        pathId="path-" + "a" * 32,
        status="conflicted",
        hops=[GraphPathStep(
            fromEntity="type 2 diabetes",
            relation="ASSOCIATED_WITH",
            toEntity="Akkermansia muciniphila",
            evidenceChunkId="PMC-REVIEW-A001",
            supportStatus="conflicted",
        )],
        sourceDocumentIds=["PMCID:PMC-REVIEW-A"],
        confidence=0.71,
    )
    view = project_evidence_path(path)
    assert view.reviewRequired is True
    assert view.hops[0].evidenceChunkId == "PMC-REVIEW-A001"
    unsafe = path.model_copy(update={"hops": [GraphPathStep(
            fromEntity="sourceSampleId=secret",
            relation="ASSOCIATED_WITH",
            toEntity="taxon",
            evidenceChunkId="chunk",
        )]})
    with pytest.raises(ValueError):
        project_evidence_path(unsafe)
