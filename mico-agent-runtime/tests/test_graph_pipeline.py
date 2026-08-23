from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
from mico_agent_runtime.knowledge.graph_pipeline import (
    GRAPH_PIPELINE_VERSION,
    EntityResolver,
    GraphPipelineError,
    approve_graph_manifest,
    build_versioned_graph,
    normalize_entity_text,
    publish_graph_manifest,
    validate_graph_record,
)
from mico_agent_runtime.knowledge.ingest import validate_versioned_graph_ingestion
from mico_agent_runtime.knowledge.graph_review import (
    apply_review_decisions,
    build_graph_review_queue,
    build_publication_approval,
)
from mico_agent_runtime.contracts.review import GraphReviewDecision


def _rows() -> list[dict[str, str]]:
    return [
        {
            "pmcid": "PMC-A",
            "chunkId": "PMC-A-A001",
            "title": "Microbiome and diabetes",
            "text": "Akkermansia muciniphila is associated with type 2 diabetes.",
            "topic": "t2d",
            "section": "Results",
            "sourceUrl": "https://example.invalid/a",
        },
        {
            "pmcid": "PMC-B",
            "chunkId": "PMC-B-A001",
            "title": "Contrary evidence",
            "text": "Akkermansia muciniphila was not associated with type 2 diabetes.",
            "topic": "t2d",
            "section": "Results",
            "sourceUrl": "https://example.invalid/b",
        },
    ]


def test_entity_normalization_is_format_stable_and_controlled() -> None:
    assert normalize_entity_text("Type-2_diabetes") == "type 2 diabetes"
    resolved = EntityResolver().resolve("T2DM", "Disease")
    assert resolved.normalizationStatus == "resolved"
    assert resolved.canonicalEntityId == "mico:disease:type_2_diabetes"
    assert resolved.ontologySource == "mico-controlled-alias-v1"


def test_unknown_taxon_remains_candidate_not_formal_alignment() -> None:
    candidate = EntityResolver().resolve("Novelus exampleus", "Taxon")
    assert candidate.normalizationStatus == "candidate"
    assert candidate.ontologySource == "taxonomy-candidate-v1"
    assert candidate.ontologyId is None


def test_versioned_build_has_source_bound_quality_manifest() -> None:
    result = build_versioned_graph(_rows())
    assert result.manifest.graphVersion == GRAPH_PIPELINE_VERSION
    assert result.manifest.status == "review_pending"
    assert result.manifest.buildRunId.startswith("graph-build-")
    assert result.manifest.inputFingerprint.startswith("sha256:")
    assert all(record["graphVersion"] == GRAPH_PIPELINE_VERSION for record in result.records)
    assert all(record["graphBuildRunId"] == result.manifest.buildRunId for record in result.records)
    semantic_edges = [
        record for record in result.records
        if record.get("recordType") == "edge" and record.get("relationClass") != "structural"
    ]
    assert semantic_edges
    assert len({edge["edgeId"] for edge in semantic_edges}) == len(semantic_edges)
    assert any(edge["assertionStatus"] == "conflicted" for edge in semantic_edges)
    assert result.manifest.reviewRequiredRelationCount >= 1
    assert result.manifest.qualityIssueCounts == {}


def test_loaded_manifest_with_review_relations_is_not_treated_as_publishable(tmp_path: Path) -> None:
    result = build_versioned_graph(_rows())
    manifest = result.manifest.model_copy(update={"status": "staging"})
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    from mico_agent_runtime.knowledge.graph_pipeline import load_graph_manifest

    loaded = load_graph_manifest(path)
    assert loaded.status == "review_pending"


def test_versioned_ids_allow_v3_and_v4_to_coexist() -> None:
    result = build_versioned_graph(_rows())
    node_ids = [record["nodeId"] for record in result.records if record.get("recordType") == "node"]
    assert node_ids
    assert all(node_id.startswith("v4:") for node_id in node_ids)
    assert all(not node_id.startswith("v3:") for node_id in node_ids)


def test_quality_gate_rejects_missing_evidence_and_endpoint() -> None:
    issues = validate_graph_record(
        {
            "recordType": "edge",
            "edgeId": "edge-1",
            "source": "missing",
            "target": "missing-target",
            "relation": "ASSOCIATED_WITH",
            "relationClass": "association",
            "assertionStatus": "asserted",
            "confidence": 0.9,
        },
        set(),
    )
    assert {issue.issueCode for issue in issues} == {"MISSING_ENDPOINT", "MISSING_EVIDENCE"}
    assert all(issue.severity == "error" for issue in issues)


def test_publish_requires_clean_build_and_writes_explicit_registry(tmp_path: Path) -> None:
    result = build_versioned_graph(_rows())
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "registry.json"
    manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")
    queue = build_graph_review_queue(result)
    queue = apply_review_decisions(queue, [
        GraphReviewDecision(
            reviewId=item.reviewId,
            decision="APPROVED",
            decisionCode="RELATION_ACCEPTED",
            decidedBy="principal-" + "1" * 32,
            decidedAt=datetime.now(timezone.utc),
        ) for item in queue.items
    ])
    approval = build_publication_approval(
        queue, approved_by="principal-" + "1" * 32, approved_at=datetime.now(timezone.utc)
    )
    manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")
    approve_graph_manifest(manifest_path, queue, approval)
    registry = publish_graph_manifest(manifest_path, registry_path)
    assert registry.currentGraphVersion == GRAPH_PIPELINE_VERSION
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["status"] == "published"
    assert saved["publishedAt"]


def test_publish_rejects_build_with_rejected_records(tmp_path: Path) -> None:
    result = build_versioned_graph(_rows())
    bad = result.manifest.model_copy(update={"rejectedRecordCount": 1})
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "registry.json"
    manifest_path.write_text(bad.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(GraphPipelineError, match="GRAPH_VERSION_HAS_REJECTED_RECORDS"):
        publish_graph_manifest(manifest_path, registry_path)


def test_database_ingestion_cannot_bypass_review_with_publish_flag() -> None:
    result = build_versioned_graph(_rows())
    with pytest.raises(GraphPipelineError, match="KNOWLEDGE_GRAPH_PUBLICATION_APPROVAL_REQUIRED"):
        validate_versioned_graph_ingestion(result.manifest, publish=True)


def test_database_ingestion_accepts_review_pending_only_for_staging() -> None:
    result = build_versioned_graph(_rows())
    validate_versioned_graph_ingestion(result.manifest, publish=False)


def test_graph_version_is_explicitly_configurable_without_changing_default() -> None:
    env = {
        "MICO_KNOWLEDGE_VECTOR_ENABLED": "true",
        "MICO_KNOWLEDGE_GRAPH_ENABLED": "true",
        "MICO_KNOWLEDGE_VECTOR_DATABASE_URL": "postgresql://test_user@127.0.0.1:55432/mico_knowledge",
        "MICO_KNOWLEDGE_NEO4J_URI": "bolt://127.0.0.1:57687",
        "MICO_KNOWLEDGE_NEO4J_USER": "neo4j",
        "MICO_KNOWLEDGE_NEO4J_PASSWORD": "test_only_password",
        "MICO_LOCAL_KNOWLEDGE_INDEX_DIR": "references/knowledge/medical/rag",
    }
    assert KnowledgeStoreConfiguration.from_environment(env).graphVersion.endswith("v3")
    env["MICO_KNOWLEDGE_GRAPH_VERSION"] = GRAPH_PIPELINE_VERSION
    assert KnowledgeStoreConfiguration.from_environment(env).graphVersion == GRAPH_PIPELINE_VERSION
    env["MICO_KNOWLEDGE_GRAPH_VERSION"] = "../patient_data_manager"
    with pytest.raises(ValueError, match="KNOWLEDGE_GRAPH_VERSION_INVALID"):
        KnowledgeStoreConfiguration.from_environment(env)
