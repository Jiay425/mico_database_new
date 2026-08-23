from __future__ import annotations

"""Versioned knowledge-graph build, quality-gate and publication contracts.

This module deliberately keeps graph construction separate from retrieval.  A
build produces a staging manifest and source-bound records; a caller must
explicitly publish a validated version before retrieval is pointed at it.
No business MySQL data is read here.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import Field, StringConstraints, field_validator

from mico_agent_runtime.contracts.base import ClosedModel
from mico_agent_runtime.contracts.review import GraphPublicationApproval, GraphReviewQueue

from .graph_v3 import ENTITY_ALIASES, GRAPH_VERSION as V3_GRAPH_VERSION, build_v3_records


GRAPH_PIPELINE_VERSION = "fulltext-provenance-graphrag-v4"
GRAPH_BUILD_ID_RE = re.compile(r"^graph-build-[0-9a-f]{32}$")
GRAPH_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,79}$")

EntityType = Literal[
    "Disease", "Taxon", "Metabolite", "Gene", "Protein", "Phenotype",
    "Study", "Host", "BodySite", "Pathway", "Concept", "HostProcess",
    "Paper", "Chunk", "Section", "Topic",
]
NormalizationStatus = Literal[
    "resolved", "candidate", "ambiguous", "unmapped", "rejected",
    "source_identifier", "controlled_topic",
]
AssertionStatus = Literal["asserted", "negated", "speculative", "conflicted"]
QualityStatus = Literal["accepted", "review_required", "rejected"]
BuildStatus = Literal[
    "staging", "review_pending", "approved", "published", "retired", "rejected"
]


class CanonicalEntity(ClosedModel):
    canonicalEntityId: str = Field(min_length=1, max_length=160)
    entityType: EntityType
    canonicalLabel: str = Field(min_length=1, max_length=512)
    ontologySource: str | None = Field(default=None, max_length=128)
    ontologyId: str | None = Field(default=None, max_length=160)
    normalizationStatus: NormalizationStatus
    aliases: list[str] = Field(default_factory=list, max_length=32)


class GraphQualityIssue(ClosedModel):
    issueCode: Literal[
        "ENTITY_UNMAPPED", "ENTITY_AMBIGUOUS", "MISSING_EVIDENCE",
        "LOW_CONFIDENCE", "UNSUPPORTED_RELATION", "INVALID_ASSERTION",
        "MISSING_ENDPOINT", "CONFLICT_REQUIRES_REVIEW",
    ]
    recordType: Literal["node", "edge"]
    recordId: str = Field(min_length=1, max_length=192)
    severity: Literal["warning", "error"]


class GraphBuildManifest(ClosedModel):
    graphVersion: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,79}$")
    buildRunId: str = Field(pattern=r"^graph-build-[0-9a-f]{32}$")
    status: BuildStatus = "staging"
    createdAt: str = Field(min_length=20, max_length=64)
    publishedAt: str | None = Field(default=None, max_length=64)
    approvedAt: str | None = Field(default=None, max_length=64)
    approvedBy: str | None = Field(default=None, pattern=r"^principal-[0-9a-f]{32}$")
    publicationApprovalHash: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    corpusScope: Literal["fulltext_only"] = "fulltext_only"
    evidenceTier: Literal["fulltext"] = "fulltext"
    inputAsset: str = Field(min_length=1, max_length=256)
    inputFingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    paperCount: int = Field(ge=0)
    chunkCount: int = Field(ge=0)
    nodeCount: int = Field(ge=0)
    edgeCount: int = Field(ge=0)
    semanticRelationCount: int = Field(ge=0)
    resolvedEntityCount: int = Field(ge=0)
    candidateEntityCount: int = Field(ge=0)
    unmappedEntityCount: int = Field(ge=0)
    reviewRequiredRelationCount: int = Field(ge=0)
    rejectedRecordCount: int = Field(ge=0)
    relationCounts: dict[str, int] = Field(default_factory=dict)
    assertionCounts: dict[str, int] = Field(default_factory=dict)
    qualityIssueCounts: dict[str, int] = Field(default_factory=dict)

    @field_validator("createdAt", "publishedAt", "approvedAt")
    @classmethod
    def require_utc_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("graph timestamps must include timezone")
        return parsed.astimezone(timezone.utc).isoformat()


class GraphPublicationRegistry(ClosedModel):
    currentGraphVersion: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,79}$")
    updatedAt: str = Field(min_length=20, max_length=64)
    publishedVersions: list[str] = Field(min_length=1, max_length=32)


class GraphBuildResult(ClosedModel):
    records: list[dict[str, Any]] = Field(min_length=1)
    manifest: GraphBuildManifest
    issues: list[GraphQualityIssue] = Field(default_factory=list, max_length=10000)


class GraphPipelineError(ValueError):
    """A graph build or publication contract cannot be accepted."""


def normalize_entity_text(value: str) -> str:
    """Normalize aliases without changing scientific display labels."""
    normalized = value.strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _opaque_id(prefix: str, value: str) -> str:
    return f"mico:{prefix}:" + hashlib.sha256(normalize_entity_text(value).encode("utf-8")).hexdigest()[:32]


def _build_run_id(rows: list[Mapping[str, Any]], graph_version: str) -> str:
    chunks = sorted(str(row.get("chunkId") or "") for row in rows)
    payload = json.dumps({"graphVersion": graph_version, "chunks": chunks}, sort_keys=True).encode("utf-8")
    return "graph-build-" + hashlib.sha256(payload).hexdigest()[:32]


def _input_fingerprint(rows: list[Mapping[str, Any]]) -> str:
    canonical = "\n".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EntityResolver:
    """Resolve only known aliases; unknown entities remain explicit candidates."""

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}
        for canonical, (_entity_type, _label, aliases) in ENTITY_ALIASES.items():
            for alias in (canonical, *_label.split("|"), *aliases):
                self._aliases[normalize_entity_text(alias)] = canonical

    def resolve(self, label: str, entity_type: str) -> CanonicalEntity:
        normalized = normalize_entity_text(label)
        canonical = self._aliases.get(normalized)
        if canonical is not None:
            expected_type, canonical_label, aliases = ENTITY_ALIASES[canonical]
            if expected_type != entity_type:
                return CanonicalEntity(
                    canonicalEntityId=_opaque_id("candidate", f"{entity_type}|{label}"),
                    entityType=entity_type,  # type: ignore[arg-type]
                    canonicalLabel=label,
                    normalizationStatus="ambiguous",
                    aliases=[label],
                )
            return CanonicalEntity(
                canonicalEntityId=f"mico:{entity_type.lower()}:{canonical}",
                entityType=expected_type,  # type: ignore[arg-type]
                canonicalLabel=canonical_label,
                ontologySource="mico-controlled-alias-v1",
                ontologyId=canonical,
                normalizationStatus="resolved",
                aliases=sorted(set([label, *aliases]))[:32],
            )
        if entity_type == "Taxon":
            return CanonicalEntity(
                canonicalEntityId=_opaque_id("candidate-taxon", normalized),
                entityType="Taxon",
                canonicalLabel=label,
                ontologySource="taxonomy-candidate-v1",
                ontologyId=None,
                normalizationStatus="candidate",
                aliases=[label],
            )
        return CanonicalEntity(
            canonicalEntityId=_opaque_id("unmapped", f"{entity_type}|{normalized}"),
            entityType=entity_type,  # type: ignore[arg-type]
            canonicalLabel=label,
            normalizationStatus="unmapped",
            aliases=[label],
        )


_ALLOWED_RELATIONS = {
    "CAUSES", "MEDIATES", "PROMOTES", "INHIBITS", "INCREASED_IN",
    "DECREASED_IN", "ASSOCIATED_WITH", "PART_OF", "IN_SECTION",
    "HAS_TOPIC", "MENTIONS_ENTITY",
}
_ALLOWED_RELATION_CLASSES = {"causal", "directional", "association", "structural"}
_ALLOWED_ASSERTIONS = {"asserted", "negated", "speculative", "conflicted"}


def _record_issue(record: Mapping[str, Any], code: str, severity: str) -> GraphQualityIssue:
    return GraphQualityIssue(
        issueCode=code,  # type: ignore[arg-type]
        recordType=record["recordType"],  # type: ignore[arg-type]
        recordId=str(record.get("nodeId") or record.get("edgeId") or "unknown"),
        severity=severity,  # type: ignore[arg-type]
    )


def validate_graph_record(record: Mapping[str, Any], node_ids: set[str]) -> list[GraphQualityIssue]:
    issues: list[GraphQualityIssue] = []
    record_type = record.get("recordType")
    if record_type == "node":
        if not record.get("nodeId") or not record.get("entityType"):
            issues.append(_record_issue(record, "MISSING_ENDPOINT", "error"))
        status = record.get("normalizationStatus")
        if status == "unmapped":
            issues.append(_record_issue(record, "ENTITY_UNMAPPED", "warning"))
        if status == "ambiguous":
            issues.append(_record_issue(record, "ENTITY_AMBIGUOUS", "error"))
        return issues
    if record_type != "edge":
        issues.append(_record_issue(record, "MISSING_ENDPOINT", "error"))
        return issues
    if not record.get("source") or not record.get("target"):
        issues.append(_record_issue(record, "MISSING_ENDPOINT", "error"))
    elif str(record["source"]) not in node_ids or str(record["target"]) not in node_ids:
        issues.append(_record_issue(record, "MISSING_ENDPOINT", "error"))
    relation = str(record.get("relation") or "")
    if relation not in _ALLOWED_RELATIONS:
        issues.append(_record_issue(record, "UNSUPPORTED_RELATION", "error"))
    if str(record.get("relationClass") or "") not in _ALLOWED_RELATION_CLASSES:
        issues.append(_record_issue(record, "UNSUPPORTED_RELATION", "error"))
    if str(record.get("assertionStatus") or "") not in _ALLOWED_ASSERTIONS:
        issues.append(_record_issue(record, "INVALID_ASSERTION", "error"))
    if not record.get("evidenceChunkId") or (
        record.get("relationClass") != "structural" and not record.get("evidenceText")
    ):
        issues.append(_record_issue(record, "MISSING_EVIDENCE", "error"))
    try:
        confidence = float(record.get("confidence"))
    except (TypeError, ValueError):
        confidence = -1.0
    if confidence < 0.65:
        issues.append(_record_issue(record, "LOW_CONFIDENCE", "error"))
    return issues


def _mark_conflicts(records: list[dict[str, Any]]) -> None:
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("recordType") == "edge" and record.get("relationClass") != "structural":
            grouped[(str(record.get("source")), str(record.get("target")), str(record.get("relation")))].append(record)
    for group in grouped.values():
        statuses = {str(item.get("assertionStatus")) for item in group}
        if "asserted" in statuses and "negated" in statuses:
            for item in group:
                item["assertionStatus"] = "conflicted"
                item["qualityStatus"] = "review_required"


def build_versioned_graph(
    rows: Iterable[Mapping[str, Any]],
    graph_version: str = GRAPH_PIPELINE_VERSION,
) -> GraphBuildResult:
    if not GRAPH_VERSION_RE.fullmatch(graph_version):
        raise GraphPipelineError("GRAPH_VERSION_INVALID")
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise GraphPipelineError("FULLTEXT_CORPUS_EMPTY")
    base_records, base_counts = build_v3_records(materialized)
    resolver = EntityResolver()
    build_run_id = _build_run_id(materialized, graph_version)
    version_tag = graph_version.rsplit("-", 1)[-1]
    version_tag = re.sub(r"[^a-z0-9]", "", version_tag.lower()) or "graph"
    id_map: dict[str, str] = {}
    for raw in base_records:
        if raw.get("recordType") != "node":
            continue
        original_id = str(raw.get("nodeId"))
        id_map[original_id] = f"{version_tag}:{original_id.removeprefix('v3:')}"
    records: list[dict[str, Any]] = []
    entity_counts: Counter[str] = Counter()
    for raw in base_records:
        record = dict(raw)
        record["graphVersion"] = graph_version
        record["graphBuildRunId"] = build_run_id
        if record.get("recordType") == "node":
            record["nodeId"] = id_map[str(record["nodeId"])]
        elif record.get("recordType") == "edge":
            original_source = str(record.get("source"))
            original_target = str(record.get("target"))
            record["source"] = id_map.get(original_source, original_source)
            record["target"] = id_map.get(original_target, original_target)
            edge_identity = json.dumps(
                {
                    "graphVersion": graph_version,
                    "originalEdgeId": record.get("edgeId"),
                    "source": record["source"],
                    "relation": record.get("relation"),
                    "target": record["target"],
                    "evidenceChunkId": record.get("evidenceChunkId"),
                    "evidenceText": record.get("evidenceText"),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            record["edgeId"] = version_tag + "-edge-" + hashlib.sha256(
                edge_identity.encode("utf-8")
            ).hexdigest()[:32]
        if record.get("recordType") == "node" and record.get("entityType") in {
            "Disease", "Taxon", "Metabolite", "Pathway", "Concept", "HostProcess",
        }:
            resolved = resolver.resolve(str(record.get("label") or ""), str(record["entityType"]))
            record.update(resolved.model_dump(mode="json"))
            entity_counts[resolved.normalizationStatus] += 1
        elif record.get("recordType") == "node":
            record["qualityStatus"] = "accepted"
        records.append(record)

    _mark_conflicts(records)
    node_ids = {str(record["nodeId"]) for record in records if record.get("recordType") == "node"}
    issues: list[GraphQualityIssue] = []
    accepted: list[dict[str, Any]] = []
    rejected_ids: set[str] = set()
    for record in records:
        record_issues = validate_graph_record(record, node_ids)
        issues.extend(record_issues)
        errors = [issue for issue in record_issues if issue.severity == "error"]
        if errors:
            record["qualityStatus"] = "rejected"
            rejected_ids.add(str(record.get("nodeId") or record.get("edgeId")))
            continue
        if record.get("recordType") == "edge" and record.get("qualityStatus") != "review_required":
            record["qualityStatus"] = "accepted"
        accepted.append(record)

    # Do not leave edges pointing to a rejected node in a staging asset.
    accepted_node_ids = {str(record["nodeId"]) for record in accepted if record.get("recordType") == "node"}
    final_records = [
        record for record in accepted
        if record.get("recordType") != "edge"
        or (str(record.get("source")) in accepted_node_ids and str(record.get("target")) in accepted_node_ids)
    ]
    semantic_edges = [
        record for record in final_records
        if record.get("recordType") == "edge" and record.get("relationClass") != "structural"
    ]
    review_count = sum(record.get("qualityStatus") == "review_required" for record in semantic_edges)
    issue_counts = Counter(issue.issueCode for issue in issues)
    assertion_counts = Counter(str(record.get("assertionStatus")) for record in semantic_edges)
    relation_counts = Counter(str(record.get("relation")) for record in semantic_edges)
    node_count = sum(record.get("recordType") == "node" for record in final_records)
    edge_count = sum(record.get("recordType") == "edge" for record in final_records)
    manifest = GraphBuildManifest(
        graphVersion=graph_version,
        buildRunId=build_run_id,
        createdAt=datetime.now(timezone.utc).isoformat(),
        inputAsset="medical_chunks.jsonl",
        inputFingerprint=_input_fingerprint(materialized),
        paperCount=base_counts.get("papers", 0),
        chunkCount=base_counts.get("chunks", 0),
        nodeCount=node_count,
        edgeCount=edge_count,
        semanticRelationCount=len(semantic_edges),
        resolvedEntityCount=entity_counts.get("resolved", 0),
        candidateEntityCount=entity_counts.get("candidate", 0),
        unmappedEntityCount=entity_counts.get("unmapped", 0) + entity_counts.get("ambiguous", 0),
        reviewRequiredRelationCount=review_count,
        rejectedRecordCount=len(rejected_ids),
        relationCounts=dict(sorted(relation_counts.items())),
        assertionCounts=dict(sorted(assertion_counts.items())),
        qualityIssueCounts=dict(sorted(issue_counts.items())),
        status="review_pending" if review_count else "staging",
    )
    return GraphBuildResult(records=final_records, manifest=manifest, issues=issues)


def load_graph_manifest(path: Path) -> GraphBuildManifest:
    try:
        manifest = GraphBuildManifest.model_validate_json(path.read_text(encoding="utf-8"))
        # Older generated v4 manifests may still say ``staging`` even though
        # they contain review-required relations.  Normalize that derived
        # state at the trust boundary; it never grants publication approval.
        if manifest.status == "staging" and manifest.reviewRequiredRelationCount:
            manifest = manifest.model_copy(update={"status": "review_pending"})
        return manifest
    except Exception as exc:
        raise GraphPipelineError("GRAPH_MANIFEST_INVALID") from exc


def _validate_publication_gate(
    manifest: GraphBuildManifest,
    queue: GraphReviewQueue,
    approval: GraphPublicationApproval,
) -> None:
    if manifest.status not in {"review_pending", "approved"}:
        raise GraphPipelineError("GRAPH_VERSION_NOT_REVIEWABLE")
    if manifest.graphVersion != queue.graphVersion or manifest.buildRunId != queue.buildRunId:
        raise GraphPipelineError("GRAPH_REVIEW_QUEUE_MISMATCH")
    if manifest.graphVersion != approval.graphVersion or manifest.buildRunId != approval.buildRunId:
        raise GraphPipelineError("GRAPH_PUBLICATION_APPROVAL_MISMATCH")
    if queue.status != "CLOSED" or queue.pendingCount or queue.rejectedCount:
        raise GraphPipelineError("GRAPH_REVIEW_NOT_CLOSED")
    if approval.reviewQueueHash != queue.queueHash:
        raise GraphPipelineError("GRAPH_REVIEW_QUEUE_HASH_MISMATCH")
    if approval.reviewedItemCount != len(queue.items):
        raise GraphPipelineError("GRAPH_REVIEW_COUNT_MISMATCH")
    if approval.decision != "APPROVED":
        raise GraphPipelineError("GRAPH_PUBLICATION_NOT_APPROVED")


def approve_graph_manifest(
    path: Path,
    queue: GraphReviewQueue,
    approval: GraphPublicationApproval,
) -> GraphBuildManifest:
    """Persist an approval marker without switching the active graph version."""

    manifest = load_graph_manifest(path)
    if manifest.rejectedRecordCount:
        raise GraphPipelineError("GRAPH_VERSION_HAS_REJECTED_RECORDS")
    _validate_publication_gate(manifest, queue, approval)
    updated = manifest.model_copy(update={
        "status": "approved",
        "approvedAt": approval.approvedAt.isoformat(),
        "approvedBy": approval.approvedBy,
        "publicationApprovalHash": approval.reviewQueueHash,
    })
    path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    return updated


def publish_graph_manifest(
    path: Path,
    registry_path: Path,
) -> GraphPublicationRegistry:
    manifest = load_graph_manifest(path)
    if manifest.rejectedRecordCount:
        raise GraphPipelineError("GRAPH_VERSION_HAS_REJECTED_RECORDS")
    if manifest.reviewRequiredRelationCount:
        if manifest.status != "approved":
            raise GraphPipelineError("GRAPH_REVIEW_REQUIRED")
        if not manifest.publicationApprovalHash or not manifest.approvedBy:
            raise GraphPipelineError("GRAPH_PUBLICATION_APPROVAL_MISSING")
    elif manifest.status not in {"staging", "approved", "published"}:
        raise GraphPipelineError("GRAPH_VERSION_NOT_PUBLISHABLE")
    now = datetime.now(timezone.utc).isoformat()
    updated = manifest.model_copy(update={"status": "published", "publishedAt": now})
    path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    published: list[str] = []
    if registry_path.exists():
        try:
            existing = GraphPublicationRegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))
            published.extend(existing.publishedVersions)
        except Exception as exc:
            raise GraphPipelineError("GRAPH_REGISTRY_INVALID") from exc
    if manifest.graphVersion not in published:
        published.append(manifest.graphVersion)
    registry = GraphPublicationRegistry(
        currentGraphVersion=manifest.graphVersion,
        updatedAt=now,
        publishedVersions=published[-32:],
    )
    registry_path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    return registry


def save_graph_build(result: GraphBuildResult, output_path: Path, manifest_path: Path) -> None:
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in result.records),
        encoding="utf-8",
    )
    manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")
