from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from .base import ClosedModel, NonEmptyText
from .evidence import LiteratureEvidenceItem
from .graph_rag import ReasoningPath
from .retrieval import RetrievalPlan
from .tools import JavaTransientSnapshotId

if TYPE_CHECKING:
    from .research import Observation


UnifiedEvidenceRoute = Literal["vector", "graph", "java"]
UnifiedEvidenceKind = Literal["literature_chunk", "java_observation"]
UnifiedEvidenceStatus = Literal["supported", "speculative", "conflicted", "partial", "unsupported"]
UnifiedEvidenceId = Annotated[str, Field(pattern=r"^evidence-[0-9a-f]{32}$")]
UnifiedBindingId = Annotated[str, Field(pattern=r"^binding-[0-9a-f]{32}$")]
ObservationReference = Annotated[str, Field(pattern=r"^observation-[0-9a-f]{32}$")]
Sha256Reference = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
VersionReference = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"),
]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(timezone.utc)


class UnifiedEvidenceSourceBinding(ClosedModel):
    """One auditable source contribution to a unified evidence candidate."""

    bindingId: UnifiedBindingId
    route: UnifiedEvidenceRoute
    origin: Literal["pgvector", "neo4j", "java_controlled_read"]
    evidenceId: UnifiedEvidenceId | None = None
    observationId: ObservationReference | None = None
    dataSnapshotId: JavaTransientSnapshotId | None = None
    snapshotPersistence: Literal["transient"] | None = None
    queryHash: Sha256Reference | None = None
    schemaVersion: VersionReference | None = None
    sourceChunkId: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    generatedAt: datetime

    @model_validator(mode="after")
    def validate_reference_boundary(self) -> "UnifiedEvidenceSourceBinding":
        references = [self.evidenceId is not None, self.observationId is not None]
        if sum(references) != 1:
            raise ValueError("a source binding must reference one evidence or observation")
        if self.route == "vector" and self.origin != "pgvector":
            raise ValueError("vector evidence must be bound to pgvector")
        if self.route == "graph" and self.origin != "neo4j":
            raise ValueError("graph evidence must be bound to Neo4j")
        if self.route == "java" and self.origin != "java_controlled_read":
            raise ValueError("Java evidence must be bound to the controlled Java source")
        if self.observationId is not None and self.dataSnapshotId is None:
            raise ValueError("Java observation bindings require a transient snapshot")
        if self.dataSnapshotId is not None and self.snapshotPersistence != "transient":
            raise ValueError("snapshot persistence must remain transient")
        if self.observationId is None and self.dataSnapshotId is not None:
            raise ValueError("literature bindings cannot carry a Java snapshot")
        return self


class UnifiedRerankBreakdown(ClosedModel):
    """Deterministic, inspectable score components for all three routes."""

    vectorContribution: float = Field(ge=0.0, le=1.0)
    graphContribution: float = Field(ge=0.0, le=1.0)
    javaContribution: float = Field(ge=0.0, le=1.0)
    reciprocalRankContribution: float = Field(ge=0.0, le=1.0)
    pathSupportContribution: float = Field(ge=0.0, le=1.0)
    sourceDiversityContribution: float = Field(ge=0.0, le=1.0)
    hopPenalty: float = Field(ge=0.0, le=1.0)
    conflictPenalty: float = Field(ge=0.0, le=1.0)
    finalScore: float = Field(ge=0.0, le=1.0)


class UnifiedEvidenceCandidate(ClosedModel):
    """One source-bound candidate after vector/graph/Java fusion."""

    candidateId: UnifiedEvidenceId
    kind: UnifiedEvidenceKind
    title: Annotated[str, Field(min_length=1, max_length=512)]
    summary: Annotated[str, Field(min_length=1, max_length=2000)]
    supportStatus: UnifiedEvidenceStatus
    sourceRoutes: list[UnifiedEvidenceRoute] = Field(min_length=1, max_length=3)
    sourceBindings: list[UnifiedEvidenceSourceBinding] = Field(min_length=1, max_length=3)
    reasoningPaths: list[ReasoningPath] = Field(default_factory=list, max_length=4)
    vectorScore: float = Field(default=0.0, ge=0.0, le=1.0)
    graphScore: float = Field(default=0.0, ge=0.0, le=1.0)
    javaScore: float = Field(default=0.0, ge=0.0, le=1.0)
    rerankScore: float = Field(default=0.0, ge=0.0, le=1.0)
    rerankBreakdown: UnifiedRerankBreakdown
    generatedAt: datetime

    @model_validator(mode="after")
    def validate_candidate_routes(self) -> "UnifiedEvidenceCandidate":
        if len(self.sourceRoutes) != len(set(self.sourceRoutes)):
            raise ValueError("source routes must be unique")
        binding_routes = {binding.route for binding in self.sourceBindings}
        if not binding_routes.issubset(set(self.sourceRoutes)):
            raise ValueError("source binding route is not present in sourceRoutes")
        if self.kind == "java_observation" and self.sourceRoutes != ["java"]:
            raise ValueError("Java observation candidates cannot be merged with literature chunks")
        if self.kind == "java_observation" and not any(
            binding.observationId is not None for binding in self.sourceBindings
        ):
            raise ValueError("Java observation candidates require an observation binding")
        if self.reasoningPaths and "graph" not in self.sourceRoutes:
            raise ValueError("reasoning paths require a graph source")
        return self


def _binding_id(reference: str, route: str) -> str:
    import hashlib

    return "binding-" + hashlib.sha256(f"{reference}|{route}".encode("utf-8")).hexdigest()[:32]


def _status(paths: list[ReasoningPath]) -> UnifiedEvidenceStatus:
    statuses = {path.status for path in paths}
    statuses.update(hop.supportStatus for path in paths for hop in path.hops)
    if "conflicted" in statuses:
        return "conflicted"
    if "unsupported" in statuses:
        return "unsupported"
    if "speculative" in statuses:
        return "speculative"
    if "partial" in statuses:
        return "partial"
    return "supported"


def _literature_candidate(item: LiteratureEvidenceItem, route: UnifiedEvidenceRoute) -> UnifiedEvidenceCandidate:
    # A vector-only candidate may carry graph paths as retrieval metadata from
    # a hybrid reranker. Those paths are not bound to a graph route until the
    # graph branch is merged; retaining them here violates the unified source
    # contract. The graph branch adds them when it is available.
    paths = [] if route == "vector" else (
        item.reasoningPaths or [ReasoningPath.from_graph_path(path) for path in item.graphPaths]
    )
    generated = datetime.now(timezone.utc)
    binding = UnifiedEvidenceSourceBinding(
        bindingId=_binding_id(item.evidenceId, route),
        route=route,
        origin="pgvector" if route == "vector" else "neo4j",
        evidenceId=item.evidenceId,
        queryHash=None,
        sourceChunkId=item.sourceChunkId,
        generatedAt=generated,
    )
    return UnifiedEvidenceCandidate(
        candidateId=item.evidenceId,
        kind="literature_chunk",
        title=item.title,
        summary=item.summary,
        supportStatus=_status(paths),
        sourceRoutes=[route],
        sourceBindings=[binding],
        reasoningPaths=paths[:4],
        vectorScore=item.vectorScore if route == "vector" else 0.0,
        graphScore=item.graphScore if route == "graph" else 0.0,
        rerankBreakdown=UnifiedRerankBreakdown(
            vectorContribution=item.vectorScore if route == "vector" else 0.0,
            graphContribution=item.graphScore if route == "graph" else 0.0,
            javaContribution=0.0,
            reciprocalRankContribution=0.0,
            pathSupportContribution=max((path.pathScore for path in paths), default=0.0),
            sourceDiversityContribution=0.5,
            hopPenalty=0.0,
            conflictPenalty=0.0,
            finalScore=0.0,
        ),
        generatedAt=generated,
    )


def _java_candidate(observation: Observation) -> UnifiedEvidenceCandidate:
    import hashlib

    candidate_id = "evidence-" + hashlib.sha256(
        f"java|{observation.observationId}".encode("utf-8")
    ).hexdigest()[:32]
    binding = UnifiedEvidenceSourceBinding(
        bindingId=_binding_id(observation.observationId, "java"),
        route="java",
        origin="java_controlled_read",
        observationId=observation.observationId,
        dataSnapshotId=observation.dataSnapshotId,
        snapshotPersistence=observation.snapshotPersistence,
        queryHash=observation.queryHash,
        schemaVersion=observation.schemaVersion,
        generatedAt=observation.generatedAt,
    )
    score = 1.0 if observation.status == "VALIDATED" else 0.5 if observation.status == "PARTIAL" else 0.0
    return UnifiedEvidenceCandidate(
        candidateId=candidate_id,
        kind="java_observation",
        title="Java controlled read observation",
        summary="Metadata-only observation from the Java business read boundary.",
        supportStatus="supported" if observation.status == "VALIDATED" else "partial",
        sourceRoutes=["java"],
        sourceBindings=[binding],
        javaScore=score,
        rerankBreakdown=UnifiedRerankBreakdown(
            vectorContribution=0.0,
            graphContribution=0.0,
            javaContribution=score,
            reciprocalRankContribution=0.0,
            pathSupportContribution=0.0,
            sourceDiversityContribution=0.5,
            hopPenalty=0.0,
            conflictPenalty=0.0,
            finalScore=0.0,
        ),
        generatedAt=observation.generatedAt,
    )


def merge_unified_evidence(
    *,
    vector_results: list[LiteratureEvidenceItem],
    graph_results: list[LiteratureEvidenceItem],
    java_observations: list[Observation],
    plan: RetrievalPlan | None = None,
    limit: int = 20,
) -> list[UnifiedEvidenceCandidate]:
    """Fuse three source routes while preserving every source binding and path."""
    limit = max(1, min(20, limit if plan is None else min(limit, plan.topK)))
    candidates: dict[str, UnifiedEvidenceCandidate] = {}
    ranks: dict[str, dict[str, int]] = {"vector": {}, "graph": {}, "java": {}}

    for route, items in (("vector", vector_results), ("graph", graph_results)):
        for rank, item in enumerate(items, start=1):
            key = item.sourceChunkId or item.externalId or item.evidenceId
            ranks[route].setdefault(key, rank)
            candidate = _literature_candidate(item, route)  # type: ignore[arg-type]
            if key not in candidates:
                candidates[key] = candidate
                continue
            current = candidates[key]
            path_map = {path.pathId: path for path in current.reasoningPaths}
            path_map.update({path.pathId: path for path in candidate.reasoningPaths})
            binding_map = {binding.route: binding for binding in current.sourceBindings}
            binding_map.update({binding.route: binding for binding in candidate.sourceBindings})
            routes = list(dict.fromkeys(current.sourceRoutes + candidate.sourceRoutes))
            candidates[key] = current.model_copy(update={
                "sourceRoutes": routes,
                "sourceBindings": list(binding_map.values())[:3],
                "reasoningPaths": list(path_map.values())[:4],
                "supportStatus": _status(list(path_map.values())[:4]),
                "vectorScore": max(current.vectorScore, candidate.vectorScore),
                "graphScore": max(current.graphScore, candidate.graphScore),
            })

    for observation in java_observations:
        candidates[observation.observationId] = _java_candidate(observation)
        ranks["java"].setdefault(observation.observationId, len(ranks["java"]) + 1)

    final: list[UnifiedEvidenceCandidate] = []
    for key, candidate in candidates.items():
        paths = candidate.reasoningPaths
        reciprocal_terms = [60.0 / (60.0 + ranks[route][key]) for route in candidate.sourceRoutes if key in ranks[route]]
        reciprocal = sum(reciprocal_terms) / len(reciprocal_terms) if reciprocal_terms else 0.0
        path_support = max((path.pathScore for path in paths), default=0.0)
        max_hops = max((path.hopCount for path in paths), default=0)
        hop_penalty = min(1.0, max(0, max_hops - 1) / 3.0)
        conflict_penalty = 0.9 if candidate.supportStatus == "conflicted" else 0.45 if candidate.supportStatus == "speculative" else 0.2 if candidate.supportStatus in {"partial", "unsupported"} else 0.0
        diversity = min(1.0, len(candidate.sourceRoutes) / 3.0)
        raw = (
            0.24 * candidate.vectorScore
            + 0.22 * candidate.graphScore
            + 0.24 * candidate.javaScore
            + 0.14 * reciprocal
            + 0.10 * path_support
            + 0.06 * diversity
            - 0.06 * hop_penalty
            - 0.14 * conflict_penalty
        )
        score = round(max(0.0, min(1.0, raw)), 8)
        breakdown = candidate.rerankBreakdown.model_copy(update={
            "vectorContribution": round(candidate.vectorScore, 8),
            "graphContribution": round(candidate.graphScore, 8),
            "javaContribution": round(candidate.javaScore, 8),
            "reciprocalRankContribution": round(reciprocal, 8),
            "pathSupportContribution": round(path_support, 8),
            "sourceDiversityContribution": round(diversity, 8),
            "hopPenalty": round(hop_penalty, 8),
            "conflictPenalty": round(conflict_penalty, 8),
            "finalScore": score,
        })
        final.append(candidate.model_copy(update={"rerankScore": score, "rerankBreakdown": breakdown}))

    final.sort(key=lambda item: (-item.rerankScore, item.candidateId))
    if len(final) > limit and len({route for item in final for route in item.sourceRoutes}) > 1:
        selected = final[:limit]
        for route in ("vector", "graph", "java"):
            if not any(route in item.sourceRoutes for item in selected):
                replacement = next((item for item in final if route in item.sourceRoutes), None)
                if replacement is not None:
                    selected[-1] = replacement
        final = sorted({item.candidateId: item for item in selected}.values(), key=lambda item: (-item.rerankScore, item.candidateId))
    return final[:limit]
