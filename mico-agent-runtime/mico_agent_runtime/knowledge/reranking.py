from __future__ import annotations

from collections import OrderedDict

from mico_agent_runtime.contracts.evidence import EvidenceQuery, LiteratureEvidenceItem
from mico_agent_runtime.contracts.graph_rag import ReasoningPath
from mico_agent_runtime.contracts.retrieval import RerankBreakdown, RetrievalPlan


def _paths(item: LiteratureEvidenceItem) -> list[ReasoningPath]:
    if item.reasoningPaths:
        return item.reasoningPaths[:4]
    return [ReasoningPath.from_graph_path(path) for path in item.graphPaths[:4]]


def _status_penalty(paths: list[ReasoningPath]) -> float:
    statuses = {path.status for path in paths}
    hop_statuses = {hop.supportStatus for path in paths for hop in path.hops}
    combined = statuses | hop_statuses
    if "conflicted" in combined:
        return 0.75
    if "speculative" in combined:
        return 0.45
    if "partial" in combined or "unsupported" in combined:
        return 0.25
    return 0.0


def _score_item(
    item: LiteratureEvidenceItem,
    vector_rank: int | None,
    graph_rank: int | None,
    sources: list[str],
    vector_score: float,
    graph_score: float,
    paths: list[ReasoningPath],
) -> tuple[float, RerankBreakdown]:
    vector_contribution = min(1.0, max(0.0, vector_score))
    graph_contribution = min(1.0, max(0.0, graph_score))
    rank_values = []
    for rank in (vector_rank, graph_rank):
        if rank is not None:
            rank_values.append(60.0 / (60.0 + rank))
    reciprocal = sum(rank_values) / len(rank_values) if rank_values else 0.0
    path_support = max((path.pathScore for path in paths), default=0.0)
    source_diversity = min(1.0, len(set(sources)) / 2.0)
    max_hops = max((path.hopCount for path in paths), default=0)
    hop_penalty = min(1.0, max(0, max_hops - 1) / 3.0)
    status_penalty = _status_penalty(paths)
    final = (
        0.32 * vector_contribution
        + 0.25 * graph_contribution
        + 0.18 * reciprocal
        + 0.15 * path_support
        + 0.10 * source_diversity
        - 0.06 * hop_penalty
        - 0.14 * status_penalty
    )
    final = min(1.0, max(0.0, final))
    return round(final, 8), RerankBreakdown(
        vectorContribution=round(vector_contribution, 8),
        graphContribution=round(graph_contribution, 8),
        reciprocalRankContribution=round(reciprocal, 8),
        pathSupportContribution=round(path_support, 8),
        sourceDiversityContribution=round(source_diversity, 8),
        hopPenalty=round(hop_penalty, 8),
        statusPenalty=round(status_penalty, 8),
        finalScore=round(final, 8),
    )


def merge_and_rerank(
    query: EvidenceQuery,
    vector_results: list[LiteratureEvidenceItem],
    graph_results: list[LiteratureEvidenceItem],
    plan: RetrievalPlan | None = None,
) -> list[LiteratureEvidenceItem]:
    """Fuse branch results into one bounded, provenance-preserving ranking."""
    is_hybrid = plan.retrievalMode == "hybrid" if plan is not None else bool(vector_results and graph_results)
    vector_rank = {
        (item.sourceChunkId or item.externalId): index
        for index, item in enumerate(vector_results, start=1)
    }
    graph_rank = {
        (item.sourceChunkId or item.externalId): index
        for index, item in enumerate(graph_results, start=1)
    }
    merged: OrderedDict[str, LiteratureEvidenceItem] = OrderedDict()
    for item in vector_results + graph_results:
        key = item.sourceChunkId or item.externalId
        if key not in merged:
            merged[key] = item
            continue
        current = merged[key]
        paths = {path.pathId: path for path in _paths(current)}
        paths.update({path.pathId: path for path in _paths(item)})
        merged[key] = current.model_copy(update={
            "retrievalRoute": "hybrid",
            "retrievalSources": sorted(set(current.retrievalSources + item.retrievalSources)),
            "vectorScore": max(current.vectorScore, item.vectorScore),
            "graphScore": max(current.graphScore, item.graphScore),
            "graphPaths": [path.to_graph_path() for path in list(paths.values())[:4]],
            "reasoningPaths": list(paths.values())[:4],
            "summary": "Full-text evidence jointly ranked by vector and graph branches.",
        })

    final_items: list[LiteratureEvidenceItem] = []
    for key, item in merged.items():
        sources = sorted(set(item.retrievalSources))
        paths = _paths(item)
        final_score, breakdown = _score_item(
            item,
            vector_rank.get(key),
            graph_rank.get(key),
            sources,
            item.vectorScore,
            item.graphScore,
            paths,
        )
        final_items.append(item.model_copy(update={
            "retrievalRoute": "hybrid" if is_hybrid else item.retrievalRoute,
            "retrievalSources": sources,
            "rerankScore": final_score,
            "retrievalScore": final_score,
            "rerankBreakdown": breakdown,
            "graphPaths": [path.to_graph_path() for path in paths[:4]],
            "reasoningPaths": paths[:4],
            "summary": (
                "Full-text evidence jointly ranked by vector and graph branches."
                if is_hybrid
                else item.summary
            ),
        }))
    final_items.sort(key=lambda item: (-item.rerankScore, item.externalId))
    limit = min(query.limit, plan.topK if plan is not None else query.limit)
    if vector_results and graph_results and limit > 1:
        selected = final_items[:limit]
        if not any("vector" in item.retrievalSources for item in selected):
            selected[-1] = next(item for item in final_items if "vector" in item.retrievalSources)
        if not any("graph" in item.retrievalSources for item in selected):
            selected[-1] = next(item for item in final_items if "graph" in item.retrievalSources)
        final_items = sorted(selected, key=lambda item: (-item.rerankScore, item.externalId))
    return final_items[:limit]
