from __future__ import annotations

from collections import OrderedDict
import re
from typing import Any

from mico_agent_runtime.contracts.evidence import EvidenceQuery, LiteratureEvidenceItem
from mico_agent_runtime.contracts.graph_rag import ReasoningPath
from mico_agent_runtime.contracts.retrieval import RerankBreakdown, RetrievalPlan


def _rrf_weights(plan: RetrievalPlan | None) -> tuple[float, float, float]:
    """Calibrated dense/sparse/graph weights by query class.

    These are safe starting priors only. The tuning job refuses to promote
    them until domain-reviewed qrels are approved.
    """
    if plan is not None and plan.fusionVersion == "rrf-standard-v1":
        return (1.0, 1.0, 1.0)
    query_type = plan.queryType if plan is not None else "semantic_fact"
    return {
        "semantic_fact": (0.55, 0.25, 0.20),
        "relation": (0.35, 0.20, 0.45),
        "multi_hop": (0.20, 0.15, 0.65),
        "composite": (0.45, 0.35, 0.20),
    }[query_type]


def _rrf_score(
    vector_rank: int | None,
    sparse_rank: int | None,
    graph_rank: int | None,
    plan: RetrievalPlan | None,
    active_weight: float | None = None,
) -> float:
    """Weighted reciprocal-rank fusion normalized to [0, 1]."""
    vector_weight, sparse_weight, graph_weight = _rrf_weights(plan)
    k = 60.0
    score = 0.0
    # Normalize over branches available for the whole candidate set, not over
    # branches present on one item.  Per-item normalization makes every
    # single-source candidate score 1.0 and lets a graph/listwise feature
    # displace a stronger dense hit merely because the other item came from a
    # different branch.
    if active_weight is None:
        active_weight = vector_weight + sparse_weight + graph_weight
    normalization = 1.0 / active_weight if active_weight else 0.0
    if vector_rank is not None:
        score += normalization * vector_weight / (k + vector_rank)
    if sparse_rank is not None:
        score += normalization * sparse_weight / (k + sparse_rank)
    if graph_rank is not None:
        score += normalization * graph_weight / (k + graph_rank)
    return min(1.0, score * (k + 1.0))


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


def _listwise_score(query: EvidenceQuery, item: LiteratureEvidenceItem) -> float:
    """Small, deterministic listwise feature used when no model endpoint is configured.

    It scores the *whole candidate representation* (title + excerpt + graph
    path labels) against the query. This is deliberately a ranking feature,
    not a relevance label and not a source generator. A production deployment
    can replace it with a cross-encoder through the same merge boundary.
    """
    query_tokens = {token for token in re.findall(r"[a-z][a-z0-9-]+", query.topic.lower()) if len(token) > 2}
    if not query_tokens:
        return 0.0
    text = " ".join([
        item.title or "",
        item.sourceExcerpt or "",
        *(hop.fromEntity + " " + hop.toEntity + " " + hop.relation for path in _paths(item) for hop in path.hops),
    ]).lower()
    hits = sum(token in text for token in query_tokens)
    coverage = hits / len(query_tokens)
    phrase = 1.0 if query.topic.lower().strip() in text else 0.0
    return min(1.0, 0.85 * coverage + 0.15 * phrase)


def _score_item(
    item: LiteratureEvidenceItem,
    vector_rank: int | None,
    sparse_rank: int | None,
    graph_rank: int | None,
    sources: list[str],
    vector_score: float,
    graph_score: float,
    sparse_score: float,
    paths: list[ReasoningPath],
) -> tuple[float, RerankBreakdown]:
    vector_contribution = min(1.0, max(0.0, vector_score))
    graph_contribution = min(1.0, max(0.0, graph_score))
    sparse_contribution = min(1.0, max(0.0, sparse_score))
    rank_values = [60.0 / (60.0 + rank) for rank in (vector_rank, sparse_rank, graph_rank) if rank is not None]
    reciprocal = sum(rank_values) / len(rank_values) if rank_values else 0.0
    path_support = max((path.pathScore for path in paths), default=0.0)
    source_diversity = min(1.0, len(set(sources)) / 3.0)
    max_hops = max((path.hopCount for path in paths), default=0)
    hop_penalty = min(1.0, max(0, max_hops - 1) / 3.0)
    status_penalty = _status_penalty(paths)
    # A single branch keeps its native ordering. Hybrid uses RRF below and
    # never compares cosine, FTS and path scores as if they were probabilities.
    if vector_rank is not None and sparse_rank is None and graph_rank is None:
        final = 0.70 * vector_contribution + 0.10 * (60.0 / (60.0 + vector_rank)) + 0.02
    elif sparse_rank is not None and vector_rank is None and graph_rank is None:
        final = 0.80 * sparse_contribution + 0.10 * (60.0 / (60.0 + sparse_rank))
    elif graph_rank is not None and vector_rank is None and sparse_rank is None:
        final = (
            0.28 * graph_contribution + 0.12 * reciprocal + 0.10 * path_support
            + 0.10 * source_diversity - 0.05 * hop_penalty - 0.14 * status_penalty
        )
    else:
        final = reciprocal
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
        sparseContribution=round(sparse_contribution, 8),
    )


def merge_and_rerank(
    query: EvidenceQuery,
    vector_results: list[LiteratureEvidenceItem],
    graph_results: list[LiteratureEvidenceItem],
    plan: RetrievalPlan | None = None,
    sparse_results: list[LiteratureEvidenceItem] | None = None,
    model_reranker: Any | None = None,
) -> list[LiteratureEvidenceItem]:
    """Fuse Dense/Sparse/Graph candidates and apply evidence constraints."""
    sparse_results = sparse_results or []
    is_hybrid = plan.retrievalMode == "hybrid" if plan is not None else (
        sum(bool(branch) for branch in (vector_results, sparse_results, graph_results)) > 1
    )
    vector_rank = {(item.sourceChunkId or item.externalId): i for i, item in enumerate(vector_results, 1)}
    sparse_rank = {(item.sourceChunkId or item.externalId): i for i, item in enumerate(sparse_results, 1)}
    graph_rank = {(item.sourceChunkId or item.externalId): i for i, item in enumerate(graph_results, 1)}
    vector_weight, sparse_weight, graph_weight = _rrf_weights(plan)
    active_weight = sum(
        weight for weight, present in (
            (vector_weight, bool(vector_results)),
            (sparse_weight, bool(sparse_results)),
            (graph_weight, bool(graph_results)),
        ) if present
    )
    merged: OrderedDict[str, LiteratureEvidenceItem] = OrderedDict()
    for item in vector_results + sparse_results + graph_results:
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
            "sparseScore": max(current.sparseScore, item.sparseScore),
            "graphScore": max(current.graphScore, item.graphScore),
            "graphPaths": [path.to_graph_path() for path in list(paths.values())[:4]],
            "reasoningPaths": list(paths.values())[:4],
            "summary": "Full-text evidence jointly ranked by dense, sparse and graph branches.",
        })

    final_items: list[LiteratureEvidenceItem] = []
    for key, item in merged.items():
        sources = sorted(set(item.retrievalSources))
        paths = _paths(item)
        final_score, breakdown = _score_item(
            item, vector_rank.get(key), sparse_rank.get(key), graph_rank.get(key),
            sources, item.vectorScore, item.graphScore, item.sparseScore, paths,
        )
        if is_hybrid:
            rrf_score = round(max(0.0, _rrf_score(
                vector_rank.get(key), sparse_rank.get(key), graph_rank.get(key),
                plan, active_weight,
            ) - 0.14 * _status_penalty(paths)), 8)
            # Listwise reranking is intentionally applied below to the RRF
            # Top-50 only. This preserves the candidate-recall boundary.
            final_score = rrf_score
            breakdown = breakdown.model_copy(update={
                "reciprocalRankContribution": rrf_score,
                "finalScore": final_score,
            })
        final_items.append(item.model_copy(update={
            "retrievalRoute": "hybrid" if is_hybrid else item.retrievalRoute,
            "retrievalSources": sources,
            "rerankScore": final_score,
            "retrievalScore": final_score,
            "rerankBreakdown": breakdown,
            "graphPaths": [path.to_graph_path() for path in paths[:4]],
            "reasoningPaths": paths[:4],
            "summary": (
                "Full-text evidence jointly ranked by dense, sparse and graph branches."
                if is_hybrid else item.summary
            ),
        }))

    if is_hybrid and final_items and (plan is None or plan.fusionVersion != "rrf-standard-v1"):
        # Stage 2: deterministic/listwise feature rerank over only RRF Top-50.
        final_items.sort(key=lambda item: (-item.rerankScore, item.externalId))
        rerank_pool = final_items[:50]
        tail = final_items[50:]
        final_items = []
        for item in rerank_pool:
            listwise_score = _listwise_score(query, item)
            # Keep lexical listwise features as a tie-breaker on the degraded
            # path; with dense candidates present it gets the calibrated 15%
            # influence and can correct branch-rank ties.
            listwise_weight = 0.05 if not vector_results else 0.15
            score = round((1.0 - listwise_weight) * item.rerankScore + listwise_weight * listwise_score, 8)
            final_items.append(item.model_copy(update={
                "rerankScore": score,
                "retrievalScore": score,
                "rerankBreakdown": item.rerankBreakdown.model_copy(update={
                    "listwiseContribution": round(listwise_score, 8),
                    "finalScore": score,
                }) if item.rerankBreakdown is not None else None,
            }))
        final_items.extend(tail)

    if is_hybrid and model_reranker is not None and final_items:
        # Stage 3: optional learned cross-encoder over the same bounded pool.
        # The adapter may return a partial map; missing scores are preserved.
        model_pool = final_items[:50]
        payload = [
            {
                "chunkId": str(item.sourceChunkId or item.externalId),
                "title": item.title,
                "text": item.sourceExcerpt or item.summary,
            }
            for item in model_pool
        ]
        try:
            model_scores = model_reranker.rerank(query.topic, payload)
        except Exception:
            model_scores = {}
        if model_scores:
            updated: list[LiteratureEvidenceItem] = []
            for item in final_items:
                key = str(item.sourceChunkId or item.externalId)
                if key not in model_scores:
                    updated.append(item)
                    continue
                model_score = max(0.0, min(1.0, float(model_scores[key])))
                score = round(0.70 * item.rerankScore + 0.30 * model_score, 8)
                updated.append(item.model_copy(update={
                    "rerankScore": score,
                    "retrievalScore": score,
                    "rerankBreakdown": item.rerankBreakdown.model_copy(update={
                        "listwiseContribution": model_score,
                        "finalScore": score,
                    }) if item.rerankBreakdown is not None else None,
                }))
            final_items = updated

    final_items.sort(key=lambda item: (-item.rerankScore, item.externalId))
    limit = min(query.limit, plan.topK if plan is not None else query.limit)
    if is_hybrid and plan is not None and plan.queryType == "multi_hop" and graph_results:
        final_items.sort(key=lambda item: (
            0 if "graph" in item.retrievalSources else 1,
            graph_rank.get(item.sourceChunkId or item.externalId, 10**9),
            -item.rerankScore,
            item.externalId,
        ))
    return _constrained_select(final_items, limit, plan)


def _document_key(item: LiteratureEvidenceItem) -> str:
    return (item.externalId or "").split("#", 1)[0]


def _has_uncertain_path(item: LiteratureEvidenceItem) -> bool:
    paths = _paths(item)
    statuses = {path.status for path in paths}
    statuses.update(hop.supportStatus for path in paths for hop in path.hops)
    return bool(statuses & {"speculative", "conflicted", "unsupported"})


def _constrained_select(
    ranked: list[LiteratureEvidenceItem],
    limit: int,
    plan: RetrievalPlan | None,
) -> list[LiteratureEvidenceItem]:
    """Select source-bound evidence under duplicate/path/status constraints."""
    if not ranked or limit <= 0:
        return []
    query_type = plan.queryType if plan is not None else "semantic_fact"
    selected: list[LiteratureEvidenceItem] = []
    counts: dict[str, int] = {}
    deferred: list[LiteratureEvidenceItem] = []

    def add(item: LiteratureEvidenceItem) -> bool:
        key = _document_key(item)
        if counts.get(key, 0) >= 2:
            return False
        selected.append(item)
        counts[key] = counts.get(key, 0) + 1
        return True

    for item in ranked:
        if _has_uncertain_path(item):
            deferred.append(item)
            continue
        if len(selected) >= limit:
            break
        add(item)

    if query_type == "multi_hop" and not any("graph" in item.retrievalSources for item in selected):
        graph_item = next((item for item in ranked if "graph" in item.retrievalSources), None)
        if graph_item is not None:
            if selected:
                selected[-1] = graph_item
            else:
                add(graph_item)

    if query_type == "composite" and len({_document_key(item) for item in selected}) < 2:
        diverse = next((item for item in ranked if _document_key(item) not in counts), None)
        if diverse is not None and selected:
            selected[-1] = diverse

    if len(selected) < limit:
        for item in deferred:
            if len(selected) >= limit:
                break
            add(item)
    return selected[:limit]
