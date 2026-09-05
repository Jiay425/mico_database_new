"""Query-specific graph evidence ranking for the fixed GraphRAG evaluation.

This intentionally replaces the earlier chunk-level proxy (token overlap +
edge counts) only in the evaluation candidate pool.  It ranks evidence chunks
through accepted semantic edges and bounded 1--3 semantic-edge paths.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from mico_agent_runtime.knowledge.graph_v3 import extract_mentions
from mico_agent_runtime.knowledge.local_retriever import _tokens


SEMANTIC_CLASSES = {"association", "causal", "directional"}
STRUCTURAL_RELATIONS = {"PART_OF", "IN_SECTION", "HAS_TOPIC", "MENTIONS_ENTITY"}


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    relation: str
    confidence: float
    assertion: str
    chunk_id: str


def _stable_entity_key(node_id: str) -> str:
    """Make a v3 query entity ID comparable with a versioned graph node ID."""
    return node_id.split(":", 1)[-1]


def _intent(question: str, router_type: str) -> str:
    value = question.lower()
    if router_type == "multi_hop" or any(marker in value for marker in (
        "mechanism", "how does", "how do", "lead to", "pathway",
    )):
        return "causal"
    if any(marker in value for marker in ("increased", "decreased", "higher", "lower", "enriched", "depleted")):
        return "directional"
    return "relation"


def _relation_match(relation: str, intent: str) -> float:
    relation = relation.upper()
    if intent == "causal":
        return {
            "CAUSES": 1.0, "PROMOTES": 0.9, "INHIBITS": 0.9,
            "MEDIATES": 0.8, "ASSOCIATED_WITH": 0.4,
            "INCREASED_IN": 0.4, "DECREASED_IN": 0.4,
        }.get(relation, 0.0)
    if intent == "directional":
        return {
            "INCREASED_IN": 1.0, "DECREASED_IN": 1.0,
            "ASSOCIATED_WITH": 0.45, "CAUSES": 0.5,
            "PROMOTES": 0.5, "INHIBITS": 0.5, "MEDIATES": 0.5,
        }.get(relation, 0.0)
    # A question that asks generically for a relationship accepts every
    # supported semantic relation, but never structural provenance edges.
    return 1.0 if relation in {
        "CAUSES", "MEDIATES", "PROMOTES", "INHIBITS", "INCREASED_IN",
        "DECREASED_IN", "ASSOCIATED_WITH",
    } else 0.0


def _assertion_factor(assertion: str) -> float:
    return {"asserted": 1.0, "speculative": 0.5, "negated": 0.25, "conflicted": 0.0}.get(assertion.lower(), 0.0)


def _path_score(path: tuple[_Edge, ...], seed_ids: set[str], intent: str) -> float:
    """Score one evidence path; every term is query-specific and semantic."""
    visited = {path[0].source, path[-1].target}
    covered = len(visited & seed_ids)
    # A two-entity relation requires both endpoints for a full entity score.
    entity_coverage = 1.0 if covered >= min(2, len(seed_ids)) else (0.5 if covered else 0.0)
    relation_match = sum(_relation_match(edge.relation, intent) for edge in path) / len(path)
    semantic_confidence = min(edge.confidence * _assertion_factor(edge.assertion) for edge in path)
    path_quality = {1: 1.0, 2: 0.75, 3: 0.55}.get(len(path), 0.0)
    return 0.35 * entity_coverage + 0.30 * relation_match + 0.20 * semantic_confidence + 0.15 * path_quality


def rank_path_aware_graph(
    graph_rows: list[dict[str, Any]], question: str, router_type: str, limit: int = 20,
) -> list[dict[str, Any]]:
    """Return evidence chunks ranked by accepted 1--3 hop semantic paths.

    Unlike the old proxy, structural edges never contribute a score.  A chunk
    is returned only because it is the recorded evidence for a semantic edge
    on a bounded path starting from an entity mentioned in the query.
    """
    nodes = {str(row.get("nodeId")): row for row in graph_rows if row.get("recordType") == "node"}
    query_mentions = extract_mentions(question)
    seed_suffixes = {_stable_entity_key(mention.node_id) for mention in query_mentions}
    seed_ids = {
        node_id for node_id in nodes
        if _stable_entity_key(node_id) in seed_suffixes
    }
    # Fall back to exact label-token overlap only when the controlled entity
    # resolver found no query entity.  This keeps the ranker usable for an
    # incomplete vocabulary without reverting to chunk-level edge counting.
    if not seed_ids:
        wanted = set(_tokens(question))
        seed_ids = {
            node_id for node_id, node in nodes.items()
            if wanted & set(_tokens(str(node.get("label") or "")))
            and str(node.get("nodeType") or "") not in {"chunk", "section", "paper", "topic"}
        }
    if not seed_ids:
        return []

    adjacency: dict[str, list[_Edge]] = defaultdict(list)
    for row in graph_rows:
        if row.get("recordType") != "edge":
            continue
        if str(row.get("relationClass") or "") not in SEMANTIC_CLASSES:
            continue
        if str(row.get("qualityStatus") or "accepted") != "accepted":
            continue
        confidence = float(row.get("confidence") or 0.0)
        chunk_id = str(row.get("evidenceChunkId") or "")
        source, target = str(row.get("source") or ""), str(row.get("target") or "")
        if not chunk_id or not source or not target or confidence < 0.65:
            continue
        edge = _Edge(source, target, str(row.get("relation") or ""), confidence, str(row.get("assertionStatus") or "asserted"), chunk_id)
        adjacency[source].append(edge)
        adjacency[target].append(_Edge(target, source, edge.relation, edge.confidence, edge.assertion, edge.chunk_id))

    intent = _intent(question, router_type)
    best_by_chunk: dict[str, float] = {}
    # Explore fixed-length paths from query entities.  No repeated node is
    # allowed, so a high-degree node cannot create circular evidence paths.
    for seed in sorted(seed_ids):
        queue: deque[tuple[str, tuple[_Edge, ...], frozenset[str]]] = deque([(seed, (), frozenset({seed}))])
        # A microbiome concept may have hundreds of neighbors.  Bounded hops
        # alone are not enough: 100 x 100 x 100 is still unusable.  The
        # deterministic caps keep this a retrieval operation, not exhaustive
        # graph enumeration.  Edges are ordered by semantic confidence before
        # the cap, never by incidental input-file order.
        expanded = 0
        while queue and expanded < 1_500:
            current, path, seen = queue.popleft()
            expanded += 1
            if path:
                score = _path_score(path, seed_ids, intent)
                for edge in path:
                    best_by_chunk[edge.chunk_id] = max(best_by_chunk.get(edge.chunk_id, 0.0), score)
            if len(path) == 3:
                continue
            neighbors = sorted(
                adjacency.get(current, []),
                key=lambda edge: (-edge.confidence * _assertion_factor(edge.assertion), edge.relation, edge.target, edge.chunk_id),
            )[:32]
            for edge in neighbors:
                if edge.target in seen:
                    continue
                queue.append((edge.target, path + (edge,), seen | {edge.target}))

    ranked = sorted(best_by_chunk.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [{"chunkId": chunk_id, "score": round(score, 8), "rank": index}
            for index, (chunk_id, score) in enumerate(ranked, 1)]
