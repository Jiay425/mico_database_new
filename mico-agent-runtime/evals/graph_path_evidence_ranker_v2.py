"""Two-stage query-specific graph retrieval over the versioned graph export.

Stage A ranks bounded semantic paths, not chunks.  Stage B projects the
evidence chunks carried by the winning paths.  This is intentionally separate
from the historical chunk-edge-count proxy so both can be evaluated fairly.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from mico_agent_runtime.knowledge.database_retriever import _normalize_graph_query
from mico_agent_runtime.knowledge.graph_v3 import extract_mentions
from mico_agent_runtime.knowledge.local_retriever import _tokens


SEMANTIC_CLASSES = {"association", "causal", "directional"}


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str
    confidence: float
    assertion: str
    evidence_chunk_id: str


@dataclass(frozen=True)
class Path:
    edges: tuple[Edge, ...]
    nodes: tuple[str, ...]
    score: float
    entity_coverage: float
    relation_match: float
    path_confidence: float
    hop_quality: float
    hub_penalty: float


def _stable_entity_key(node_id: str) -> str:
    return node_id.split(":", 1)[-1]


def _assertion_factor(assertion: str) -> float:
    return {"asserted": 1.0, "speculative": 0.50, "negated": 0.25, "conflicted": 0.0}.get(assertion.lower(), 0.0)


def _intent(question: str, router_type: str) -> str:
    value = question.lower()
    if router_type == "multi_hop" or any(marker in value for marker in (
        "mechanism", "how does", "how do", "through what", "pathway", "lead to",
    )):
        return "causal"
    if any(marker in value for marker in ("increased", "decreased", "higher", "lower", "enriched", "depleted")):
        return "directional"
    return "association"


def _relation_match(relation: str, intent: str) -> float:
    relation = relation.upper()
    if intent == "causal":
        return {"CAUSES": 1.0, "PROMOTES": 0.9, "INHIBITS": 0.9, "MEDIATES": 0.8,
                "ASSOCIATED_WITH": 0.4, "INCREASED_IN": 0.4, "DECREASED_IN": 0.4}.get(relation, 0.0)
    if intent == "directional":
        return {"INCREASED_IN": 1.0, "DECREASED_IN": 1.0, "CAUSES": 0.5, "PROMOTES": 0.5,
                "INHIBITS": 0.5, "MEDIATES": 0.5, "ASSOCIATED_WITH": 0.45}.get(relation, 0.0)
    return 1.0 if relation in {"CAUSES", "MEDIATES", "PROMOTES", "INHIBITS", "INCREASED_IN", "DECREASED_IN", "ASSOCIATED_WITH"} else 0.0


def _link_query_entities(nodes: dict[str, dict[str, Any]], question: str) -> set[str]:
    """Link controlled query mentions to their exact versioned graph nodes."""
    mention_keys = {_stable_entity_key(mention.node_id) for mention in extract_mentions(question)}
    linked = {node_id for node_id in nodes if _stable_entity_key(node_id) in mention_keys}
    if linked:
        return linked
    # Vocabulary fallback remains node-level rather than chunk-level: it may
    # find a graph entity but never treats arbitrary chunk text as an entity.
    query_tokens = set(_tokens(question))
    return {
        node_id for node_id, node in nodes.items()
        if str(node.get("nodeType") or "") in {"disease", "concept", "metabolite", "pathway", "hostprocess", "taxon"}
        and query_tokens & set(_tokens(str(node.get("label") or "")))
    }


def _semantic_adjacency(rows: list[dict[str, Any]]) -> dict[str, list[Edge]]:
    adjacency: dict[str, list[Edge]] = defaultdict(list)
    for row in rows:
        if row.get("recordType") != "edge" or str(row.get("relationClass") or "") not in SEMANTIC_CLASSES:
            continue
        if str(row.get("qualityStatus") or "accepted") != "accepted":
            continue
        source, target = str(row.get("source") or ""), str(row.get("target") or "")
        chunk_id, confidence = str(row.get("evidenceChunkId") or ""), float(row.get("confidence") or 0.0)
        if not source or not target or not chunk_id or confidence < 0.65:
            continue
        edge = Edge(source, target, str(row.get("relation") or ""), confidence, str(row.get("assertionStatus") or "asserted"), chunk_id)
        adjacency[source].append(edge)
        # Traversal is undirected for retrieval.  The edge retains its
        # original relation/evidence; direction is exposed in path metadata.
        adjacency[target].append(Edge(target, source, edge.relation, edge.confidence, edge.assertion, edge.evidence_chunk_id))
    for node_id in adjacency:
        adjacency[node_id].sort(key=lambda edge: (-edge.confidence * _assertion_factor(edge.assertion), edge.relation, edge.target, edge.evidence_chunk_id))
    return adjacency


def _score_path(
    edges: tuple[Edge, ...], nodes: tuple[str, ...], seed_ids: set[str], intent: str,
    degrees: dict[str, int], max_degree: int,
) -> Path:
    covered = len(set(nodes) & seed_ids)
    required = min(2, len(seed_ids))
    entity_coverage = 1.0 if covered >= required else (0.5 if covered else 0.0)
    relation_match = sum(_relation_match(edge.relation, intent) for edge in edges) / len(edges)
    # Geometric mean rewards consistently supported paths; the minimum edge
    # remains a hard upper bound so one weak assertion cannot be washed out.
    weighted = [edge.confidence * _assertion_factor(edge.assertion) for edge in edges]
    geometric = math.prod(weighted) ** (1.0 / len(weighted))
    path_confidence = min(min(weighted), geometric)
    hop_quality = {1: 1.0, 2: 0.75, 3: 0.55}[len(edges)]
    # Only intermediates are subject to hub penalty.  Query entities may be
    # frequent and are not evidence of an irrelevant bridge by themselves.
    hub_ratio = max((degrees.get(node_id, 0) / max(1, max_degree) for node_id in nodes[1:-1]), default=0.0)
    hub_penalty = 0.20 * hub_ratio
    score = (
        0.35 * entity_coverage
        + 0.30 * relation_match
        + 0.20 * path_confidence
        + 0.15 * hop_quality
        - hub_penalty
    )
    return Path(edges, nodes, round(max(0.0, score), 8), entity_coverage, relation_match, path_confidence, hop_quality, hub_penalty)


def _valid_complete_path(path_nodes: tuple[str, ...], path_edges: tuple[Edge, ...], seed_ids: set[str], router_type: str) -> bool:
    if router_type == "relation" and len(seed_ids) >= 2:
        # A relation query may use two or three edge bridge evidence, but its
        # path must genuinely connect two entities stated in the query.
        return len(set(path_nodes) & seed_ids) >= 2
    if router_type == "multi_hop":
        # With one explicit entity, bridge evidence needs at least two
        # semantic edges; a lone adjacent fact is not multi-hop evidence.
        return len(path_edges) >= 2
    return bool(path_edges)


def rank_path_evidence_graph(rows: list[dict[str, Any]], question: str, router_type: str, limit: int = 20) -> list[dict[str, Any]]:
    nodes = {str(row.get("nodeId")): row for row in rows if row.get("recordType") == "node"}
    normalized_question = _normalize_graph_query(question)
    seed_ids = _link_query_entities(nodes, normalized_question)
    if not seed_ids:
        return []
    adjacency = _semantic_adjacency(rows)
    degrees = {node_id: len(edges) for node_id, edges in adjacency.items()}
    max_degree = max(degrees.values(), default=1)
    intent = _intent(normalized_question, router_type)
    paths: dict[tuple[tuple[str, str, str], ...], Path] = {}
    for seed in sorted(seed_ids):
        queue: deque[tuple[str, tuple[Edge, ...], tuple[str, ...], frozenset[str]]] = deque([(seed, (), (seed,), frozenset({seed}))])
        expanded = 0
        while queue and expanded < 1_500:
            current, path_edges, path_nodes, seen = queue.popleft()
            expanded += 1
            if path_edges and _valid_complete_path(path_nodes, path_edges, seed_ids, router_type):
                path = _score_path(path_edges, path_nodes, seed_ids, intent, degrees, max_degree)
                signature = tuple((edge.source, edge.relation, edge.target) for edge in path_edges)
                previous = paths.get(signature)
                if previous is None or path.score > previous.score:
                    paths[signature] = path
            if len(path_edges) == 3:
                continue
            for edge in adjacency.get(current, [])[:32]:
                if edge.target in seen:
                    continue
                queue.append((edge.target, path_edges + (edge,), path_nodes + (edge.target,), seen | {edge.target}))

    # Stage B: project each accepted path to the evidence chunks of every
    # semantic edge.  A bridge chunk need not mention both query endpoints;
    # it is retained because the full path connects them.
    support: dict[str, list[Path]] = defaultdict(list)
    for path in paths.values():
        for edge in path.edges:
            support[edge.evidence_chunk_id].append(path)
    candidates: list[dict[str, Any]] = []
    for chunk_id, chunk_paths in support.items():
        best = max(chunk_paths, key=lambda path: path.score)
        distinct_paths = {tuple((edge.source, edge.relation, edge.target) for edge in path.edges) for path in chunk_paths}
        score = min(1.0, best.score + min(0.05, 0.02 * max(0, len(distinct_paths) - 1)))
        candidates.append({
            "chunkId": chunk_id,
            "score": round(score, 8),
            "pathEvidence": {
                "pathScore": best.score,
                "hopCount": len(best.edges),
                "entityCoverage": best.entity_coverage,
                "relationMatch": best.relation_match,
                "pathConfidence": best.path_confidence,
                "hubPenalty": best.hub_penalty,
                "supportingPathCount": len(distinct_paths),
                "evidenceChunkIds": list(dict.fromkeys(edge.evidence_chunk_id for edge in best.edges)),
            },
        })
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["chunkId"])))
    return [{**row, "rank": index} for index, row in enumerate(candidates[:limit], 1)]
