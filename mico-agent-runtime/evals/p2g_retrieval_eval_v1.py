"""GraphRAG retrieval evaluation with a reproducible local v3 corpus.

This is an evidence/provenance evaluation, not a human relevance benchmark.
It compares the same cases under vector, graph and hybrid routing and keeps
the distinction explicit in the output.  A future human-reviewed query set
can replace the generated provenance gold labels without changing the metric
or report format.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from mico_agent_runtime.contracts.evidence import EvidenceQuery
from mico_agent_runtime.knowledge.local_retriever import (
    LocalKnowledgeIndexConfiguration,
    LocalKnowledgeSearchPort,
)


SEMANTIC_CLASSES = {"association", "causal", "directional"}
ENTITY_TYPES = {
    "concept",
    "disease",
    "hostprocess",
    "metabolite",
    "pathway",
    "taxon",
    "topic",
}
BRANCHES = ("vector", "graph", "hybrid")
CATEGORIES = ("single_fact", "relation", "multi_hop", "document_synthesis")


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _unversioned(value: str) -> str:
    return re.sub(r"^v[0-9]+:", "", value)


def _chunk_from_evidence(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _pmcid_from_chunk(chunk_id: str) -> str:
    return chunk_id.split("-", 1)[0]


def _load_graph(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("recordType") == "node":
                nodes[str(row["nodeId"])] = row
            elif row.get("recordType") == "edge":
                edges.append(row)
    if not nodes or not edges:
        raise ValueError(f"empty graph asset: {path}")
    return nodes, edges


def _semantic_edges(nodes: dict[str, dict[str, Any]], edges: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for edge in edges:
        if edge.get("relationClass") not in SEMANTIC_CLASSES:
            continue
        source = nodes.get(str(edge.get("source")))
        target = nodes.get(str(edge.get("target")))
        if not source or not target:
            continue
        if source.get("nodeType") not in ENTITY_TYPES or target.get("nodeType") not in ENTITY_TYPES:
            continue
        if _norm(str(source.get("label") or "")) == _norm(str(target.get("label") or "")):
            continue
        if not _chunk_from_evidence(edge.get("evidenceChunkId")):
            continue
        result.append(edge)
    return sorted(result, key=lambda row: str(row.get("edgeId") or ""))


def _edge_labels(nodes: dict[str, dict[str, Any]], edge: dict[str, Any]) -> tuple[str, str] | None:
    source = nodes.get(str(edge.get("source")))
    target = nodes.get(str(edge.get("target")))
    if not source or not target:
        return None
    left = str(source.get("label") or "").strip()
    right = str(target.get("label") or "").strip()
    if len(left) < 4 or len(right) < 4:
        return None
    return left, right


def _find_pair(
    nodes: dict[str, dict[str, Any]], semantic: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any] | None:
    wanted = {_norm(left), _norm(right)}
    for edge in semantic:
        labels = _edge_labels(nodes, edge)
        if labels and {_norm(labels[0]), _norm(labels[1])} == wanted:
            return edge
    return None


def _case_from_edge(
    case_id: str,
    category: str,
    nodes: dict[str, dict[str, Any]],
    edge: dict[str, Any],
    query_prefix: str,
) -> dict[str, Any] | None:
    labels = _edge_labels(nodes, edge)
    chunk = _chunk_from_evidence(edge.get("evidenceChunkId"))
    if not labels or not chunk:
        return None
    left, right = labels
    return {
        "caseId": case_id,
        "category": category,
        "query": f"{query_prefix} {left} and {right}",
        "goldKind": "chunk",
        "goldChunkIds": [chunk],
        "goldRelation": str(edge.get("relation") or ""),
        "labels": [left, right],
    }


def _load_chunk_texts(vector_path: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    with vector_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = str(row.get("chunkId") or "")
            if chunk_id:
                texts[chunk_id] = str(row.get("text") or "").lower()
    return texts


def _label_tokens(label: str) -> list[str]:
    return [token for token in re.findall(r"[a-z][a-z0-9-]+", label.lower()) if len(token) > 2]


def _pair_gold_chunks(
    edge: dict[str, Any], labels: tuple[str, str], chunk_texts: dict[str, str]
) -> list[str]:
    """Use the source edge plus chunks whose text contains both entities.

    This avoids the invalid assumption that one relation edge has exactly one
    acceptable answer chunk.  It remains a deterministic textual/provenance
    oracle, not a substitute for human relevance judgments.
    """
    gold = set()
    evidence_chunk = _chunk_from_evidence(edge.get("evidenceChunkId"))
    if evidence_chunk:
        gold.add(evidence_chunk)
    left_tokens = _label_tokens(labels[0])
    right_tokens = _label_tokens(labels[1])
    if left_tokens and right_tokens:
        for chunk_id, text in chunk_texts.items():
            if all(token in text for token in left_tokens + right_tokens):
                gold.add(chunk_id)
    return sorted(gold)


def _build_cases(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    chunk_texts: dict[str, str],
) -> list[dict[str, Any]]:
    semantic = _semantic_edges(nodes, edges)
    cases: list[dict[str, Any]] = []

    single_pairs = [
        ("type 2 diabetes", "gut microbiome"),
        ("colorectal cancer", "Fusobacterium nucleatum"),
        ("inflammatory bowel disease", "dysbiosis"),
        ("obesity", "bile acid"),
        ("cirrhosis", "lipopolysaccharide"),
        ("multiple sclerosis", "gut microbiome"),
    ]
    relation_pairs = [
        ("alzheimer disease", "inflammation"),
        ("fatty liver disease", "short-chain fatty acids"),
        ("coronary artery disease", "inflammation"),
        ("type 2 diabetes", "butyrate"),
        ("obesity", "inflammation"),
        ("colorectal cancer", "bile acid"),
    ]
    for index, pair in enumerate(single_pairs, start=1):
        edge = _find_pair(nodes, semantic, *pair)
        if edge:
            case = _case_from_edge(
                f"single-{index:02d}", "single_fact", nodes, edge, "Find full-text evidence about"
            )
            if case:
                case["goldChunkIds"] = _pair_gold_chunks(edge, tuple(case["labels"]), chunk_texts)
                cases.append(case)
    for index, pair in enumerate(relation_pairs, start=1):
        edge = _find_pair(nodes, semantic, *pair)
        if edge:
            case = _case_from_edge(
                f"relation-{index:02d}", "relation", nodes, edge, "What relationship is reported between"
            )
            if case:
                case["goldChunkIds"] = _pair_gold_chunks(edge, tuple(case["labels"]), chunk_texts)
                cases.append(case)

    # Build deterministic two-edge states for multi-hop evaluation.  The
    # provenance gold is the union of both supporting chunks.
    adjacency: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in semantic:
        adjacency[str(edge["source"])].append(edge)
        adjacency[str(edge["target"])].append(edge)
    multi_added = 0
    seen_multi: set[tuple[str, str]] = set()
    for center in sorted(adjacency):
        incident = adjacency[center]
        for first_index, first in enumerate(incident):
            first_other = str(first["target"] if str(first["source"]) == center else first["source"])
            for second in incident[first_index + 1 :]:
                second_other = str(second["target"] if str(second["source"]) == center else second["source"])
                if first_other == second_other:
                    continue
                chunks = [
                    _chunk_from_evidence(first.get("evidenceChunkId")),
                    _chunk_from_evidence(second.get("evidenceChunkId")),
                ]
                if None in chunks or chunks[0] == chunks[1]:
                    continue
                labels = []
                for node_id in (first_other, center, second_other):
                    node = nodes.get(node_id)
                    label = str(node.get("label") or "").strip() if node else ""
                    if len(label) < 4:
                        labels = []
                        break
                    labels.append(label)
                if len(labels) != 3 or len({_norm(label) for label in labels}) != 3:
                    continue
                signature = (_norm(labels[0]), _norm(labels[2]))
                if signature in seen_multi:
                    continue
                seen_multi.add(signature)
                case = {
                    "caseId": f"multi-hop-{multi_added + 1:02d}",
                    "category": "multi_hop",
                    "query": f"Explain the multi-hop mechanism linking {labels[0]}, {labels[1]}, and {labels[2]}",
                    "goldKind": "chunk",
                    "goldChunkIds": list(dict.fromkeys(chunks)),
                    "labels": labels,
                }
                text_gold = [
                    chunk_id for chunk_id, text in chunk_texts.items()
                    if all(token in text for label in labels for token in _label_tokens(label))
                ]
                case["goldChunkIds"] = sorted(set(case["goldChunkIds"]) | set(text_gold))
                cases.append(case)
                multi_added += 1
                if multi_added >= 4:
                    break
            if multi_added >= 4:
                break
        if multi_added >= 4:
            break

    # Document synthesis cases use several source documents around a shared
    # entity.  They evaluate document diversity rather than pretending that a
    # single chunk is the only valid answer.
    by_node_docs: defaultdict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for edge in semantic:
        chunk = _chunk_from_evidence(edge.get("evidenceChunkId"))
        if not chunk:
            continue
        doc = _pmcid_from_chunk(chunk)
        for node_id in (str(edge["source"]), str(edge["target"])):
            by_node_docs[node_id][doc].add(chunk)
    synthesis_added = 0
    for node_id in sorted(by_node_docs):
        docs = by_node_docs[node_id]
        if len(docs) < 3:
            continue
        node = nodes.get(node_id)
        center_label = str(node.get("label") or "").strip() if node else ""
        if len(center_label) < 4:
            continue
        selected_docs = sorted(docs)[:3]
        selected_chunks = [sorted(docs[doc])[0] for doc in selected_docs]
        cases.append({
            "caseId": f"synthesis-{synthesis_added + 1:02d}",
            "category": "document_synthesis",
            "query": f"Synthesize full-text evidence across studies about {center_label}",
            "goldKind": "document",
            "goldChunkIds": selected_chunks,
            "goldDocumentIds": selected_docs,
            "labels": [center_label],
        })
        synthesis_added += 1
        if synthesis_added >= 4:
            break

    if len(cases) < 16:
        raise RuntimeError(f"could only construct {len(cases)} reproducible cases")
    return cases


def _prepare_local_index(source_dir: Path, graph_file: Path) -> tempfile.TemporaryDirectory[str]:
    temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="mico-p2g-eval-")
    target = Path(temp_dir.name)
    required = ("medical_knowledge_manifest.json", "medical_vector_meta.json", "medical_vector_index.jsonl")
    for name in required:
        shutil.copyfile(source_dir / name, target / name)
    shutil.copyfile(graph_file, target / "medical_knowledge_graph.jsonl")
    return temp_dir


class _CachedEmbeddingPort:
    """Avoid charging a second query embedding for vector + hybrid A/B."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.modelName = delegate.modelName
        self._query_cache: dict[str, list[float]] = {}

    def embed_query(self, query: str) -> list[float]:
        if query not in self._query_cache:
            self._query_cache[query] = self._delegate.embed_query(query)
        return self._query_cache[query]

    def embed_document(self, title: str | None, text: str) -> list[float]:
        return self._delegate.embed_document(title, text)

    def close(self) -> None:
        close = getattr(self._delegate, "close", None)
        if callable(close):
            close()


def _document_id(item: Any) -> str | None:
    external = str(getattr(item, "externalId", ""))
    match = re.search(r"PMCID:([^#]+)", external)
    return match.group(1) if match else None


def _metric_for_case(results: list[Any], case: dict[str, Any], limit: int = 10) -> dict[str, Any]:
    returned_chunks = [str(item.sourceChunkId) for item in results[:limit] if item.sourceChunkId]
    returned_docs = [doc for doc in (_document_id(item) for item in results[:limit]) if doc]
    if case["goldKind"] == "document":
        gold = set(case["goldDocumentIds"])
        ranked = returned_docs
    else:
        gold = set(case["goldChunkIds"])
        ranked = returned_chunks
    found = [value for value in ranked if value in gold]
    first_rank = next((index + 1 for index, value in enumerate(ranked) if value in gold), None)
    recall = len(set(found)) / len(gold) if gold else 0.0
    dcg = sum(1.0 / math.log2(index + 2) for index, value in enumerate(ranked) if value in gold)
    ideal_count = min(len(gold), limit)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count)) or 1.0
    ranked5 = ranked[:5]
    found5 = [value for value in ranked5 if value in gold]
    first_rank5 = next((index + 1 for index, value in enumerate(ranked5) if value in gold), None)
    recall5 = len(set(found5)) / len(gold) if gold else 0.0
    dcg5 = sum(1.0 / math.log2(index + 2) for index, value in enumerate(ranked5) if value in gold)
    idcg5 = sum(1.0 / math.log2(index + 2) for index in range(min(len(gold), 5))) or 1.0
    return {
        "hitAt5": bool(found5),
        "hitAt10": bool(found),
        "recallAt5": round(recall5, 8),
        "recallAt10": round(recall, 8),
        "mrr": round(1.0 / first_rank, 8) if first_rank else 0.0,
        "ndcgAt5": round(dcg5 / idcg5, 8),
        "ndcgAt10": round(dcg / idcg, 8),
        "returnedChunks": returned_chunks,
        "returnedDocuments": returned_docs,
        "firstRelevantRank": first_rank,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("error") is None]
    if not successful:
        return {"caseCount": len(rows), "successfulSearchCount": 0, "errorCount": len(rows)}
    return {
        "caseCount": len(rows),
        "successfulSearchCount": len(successful),
        "errorCount": len(rows) - len(successful),
        "hitRateAt5": round(sum(row["metrics"]["hitAt5"] for row in successful) / len(successful), 8),
        "hitRateAt10": round(sum(row["metrics"]["hitAt10"] for row in successful) / len(successful), 8),
        "meanRecallAt5": round(sum(row["metrics"]["recallAt5"] for row in successful) / len(successful), 8),
        "meanRecallAt10": round(sum(row["metrics"]["recallAt10"] for row in successful) / len(successful), 8),
        "mrr": round(sum(row["metrics"]["mrr"] for row in successful) / len(successful), 8),
        "ndcgAt5": round(sum(row["metrics"]["ndcgAt5"] for row in successful) / len(successful), 8),
        "ndcgAt10": round(sum(row["metrics"]["ndcgAt10"] for row in successful) / len(successful), 8),
    }


def _quality_audit(
    manifest: dict[str, Any], nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    semantic = [edge for edge in edges if edge.get("relationClass") in SEMANTIC_CLASSES]
    chunk_nodes = {
        str(node.get("nodeId")) for node in nodes.values() if node.get("nodeType") == "chunk"
    }
    semantic_chunk_ids = {
        str(edge.get("evidenceChunkId")) for edge in semantic if edge.get("evidenceChunkId")
    }
    manifest_semantic = int(manifest.get("semanticRelationCount") or 0)
    actual_assertions = Counter(str(edge.get("assertionStatus") or "missing") for edge in semantic)
    actual_relation_counts = Counter(str(edge.get("relation") or "missing") for edge in semantic)
    endpoint_missing = sum(
        str(edge.get("source")) not in nodes or str(edge.get("target")) not in nodes for edge in edges
    )
    missing_evidence = sum(not _chunk_from_evidence(edge.get("evidenceChunkId")) for edge in edges)
    low_confidence = sum(float(edge.get("confidence") or 0.0) < 0.65 for edge in semantic)
    edge_ids = [str(edge.get("edgeId") or "") for edge in edges]
    quality = {
        "nodeCount": len(nodes),
        "uniqueNodeId": len(nodes) == len(set(nodes)),
        "edgeCount": len(edges),
        "uniqueEdgeId": len(edge_ids) == len(set(edge_ids)),
        "endpointIntegrity": endpoint_missing == 0,
        "endpointMissingCount": endpoint_missing,
        "edgeEvidenceBindingRate": round(
            (len(edges) - missing_evidence) / len(edges), 8
        ) if edges else 0.0,
        "semanticRelationCount": len(semantic),
        "manifestSemanticRelationCount": manifest_semantic,
        "manifestSemanticCountMatches": len(semantic) == manifest_semantic,
        "semanticChunkCoverage": round(len(semantic_chunk_ids) / len(chunk_nodes), 8) if chunk_nodes else 0.0,
        "semanticChunkCount": len(semantic_chunk_ids),
        "chunkNodeCount": len(chunk_nodes),
        "minimumSemanticConfidence": min(
            (float(edge.get("confidence") or 0.0) for edge in semantic), default=0.0
        ),
        "lowConfidenceSemanticCount": low_confidence,
        "assertionCounts": dict(sorted(actual_assertions.items())),
        "relationCounts": dict(sorted(actual_relation_counts.items())),
        "accuracyMetrics": {
            "entityAccuracy": "not_measured_without_human_gold_labels",
            "relationAccuracy": "not_measured_without_human_gold_labels",
            "entityAlignmentAccuracy": "not_measured_without_human_gold_labels",
        },
    }
    quality["qualityGatePass"] = all([
        quality["uniqueNodeId"],
        quality["uniqueEdgeId"],
        quality["endpointIntegrity"],
        quality["edgeEvidenceBindingRate"] == 1.0,
        quality["manifestSemanticCountMatches"],
        quality["minimumSemanticConfidence"] >= 0.65,
    ])
    return quality


def run(
    index_dir: Path,
    output_path: Path,
    backend: str = "local",
    graph_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run the smoke evaluator against an explicitly versioned graph.

    The historical evaluator implicitly paired v3 graph gold with whichever
    chunk index was configured.  That silently scores every chunk-v2 evidence
    id as a miss.  Explicit graph/manifest paths keep the provenance oracle
    aligned with the chunk variant under test while preserving v3 defaults.
    """
    graph_path = graph_path or (index_dir / "medical_knowledge_graph_v3.jsonl")
    manifest_path = manifest_path or (index_dir / "medical_knowledge_graph_v3_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes, edges = _load_graph(graph_path)
    chunk_texts = _load_chunk_texts(index_dir / "medical_vector_index.jsonl")
    cases = _build_cases(nodes, edges, chunk_texts)
    temp_dir = _prepare_local_index(index_dir, graph_path) if backend == "local" else None
    try:
        if backend == "local":
            assert temp_dir is not None
            port = LocalKnowledgeSearchPort(
                LocalKnowledgeIndexConfiguration(Path(temp_dir.name), "tfidf")
            )
        elif backend == "database":
            from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
            from mico_agent_runtime.knowledge.database_retriever import DatabaseKnowledgeSearchPort
            from mico_agent_runtime.knowledge.embeddings import GeminiEmbeddingPort

            embedding = _CachedEmbeddingPort(GeminiEmbeddingPort.from_environment())
            port = DatabaseKnowledgeSearchPort(
                KnowledgeStoreConfiguration.from_environment(),
                embedding,
                query_embedding_cache_path=index_dir / ".p2g-query-embedding-cache-v1.json",
            )
        else:
            raise ValueError(f"unsupported backend: {backend}")
        rows_by_branch: dict[str, list[dict[str, Any]]] = {branch: [] for branch in BRANCHES}
        provenance: Counter[str] = Counter()
        hybrid_branch_coverage = 0
        branch_eligible = 0
        case_results: list[dict[str, Any]] = []
        for case in cases:
            query_results: dict[str, Any] = {"case": case}
            branch_rows: dict[str, Any] = {}
            for branch in BRANCHES:
                query = EvidenceQuery(
                    topic=case["query"],
                    direction="context",
                    retrievalMode=branch,
                    limit=10,
                )
                try:
                    results = port.search(query)
                    metrics = _metric_for_case(results, case)
                    graph_path_count = sum(len(item.graphPaths) for item in results)
                    path_contract_pass = all(
                        1 <= path.hopCount <= 3
                        and all(hop.evidenceChunkId for hop in path.hops)
                        and bool(path.sourceDocumentIds)
                        for item in results
                        for path in item.reasoningPaths
                    )
                    source_bound = all(bool(item.sourceChunkId) for item in results)
                    route_sources = sorted({source for item in results for source in item.retrievalSources})
                    row = {
                        "metrics": metrics,
                        "resultCount": len(results),
                        "routeSources": route_sources,
                        "sourceBoundRate": round(
                            sum(bool(item.sourceChunkId) for item in results) / len(results), 8
                        ) if results else 0.0,
                        "reasoningPathCount": graph_path_count,
                        "reasoningPathContractPass": path_contract_pass,
                        "sourceBound": source_bound,
                    }
                    rows_by_branch[branch].append(row)
                    branch_rows[branch] = row
                    if branch == "hybrid":
                        has_vector = "vector" in route_sources
                        has_graph = "graph" in route_sources
                        if has_vector or has_graph:
                            branch_eligible += 1
                        if has_vector and has_graph:
                            hybrid_branch_coverage += 1
                    provenance["source_bound_pass"] += int(source_bound)
                    provenance["path_contract_pass"] += int(path_contract_pass)
                except Exception as exc:  # keep all case failures in the report
                    row = {"error": f"{type(exc).__name__}: {exc}"}
                    rows_by_branch[branch].append(row)
                    branch_rows[branch] = row
            query_results["branches"] = branch_rows
            case_results.append(query_results)
        port.close()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    aggregate = {
        branch: _aggregate(rows)
        for branch, rows in rows_by_branch.items()
    }
    by_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        by_category[category] = {
            branch: _aggregate([
                row["branches"][branch]
                for row in case_results
                if row["case"]["category"] == category
            ])
            for branch in BRANCHES
        }
    for branch in BRANCHES:
        successful = [row for row in rows_by_branch[branch] if row.get("error") is None]
        aggregate[branch]["sourceBoundRate"] = round(
            sum(row["sourceBoundRate"] for row in successful) / len(successful), 8
        ) if successful else 0.0
        aggregate[branch]["reasoningPathContractRate"] = round(
            sum(row["reasoningPathContractPass"] for row in successful) / len(successful), 8
        ) if successful else 0.0
    vector_mrr = aggregate["vector"].get("mrr", 0.0)
    graph_mrr = aggregate["graph"].get("mrr", 0.0)
    hybrid_mrr = aggregate["hybrid"].get("mrr", 0.0)
    report = {
        "reportVersion": "p2g-retrieval-eval-v1",
        "status": "COMPLETED",
        "evaluationBoundary": {
            "corpus": str(manifest.get("graphVersion") or graph_path.stem),
            "caseCount": len(cases),
            "caseSource": "deterministic provenance-derived cases",
            "relevanceLimitation": "not a human-reviewed relevance benchmark",
            "vectorImplementation": (
                "local full-text TF-IDF baseline"
                if backend == "local"
                else "PostgreSQL pgvector with Gemini Embedding 2"
            ),
            "graphImplementation": (
                f"local {manifest.get('graphVersion') or graph_path.stem} provenance graph with bounded traversal"
                if backend == "local"
                else f"Neo4j configured graph {manifest.get('graphVersion') or graph_path.stem} with bounded traversal"
            ),
            "backend": backend,
        },
        "qualityAudit": _quality_audit(manifest, nodes, edges),
        "caseCountsByCategory": dict(Counter(case["category"] for case in cases)),
        "aggregateByBranch": aggregate,
        "aggregateByCategory": by_category,
        "hybridComparison": {
            "hybridVsVectorMrrDelta": round(hybrid_mrr - vector_mrr, 8),
            "hybridVsGraphMrrDelta": round(hybrid_mrr - graph_mrr, 8),
            "hybridBranchCoverageEligibleCases": branch_eligible,
            "hybridBothBranchCoverageCount": hybrid_branch_coverage,
            "hybridBothBranchCoverageRate": round(
                hybrid_branch_coverage / branch_eligible, 8
            ) if branch_eligible else 0.0,
        },
        "provenanceAudit": {
            "sourceBoundPassCount": provenance["source_bound_pass"],
            "pathContractPassCount": provenance["path_contract_pass"],
            "totalBranchRuns": len(cases) * len(BRANCHES),
        },
        "cases": case_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "mico_database_new" / "references" / "knowledge" / "medical" / "rag",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "p2g-retrieval-eval-v1.json",
    )
    parser.add_argument("--backend", choices=("local", "database"), default="local")
    parser.add_argument(
        "--graph",
        type=Path,
        help="explicit graph JSONL used for case construction and provenance audit",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="manifest matching --graph; defaults to the historical v3 manifest",
    )
    args = parser.parse_args()
    report = run(args.index_dir, args.output, args.backend, args.graph, args.manifest)
    print(json.dumps({
        "status": report["status"],
        "caseCount": report["evaluationBoundary"]["caseCount"],
        "caseCountsByCategory": report["caseCountsByCategory"],
        "qualityGatePass": report["qualityAudit"]["qualityGatePass"],
        "aggregateByBranch": report["aggregateByBranch"],
        "hybridComparison": report["hybridComparison"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
