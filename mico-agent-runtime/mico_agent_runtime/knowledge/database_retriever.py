from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from mico_agent_runtime.contracts.evidence import EvidenceQuery, LiteratureEvidenceItem
from mico_agent_runtime.contracts.graph_rag import GraphEvidencePath, GraphPathStep, ReasoningPath
from mico_agent_runtime.contracts.retrieval import RetrievalPlan, build_retrieval_plan
from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
from mico_agent_runtime.knowledge.embeddings import EmbeddingPort, GeminiEmbeddingPort
from mico_agent_runtime.knowledge.reranking import merge_and_rerank

from .local_retriever import QUERY_TERM_ALIASES, STOPWORDS, TOKEN_RE


def _tokens(value: str) -> list[str]:
    result: list[str] = []
    for token in TOKEN_RE.findall(value.lower()):
        if token.isascii() and token in STOPWORDS:
            continue
        if len(token) > 1:
            result.append(token)
    return result


def _query_terms(value: str) -> list[str]:
    terms = _tokens(value)
    compact = re.sub(r"\s+", "", value.lower())
    for source, aliases in QUERY_TERM_ALIASES.items():
        if source in compact:
            terms.extend(aliases)
    return list(dict.fromkeys(terms))


def _opaque_id(value: str) -> str:
    return "evidence-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _path_id(steps: list[GraphPathStep]) -> str:
    import json

    return "path-" + hashlib.sha256(
        json.dumps([step.model_dump(mode="json") for step in steps], sort_keys=True).encode("utf-8")
    ).hexdigest()[:32]


def _display(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("label") or value.get("nodeId") or "entity")[:512]
    getter = getattr(value, "get", None)
    if callable(getter):
        return str(getter("label") or getter("nodeId") or "entity")[:512]
    return str(value)[:512]


def _node_id(value: Any) -> str:
    node_id = ""
    if isinstance(value, Mapping):
        node_id = str(value.get("nodeId") or "")
    else:
        getter = getattr(value, "get", None)
        node_id = str(getter("nodeId") or "") if callable(getter) else ""
    # Versioned graphs use version-prefixed node IDs so multiple published or
    # staging builds can coexist.  The evidence chunk identifier exposed to
    # the relational evidence store remains the original chunk:<id> namespace.
    return re.sub(r"^v[0-9]+:", "", node_id)


def _relation(value: Any) -> str:
    getter = getattr(value, "get", None)
    return str(getter("relation") or "RELATED_TO") if callable(getter) else "RELATED_TO"


def _prop(value: Any, name: str) -> str:
    getter = getattr(value, "get", None)
    return str(getter(name) or "") if callable(getter) else ""


def _hop_status(target: str, relation: str, assertion: str = "") -> Literal[
    "supported", "speculative", "conflicted", "partial", "unsupported"
]:
    marker = f"{relation} {assertion}".lower()
    if any(value in marker for value in ("conflict", "contradict", "negat")):
        return "conflicted"
    if "speculative" in marker:
        return "speculative"
    if target.startswith(("candidate_taxon:", "v3:taxon:", "taxon:")) or "candidate" in relation.lower():
        return "speculative"
    return "supported"


def _path_status(statuses: list[str]) -> Literal[
    "supported", "speculative", "conflicted", "partial", "unsupported"
]:
    if "conflicted" in statuses:
        return "conflicted"
    if "speculative" in statuses:
        return "speculative"
    return "supported" if statuses and all(value == "supported" for value in statuses) else "partial"


@dataclass(frozen=True)
class _GraphPathRow:
    chunkId: str
    path: GraphEvidencePath


class DatabaseKnowledgeSearchPort:
    """Real pgvector + Neo4j retrieval port for the independent literature store."""

    def __init__(
        self,
        configuration: KnowledgeStoreConfiguration,
        embedding_port: EmbeddingPort,
    ) -> None:
        self._configuration = configuration
        self._embedding_port = embedding_port

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "DatabaseKnowledgeSearchPort":
        configuration = KnowledgeStoreConfiguration.from_environment(env)
        return cls(configuration, GeminiEmbeddingPort.from_environment(env))

    def close(self) -> None:
        close = getattr(self._embedding_port, "close", None)
        if callable(close):
            close()

    def search(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        return self.search_parallel(query, branches=("vector", "graph"))

    def search_parallel(
        self,
        query: EvidenceQuery,
        branches: tuple[Literal["vector", "graph"], ...] = ("vector", "graph"),
        plan: RetrievalPlan | None = None,
    ) -> list[LiteratureEvidenceItem]:
        requested = tuple(dict.fromkeys(branches))
        if not requested or any(branch not in {"vector", "graph"} for branch in requested):
            raise ValueError("unsupported retrieval branch")
        if plan is None:
            mode = "hybrid" if len(requested) == 2 else requested[0]
            plan = build_retrieval_plan(
                query_summary=query.topic,
                query_type="multi_hop" if "graph" in requested else "semantic_fact",
                retrieval_mode=mode,
                retrieval_branches=list(requested),
                top_k=query.limit,
                graph_version=self._configuration.graphVersion,
            )
        terms = _query_terms(f"{query.topic} {query.taxonName or ''}")
        if not terms:
            return []
        if len(requested) == 2:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mico-knowledge") as pool:
                vector_future = pool.submit(self._vector_branch, query, terms)
                graph_future = pool.submit(
                    self._graph_branch, query, terms, plan.maxHops or 3, plan.minConfidence
                )
                vector = vector_future.result()
                graph = graph_future.result()
            return merge_and_rerank(query, vector, graph, plan)
        if requested[0] == "vector":
            return merge_and_rerank(query, self._vector_branch(query, terms), [], plan)
        return merge_and_rerank(
            query, [], self._graph_branch(query, terms, plan.maxHops or 3, plan.minConfidence), plan
        )

    def _vector_branch(self, query: EvidenceQuery, terms: list[str]) -> list[LiteratureEvidenceItem]:
        import psycopg
        from pgvector import HalfVector
        from pgvector.psycopg import register_vector

        query_text = " ".join(terms)
        vector = self._embedding_port.embed_query(query_text)
        with psycopg.connect(self._configuration.vectorDatabaseUrl) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT document_id, pmcid, pmid, doi, title, journal,
                           publication_year, source_url, embedding_model,
                           1 - (embedding <=> %s) AS score
                    FROM knowledge_document
                    WHERE evidence_tier = 'fulltext'
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (HalfVector(vector), HalfVector(vector), min(query.limit, self._configuration.vectorTopK)),
                )
                documents = cursor.fetchall()
                results: list[LiteratureEvidenceItem] = []
                for row in documents:
                    cursor.execute(
                        """
                        SELECT chunk_id, section, source_url, text
                        FROM knowledge_chunk
                        WHERE document_id = %s
                        ORDER BY ts_rank(
                            to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, '')),
                            plainto_tsquery('english', %s)
                        ) DESC, ordinal
                        LIMIT 1
                        """,
                        (row[0], query_text),
                    )
                    chunk = cursor.fetchone()
                    if chunk is None:
                        continue
                    score = max(0.0, float(row[9] or 0.0))
                    results.append(self._item_from_rows(
                        query, row, chunk, score, "vector", [], "pgvector-gemini-embedding-2"
                    ))
        return results

    def _graph_branch(
        self,
        query: EvidenceQuery,
        terms: list[str],
        max_hops: int = 3,
        min_confidence: float = 0.65,
    ) -> list[LiteratureEvidenceItem]:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            self._configuration.neo4jUri,
            auth=(self._configuration.neo4jUser, self._configuration.neo4jPassword),
        )
        try:
            path_rows: list[_GraphPathRow] = []
            with driver.session() as session:
                graph_version = self._configuration.graphVersion
                seed_result = session.run(
                    """
                    MATCH (build:KnowledgeGraphBuild)
                    WHERE build.graphVersion = $graph_version
                      AND build.status = 'published'
                    MATCH (seed:KnowledgeEntity)
                    WHERE seed.graphVersion = $graph_version
                      AND seed.nodeType IN ['disease', 'concept', 'metabolite', 'pathway', 'hostprocess', 'taxon']
                      AND any(term IN $terms WHERE toLower(seed.label) CONTAINS toLower(term))
                    RETURN seed.nodeId AS node_id
                    LIMIT $seed_limit
                    """,
                    terms=terms[:16],
                    graph_version=graph_version,
                    seed_limit=min(32, max(8, query.limit * 3)),
                )
                seed_ids = [str(record["node_id"]) for record in seed_result]
                if not seed_ids:
                    return []
                query_params = {
                    "terms": terms[:16],
                    "graph_version": graph_version,
                    "min_confidence": min_confidence,
                    "seed_ids": seed_ids,
                    "path_limit": min(120, max(20, query.limit * 12)),
                }
                # Do not use an unrestricted variable-length pattern here.
                # The graph contains shared entities, so a broad 1..3 hop
                # expansion can enumerate a combinatorial number of paths.
                # These three fixed shapes preserve 1/2/3-hop semantics while
                # anchoring every path at a source seed and an evidence chunk.
                path_queries = (
                    """
                    MATCH p=(seed:KnowledgeEntity)-[:KNOWLEDGE_RELATION]-(chunk:KnowledgeEntity)
                    WHERE seed.graphVersion = $graph_version
                      AND seed.nodeId IN $seed_ids
                      AND chunk.graphVersion = $graph_version
                      AND chunk.nodeType = 'chunk'
                      AND seed.nodeType <> 'chunk'
                      AND any(term IN $terms WHERE toLower(seed.label) CONTAINS toLower(term))
                    AND all(rel IN relationships(p) WHERE rel.graphVersion = $graph_version
                              AND rel.evidenceChunkId IS NOT NULL AND coalesce(rel.confidence, 1.0) >= $min_confidence)
                    RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels
                    LIMIT $path_limit
                    """,
                    """
                    MATCH p=(seed:KnowledgeEntity)-[:KNOWLEDGE_RELATION]-(mid:KnowledgeEntity)-[:KNOWLEDGE_RELATION]-(chunk:KnowledgeEntity)
                    WHERE seed.graphVersion = $graph_version
                      AND seed.nodeId IN $seed_ids
                      AND mid.graphVersion = $graph_version
                      AND chunk.graphVersion = $graph_version
                      AND seed.nodeType <> 'chunk' AND mid.nodeType <> 'chunk' AND chunk.nodeType = 'chunk'
                      AND any(term IN $terms WHERE toLower(seed.label) CONTAINS toLower(term))
                      AND all(rel IN relationships(p) WHERE rel.graphVersion = $graph_version
                              AND rel.evidenceChunkId IS NOT NULL AND coalesce(rel.confidence, 1.0) >= $min_confidence)
                      AND relationships(p)[-1].relation = 'MENTIONS_ENTITY'
                    RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels
                    LIMIT $path_limit
                    """,
                    """
                    MATCH p=(seed:KnowledgeEntity)-[:KNOWLEDGE_RELATION]-(mid1:KnowledgeEntity)-[:KNOWLEDGE_RELATION]-(mid2:KnowledgeEntity)-[:KNOWLEDGE_RELATION]-(chunk:KnowledgeEntity)
                    WHERE seed.graphVersion = $graph_version
                      AND seed.nodeId IN $seed_ids
                      AND mid1.graphVersion = $graph_version
                      AND mid2.graphVersion = $graph_version
                      AND chunk.graphVersion = $graph_version
                      AND seed.nodeType <> 'chunk' AND mid1.nodeType <> 'chunk'
                      AND mid2.nodeType <> 'chunk' AND chunk.nodeType = 'chunk'
                      AND any(term IN $terms WHERE toLower(seed.label) CONTAINS toLower(term))
                      AND all(rel IN relationships(p) WHERE rel.graphVersion = $graph_version
                              AND rel.evidenceChunkId IS NOT NULL AND coalesce(rel.confidence, 1.0) >= $min_confidence)
                      AND relationships(p)[-1].relation = 'MENTIONS_ENTITY'
                      AND relationships(p)[-2].relation <> 'MENTIONS_ENTITY'
                    RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels
                    LIMIT $path_limit
                    """,
                )
                records = []
                for path_query in path_queries[: max(1, min(3, max_hops))]:
                    records.extend(session.run(path_query, **query_params))
                for record in records:
                    nodes = list(record["path_nodes"] or [])
                    relationships = list(record["path_rels"] or [])
                    if len(nodes) < 2 or not relationships:
                        continue
                    chunk_node = nodes[-1]
                    chunk_id = _node_id(chunk_node).removeprefix("chunk:")
                    steps: list[GraphPathStep] = []
                    statuses: list[str] = []
                    for index, relationship in enumerate(relationships):
                        source_node = nodes[index]
                        target_node = nodes[index + 1]
                        source_id = _node_id(source_node)
                        target_id = _node_id(target_node)
                        relation = _relation(relationship)
                        if str(getattr(relationship, "start_node", "")) != str(source_node):
                            relation = relation + "_REVERSE"
                        status = _hop_status(target_id, relation, _prop(relationship, "assertionStatus"))
                        statuses.append(status)
                        steps.append(GraphPathStep(
                            fromEntity=_display(source_node),
                            relation=relation,
                            toEntity=_display(target_node),
                            evidenceChunkId=_prop(relationship, "evidenceChunkId") or chunk_id,
                            supportStatus=status,
                        ))
                    status = _path_status(statuses)
                    structural_relations = {"PART_OF", "IN_SECTION", "HAS_TOPIC", "MENTIONS_ENTITY"}
                    semantic_hop_count = sum(
                        step.relation.removesuffix("_REVERSE") not in structural_relations
                        for step in steps
                    )
                    base_confidence = 0.55 if semantic_hop_count == 0 else min(
                        0.95, 0.65 + 0.10 * semantic_hop_count
                    )
                    path = GraphEvidencePath(
                        pathId=_path_id(steps),
                        status=status,
                        hops=steps,
                        sourceDocumentIds=["PMCID:" + chunk_id.split("-", 1)[0]],
                        confidence=round(
                            base_confidence
                            * (0.5 if status == "speculative" else 0.25 if status == "conflicted" else 1.0),
                            6,
                        ),
                    )
                    path_rows.append(_GraphPathRow(chunk_id, path))
        finally:
            driver.close()
        paths_by_chunk: defaultdict[str, list[GraphEvidencePath]] = defaultdict(list)
        for row in path_rows:
            if row.path.pathId not in {item.pathId for item in paths_by_chunk[row.chunkId]}:
                paths_by_chunk[row.chunkId].append(row.path)
        chunk_ids = list(paths_by_chunk)[: max(query.limit * 4, 20)]
        if not chunk_ids:
            return []
        chunks = self._load_chunks(chunk_ids)
        results: list[LiteratureEvidenceItem] = []
        for chunk_id, paths in paths_by_chunk.items():
            row = chunks.get(chunk_id)
            if row is None:
                continue
            score = max((path.confidence for path in paths[:4]), default=0.0)
            results.append(self._item_from_rows(
                query,
                row["document"],
                row["chunk"],
                score,
                "graph",
                paths[:4],
                "neo4j-provenance-graph-v1",
            ))
        return sorted(results, key=lambda item: (-item.rerankScore, item.externalId))[:query.limit]

    def _load_chunks(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        import psycopg

        with psycopg.connect(self._configuration.vectorDatabaseUrl) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT c.chunk_id, c.section, c.source_url, c.text,
                           d.document_id, d.pmcid, d.pmid, d.doi, d.title,
                           d.journal, d.publication_year, d.source_url AS document_url,
                           d.embedding_model
                    FROM knowledge_chunk c
                    JOIN knowledge_document d ON d.document_id = c.document_id
                    WHERE c.chunk_id = ANY(%s)
                    """,
                    (chunk_ids,),
                )
                rows = cursor.fetchall()
        return {
            str(row[0]): {
                "chunk": (row[0], row[1], row[2], row[3]),
                "document": row[4:13],
            }
            for row in rows
        }

    @staticmethod
    def _item_from_rows(
        query: EvidenceQuery,
        document: Any,
        chunk: Any,
        score: float,
        route: Literal["vector", "graph"],
        paths: list[GraphEvidencePath],
        model: str,
    ) -> LiteratureEvidenceItem:
        # document: id, pmcid, pmid, doi, title, journal, year, url, [model]
        chunk_id, section, source_url, text = chunk
        pmcid = str(document[1])
        excerpt = re.sub(r"\s+", " ", str(text or "")).strip()[:1200] or None
        return LiteratureEvidenceItem(
            evidenceId=_opaque_id(f"database|{chunk_id}"),
            taxonName=query.taxonName,
            source="internal_knowledge",
            externalId=f"PMCID:{pmcid}#{chunk_id}",
            title=str(document[4])[:512],
            journal=str(document[5] or "")[:256] or None,
            publicationYear=int(document[6] or 2000),
            direction=query.direction,
            summary=f"Full-text evidence selected by {route} database retrieval.",
            evidenceTier="fulltext",
            retrievalRoute=route,
            retrievalModel="gemini-embedding-2" if route == "vector" else None,
            sourceChunkId=str(chunk_id),
            retrievalScore=round(max(0.0, score), 8),
            sourceExcerpt=excerpt,
            vectorScore=round(max(0.0, score) if route == "vector" else 0.0, 8),
            graphScore=round(max(0.0, score) if route == "graph" else 0.0, 8),
            rerankScore=round(max(0.0, score), 8),
            graphPaths=paths[:4],
            reasoningPaths=[ReasoningPath.from_graph_path(path) for path in paths[:4]],
            retrievalSources=[route],
        )

    @staticmethod
    def _merge(
        query: EvidenceQuery,
        vector: list[LiteratureEvidenceItem],
        graph: list[LiteratureEvidenceItem],
    ) -> list[LiteratureEvidenceItem]:
        merged: dict[str, LiteratureEvidenceItem] = {}
        for item in vector + graph:
            key = item.sourceChunkId or item.externalId
            if key not in merged:
                merged[key] = item
                continue
            current = merged[key]
            paths = {path.pathId: path for path in current.graphPaths}
            paths.update({path.pathId: path for path in item.graphPaths})
            vector_score = max(current.vectorScore, item.vectorScore)
            graph_score = max(current.graphScore, item.graphScore)
            path_support = min(1.0, len(paths) / 2.0)
            score = 0.55 * vector_score + 0.35 * graph_score + 0.10 * path_support
            merged[key] = current.model_copy(update={
                "retrievalRoute": "hybrid",
                "retrievalSources": sorted(set(current.retrievalSources + item.retrievalSources)),
                "vectorScore": round(vector_score, 8),
                "graphScore": round(graph_score, 8),
                "rerankScore": round(score, 8),
                "retrievalScore": round(score, 8),
                "graphPaths": list(paths.values())[:4],
                "summary": "Full-text evidence jointly ranked by pgvector and Neo4j.",
            })
        for key, item in list(merged.items()):
            paths = {path.pathId: path for path in item.graphPaths}
            score = 0.55 * item.vectorScore + 0.35 * item.graphScore + 0.10 * min(1.0, len(paths) / 2.0)
            merged[key] = item.model_copy(update={
                "retrievalRoute": "hybrid",
                "rerankScore": round(score, 8),
                "retrievalScore": round(score, 8),
                "summary": "Full-text evidence jointly ranked by pgvector and Neo4j.",
            })
        ordered = sorted(merged.values(), key=lambda item: (-item.rerankScore, item.externalId))
        if vector and graph and query.limit > 1:
            has_vector = any("vector" in item.retrievalSources for item in ordered[:query.limit])
            has_graph = any("graph" in item.retrievalSources for item in ordered[:query.limit])
            if not has_vector:
                ordered = ordered[: max(0, query.limit - 1)] + [vector[0].model_copy(update={
                    "retrievalRoute": "hybrid",
                    "summary": "Full-text evidence jointly ranked by pgvector and Neo4j.",
                })]
            if not has_graph:
                ordered = ordered[: max(0, query.limit - 1)] + [graph[0].model_copy(update={
                    "retrievalRoute": "hybrid",
                    "summary": "Full-text evidence jointly ranked by pgvector and Neo4j.",
                })]
            ordered = sorted(ordered, key=lambda item: (-item.rerankScore, item.externalId))
        return ordered[:query.limit]
