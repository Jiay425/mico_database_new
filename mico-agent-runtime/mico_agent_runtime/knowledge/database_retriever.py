from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from mico_agent_runtime.contracts.evidence import EvidenceQuery, LiteratureEvidenceItem
from mico_agent_runtime.contracts.graph_rag import GraphEvidencePath, GraphPathStep, ReasoningPath
from mico_agent_runtime.contracts.retrieval import RetrievalPlan, RetrievalScope, build_retrieval_plan
from mico_agent_runtime.knowledge.database_config import KnowledgeStoreConfiguration
from mico_agent_runtime.knowledge.embeddings import EmbeddingPort, GeminiEmbeddingPort, QueryEmbeddingCache
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


def _query_type(topic: str) -> Literal["semantic_fact", "relation", "multi_hop", "composite"]:
    value = topic.lower()
    # Cross-document synthesis and conflict prompts often contain incidental
    # phrases such as "in relation to".  Classify these before relation cues
    # so the graph branch is not enabled solely by that wording.
    if any(marker in value for marker in (
        "across studies", "across the supplied studies", "agree or differ",
        "consistent, and where", "conflicting findings", "null, negative",
    )):
        return "composite"
    # Runtime multi-hop prompts are not always written literally as
    # "multi-hop".  Evidence-composition requests must retain this route.
    if (
        "multi-hop" in value or "multi hop" in value or "mechanism linking" in value
        or any(marker in value for marker in (
            "mechanism", "mechanisms", "pathway", "pathways", "how does", "how do", "through what",
        ))
        or ("fit together" in value and "evidence" in value)
    ):
        return "multi_hop"
    if "relationship" in value or "relation" in value or "between" in value:
        return "relation"
    if "synthesize" in value or "across studies" in value:
        return "composite"
    return "semantic_fact"


def _scope_document_ids(scope: RetrievalScope) -> list[str] | None:
    """Translate an explicit retrieval scope into a SQL/Cypher allow-list.

    ``None`` means truly global.  Document and document-set scopes are
    represented only by stable PMCIDs, so every branch can apply the same
    boundary without relying on ranking-time post-filtering.
    """
    return None if scope.scopeType == "global" else scope.allowedDocumentIds


# Retrieval-time normalization expands a user's surface form to the canonical
# label already present in the frozen graph. It does not modify graph nodes,
# aliases, or extracted relations, so r4 remains version-stable.
_QUERY_ENTITY_NORMALIZATION: dict[str, str] = {
    "t2dm": "type 2 diabetes",
    "type ii diabetes mellitus": "type 2 diabetes",
    "type ii diabetes": "type 2 diabetes",
    "type 2 diabetes mellitus": "type 2 diabetes",
    "nafld": "fatty liver disease",
    "non-alcoholic fatty liver disease": "fatty liver disease",
    "nonalcoholic fatty liver disease": "fatty liver disease",
    "ibd": "inflammatory bowel disease",
    "toll like receptor 4": "toll-like receptor 4",
    "tlr-4": "toll-like receptor 4",
    "糖尿病": "type 2 diabetes",
    "非酒精性脂肪肝": "fatty liver disease",
    "炎症性肠病": "inflammatory bowel disease",
    "肠道微生物组": "gut microbiome",
}


def _normalize_graph_query(value: str) -> str:
    normalized = value
    for surface, canonical in _QUERY_ENTITY_NORMALIZATION.items():
        if surface.isascii():
            if re.search(r"(?<![a-z0-9])" + re.escape(surface) + r"(?![a-z0-9])", normalized, re.I):
                normalized += " " + canonical
        elif surface in normalized:
            normalized += " " + canonical
    return normalized


def _lexical_coverage(value: str, terms: list[str]) -> float:
    """Return bounded query-term coverage for ranking source-bound paths.

    This is deliberately lexical and only ranks chunks already reached through
    Neo4j evidence paths.  It never creates an edge or promotes a relation.
    """
    tokens = set(_tokens(value))
    useful = [term for term in terms if len(term) > 2]
    if not useful:
        return 0.0
    matched = sum(term in tokens for term in useful)
    return matched / len(useful)


_STRUCTURAL_RELATIONS = {"MENTIONS_ENTITY", "PART_OF", "IN_SECTION", "HAS_TOPIC"}


def _relation_match_score(query_text: str, relations: list[str]) -> float:
    """Compare the requested relation intent with semantic edge types."""
    value = query_text.lower()
    semantic = {relation.removesuffix("_REVERSE") for relation in relations} - _STRUCTURAL_RELATIONS
    if not semantic:
        return 0.0
    if any(marker in value for marker in ("inhibit", "suppress", "block", "prevent")):
        return 1.0 if "INHIBITS" in semantic else 0.15
    if any(marker in value for marker in ("increase", "enrich", "elevat", "higher")):
        return 1.0 if "INCREASED_IN" in semantic else 0.20
    if any(marker in value for marker in ("decrease", "deplet", "reduc", "lower")):
        return 1.0 if "DECREASED_IN" in semantic else 0.20
    if any(marker in value for marker in ("cause", "drive", "trigger", "mechanism", "mediate", "pathway")):
        if "CAUSES" in semantic:
            return 1.0
        if semantic & {"MEDIATES", "PROMOTES"}:
            return 0.90
        if "ASSOCIATED_WITH" in semantic:
            return 0.35
        return 0.15
    # Generic relation questions accept all semantic relations, but retain a
    # distinction between direct association and causal/directional evidence.
    return 0.85 if semantic & {"ASSOCIATED_WITH", "CAUSES", "MEDIATES", "PROMOTES", "INHIBITS"} else 0.70


def _path_relevance(
    nodes: list[Any],
    relationships: list[Any],
    terms: list[str],
    query_text: str,
    hub_frequency: Mapping[str, int],
    path_count: int,
) -> float:
    """Score a query-specific graph path, not merely a graph-rich chunk."""
    entity_labels = [
        _display(node) for node in nodes
        if _prop(node, "nodeType") not in {"chunk", "section", "document"}
    ]
    query_lower = query_text.lower()
    exact_hits = sum(bool(label and label.lower() in query_lower) for label in entity_labels)
    lexical_hits = sum(_lexical_coverage(label, terms) > 0.0 for label in entity_labels)
    # Exact entity linking is preferred; lexical overlap only recovers an
    # abbreviation or partial surface form not yet present in the alias map.
    entity_coverage = min(1.0, max(exact_hits, lexical_hits) / 2.0)
    relation_names = [_relation(relation).upper() for relation in relationships]
    relation_match = _relation_match_score(query_text, relation_names)
    semantic_confidences = [
        float(_prop(relation, "confidence") or 0.65)
        for relation, name in zip(relationships, relation_names)
        if name not in _STRUCTURAL_RELATIONS
    ]
    edge_confidence = min(semantic_confidences) if semantic_confidences else 0.0
    semantic_hops = len(semantic_confidences)
    path_quality = {1: 1.0, 2: 0.75, 3: 0.55}.get(semantic_hops, 0.35)
    # Only intermediate entities are penalized. Query endpoints may be common
    # by nature; penalizing them would make every legitimate diabetes or
    # inflammation question lose equally.
    intermediate_labels = [
        _display(node).lower() for node in nodes[1:-1]
        if _prop(node, "nodeType") not in {"chunk", "section", "document"}
    ]
    hub_ratio = max((hub_frequency.get(label, 0) / max(1, path_count) for label in intermediate_labels), default=0.0)
    hub_penalty = 0.20 * hub_ratio
    return max(0.0, min(1.0,
        0.35 * entity_coverage
        + 0.30 * relation_match
        + 0.20 * edge_confidence
        + 0.15 * path_quality
        - hub_penalty
    ))


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
    directPair: bool = False


class DatabaseKnowledgeSearchPort:
    """Real pgvector + Neo4j retrieval port for the independent literature store."""

    def __init__(
        self,
        configuration: KnowledgeStoreConfiguration,
        embedding_port: EmbeddingPort,
        query_embedding_cache_path: str | Path | None = None,
    ) -> None:
        self._configuration = configuration
        self._embedding_port = embedding_port
        self._query_embedding_cache = QueryEmbeddingCache(
            getattr(embedding_port, "modelName", "unknown"),
            query_embedding_cache_path,
            configuration.vectorDimension,
        )
        # Candidate generation/fusion is being calibrated first.  A learned
        # cross-encoder is intentionally not constructed or called until the
        # Dense/BM25/Graph Top-30 pool has been validated on held-out queries.
        self._model_reranker = None

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "DatabaseKnowledgeSearchPort":
        configuration = KnowledgeStoreConfiguration.from_environment(env)
        source = os.environ if env is None else env
        cache_path = source.get("MICO_KNOWLEDGE_QUERY_EMBED_CACHE_PATH", "").strip()
        return cls(
            configuration,
            GeminiEmbeddingPort.from_environment(env),
            query_embedding_cache_path=cache_path or None,
        )

    def close(self) -> None:
        close = getattr(self._embedding_port, "close", None)
        if callable(close):
            close()

    def search(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        if query.retrievalMode == "vector":
            return self.search_parallel(query, branches=("vector",))
        if query.retrievalMode == "graph":
            return self.search_parallel(query, branches=("graph",))
        # A document-bound request is normally a close-reading task.  Dense
        # remains the primary retriever there; the three-way fusion is for the
        # global corpus where there is meaningful candidate-selection space.
        if query.retrievalScope.scopeType != "global":
            return self.search_parallel(query, branches=("vector",))
        return self.search_parallel(query, branches=("vector", "graph"))

    def search_parallel(
        self,
        query: EvidenceQuery,
        branches: tuple[Literal["vector", "sparse", "graph"], ...] = ("vector", "sparse", "graph"),
        plan: RetrievalPlan | None = None,
    ) -> list[LiteratureEvidenceItem]:
        requested = tuple(dict.fromkeys(branches))
        if not requested or any(branch not in {"vector", "sparse", "graph"} for branch in requested):
            raise ValueError("unsupported retrieval branch")
        query_type = _query_type(query.topic) if "graph" in requested else "semantic_fact"
        # Existing callers encode hybrid as (vector, graph).  Expand that
        # legacy shape internally so production candidate recall is genuinely
        # three-way without breaking the public intent contract.
        if set(requested) == {"vector", "graph"}:
            requested = ("vector", "sparse", "graph")
        if plan is None:
            mode = "hybrid" if len(requested) > 1 or requested[0] == "sparse" else requested[0]
            plan = build_retrieval_plan(
                query_summary=query.topic,
                query_type=query_type,
                retrieval_mode=mode,
                retrieval_branches=list(requested),
                top_k=query.limit,
                graph_version=self._configuration.graphVersion,
                retrieval_scope=query.retrievalScope,
            )
        elif plan.retrievalScope != query.retrievalScope:
            # The request is the authority for corpus scope. A plan from a
            # different scope could otherwise retrieve outside its boundary.
            raise ValueError("retrieval plan scope must match query scope")
        if self._model_reranker is not None:
            plan = plan.model_copy(update={"rerankerVersion": "http-cross-encoder-v1"})
        terms = _query_terms(f"{query.topic} {query.taxonName or ''}")
        graph_query_text = _normalize_graph_query(query.topic)
        if not terms:
            return []
        if len(requested) >= 2:
            # Candidate generation is deliberately fixed at Top-30 per
            # retriever. RRF is a scale-safe pool merger, not the final
            # semantic judge; Cross-Encoder reranking is added only later.
            candidate_limit = min(
                30,
                self._configuration.vectorTopK,
                self._configuration.sparseTopK,
                self._configuration.graphTopK,
            )
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="mico-knowledge") as pool:
                vector_future = pool.submit(self._vector_branch, query, terms, candidate_limit, plan.retrievalScope)
                sparse_future = pool.submit(self._sparse_branch, query, terms, candidate_limit, plan.retrievalScope)
                graph_future = pool.submit(
                    self._graph_branch, query, terms, plan.maxHops or 3, plan.minConfidence, graph_query_text,
                    candidate_limit, plan.retrievalScope,
                )
                # Hybrid is allowed to degrade to the surviving source-bound
                # branches when an external dense provider is temporarily
                # unavailable (quota, timeout, or TLS failure). A vector-only
                # request still fails closed; this fallback is scoped to the
                # multi-branch path so FastAPI does not return an empty 500 for
                # a transient provider outage.
                try:
                    vector = vector_future.result()
                except Exception:
                    vector = []
                try:
                    sparse = sparse_future.result()
                except Exception:
                    sparse = []
                try:
                    graph = graph_future.result()
                except Exception:
                    graph = []
            return merge_and_rerank(
                query, vector, graph, plan,
                sparse_results=sparse,
                model_reranker=self._model_reranker,
            )
        if requested[0] == "vector":
            return merge_and_rerank(
                query,
                self._vector_branch(query, terms, min(30, self._configuration.vectorTopK), plan.retrievalScope),
                [], plan,
                model_reranker=self._model_reranker,
            )
        if requested[0] == "sparse":
            return merge_and_rerank(
                query, [], [], plan,
                sparse_results=self._sparse_branch(query, terms, min(30, self._configuration.sparseTopK), plan.retrievalScope),
                model_reranker=self._model_reranker,
            )
        return merge_and_rerank(
            query, [], self._graph_branch(
                query, terms, plan.maxHops or 3, plan.minConfidence, graph_query_text,
                min(30, self._configuration.graphTopK), plan.retrievalScope,
            ), plan
        )

    def _vector_branch(
        self, query: EvidenceQuery, terms: list[str], candidate_limit: int, scope: RetrievalScope,
    ) -> list[LiteratureEvidenceItem]:
        import psycopg
        from pgvector import HalfVector
        from pgvector.psycopg import register_vector

        query_text = " ".join(terms)
        vector = self._query_embedding_cache.get_or_compute(
            query_text,
            lambda: self._embedding_port.embed_query(query_text),
        )
        with psycopg.connect(self._configuration.vectorDatabaseUrl) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                documents = []
                results: list[LiteratureEvidenceItem] = []
                # Preferred path: true chunk-level dense retrieval.  The
                # column is nullable during migration, so an old database or
                # an unbackfilled corpus safely falls through to the legacy
                # document-vector path below.
                try:
                    query_vector = HalfVector(vector)
                    cursor.execute(
                        """
                        SELECT d.document_id, d.pmcid, d.pmid, d.doi, d.title, d.journal,
                               d.publication_year, d.source_url, d.embedding_model,
                               1 - (c.embedding <=> %s) AS score,
                               c.chunk_id, c.section, c.source_url, c.text
                        FROM knowledge_chunk c
                        JOIN knowledge_document d ON d.document_id = c.document_id
                        WHERE c.evidence_tier = 'fulltext' AND d.evidence_tier = 'fulltext'
                          AND c.chunk_version = %s AND c.chunk_variant = %s
                          AND c.embedding IS NOT NULL
                          AND (%s::text[] IS NULL OR d.pmcid = ANY(%s::text[]))
                        ORDER BY c.embedding <=> %s, c.chunk_id
                        LIMIT %s
                        """,
                        (
                            query_vector,
                            self._configuration.chunkVersion,
                            self._configuration.chunkVariant,
                            _scope_document_ids(scope),
                            _scope_document_ids(scope),
                            query_vector,
                            candidate_limit,
                        ),
                    )
                    chunk_rows = cursor.fetchall()
                except psycopg.errors.UndefinedColumn:
                    # PostgreSQL marks the transaction failed after an
                    # undefined-column probe; clear it before the compatible
                    # document-vector fallback query.
                    connection.rollback()
                    chunk_rows = []
                if chunk_rows:
                    for row in chunk_rows:
                        results.append(self._item_from_rows(
                            query,
                            row[:9],
                            row[10:14],
                            max(0.0, float(row[9] or 0.0)),
                            "vector",
                            [],
                            "gemini-embedding-2",
                        ))
                    return results
                cursor.execute(
                    """
                    SELECT document_id, pmcid, pmid, doi, title, journal,
                           publication_year, source_url, embedding_model,
                           1 - (embedding <=> %s) AS score
                    FROM knowledge_document
                    WHERE evidence_tier = 'fulltext'
                      AND (%s::text[] IS NULL OR pmcid = ANY(%s::text[]))
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (
                        HalfVector(vector),
                        _scope_document_ids(scope),
                        _scope_document_ids(scope),
                        HalfVector(vector),
                        candidate_limit,
                    ),
                )
                documents = cursor.fetchall()
                per_document: list[list[LiteratureEvidenceItem]] = []
                for row in documents:
                    cursor.execute(
                        """
                        SELECT chunk_id, section, source_url, text
                        FROM knowledge_chunk
                        WHERE document_id = %s
                          AND chunk_version = %s AND chunk_variant = %s
                        ORDER BY ts_rank(
                            to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, '')),
                            plainto_tsquery('english', %s)
                        ) DESC, ordinal
                        LIMIT %s
                        """,
                        (
                            row[0],
                            self._configuration.chunkVersion,
                            self._configuration.chunkVariant,
                            query_text,
                            min(4, candidate_limit),
                        ),
                    )
                    score = max(0.0, float(row[9] or 0.0))
                    chunks = cursor.fetchall()
                    per_document.append([
                        self._item_from_rows(
                            query, row, chunk, score, "vector", [], "gemini-embedding-2"
                        )
                        for chunk in chunks
                    ])
                # Round-robin chunks from the strongest documents. This gives
                # chunk-level recall in the migration fallback without
                # allowing one long paper to consume all Top-50 slots.
                for offset in range(min(4, candidate_limit)):
                    for document_chunks in per_document:
                        if offset < len(document_chunks) and len(results) < candidate_limit:
                            results.append(document_chunks[offset])
        return results

    def _sparse_branch(
        self,
        query: EvidenceQuery,
        terms: list[str],
        candidate_limit: int,
        scope: RetrievalScope,
    ) -> list[LiteratureEvidenceItem]:
        """Retrieve chunk-level lexical evidence with PostgreSQL FTS/BM25.

        ``ts_rank_cd`` is used instead of comparing its raw value with cosine
        similarity.  The score is retained only as an auditable feature; RRF
        consumes this branch's rank, which makes the fusion scale-safe.
        """
        import psycopg

        fts_terms = [
            term for term in dict.fromkeys(terms)
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,63}", term)
        ][:32]
        query_text = " | ".join(f"{term}:*" for term in fts_terms)
        if not fts_terms:
            return []
        with psycopg.connect(self._configuration.vectorDatabaseUrl) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT d.document_id, d.pmcid, d.pmid, d.doi, d.title, d.journal,
                           d.publication_year, d.source_url, d.embedding_model,
                           ts_rank_cd(
                               to_tsvector('english', coalesce(c.title, '') || ' ' || coalesce(c.text, '')),
                               to_tsquery('english', %s)
                           ) AS score,
                           c.chunk_id, c.section, c.source_url, c.text
                    FROM knowledge_chunk c
                    JOIN knowledge_document d ON d.document_id = c.document_id
                    WHERE c.evidence_tier = 'fulltext'
                      AND d.evidence_tier = 'fulltext'
                      AND c.chunk_version = %s AND c.chunk_variant = %s
                      AND (%s::text[] IS NULL OR d.pmcid = ANY(%s::text[]))
                      AND to_tsvector('english', coalesce(c.title, '') || ' ' || coalesce(c.text, ''))
                          @@ to_tsquery('english', %s)
                    ORDER BY score DESC, c.chunk_id
                    LIMIT %s
                    """,
                    (
                        query_text,
                        self._configuration.chunkVersion,
                        self._configuration.chunkVariant,
                        _scope_document_ids(scope),
                        _scope_document_ids(scope),
                        query_text,
                        candidate_limit,
                    ),
                )
                rows = cursor.fetchall()
        results: list[LiteratureEvidenceItem] = []
        for row in rows:
            raw_score = max(0.0, float(row[9] or 0.0))
            # Saturating transform keeps the diagnostic score bounded while
            # preserving monotonicity; ranking itself remains RRF-based.
            score = raw_score / (1.0 + raw_score)
            results.append(self._item_from_rows(
                query,
                row[:9],
                row[10:14],
                score,
                "sparse",
                [],
                "postgres-fulltext-bm25-v1",
            ))
        return results

    def _graph_branch(
        self,
        query: EvidenceQuery,
        terms: list[str],
        max_hops: int = 3,
        min_confidence: float = 0.65,
        query_text: str = "",
        candidate_limit: int = 10,
        scope: RetrievalScope | None = None,
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
                      AND (
                          toLower($query_text) CONTAINS toLower(seed.label)
                          OR any(term IN $terms WHERE toLower(seed.label) CONTAINS toLower(term))
                      )
                    RETURN seed.nodeId AS node_id
                    ORDER BY CASE WHEN toLower($query_text) CONTAINS toLower(seed.label) THEN 0 ELSE 1 END,
                             size(toLower(seed.label)) DESC, seed.nodeId
                    LIMIT $seed_limit
                    """,
                    terms=terms[:16],
                    query_text=query_text.lower(),
                    graph_version=graph_version,
                    seed_limit=min(32, max(8, query.limit * 3)),
                )
                seed_ids = [str(record["node_id"]) for record in seed_result]
                if not seed_ids:
                    return []
                query_params = {
                    "terms": terms[:16],
                    "query_text": query_text.lower(),
                    "graph_version": graph_version,
                    "min_confidence": min_confidence,
                    "seed_ids": seed_ids,
                    "path_limit": min(400, max(60, candidate_limit * 8)),
                    "allowed_pmcids": _scope_document_ids(scope or RetrievalScope()),
                }
                # Do not use an unrestricted variable-length pattern here.
                # The graph contains shared entities, so a broad 1..3 hop
                # expansion can enumerate a combinatorial number of paths.
                # These three fixed shapes preserve 1/2/3-hop semantics while
                # anchoring every path at a source seed and an evidence chunk.
                path_queries = (
                    # Highest precision: both relation endpoints occur in the
                    # request itself.  Run this before broad seed expansion so
                    # high-degree entities cannot fill the bounded candidate
                    # set before their direct relation is considered.
                    """
                    MATCH p=(seed:KnowledgeEntity)-[rel:KNOWLEDGE_RELATION]-(target:KnowledgeEntity)
                    WHERE seed.graphVersion = $graph_version
                      AND target.graphVersion = $graph_version
                      AND seed.nodeType <> 'chunk' AND target.nodeType <> 'chunk'
                      AND toLower($query_text) CONTAINS toLower(seed.label)
                      AND toLower($query_text) CONTAINS toLower(target.label)
                      AND seed.nodeId <> target.nodeId
                      AND rel.graphVersion = $graph_version
                      AND rel.evidenceChunkId IS NOT NULL
                      AND ($allowed_pmcids IS NULL OR split(rel.evidenceChunkId, '-')[0] IN $allowed_pmcids)
                      AND coalesce(rel.confidence, 1.0) >= $min_confidence
                      AND coalesce(rel.qualityStatus, 'accepted') = 'accepted'
                      AND rel.relationClass IN ['association', 'causal', 'directional']
                    RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels,
                           rel.evidenceChunkId AS evidence_chunk_id
                    ORDER BY coalesce(rel.confidence, 1.0) DESC, rel.evidenceChunkId
                    LIMIT $path_limit
                    """,
                    # A semantic edge already carries the evidence chunk that
                    # supports it.  Start here for relation questions instead
                    # of first walking through structural MENTIONS_ENTITY
                    # edges; the old traversal discarded this most specific
                    # source and returned arbitrary adjacent chunks.
                    """
                    MATCH p=(seed:KnowledgeEntity)-[rel:KNOWLEDGE_RELATION]-(target:KnowledgeEntity)
                    WHERE seed.graphVersion = $graph_version
                      AND seed.nodeId IN $seed_ids
                      AND target.graphVersion = $graph_version
                      AND seed.nodeType <> 'chunk' AND target.nodeType <> 'chunk'
                      AND seed.nodeId <> target.nodeId
                      AND rel.graphVersion = $graph_version
                      AND rel.evidenceChunkId IS NOT NULL
                      AND ($allowed_pmcids IS NULL OR split(rel.evidenceChunkId, '-')[0] IN $allowed_pmcids)
                      AND coalesce(rel.confidence, 1.0) >= $min_confidence
                      AND coalesce(rel.qualityStatus, 'accepted') = 'accepted'
                      AND rel.relationClass IN ['association', 'causal', 'directional']
                    RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels,
                           rel.evidenceChunkId AS evidence_chunk_id
                    ORDER BY coalesce(rel.confidence, 1.0) DESC, rel.evidenceChunkId
                    LIMIT $path_limit
                    """,
                    """
                    MATCH p=(seed:KnowledgeEntity)-[:KNOWLEDGE_RELATION]-(chunk:KnowledgeEntity)
                    WHERE seed.graphVersion = $graph_version
                      AND seed.nodeId IN $seed_ids
                      AND chunk.graphVersion = $graph_version
                      AND chunk.nodeType = 'chunk'
                      AND seed.nodeType <> 'chunk'
                      AND any(term IN $terms WHERE toLower(seed.label) CONTAINS toLower(term))
                    AND all(rel IN relationships(p) WHERE rel.graphVersion = $graph_version
                              AND rel.evidenceChunkId IS NOT NULL AND coalesce(rel.confidence, 1.0) >= $min_confidence
                              AND ($allowed_pmcids IS NULL OR split(rel.evidenceChunkId, '-')[0] IN $allowed_pmcids)
                              AND coalesce(rel.qualityStatus, 'accepted') = 'accepted')
                    RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels,
                           NULL AS evidence_chunk_id
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
                              AND rel.evidenceChunkId IS NOT NULL AND coalesce(rel.confidence, 1.0) >= $min_confidence
                              AND ($allowed_pmcids IS NULL OR split(rel.evidenceChunkId, '-')[0] IN $allowed_pmcids)
                              AND coalesce(rel.qualityStatus, 'accepted') = 'accepted')
                      AND relationships(p)[-1].relation = 'MENTIONS_ENTITY'
                    RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels,
                           NULL AS evidence_chunk_id
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
                              AND rel.evidenceChunkId IS NOT NULL AND coalesce(rel.confidence, 1.0) >= $min_confidence
                              AND ($allowed_pmcids IS NULL OR split(rel.evidenceChunkId, '-')[0] IN $allowed_pmcids)
                              AND coalesce(rel.qualityStatus, 'accepted') = 'accepted')
                      AND relationships(p)[-1].relation = 'MENTIONS_ENTITY'
                      AND relationships(p)[-2].relation <> 'MENTIONS_ENTITY'
                    RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels,
                           NULL AS evidence_chunk_id
                    LIMIT $path_limit
                    """,
                )
                records = []
                # Exact and broad semantic evidence are always allowed;
                # max_hops limits only the additional structural path shapes.
                for query_index, path_query in enumerate(path_queries[: 2 + max(1, min(3, max_hops))]):
                    for record in session.run(path_query, **query_params):
                        records.append((query_index < 2, record))
                # Estimate hubness only from intermediate nodes present in
                # this bounded query-specific candidate set. This prevents a
                # global high-degree concept from connecting unrelated papers
                # while never penalizing an entity simply for being queried.
                hub_frequency: Counter[str] = Counter()
                for _, record in records:
                    path_nodes = list(record["path_nodes"] or [])
                    for node in path_nodes[1:-1]:
                        if _prop(node, "nodeType") not in {"chunk", "section", "document"}:
                            hub_frequency[_display(node).lower()] += 1
                for direct_pair, record in records:
                    nodes = list(record["path_nodes"] or [])
                    relationships = list(record["path_rels"] or [])
                    if len(nodes) < 2 or not relationships:
                        continue
                    direct_chunk_id = str(record.get("evidence_chunk_id") or "")
                    chunk_node = nodes[-1]
                    chunk_id = direct_chunk_id or _node_id(chunk_node).removeprefix("chunk:")
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
                    semantic_confidences = [
                        float(_prop(relationship, "confidence") or 0.65)
                        for relationship in relationships
                        if _relation(relationship).upper() not in structural_relations
                    ]
                    # A multi-hop claim is only as trustworthy as its weakest
                    # semantic edge; structural provenance links do not lift
                    # the confidence of a scientific relation.
                    base_confidence = min(semantic_confidences) if semantic_confidences else 0.55
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
                    # Graph path confidence expresses support quality; combine
                    # it with entity coverage before sorting candidates.  The
                    # previous implementation assigned nearly the same score
                    # to every supported path, leaving Neo4j's unordered
                    # traversal to decide Top-K.
                    relevance = _path_relevance(
                        nodes, relationships, terms, query_text, hub_frequency, len(records),
                    )
                    path_rows.append(_GraphPathRow(
                        chunk_id,
                        path.model_copy(update={
                            "confidence": round(min(1.0, 0.35 * path.confidence + 0.65 * relevance), 6)
                        }),
                        directPair=direct_pair,
                    ))
        finally:
            driver.close()
        paths_by_chunk: defaultdict[str, list[GraphEvidencePath]] = defaultdict(list)
        for row in path_rows:
            if row.path.pathId not in {item.pathId for item in paths_by_chunk[row.chunkId]}:
                paths_by_chunk[row.chunkId].append(row.path)
        direct_chunks = {row.chunkId for row in path_rows if row.directPair}
        if direct_chunks and _query_type(query.topic) == "relation":
            # For an explicit two-entity question, broad seed expansion is a
            # source of false positives. Keep the exact pair candidates in
            # the graph branch; multi-hop/composite queries still use the
            # bounded expansion below.
            paths_by_chunk = {
                chunk_id: paths for chunk_id, paths in paths_by_chunk.items()
                if chunk_id in direct_chunks
            }
        # Load the strongest path candidates first; insertion order reflects
        # Cypher traversal order and is not a relevance signal.
        chunk_ids = [
            chunk_id
            for chunk_id, _ in sorted(
                paths_by_chunk.items(),
                key=lambda item: (-max((path.confidence for path in item[1]), default=0.0), item[0]),
            )[: max(candidate_limit * 4, 50)]
        ]
        if not chunk_ids:
            return []
        chunks = self._load_chunks(chunk_ids, scope or RetrievalScope())
        results: list[LiteratureEvidenceItem] = []
        for chunk_id, paths in paths_by_chunk.items():
            row = chunks.get(chunk_id)
            if row is None:
                continue
            # Rank graph-reachable evidence by both path relevance and the
            # source chunk's lexical coverage.  The latter resolves ties among
            # valid paths without falling back to vector search.
            source_text = " ".join((
                str(row["document"][4] or ""),
                str(row["chunk"][3] or ""),
            ))
            path_score = max((path.confidence for path in paths[:4]), default=0.0)
            lexical_score = _lexical_coverage(source_text, terms)
            score = 0.78 * path_score + 0.22 * lexical_score
            results.append(self._item_from_rows(
                query,
                row["document"],
                row["chunk"],
                score,
                "graph",
                paths[:4],
                "neo4j-provenance-graph-v1",
            ))
        return sorted(results, key=lambda item: (-item.rerankScore, item.externalId))[:candidate_limit]

    def _load_chunks(self, chunk_ids: list[str], scope: RetrievalScope) -> dict[str, dict[str, Any]]:
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
                      AND c.chunk_version = %s AND c.chunk_variant = %s
                      AND (%s::text[] IS NULL OR d.pmcid = ANY(%s::text[]))
                    """,
                    (
                        chunk_ids,
                        self._configuration.chunkVersion,
                        self._configuration.chunkVariant,
                        _scope_document_ids(scope),
                        _scope_document_ids(scope),
                    ),
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
        route: Literal["vector", "sparse", "graph"],
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
            retrievalModel=model if model in {
                "gemini-embedding-2", "fulltext-tfidf-cosine-v1", "postgres-fulltext-bm25-v1"
            } else None,
            sourceChunkId=str(chunk_id),
            retrievalScore=round(max(0.0, score), 8),
            sourceExcerpt=excerpt,
            vectorScore=round(max(0.0, score) if route == "vector" else 0.0, 8),
            sparseScore=round(max(0.0, score) if route == "sparse" else 0.0, 8),
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
