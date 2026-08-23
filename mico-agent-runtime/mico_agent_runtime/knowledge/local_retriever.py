from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from mico_agent_runtime.contracts.evidence import EvidenceQuery, LiteratureEvidenceItem
from mico_agent_runtime.contracts.graph_rag import GraphEvidencePath, GraphPathStep, ReasoningPath
from mico_agent_runtime.contracts.retrieval import RetrievalPlan, build_retrieval_plan
from mico_agent_runtime.knowledge.embeddings import EmbeddingPort, GeminiEmbeddingPort
from mico_agent_runtime.knowledge.reranking import merge_and_rerank


class LocalKnowledgeIndexConfigurationError(ValueError):
    """The local full-text index is missing or unsafe to use."""


@dataclass(frozen=True)
class LocalKnowledgeIndexConfiguration:
    indexDirectory: Path
    retrievalBackend: Literal["tfidf", "gemini"] = "tfidf"

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "LocalKnowledgeIndexConfiguration":
        import os

        source = os.environ if env is None else env
        if source.get("MICO_LOCAL_KNOWLEDGE_ENABLED", "").strip().lower() != "true":
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_DISABLED")
        value = source.get("MICO_LOCAL_KNOWLEDGE_INDEX_DIR", "").strip()
        if not value or "\x00" in value:
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_INDEX_DIR_MISSING")
        backend = source.get("MICO_LOCAL_KNOWLEDGE_RETRIEVAL_BACKEND", "tfidf").strip().lower()
        if backend not in {"tfidf", "gemini"}:
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_BACKEND_UNSUPPORTED")
        return cls(Path(value).expanduser(), backend)


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{1,}")
STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "for", "on", "with", "by", "is",
    "are", "was", "were", "be", "as", "at", "from", "that", "this", "it", "its", "into",
    "their", "than", "then", "but", "if", "we", "our", "can", "could", "may", "might",
    "not", "no", "have", "has", "had", "also", "these", "those", "which", "who", "what",
    "when", "where", "why", "how", "such", "using", "used", "use", "between", "within",
    "without", "about", "after", "before", "during", "over", "under", "all", "any", "each",
    "other", "more", "most", "some", "many", "much", "via", "per", "both", "new", "one",
    "two", "three",
}

# Small language-bridge vocabulary for the local corpus.  This is retrieval
# normalization only: it does not classify a disease, create a cohort, or
# assert an ontology relationship.
QUERY_TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "糖尿病": ("diabetes", "diabetic"),
    "2型糖尿病": ("type", "diabetes"),
    "微生物组": ("microbiome", "microbiota"),
    "微生物": ("microbiome", "microbiota", "microbial"),
    "肠道": ("gut", "intestinal"),
    "文献": ("literature", "evidence"),
    "证据": ("evidence", "study"),
    "研究": ("research", "study"),
    "机制": ("mechanism", "pathway"),
    "健康": ("healthy", "control"),
    "对照": ("control", "healthy"),
}


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
    return terms


def _opaque_id(value: str) -> str:
    return "evidence-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


class LocalKnowledgeSearchPort:
    """Search the versioned 65-paper full-text corpus.

    The vector side is deterministic TF-IDF cosine.  Graph matches are
    provenance edges from chunks to controlled topics or candidate terms. The
    port never invents a relation and never treats a candidate taxon as a
    reviewed ontology assertion.
    """

    def __init__(
        self,
        configuration: LocalKnowledgeIndexConfiguration,
        embedding_port: EmbeddingPort | None = None,
    ) -> None:
        self._directory = configuration.indexDirectory
        self._backend = configuration.retrievalBackend
        self._embedding_port = embedding_port
        self._dense_vectors: dict[str, list[float]] = {}
        self._dense_dimension: int | None = None
        self._retrieval_model = "fulltext-tfidf-cosine-v1"
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, object]] = []
        self._graph_labels: dict[str, set[str]] = {}
        self._graph_nodes: dict[str, dict[str, object]] = {}
        self._graph_edges: list[dict[str, object]] = []
        self._graph_adjacency: defaultdict[str, list[tuple[str, dict[str, object], bool]]] = defaultdict(list)
        self._graph_paths_by_chunk: dict[str, list[GraphEvidencePath]] = {}
        self._load()

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "LocalKnowledgeSearchPort":
        configuration = LocalKnowledgeIndexConfiguration.from_environment(env)
        embedding_port = (
            GeminiEmbeddingPort.from_environment(env)
            if configuration.retrievalBackend == "gemini"
            else None
        )
        return cls(configuration, embedding_port=embedding_port)

    def _load(self) -> None:
        manifest_path = self._directory / "medical_knowledge_manifest.json"
        vector_meta_path = self._directory / "medical_vector_meta.json"
        vector_path = self._directory / "medical_vector_index.jsonl"
        graph_path = self._directory / "medical_knowledge_graph.jsonl"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            meta = json.loads(vector_meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_INDEX_UNAVAILABLE") from exc
        if manifest.get("corpusScope") != "fulltext_only" or manifest.get("evidenceTier") != "fulltext":
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_CORPUS_NOT_FULLTEXT")
        if meta.get("indexVersion") != "fulltext-tfidf-cosine-v1":
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_VECTOR_VERSION_UNSUPPORTED")
        raw_idf = meta.get("idf")
        if not isinstance(raw_idf, dict) or not raw_idf:
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_VECTOR_METADATA_INVALID")
        self._idf = {str(key): float(value) for key, value in raw_idf.items()}
        try:
            with vector_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("evidenceTier") != "fulltext" or not row.get("chunkId"):
                            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_VECTOR_RECORD_INVALID")
                        self._vectors.append(row)
            with graph_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("recordType") == "node":
                        self._graph_nodes[str(row["nodeId"])] = row
                        labels = _tokens(str(row.get("label", "")))
                        aliases = row.get("aliases")
                        if isinstance(aliases, list):
                            for alias in aliases:
                                if isinstance(alias, str):
                                    labels.extend(_tokens(alias))
                        self._graph_labels[str(row["nodeId"])] = set(labels)
                    elif row.get("recordType") == "edge":
                        self._graph_edges.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_GRAPH_INVALID") from exc
        if not self._vectors or not self._graph_edges:
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_INDEX_EMPTY")
        for node_id, node in self._graph_nodes.items():
            self._graph_labels.setdefault(node_id, set()).update(
                _tokens(str(node.get("label") or ""))
            )
        for edge in self._graph_edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if source and target:
                self._graph_adjacency[source].append((target, edge, False))
                self._graph_adjacency[target].append((source, edge, True))
        if self._backend == "gemini":
            if self._embedding_port is None:
                raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_EMBEDDING_NOT_CONFIGURED")
            dense_meta_path = self._directory / "medical_gemini_paper_embedding_meta.json"
            dense_path = self._directory / "medical_gemini_paper_embedding_index.jsonl"
            try:
                dense_meta = json.loads(dense_meta_path.read_text(encoding="utf-8"))
                if dense_meta.get("indexVersion") != "fulltext-gemini-paper-embedding-v1":
                    raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_EMBEDDING_VERSION_UNSUPPORTED")
                if dense_meta.get("model") != "gemini-embedding-2":
                    raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_EMBEDDING_MODEL_MISMATCH")
                if dense_meta.get("granularity") != "paper":
                    raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_EMBEDDING_GRANULARITY_UNSUPPORTED")
                self._dense_dimension = int(dense_meta.get("dimension"))
                if self._dense_dimension <= 0:
                    raise ValueError("invalid dimension")
                with dense_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        paper_id = str(row["paperId"])
                        values = row.get("embedding")
                        if (
                            row.get("evidenceTier") != "fulltext"
                            or not isinstance(values, list)
                            or len(values) != self._dense_dimension
                        ):
                            raise ValueError("invalid dense record")
                        parsed = [float(value) for value in values]
                        if not all(math.isfinite(value) for value in parsed):
                            raise ValueError("invalid dense vector")
                        self._dense_vectors[paper_id] = parsed
            except LocalKnowledgeIndexConfigurationError:
                raise
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_EMBEDDING_INDEX_INVALID") from exc
            if len(self._dense_vectors) != len({str(row.get("pmcid")) for row in self._vectors}):
                raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_EMBEDDING_INDEX_INCOMPLETE")
            self._retrieval_model = "gemini-embedding-2"

    def search(self, query: EvidenceQuery) -> list[LiteratureEvidenceItem]:
        """Execute the requested branch or both branches and rerank once."""
        if query.retrievalMode == "vector":
            return self.search_parallel(query, branches=("vector",))
        if query.retrievalMode == "graph":
            return self.search_parallel(query, branches=("graph",))
        return self.search_parallel(query, branches=("vector", "graph"))

    def search_parallel(
        self,
        query: EvidenceQuery,
        branches: tuple[Literal["vector", "graph"], ...] = ("vector", "graph"),
        plan: RetrievalPlan | None = None,
    ) -> list[LiteratureEvidenceItem]:
        """Run vector and graph retrieval independently, then unify their results.

        The index is immutable after construction, so bounded worker threads are
        safe here.  The returned model keeps branch provenance on every item and
        exposes only the final deterministic rerank score.
        """
        terms = _query_terms(f"{query.topic} {query.taxonName or ''}")
        if not terms:
            return []
        query_text = f"{query.topic} {query.taxonName or ''}".strip()
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
            )
        if len(requested) == 2:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mico-rag") as pool:
                vector_future = pool.submit(
                    self._dense_scores if self._backend == "gemini" else self._vector_scores,
                    query_text if self._backend == "gemini" else terms,
                )
                graph_future = pool.submit(
                    self._graph_scores, terms, plan.maxHops or 3, plan.minConfidence
                )
                vector_scores = vector_future.result()
                graph_scores, graph_paths = graph_future.result()
            vector_results = self._build_branch_results(
                query, vector_scores, {}, {}, "vector", max_items=query.limit
            )
            graph_results = self._build_branch_results(
                query, {}, graph_scores, graph_paths, "graph", max_items=query.limit
            )
            return merge_and_rerank(query, vector_results, graph_results, plan)

        if requested[0] == "vector":
            vector_scores = (
                self._dense_scores(query_text)
                if self._backend == "gemini"
                else self._vector_scores(terms)
            )
            return merge_and_rerank(query, self._build_branch_results(
                query, vector_scores, {}, {}, "vector", max_items=query.limit
            ), [], plan)
        graph_scores, graph_paths = self._graph_scores(terms, plan.maxHops or 3, plan.minConfidence)
        return merge_and_rerank(query, [], self._build_branch_results(
            query, {}, graph_scores, graph_paths, "graph", max_items=query.limit
        ), plan)

    def _build_branch_results(
        self,
        query: EvidenceQuery,
        vector_scores: dict[str, float],
        graph_scores: dict[str, float],
        graph_paths: dict[str, list[GraphEvidencePath]],
        route: Literal["vector", "graph", "hybrid"],
        *,
        max_items: int,
    ) -> list[LiteratureEvidenceItem]:
        scores: dict[str, float] = {}
        for chunk_id in set(vector_scores) | set(graph_scores):
            vector_score = vector_scores.get(chunk_id, 0.0)
            graph_score = graph_scores.get(chunk_id, 0.0)
            path_support = min(1.0, len(graph_paths.get(chunk_id, [])) / 2.0)
            if route == "vector":
                score = vector_score
            elif route == "graph":
                score = graph_score
            else:
                score = 0.55 * vector_score + 0.35 * graph_score + 0.10 * path_support
            if score > 0:
                scores[chunk_id] = score

        by_chunk = {str(row["chunkId"]): row for row in self._vectors}
        results: list[LiteratureEvidenceItem] = []
        seen_papers: set[str] = set()
        for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
            row = by_chunk.get(chunk_id)
            if row is None or str(row.get("pmcid")) in seen_papers:
                continue
            seen_papers.add(str(row.get("pmcid")))
            year = int(row.get("year") or 2000)
            excerpt = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()[:1200] or None
            results.append(LiteratureEvidenceItem(
                evidenceId=_opaque_id(f"local|{chunk_id}"),
                taxonName=query.taxonName,
                source="internal_knowledge",
                externalId=f"PMCID:{row.get('pmcid')}#{chunk_id}",
                title=str(row.get("title") or row.get("pmcid"))[:512],
                journal=str(row.get("journal") or "")[:256] or None,
                publicationYear=year if 1900 <= year <= 2100 else 2000,
                direction=query.direction,
                summary=(
                    "Full-text evidence chunk selected by " + route
                    + "; source chunk is retained for traceability."
                ),
                evidenceTier="fulltext",
                retrievalRoute=route,
                retrievalModel=self._retrieval_model,
                sourceChunkId=chunk_id,
                retrievalScore=round(score, 8),
                sourceExcerpt=excerpt,
                vectorScore=round(vector_scores.get(chunk_id, 0.0), 8),
                graphScore=round(graph_scores.get(chunk_id, 0.0), 8),
                rerankScore=round(score, 8),
                graphPaths=graph_paths.get(chunk_id, [])[:4],
                reasoningPaths=[
                    ReasoningPath.from_graph_path(path)
                    for path in graph_paths.get(chunk_id, [])[:4]
                ],
                retrievalSources=[route],
            ))
            if len(results) >= max_items:
                break
        return results

    @staticmethod
    def _merge_branch_results(
        query: EvidenceQuery,
        vector_results: list[LiteratureEvidenceItem],
        graph_results: list[LiteratureEvidenceItem],
    ) -> list[LiteratureEvidenceItem]:
        merged: dict[str, LiteratureEvidenceItem] = {}
        for item in vector_results + graph_results:
            key = item.sourceChunkId or item.externalId
            if key not in merged:
                merged[key] = item
                continue
            current = merged[key]
            paths = {path.pathId: path for path in current.graphPaths}
            paths.update({path.pathId: path for path in item.graphPaths})
            sources = sorted(set(current.retrievalSources) | set(item.retrievalSources))
            vector_score = max(current.vectorScore, item.vectorScore)
            graph_score = max(current.graphScore, item.graphScore)
            path_support = min(1.0, len(paths) / 2.0)
            score = 0.55 * vector_score + 0.35 * graph_score + 0.10 * path_support
            merged[key] = current.model_copy(update={
                "retrievalRoute": "hybrid",
                "retrievalSources": sources,
                "vectorScore": round(vector_score, 8),
                "graphScore": round(graph_score, 8),
                "rerankScore": round(score, 8),
                "retrievalScore": round(score, 8),
                "graphPaths": list(paths.values())[:4],
                "summary": "Full-text evidence jointly ranked by vector and graph branches.",
            })
        # A hybrid request is a unified result model even when only one branch
        # found a particular paper.  Keep the branch origin separately instead
        # of exposing an accidental vector/graph route label.
        for key, item in list(merged.items()):
            paths = {path.pathId: path for path in item.graphPaths}
            vector_score = item.vectorScore
            graph_score = item.graphScore
            path_support = min(1.0, len(paths) / 2.0)
            score = 0.55 * vector_score + 0.35 * graph_score + 0.10 * path_support
            merged[key] = item.model_copy(update={
                "retrievalRoute": "hybrid",
                "retrievalSources": sorted(set(item.retrievalSources)),
                "rerankScore": round(score, 8),
                "retrievalScore": round(score, 8),
                "summary": "Full-text evidence jointly ranked by vector and graph branches.",
            })
        ordered = sorted(merged.values(), key=lambda item: (-item.rerankScore, item.externalId))
        # A hybrid answer must keep both retrieval branches visible when both
        # branches produced candidates.  Otherwise a strong vector top-k can
        # silently erase every graph path from the returned evidence.
        if vector_results and graph_results and query.limit > 1:
            has_vector = any("vector" in item.retrievalSources for item in ordered[:query.limit])
            has_graph = any("graph" in item.retrievalSources for item in ordered[:query.limit])
            if not has_vector:
                ordered = ordered[: max(0, query.limit - 1)] + [vector_results[0].model_copy(update={
                    "retrievalRoute": "hybrid",
                    "summary": "Full-text evidence jointly ranked by vector and graph branches.",
                })]
            if not has_graph:
                ordered.append(graph_results[0].model_copy(update={
                    "retrievalRoute": "hybrid",
                    "summary": "Full-text evidence jointly ranked by vector and graph branches.",
                }))
                ordered = ordered[:query.limit]
            ordered = sorted(ordered, key=lambda item: (-item.rerankScore, item.externalId))
        return [item for item in ordered[:query.limit]]

    def _dense_scores(self, query_text: str) -> dict[str, float]:
        if self._embedding_port is None:
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_EMBEDDING_NOT_CONFIGURED")
        query_vector = self._embedding_port.embed_query(query_text)
        if self._dense_dimension is None or len(query_vector) != self._dense_dimension:
            raise LocalKnowledgeIndexConfigurationError("LOCAL_KNOWLEDGE_EMBEDDING_DIMENSION_MISMATCH")
        scores: dict[str, float] = {}
        for paper_id, vector in self._dense_vectors.items():
            score = sum(left * right for left, right in zip(query_vector, vector))
            if score > 0:
                for row in self._vectors:
                    if str(row.get("pmcid")) == paper_id:
                        scores[str(row["chunkId"])] = score
        return scores

    def _vector_scores(self, terms: list[str]) -> dict[str, float]:
        counts = Counter(terms)
        weighted = {term: (1.0 + math.log(count)) * self._idf.get(term, 0.0) for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        query_vector = {term: value / norm for term, value in weighted.items() if value > 0}
        scores: dict[str, float] = {}
        for row in self._vectors:
            vector = {str(term): float(value) for term, value in row.get("vector", [])}
            score = sum(query_vector.get(term, 0.0) * value for term, value in vector.items())
            if score > 0:
                scores[str(row["chunkId"])] = score
        return scores

    @staticmethod
    def _display_node(node_id: str, nodes: dict[str, dict[str, object]]) -> str:
        node = nodes.get(node_id)
        if node is not None and node.get("label"):
            return str(node["label"])[:512]
        if node_id.startswith("chunk:"):
            return node_id[6:][:512]
        if node_id.startswith("term:"):
            return node_id[5:][:512]
        if node_id.startswith("candidate_taxon:"):
            return "candidate_taxon"
        return node_id.split(":", 1)[-1][:512]

    @staticmethod
    def _reverse_relation(relation: str) -> str:
        return {
            "MENTIONS": "MENTIONED_BY",
            "MENTIONS_CANDIDATE": "CANDIDATE_MENTIONED_BY",
            "PART_OF": "HAS_CHUNK",
            "IN_SECTION": "HAS_CHUNK",
            "HAS_TOPIC": "TOPIC_OF",
            "ASSOCIATED_WITH_CANDIDATE": "CANDIDATE_ASSOCIATED_WITH",
        }.get(relation, relation + "_REVERSE")

    @staticmethod
    def _hop_status(target: str, edge: Mapping[str, object]) -> str:
        assertion = str(edge.get("assertionStatus") or "").lower()
        relation = str(edge.get("relation") or "").upper()
        if any(marker in assertion for marker in ("conflict", "contradict", "negat")):
            return "conflicted"
        if any(marker in relation for marker in ("CONFLICT", "CONTRADICT", "NEGAT")):
            return "conflicted"
        if target.startswith("candidate_taxon:") or "CANDIDATE" in relation:
            return "speculative"
        return "supported"

    @classmethod
    def _path_status(cls, statuses: list[str]) -> str:
        if "conflicted" in statuses:
            return "conflicted"
        if "speculative" in statuses:
            return "speculative"
        if all(value == "supported" for value in statuses):
            return "supported"
        return "partial"

    def _graph_scores(
        self, terms: list[str], max_hops: int = 3, min_confidence: float = 0.65
    ) -> tuple[dict[str, float], dict[str, list[GraphEvidencePath]]]:
        max_seed_nodes = 16
        max_expansions = 4000
        max_neighbors_per_node = 80
        allowed_relations = {
            "MENTIONS",
            "MENTIONS_CANDIDATE",
            "ASSOCIATED_WITH_CANDIDATE",
            "HAS_TOPIC",
            "PART_OF",
            "IN_SECTION",
            "CONTRADICTS",
            "NEGATES",
            "CONFLICTS_WITH",
        }
        query_terms = set(terms)
        seed_nodes = {
            node_id for node_id, labels in self._graph_labels.items()
            if labels & query_terms
        }
        scores: defaultdict[str, float] = defaultdict(float)
        paths_by_chunk: defaultdict[str, list[GraphEvidencePath]] = defaultdict(list)
        if not seed_nodes:
            return {}, {}

        # Bidirectional bounded BFS makes the graph branch genuinely
        # multi-hop: term -> chunk -> candidate taxon / topic / paper.  The
        # bound prevents unrestricted graph walking and every retained hop
        # must carry a source chunk.
        expansions = 0
        for seed in sorted(seed_nodes)[:max_seed_nodes]:
            queue = deque([(seed, [], {seed})])
            while queue and expansions < max_expansions:
                node_id, edge_path, visited = queue.popleft()
                depth = len(edge_path)
                if depth >= max(1, min(3, max_hops)):
                    continue
                neighbors = sorted(
                    self._graph_adjacency.get(node_id, []),
                    key=lambda item: (
                        0 if str(item[1].get("relation")) in {"MENTIONS", "MENTIONS_CANDIDATE"} else 1,
                        str(item[0]),
                    ),
                )
                for neighbor, edge, reversed_edge in neighbors[:max_neighbors_per_node]:
                    expansions += 1
                    if expansions >= max_expansions:
                        break
                    if neighbor in visited:
                        continue
                    if str(edge.get("relation")) not in allowed_relations:
                        continue
                    try:
                        if float(edge.get("confidence", 1.0)) < min_confidence:
                            continue
                    except (TypeError, ValueError):
                        continue
                    evidence_chunk = edge.get("evidenceChunkId")
                    if not isinstance(evidence_chunk, str) or not evidence_chunk:
                        continue
                    next_path = edge_path + [(neighbor, edge, reversed_edge)]
                    next_visited = visited | {neighbor}
                    if neighbor.startswith("chunk:"):
                        chunk_id = neighbor[6:]
                        steps: list[GraphPathStep] = []
                        for index, (target, hop, reverse) in enumerate(next_path):
                            source = seed if index == 0 else next_path[index - 1][0]
                            relation = str(hop.get("relation") or "RELATED_TO")
                            if reverse:
                                relation = self._reverse_relation(relation)
                            hop_status = self._hop_status(target, hop)
                            steps.append(GraphPathStep(
                                fromEntity=self._display_node(source, self._graph_nodes),
                                relation=relation,
                                toEntity=self._display_node(target, self._graph_nodes),
                                evidenceChunkId=str(hop["evidenceChunkId"]),
                                supportStatus=hop_status,
                            ))
                        path_id = "path-" + hashlib.sha256(
                            json.dumps(
                                [step.model_dump(mode="json") for step in steps],
                                sort_keys=True,
                            ).encode("utf-8")
                        ).hexdigest()[:32]
                        path_status = self._path_status([step.supportStatus for step in steps])
                        path = GraphEvidencePath(
                            pathId=path_id,
                            status=path_status,
                            hops=steps,
                            sourceDocumentIds=["PMCID:" + chunk_id.split("-", 1)[0]],
                            confidence=round(
                                min(1.0, 1.0 / len(steps))
                                * (0.5 if path_status == "speculative" else 0.25 if path_status == "conflicted" else 1.0),
                                6,
                            ),
                        )
                        signatures = {
                            tuple((step.fromEntity, step.relation, step.toEntity, step.evidenceChunkId)
                                  for step in value.hops)
                            for value in paths_by_chunk[chunk_id]
                        }
                        signature = tuple(
                            (step.fromEntity, step.relation, step.toEntity, step.evidenceChunkId)
                            for step in path.hops
                        )
                        if signature not in signatures and len(paths_by_chunk[chunk_id]) < 4:
                            paths_by_chunk[chunk_id].append(path)
                        scores[chunk_id] += 1.0 / len(steps)
                    queue.append((neighbor, next_path, next_visited))
        if not scores:
            return {}, {}
        maximum = max(scores.values())
        normalized = {chunk: value / maximum for chunk, value in scores.items()}
        bounded_paths = {
            chunk: paths[:4]
            for chunk, paths in paths_by_chunk.items()
        }
        return normalized, bounded_paths

    @staticmethod
    def _route(mode: str, vector_scores: dict[str, float], graph_scores: dict[str, float]) -> str:
        if mode in {"vector", "graph", "hybrid"}:
            return mode
        if vector_scores and graph_scores:
            return "hybrid"
        if graph_scores:
            return "graph"
        return "vector"

    def close(self) -> None:
        return None
