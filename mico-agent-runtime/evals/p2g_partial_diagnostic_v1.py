"""Provider-free diagnostic evaluation for the 1001-chunk controlled corpus.

This is deliberately separate from the formal database evaluator.  It uses
the already materialized chunk vectors when a query vector is cached, a
deterministic BM25-style lexical branch, and a provenance-bounded graph proxy.
It is useful for inspecting ranks and fusion while the remaining 307 Gemini
vectors are unavailable; its output is marked diagnostic and must not be used
as a full-corpus model-selection result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from p2g_retrieval_eval_v1 import _build_cases, _load_graph, _load_chunk_texts
from mico_agent_runtime.knowledge.database_retriever import _query_terms


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{1,}")
STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "for", "on", "with", "by",
    "is", "are", "was", "were", "be", "as", "at", "from", "that", "this", "it", "its",
    "into", "their", "than", "then", "but", "if", "we", "our", "can", "could", "may",
    "might", "not", "no", "have", "has", "had", "also", "these", "those", "which", "who",
    "what", "when", "where", "why", "how", "such", "using", "used", "use", "between",
    "within", "without", "about", "after", "before", "during", "over", "under", "all", "any",
    "each", "other", "more", "most", "some", "many", "much", "via", "per", "both", "new",
    "one", "two", "three",
}
WEIGHTS = {
    "single_fact": (0.55, 0.25, 0.20),
    "relation": (0.35, 0.20, 0.45),
    "multi_hop": (0.20, 0.15, 0.65),
    "document_synthesis": (0.45, 0.35, 0.20),
}


def _tokens(value: str) -> list[str]:
    return [
        token for token in TOKEN_RE.findall(value.lower())
        if not (token.isascii() and token in STOPWORDS) and len(token) > 1
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _load_cache(path: Path | None) -> tuple[str, dict[str, list[float]]]:
    if path is None or not path.exists():
        return "gemini-embedding-2", {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = str(payload.get("model") or "gemini-embedding-2")
    vectors: dict[str, list[float]] = {}
    for key, value in (payload.get("vectors") or {}).items():
        if isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
            vectors[str(key)] = [float(item) for item in value]
    return model, vectors


def _cache_key(model: str, query: str) -> str:
    return hashlib.sha256((model + "\x00" + query).encode("utf-8")).hexdigest()


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    return sum(a * b for a, b in zip(left, right))


def _rank(scores: dict[str, float], limit: int = 50) -> list[dict[str, Any]]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [{"chunkId": chunk_id, "score": round(float(score), 8), "rank": index}
            for index, (chunk_id, score) in enumerate(ordered, 1)]


def _dense_rank(
    chunks: list[dict[str, Any]],
    embedding_by_id: dict[str, list[float]],
    query_vector: list[float] | None,
) -> list[dict[str, Any]]:
    if query_vector is None:
        return []
    return _rank({
        str(row["chunkId"]): _cosine(query_vector, embedding_by_id[str(row["chunkId"])])
        for row in chunks
        if str(row.get("chunkId")) in embedding_by_id
    })


def _sparse_rank(chunks: list[dict[str, Any]], query_terms: list[str]) -> list[dict[str, Any]]:
    token_sets: dict[str, list[str]] = {}
    document_frequency: Counter[str] = Counter()
    for row in chunks:
        chunk_id = str(row["chunkId"])
        values = _tokens(" ".join((str(row.get("title") or ""), str(row.get("text") or ""))))
        token_sets[chunk_id] = values
        document_frequency.update(set(values))
    n_docs = max(1, len(chunks))
    avg_len = sum(len(value) for value in token_sets.values()) / n_docs
    query = list(dict.fromkeys(term for term in query_terms if len(term) > 1))
    scores: dict[str, float] = {}
    for chunk_id, values in token_sets.items():
        counts = Counter(values)
        length = len(values)
        score = 0.0
        for term in query:
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1.0 + (n_docs - document_frequency.get(term, 0) + 0.5)
                           / (document_frequency.get(term, 0) + 0.5))
            k1, b = 1.2, 0.75
            norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * length / max(1.0, avg_len)))
            score += idf * norm
        if score > 0:
            scores[chunk_id] = score / (1.0 + score)
    return _rank(scores)


def _graph_rank(
    chunks: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    query_terms: list[str],
) -> list[dict[str, Any]]:
    selected = {str(row["chunkId"]) for row in chunks}
    nodes = {str(row.get("nodeId")): row for row in graph_rows if row.get("recordType") == "node"}
    by_chunk: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph_rows:
        chunk_id = str(edge.get("evidenceChunkId") or "")
        if edge.get("recordType") != "edge" or chunk_id not in selected:
            continue
        labels = []
        for endpoint in (edge.get("source"), edge.get("target")):
            node = nodes.get(str(endpoint)) or {}
            labels.append(str(node.get("label") or ""))
        by_chunk[chunk_id].append({
            "labels": labels,
            "relationClass": str(edge.get("relationClass") or ""),
            "confidence": float(edge.get("confidence") or 0.0),
        })
    wanted = set(query_terms)
    scores: dict[str, float] = {}
    semantic = {"association", "causal", "directional"}
    for chunk_id, evidence in by_chunk.items():
        label_tokens = set(_tokens(" ".join(label for item in evidence for label in item["labels"])))
        hits = len(wanted & label_tokens)
        semantic_count = sum(item["relationClass"] in semantic for item in evidence)
        confidence = max((item["confidence"] for item in evidence), default=0.0)
        lexical = min(1.0, hits / 4.0)
        semantic_support = min(1.0, semantic_count / 2.0)
        score = 0.50 * lexical + 0.30 * semantic_support + 0.20 * confidence
        if score > 0:
            scores[chunk_id] = score
    return _rank(scores)


def _rrf_rank(
    branch_rows: dict[str, list[dict[str, Any]]],
    weights: tuple[float, float, float],
    equal_weight: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    names = ("dense", "sparse", "graph")
    available = [index for index, name in enumerate(names) if branch_rows.get(name)]
    if not available:
        return []
    active_weights = [1.0 if equal_weight else weights[index] for index in available]
    normalizer = sum(active_weights) or 1.0
    ranks = {
        name: {row["chunkId"]: row["rank"] for row in branch_rows.get(name, [])}
        for name in names
    }
    candidates = set().union(*(set(value) for value in ranks.values()))
    scores: dict[str, float] = {}
    for chunk_id in candidates:
        score = 0.0
        for index in available:
            rank = ranks[names[index]].get(chunk_id)
            if rank is not None:
                weight = 1.0 if equal_weight else weights[index]
                score += weight / (60.0 + rank)
        scores[chunk_id] = score / normalizer
    return _rank(scores, limit=limit)


def _metrics(ranked: list[dict[str, Any]], gold: set[str]) -> dict[str, float]:
    ids = [row["chunkId"] for row in ranked]
    if not gold:
        return {"recallAt5": 0.0, "recallAt10": 0.0, "recallAt20": 0.0, "recallAt50": 0.0,
                "hitAt5": 0.0, "hitAt10": 0.0, "hitAt20": 0.0, "hitAt50": 0.0,
                "mrrAt10": 0.0, "ndcgAt10": 0.0, "precisionAt5": 0.0}
    def recall(k: int) -> float:
        return len(set(ids[:k]) & gold) / len(gold)
    def hit(k: int) -> float:
        return float(bool(set(ids[:k]) & gold))
    first = next((index for index, chunk_id in enumerate(ids[:10], 1) if chunk_id in gold), None)
    dcg = sum(1.0 / math.log2(index + 1) for index, chunk_id in enumerate(ids[:10], 1) if chunk_id in gold)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(10, len(gold)) + 1))
    return {
        "recallAt5": round(recall(5), 8), "recallAt10": round(recall(10), 8),
        "recallAt20": round(recall(20), 8), "recallAt50": round(recall(50), 8),
        "hitAt5": hit(5), "hitAt10": hit(10), "hitAt20": hit(20), "hitAt50": hit(50),
        "mrrAt10": round(1.0 / first, 8) if first else 0.0,
        "ndcgAt10": round(dcg / ideal, 8) if ideal else 0.0,
        "precisionAt5": round(len(set(ids[:5]) & gold) / 5.0, 8),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 8) for key in keys}


def run(
    index_dir: Path,
    output: Path,
    query_cache: Path | None,
    cases_input: Path | None = None,
) -> dict[str, Any]:
    manifest_path = index_dir / "p2g-controlled-corpus-v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    assets = manifest.get("assets") or {}
    chunks_path = Path(str(assets.get("chunks") or index_dir / "medical_chunks_v2_medium_partial1001.jsonl"))
    embeddings_path = Path(str(assets.get("embeddings") or index_dir / "medical_gemini_chunk_embedding_index_v2_medium_partial1001.jsonl"))
    graph_asset = assets.get("graph")
    graph_path = Path(str(graph_asset)) if graph_asset else index_dir / "medical_knowledge_graph_v4_v2m_r3_partial1001.jsonl"
    chunks = _read_jsonl(chunks_path)
    embeddings = _read_jsonl(embeddings_path)
    graph_rows = _read_jsonl(graph_path)
    embedding_by_id = {
        str(row["chunkId"]): [float(item) for item in row["embedding"]]
        for row in embeddings if isinstance(row.get("embedding"), list)
    }
    nodes = {str(row.get("nodeId")): row for row in graph_rows if row.get("recordType") == "node"}
    edges = [row for row in graph_rows if row.get("recordType") == "edge"]
    # Reuse the same deterministic smoke-case constructor, but against v2m
    # graph/chunks so gold IDs are valid for the controlled corpus.
    source = manifest.get("source") or {}
    full_graph_path = Path(str(source.get("graph") or ""))
    full_chunks_path = Path(str(source.get("chunks") or ""))
    if not full_graph_path.exists():
        full_graph_path = index_dir.parent / "medical_knowledge_graph_v4_v2m_r3.jsonl"
    if not full_chunks_path.exists():
        full_chunks_path = index_dir.parent / "medical_chunks_v2_medium.jsonl"
    full_graph_nodes, full_graph_edges = _load_graph(full_graph_path)
    chunk_texts = _load_chunk_texts(full_chunks_path)
    if cases_input is not None and cases_input.exists():
        frozen = json.loads(cases_input.read_text(encoding="utf-8"))
        cases = [dict(item["case"]) for item in (frozen.get("cases") or [])]
        case_source = f"frozen cases from {cases_input.name}"
    else:
        cases = _build_cases(full_graph_nodes, full_graph_edges, chunk_texts)
        case_source = "deterministic provenance-derived smoke cases on v2m graph"
    model, cached = _load_cache(query_cache)
    branch_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_outputs: list[dict[str, Any]] = []
    for case in cases:
        terms = _query_terms(case["query"])
        query_text = " ".join(terms)
        query_vector = cached.get(_cache_key(model, query_text))
        branches = {
            "dense": _dense_rank(chunks, embedding_by_id, query_vector),
            "sparse": _sparse_rank(chunks, terms),
            "graph": _graph_rank(chunks, graph_rows, terms),
        }
        weights = WEIGHTS.get(case["category"], WEIGHTS["single_fact"])
        branches["rrf_standard"] = _rrf_rank(branches, weights, equal_weight=True)
        branches["rrf_weighted"] = _rrf_rank(branches, weights)
        gold = set(case.get("goldChunkIds") or []) & {str(row["chunkId"]) for row in chunks}
        metrics = {name: _metrics(rows, gold) for name, rows in branches.items()}
        for name in ("dense", "sparse", "graph", "rrf_standard", "rrf_weighted"):
            branch_rows[name].append(metrics[name])
        case_outputs.append({
            "case": case,
            "queryTerms": terms,
            "denseQueryVectorCached": query_vector is not None,
            "goldChunkCountInControlledCorpus": len(gold),
            "branches": branches,
            "metrics": metrics,
        })
    aggregate = {name: _aggregate(rows) for name, rows in branch_rows.items()}
    report = {
        "reportVersion": "p2g-partial-diagnostic-v1",
        "status": "COMPLETED",
        "evaluationBoundary": {
            "corpus": "dense-partial-corpus-v1",
            "corpusChunks": len(chunks),
            "totalMediumChunks": 1308,
            "excludedChunks": 1308 - len(chunks),
            "caseCount": len(cases),
            "caseSource": case_source,
            "diagnosticOnly": True,
            "denseUnavailableReason": "query embedding cache miss does not call provider",
        },
        "assets": {
            "chunks": str(chunks_path),
            "embeddings": str(embeddings_path),
            "graph": str(graph_path),
            "queryCache": str(query_cache) if query_cache else None,
        },
        "counts": {
            "embeddedChunks": len(embedding_by_id),
            "graphNodes": len(nodes),
            "graphEdges": len(edges),
            "denseQueryCacheEntries": len(cached),
            "denseAvailableCases": sum(item["denseQueryVectorCached"] for item in case_outputs),
        },
        "aggregateByBranch": aggregate,
        "cases": case_outputs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--index-dir", type=Path, default=root / "evals" / "partial-corpus-v1")
    parser.add_argument("--query-cache", type=Path,
                        default=root / "evals" / ".p2g-query-embedding-cache-v2-medium.json")
    parser.add_argument("--output", type=Path,
                        default=root / "evals" / "p2g-partial-diagnostic-v1.json")
    parser.add_argument("--cases-input", type=Path,
                        help="reuse a prior diagnostic's case list for a fair graph/fusion comparison")
    args = parser.parse_args()
    report = run(args.index_dir, args.output, args.query_cache, args.cases_input)
    print(json.dumps({
        "status": report["status"],
        "corpusChunks": report["evaluationBoundary"]["corpusChunks"],
        "caseCount": report["evaluationBoundary"]["caseCount"],
        "counts": report["counts"],
        "aggregateByBranch": report["aggregateByBranch"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
