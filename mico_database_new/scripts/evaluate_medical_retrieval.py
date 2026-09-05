#!/usr/bin/env python3
"""Run the versioned retrieval contract/A-B harness.

This evaluates route correctness, full-text source coverage, graph-path
coverage and source diversity. It deliberately does not claim biological
relevance accuracy without a manually reviewed gold set.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from mico_agent_runtime.contracts.evidence import EvidenceQuery
from mico_agent_runtime.knowledge.embeddings import GeminiEmbeddingPort
from mico_agent_runtime.knowledge.local_retriever import (
    LocalKnowledgeIndexConfiguration,
    LocalKnowledgeSearchPort,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the medical GraphRAG retrieval contract")
    parser.add_argument("--backend", choices=("tfidf", "gemini"), default="tfidf")
    parser.add_argument("--index-dir", default="references/knowledge/medical/rag")
    parser.add_argument("--cases", default="references/knowledge/medical/rag/medical_retrieval_eval_v1.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    index_dir = (root / args.index_dir).resolve()
    cases_path = (root / args.cases).resolve()
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embedding = GeminiEmbeddingPort.from_environment() if args.backend == "gemini" else None
    port = LocalKnowledgeSearchPort(
        LocalKnowledgeIndexConfiguration(index_dir, retrievalBackend=args.backend),
        embedding_port=embedding,
    )
    results = []
    for case in cases:
        items = port.search(EvidenceQuery(
            topic=case["question"],
            direction="context",
            retrievalMode=case["retrievalMode"],
            limit=5,
        ))
        results.append({
            "caseId": case["caseId"],
            "expectedRoute": case["retrievalMode"],
            "actualRoute": items[0].retrievalRoute if items else None,
            "routeMatch": bool(items) and items[0].retrievalRoute == case["retrievalMode"],
            "resultCount": len(items),
            "fulltextSourceRate": round(
                sum(item.evidenceTier == "fulltext" and bool(item.sourceChunkId) for item in items)
                / max(1, len(items)),
                6,
            ),
            "graphPathCount": sum(len(item.graphPaths) for item in items),
            "distinctPaperCount": len({item.externalId.split("#", 1)[0] for item in items}),
        })
    output = {
        "evaluationVersion": "medical-retrieval-contract-eval-v1",
        "backend": args.backend,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "caseCount": len(results),
        "routeMatchRate": round(sum(item["routeMatch"] for item in results) / max(1, len(results)), 6),
        "fulltextSourceRate": round(sum(item["fulltextSourceRate"] for item in results) / max(1, len(results)), 6),
        "graphPathCaseRate": round(
            sum(item["graphPathCount"] > 0 for item in results) / max(1, len(results)), 6
        ),
        "cases": results,
        "interpretation": "contract_and_provenance_metrics_only_not_relevance_gold_accuracy",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
