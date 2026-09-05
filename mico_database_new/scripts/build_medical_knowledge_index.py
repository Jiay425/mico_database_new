#!/usr/bin/env python3
"""Build a full-text-only vector and provenance graph index.

This version deliberately uses the checked-in 65-paper full-text corpus.
The vector side is a deterministic TF-IDF cosine baseline so it can be rebuilt
without downloading a model.  The graph side contains provenance-safe paper,
topic, section, and controlled/candidate term relations; it is not presented
as a complete biomedical ontology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{1,}")
BINOMIAL_RE = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z][a-z-]{3,})\b")
STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "for", "on", "with", "by",
    "is", "are", "was", "were", "be", "as", "at", "from", "that", "this", "it",
    "its", "into", "their", "than", "then", "but", "if", "we", "our", "can", "could",
    "may", "might", "not", "no", "have", "has", "had", "also", "these", "those",
    "which", "who", "what", "when", "where", "why", "how", "such", "using", "used",
    "use", "between", "within", "without", "about", "after", "before", "during", "over",
    "under", "all", "any", "each", "other", "more", "most", "some", "many", "much",
    "via", "per", "both", "new", "one", "two", "three",
}
BINOMIAL_EXCLUDES = {
    "According", "Abstract", "Alzheimer", "Author", "Background", "Conclusion", "Figure",
    "Frontiers", "Methods", "Nature", "Results", "Table", "This", "Type", "United",
}

CONTROLLED_TERMS: dict[str, tuple[str, ...]] = {
    "type_2_diabetes": ("t2d", "t2dm", "type 2 diabetes", "type 2 diabetes mellitus"),
    "healthy_control": ("healthy control", "healthy controls", "healthy subject", "healthy subjects"),
    "gut_microbiome": ("gut microbiome", "gut microbiota", "intestinal microbiota", "intestinal flora"),
    "dysbiosis": ("dysbiosis", "microbial dysbiosis"),
    "cirrhosis": ("cirrhosis",),
    "colorectal_cancer": ("colorectal cancer", "crc"),
    "coronary_artery_disease": ("coronary artery disease", "cad"),
    "alzheimer_disease": ("alzheimer disease", "alzheimer's disease", "ad"),
    "multiple_sclerosis": ("multiple sclerosis", "ms"),
    "fatty_liver": ("fatty liver", "nafld", "nonalcoholic fatty liver"),
    "inflammatory_bowel_disease": ("inflammatory bowel disease", "ibd"),
    "obesity": ("obesity",),
    "short_chain_fatty_acids": ("short-chain fatty acid", "short chain fatty acid", "scfa"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-text knowledge indexes")
    parser.add_argument("--chunks", default="references/knowledge/medical/rag/medical_chunks.jsonl")
    parser.add_argument("--out-dir", default="references/knowledge/medical/rag")
    return parser.parse_args()


def tokenize(value: str) -> list[str]:
    result: list[str] = []
    for token in TOKEN_RE.findall(value.lower()):
        if token.isascii() and token in STOPWORDS:
            continue
        if len(token) > 1:
            result.append(token)
    return result


def load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("pmcid") and value.get("chunkId") and value.get("text"):
                    rows.append(value)
    return rows


def build_vector_index(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    token_rows = [tokenize(str(row.get("text", ""))) for row in rows]
    document_frequency: Counter[str] = Counter()
    for tokens in token_rows:
        document_frequency.update(set(tokens))

    document_count = len(rows)
    idf = {
        term: math.log((document_count + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }
    vector_path = out_dir / "medical_vector_index.jsonl"
    with vector_path.open("w", encoding="utf-8") as handle:
        for row, tokens in zip(rows, token_rows):
            counts = Counter(tokens)
            weighted = {
                term: (1.0 + math.log(count)) * idf[term]
                for term, count in counts.items()
            }
            norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
            vector = [[term, round(value / norm, 8)] for term, value in sorted(weighted.items())]
            payload = {
                "indexVersion": "fulltext-tfidf-cosine-v1",
                "evidenceTier": "fulltext",
                "chunkId": row["chunkId"],
                "pmcid": row["pmcid"],
                "pmid": row.get("pmid", ""),
                "topic": row.get("topic", "unknown"),
                "title": row.get("title", ""),
                "year": row.get("year", ""),
                "journal": row.get("journal", ""),
                "doi": row.get("doi", ""),
                "section": row.get("section", ""),
                "sourceUrl": row.get("sourceUrl", ""),
                "text": row["text"],
                "vector": vector,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    meta_path = out_dir / "medical_vector_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "indexVersion": "fulltext-tfidf-cosine-v1",
                "evidenceTier": "fulltext",
                "algorithm": "sublinear-tf-idf-l2-cosine",
                "documentCount": document_count,
                "vocabularySize": len(idf),
                "idf": idf,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "indexVersion": "fulltext-tfidf-cosine-v1",
        "documentCount": document_count,
        "vocabularySize": len(idf),
        "path": str(vector_path),
    }


def canonical_controlled_terms(text: str) -> list[str]:
    lowered = text.lower()
    return [
        canonical
        for canonical, aliases in CONTROLLED_TERMS.items()
        if any(alias in lowered for alias in aliases)
    ]


def candidate_taxa(text: str) -> list[str]:
    values: set[str] = set()
    for genus, species in BINOMIAL_RE.findall(text):
        if genus in BINOMIAL_EXCLUDES or species in STOPWORDS:
            continue
        values.add(f"{genus} {species}")
    return sorted(values)[:32]


def build_graph(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    edge_ids: set[str] = set()

    def add_node(node_id: str, node_type: str, label: str, **extra: Any) -> None:
        if node_id in node_ids:
            return
        node_ids.add(node_id)
        records.append({
            "recordType": "node",
            "nodeId": node_id,
            "nodeType": node_type,
            "label": label,
            "evidenceTier": "fulltext",
            **extra,
        })

    def add_edge(source: str, relation: str, target: str, chunk_id: str | None = None) -> None:
        raw = f"{source}|{relation}|{target}|{chunk_id or ''}"
        edge_id = "edge-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        if edge_id in edge_ids:
            return
        edge_ids.add(edge_id)
        records.append({
            "recordType": "edge",
            "edgeId": edge_id,
            "source": source,
            "relation": relation,
            "target": target,
            "evidenceTier": "fulltext",
            "evidenceChunkId": chunk_id,
        })

    relation_chunks = 0
    controlled_mention_count = 0
    candidate_mention_count = 0
    for row in rows:
        pmcid = str(row["pmcid"])
        chunk_id = str(row["chunkId"])
        paper_id = f"paper:{pmcid}"
        chunk_node = f"chunk:{chunk_id}"
        topic = str(row.get("topic") or "unknown").lower()
        section = str(row.get("section") or "unknown")
        section_id = f"section:{pmcid}:{hashlib.sha256(section.encode()).hexdigest()[:16]}"
        add_node(paper_id, "paper", str(row.get("title") or pmcid), pmcid=pmcid, pmid=row.get("pmid"), doi=row.get("doi"), year=row.get("year"), sourceUrl=row.get("sourceUrl"))
        add_node(chunk_node, "chunk", chunk_id, pmcid=pmcid, section=section, sourceUrl=row.get("sourceUrl"))
        add_node(f"topic:{topic}", "topic", topic)
        add_node(section_id, "section", section, pmcid=pmcid)
        add_edge(chunk_node, "PART_OF", paper_id, chunk_id)
        add_edge(chunk_node, "IN_SECTION", section_id, chunk_id)
        add_edge(paper_id, "HAS_TOPIC", f"topic:{topic}", chunk_id)

        text = f"{row.get('title', '')} {row.get('text', '')}"
        controlled_ids: list[str] = []
        for term in canonical_controlled_terms(text):
            term_id = f"term:{term}"
            controlled_ids.append(term_id)
            controlled_mention_count += 1
            add_node(
                term_id,
                "controlled_term",
                term,
                aliases=list(CONTROLLED_TERMS[term]),
                extractionMethod="controlled_alias_v1",
            )
            add_edge(chunk_node, "MENTIONS", term_id, chunk_id)

        candidate_ids: list[str] = []
        for taxon in candidate_taxa(text):
            term_id = "candidate_taxon:" + hashlib.sha256(taxon.lower().encode()).hexdigest()[:32]
            candidate_ids.append(term_id)
            candidate_mention_count += 1
            add_node(term_id, "candidate_taxon", taxon, extractionMethod="binomial_heuristic_v1")
            add_edge(chunk_node, "MENTIONS_CANDIDATE", term_id, chunk_id)

        # These are source-local association edges, not asserted biological
        # causality.  They make the graph traversable for disease/topic ->
        # evidence chunk -> candidate taxon paths and retain the supporting
        # chunk on every hop.
        if controlled_ids and candidate_ids:
            relation_chunks += 1
            for controlled_id in controlled_ids:
                for candidate_id in candidate_ids:
                    add_edge(
                        controlled_id,
                        "ASSOCIATED_WITH_CANDIDATE",
                        candidate_id,
                        chunk_id,
                    )

    graph_path = out_dir / "medical_knowledge_graph.jsonl"
    with graph_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    covered_chunks = {
        str(record.get("evidenceChunkId"))
        for record in records
        if record["recordType"] == "edge" and record.get("evidenceChunkId")
    }
    return {
        "path": str(graph_path),
        "recordCount": len(records),
        "nodeCount": sum(record["recordType"] == "node" for record in records),
        "edgeCount": sum(record["recordType"] == "edge" for record in records),
        "graphVersion": "fulltext-provenance-graphrag-v2",
        "relationChunks": relation_chunks,
        "controlledMentionCount": controlled_mention_count,
        "candidateMentionCount": candidate_mention_count,
        "coveredChunkCount": len(covered_chunks),
        "chunkSourceCoverage": round(len(covered_chunks) / max(1, len(rows)), 6),
        "sourceCoverage": round(
            sum(
                record["recordType"] == "edge" and bool(record.get("evidenceChunkId"))
                for record in records
            ) / max(1, sum(record["recordType"] == "edge" for record in records)),
            6,
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    chunks_path = (root / args.chunks).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_chunks(chunks_path)
    if not rows:
        raise SystemExit("FULLTEXT_CORPUS_EMPTY")
    vector = build_vector_index(rows, out_dir)
    graph = build_graph(rows, out_dir)
    manifest = {
        "indexVersion": "fulltext-knowledge-index-v2",
        "graphVersion": "fulltext-provenance-graphrag-v2",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "corpusScope": "fulltext_only",
        "evidenceTier": "fulltext",
        "paperCount": len({row["pmcid"] for row in rows}),
        "chunkCount": len(rows),
        "vector": vector,
        "graph": graph,
        "limitations": [
            "vector_index_is_deterministic_tfidf_not_dense_semantic_embedding",
            "graph_contains_source_local_associations_not_a_complete_biomedical_ontology",
            "candidate_taxon_edges_require_review_before_claim_level_generation",
            "multi_hop_paths_are_bounded_to_three_hops",
        ],
    }
    manifest_path = out_dir / "medical_knowledge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MEDICAL_KNOWLEDGE_INDEX_BUILD_DONE")
    print(f"papers={manifest['paperCount']}")
    print(f"chunks={manifest['chunkCount']}")
    print(f"vector_documents={vector['documentCount']}")
    print(f"graph_nodes={graph['nodeCount']}")
    print(f"graph_edges={graph['edgeCount']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
