#!/usr/bin/env python3
"""Query local medical RAG chunks with a simple BM25 retriever."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "for", "on", "with", "by", "is", "are", "was", "were",
    "be", "as", "at", "from", "that", "this", "it", "its", "into", "their", "than", "then", "but", "if", "we",
    "our", "can", "could", "may", "might", "not", "no", "yes", "have", "has", "had", "also", "these", "those",
    "which", "who", "whom", "what", "when", "where", "why", "how", "such", "using", "used", "use", "between",
    "within", "without", "about", "after", "before", "during", "over", "under", "all", "any", "each", "other",
    "more", "most", "some", "many", "much", "than", "via", "per", "both", "new", "one", "two", "three",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{1,}")


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(text.lower())
    out: List[str] = []
    for t in tokens:
        if t.isascii() and t in EN_STOPWORDS:
            continue
        if len(t) <= 1:
            continue
        out.append(t)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query local medical RAG index")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--chunks", default="references/knowledge/medical/rag/medical_chunks.jsonl")
    parser.add_argument("--bm25", default="references/knowledge/medical/rag/medical_bm25_stats.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--topic", default="", help="Filter by topic (optional)")
    return parser.parse_args()


def bm25_score(
    q_terms: List[str],
    doc_tokens: List[str],
    df: Dict[str, int],
    n_docs: int,
    avgdl: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    if not q_terms or not doc_tokens or avgdl <= 0:
        return 0.0

    score = 0.0
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)

    for term in q_terms:
        term_df = df.get(term, 0)
        if term_df <= 0:
            continue
        idf = math.log(1 + (n_docs - term_df + 0.5) / (term_df + 0.5))
        freq = tf.get(term, 0)
        if freq <= 0:
            continue
        denom = freq + k1 * (1 - b + b * dl / avgdl)
        score += idf * (freq * (k1 + 1)) / denom
    return score


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    chunk_path = (root / args.chunks).resolve()
    bm25_path = (root / args.bm25).resolve()

    bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))
    df = {k: int(v) for k, v in (bm25.get("df") or {}).items()}
    n_docs = int(bm25.get("n_docs") or 0)
    avgdl = float(bm25.get("avgdl") or 0.0)

    query_terms = tokenize(args.query)
    if not query_terms:
        print("No valid query terms.")
        return

    rows = []
    with chunk_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            if args.topic and (row.get("topic") or "").lower() != args.topic.lower():
                continue
            tokens = tokenize(row.get("text") or "")
            score = bm25_score(query_terms, tokens, df, n_docs, avgdl)
            if score <= 0:
                continue
            row["score"] = score
            rows.append(row)

    rows.sort(key=lambda x: x["score"], reverse=True)
    top = rows[: args.top_k]

    if not top:
        print("No result.")
        return

    for i, row in enumerate(top, start=1):
        snippet = (row.get("text") or "").replace("\n", " ").strip()
        snippet = snippet[:260] + ("..." if len(snippet) > 260 else "")
        print(f"[{i}] score={row['score']:.4f} topic={row.get('topic')} pmcid={row.get('pmcid')} section={row.get('section')}")
        print(f"    title: {row.get('title')}")
        print(f"    source: {row.get('sourceUrl')}")
        print(f"    snippet: {snippet}")


if __name__ == "__main__":
    main()
