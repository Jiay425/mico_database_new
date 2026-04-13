#!/usr/bin/env python3
"""Build searchable RAG chunk index from medical full-text markdown files."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "for", "on", "with", "by", "is", "are", "was", "were",
    "be", "as", "at", "from", "that", "this", "it", "its", "into", "their", "than", "then", "but", "if", "we",
    "our", "can", "could", "may", "might", "not", "no", "yes", "have", "has", "had", "also", "these", "those",
    "which", "who", "whom", "what", "when", "where", "why", "how", "such", "using", "used", "use", "between",
    "within", "without", "about", "after", "before", "during", "over", "under", "all", "any", "each", "other",
    "more", "most", "some", "many", "much", "than", "via", "per", "both", "new", "one", "two", "three",
}

SKIP_SECTION_KEYWORDS = {
    "references",
    "reference",
    "bibliography",
    "acknowledgment",
    "acknowledgement",
    "supplementary",
    "supporting information",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{1,}")


@dataclass
class Chunk:
    chunk_id: str
    pmcid: str
    pmid: str
    topic: str
    title: str
    year: str
    journal: str
    doi: str
    section: str
    source_url: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build medical RAG chunk index")
    parser.add_argument(
        "--fulltext-dir",
        default="references/knowledge/medical/papers-fulltext",
        help="Directory containing PMCID-*.md fulltext files",
    )
    parser.add_argument(
        "--fulltext-index",
        default="references/knowledge/medical/papers-fulltext/fulltext-index.jsonl",
        help="JSONL index generated during fulltext import",
    )
    parser.add_argument(
        "--out-dir",
        default="references/knowledge/medical/rag",
        help="Output directory for RAG index artifacts",
    )
    parser.add_argument("--max-chars", type=int, default=1800, help="Max chars per chunk")
    parser.add_argument("--overlap", type=int, default=250, help="Overlap chars between chunks")
    parser.add_argument("--min-chars", type=int, default=240, help="Minimum chars per chunk")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\u2009|\u2002|\u2003|\xa0", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(text.lower())
    filtered: List[str] = []
    for t in tokens:
        if t.isascii() and t in EN_STOPWORDS:
            continue
        if len(t) <= 1:
            continue
        filtered.append(t)
    return filtered


def parse_markdown_metadata(md_text: str) -> Tuple[str, Dict[str, str], str, str]:
    lines = md_text.splitlines()
    title = ""
    metadata: Dict[str, str] = {}
    fulltext_start = 0
    abstract_start = 0

    for idx, line in enumerate(lines):
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        if line.strip() == "## Metadata":
            j = idx + 1
            while j < len(lines):
                ln = lines[j].strip()
                if not ln:
                    j += 1
                    continue
                if ln.startswith("## "):
                    break
                if ln.startswith("- ") and ":" in ln:
                    key, val = ln[2:].split(":", 1)
                    metadata[key.strip().lower()] = val.strip()
                j += 1
        if line.strip() == "## Abstract":
            abstract_start = idx + 1
        if line.strip() == "## Full Text":
            fulltext_start = idx + 1
            break

    abstract_text = ""
    if abstract_start > 0:
        end = fulltext_start - 1 if fulltext_start > 0 else len(lines)
        abstract_text = normalize_text("\n".join(lines[abstract_start:end]))

    fulltext_text = normalize_text("\n".join(lines[fulltext_start:])) if fulltext_start > 0 else ""
    return title, metadata, abstract_text, fulltext_text


def split_into_sections(fulltext_text: str) -> List[Tuple[str, str]]:
    if not fulltext_text:
        return []

    sections: List[Tuple[str, str]] = []
    current_title = "正文"
    current_lines: List[str] = []

    for line in fulltext_text.splitlines():
        if line.startswith("### "):
            content = normalize_text("\n".join(current_lines))
            if content:
                sections.append((current_title, content))
            current_title = line[4:].strip() or "正文"
            current_lines = []
        else:
            current_lines.append(line)

    tail = normalize_text("\n".join(current_lines))
    if tail:
        sections.append((current_title, tail))

    cleaned: List[Tuple[str, str]] = []
    for sec_title, sec_text in sections:
        title_norm = sec_title.lower()
        if any(k in title_norm for k in SKIP_SECTION_KEYWORDS):
            continue
        cleaned.append((sec_title, sec_text))
    return cleaned


def choose_boundary(text: str, start: int, max_end: int, min_end: int) -> int:
    window = text[start:max_end]
    candidates = [m.start() for m in re.finditer(r"[。！？.!?;；]\s|\n\n", window)]
    if not candidates:
        return max_end
    best = max(candidates)
    split_pos = start + best + 1
    if split_pos < min_end:
        return max_end
    return split_pos


def split_section_to_chunks(sec_text: str, max_chars: int, overlap: int, min_chars: int) -> List[str]:
    text = normalize_text(sec_text)
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []

    chunks: List[str] = []
    start = 0
    total = len(text)

    while start < total:
        hard_end = min(total, start + max_chars)
        min_end = min(total, start + int(max_chars * 0.62))
        end = choose_boundary(text, start, hard_end, min_end)
        chunk = normalize_text(text[start:end])
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if end >= total:
            break
        start = max(0, end - overlap)

    return chunks


def load_fulltext_index(path: Path) -> Dict[str, Dict[str, str]]:
    by_pmcid: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            pmcid = (rec.get("pmcid") or "").strip()
            if not pmcid:
                continue
            by_pmcid[pmcid] = rec
    return by_pmcid


def build_chunks(
    fulltext_dir: Path,
    fulltext_index: Dict[str, Dict[str, str]],
    max_chars: int,
    overlap: int,
    min_chars: int,
) -> Tuple[List[Chunk], Dict[str, int]]:
    chunks: List[Chunk] = []
    topic_counter: Counter[str] = Counter()

    for md_file in sorted(fulltext_dir.glob("PMCID-*.md")):
        pmcid = md_file.stem.replace("PMCID-", "")
        index_rec = fulltext_index.get(pmcid) or {}
        if index_rec.get("ingested") is False:
            continue

        text = md_file.read_text(encoding="utf-8", errors="ignore")
        title, meta, abstract_text, fulltext_text = parse_markdown_metadata(text)

        pmid = meta.get("pmid") or index_rec.get("pmid") or ""
        topic = meta.get("topic") or index_rec.get("topic") or "unknown"
        year = meta.get("year", "")
        journal = meta.get("journal", "")
        doi = meta.get("doi", "")
        source_url = index_rec.get("source_url") or meta.get("europepmc", "")

        topic_counter[topic] += 1
        chunk_no = 0

        if abstract_text:
            for piece in split_section_to_chunks(abstract_text, max_chars, overlap, min_chars):
                chunk_no += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{pmcid}-A{chunk_no:03d}",
                        pmcid=pmcid,
                        pmid=pmid,
                        topic=topic,
                        title=title,
                        year=year,
                        journal=journal,
                        doi=doi,
                        section="Abstract",
                        source_url=source_url,
                        text=piece,
                    )
                )

        for sec_title, sec_text in split_into_sections(fulltext_text):
            sec_chunks = split_section_to_chunks(sec_text, max_chars, overlap, min_chars)
            for piece in sec_chunks:
                chunk_no += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{pmcid}-F{chunk_no:03d}",
                        pmcid=pmcid,
                        pmid=pmid,
                        topic=topic,
                        title=title,
                        year=year,
                        journal=journal,
                        doi=doi,
                        section=sec_title,
                        source_url=source_url,
                        text=piece,
                    )
                )

    return chunks, dict(topic_counter)


def build_bm25_stats(chunks: Iterable[Chunk]) -> Dict[str, object]:
    chunk_list = list(chunks)
    df_counter: Counter[str] = Counter()
    doc_lengths: List[int] = []

    for ch in chunk_list:
        tokens = tokenize(ch.text)
        doc_lengths.append(len(tokens))
        for tok in set(tokens):
            df_counter[tok] += 1

    n_docs = len(chunk_list)
    avgdl = (sum(doc_lengths) / n_docs) if n_docs > 0 else 0.0

    return {
        "n_docs": n_docs,
        "avgdl": round(avgdl, 6),
        "df": df_counter,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    fulltext_dir = (root / args.fulltext_dir).resolve()
    fulltext_index_path = (root / args.fulltext_index).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    index_records = load_fulltext_index(fulltext_index_path)
    chunks, topic_counter = build_chunks(
        fulltext_dir,
        index_records,
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_chars=args.min_chars,
    )

    chunk_out = out_dir / "medical_chunks.jsonl"
    with chunk_out.open("w", encoding="utf-8") as fp:
        for ch in chunks:
            tokens = tokenize(ch.text)
            row = {
                "chunkId": ch.chunk_id,
                "pmcid": ch.pmcid,
                "pmid": ch.pmid,
                "topic": ch.topic,
                "title": ch.title,
                "year": ch.year,
                "journal": ch.journal,
                "doi": ch.doi,
                "section": ch.section,
                "sourceUrl": ch.source_url,
                "charCount": len(ch.text),
                "tokenCount": len(tokens),
                "text": ch.text,
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = build_bm25_stats(chunks)
    bm25_out = out_dir / "medical_bm25_stats.json"
    bm25_payload = {
        "n_docs": stats["n_docs"],
        "avgdl": stats["avgdl"],
        "created_at": stats["created_at"],
        "df": stats["df"],
    }
    bm25_out.write_text(json.dumps(bm25_payload, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "fulltextDir": str(fulltext_dir),
            "fulltextIndex": str(fulltext_index_path),
        },
        "output": {
            "chunks": str(chunk_out),
            "bm25": str(bm25_out),
        },
        "params": {
            "maxChars": args.max_chars,
            "overlap": args.overlap,
            "minChars": args.min_chars,
        },
        "summary": {
            "chunkCount": len(chunks),
            "paperCount": sum(topic_counter.values()),
            "topicPaperCount": topic_counter,
            "bm25Docs": stats["n_docs"],
            "bm25Avgdl": stats["avgdl"],
        },
    }
    manifest_out = out_dir / "medical_rag_manifest.json"
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("RAG_INDEX_BUILD_DONE")
    print(f"chunks={len(chunks)}")
    print(f"papers={sum(topic_counter.values())}")
    print(f"topics={len(topic_counter)}")
    print(f"chunk_file={chunk_out}")
    print(f"bm25_file={bm25_out}")
    print(f"manifest_file={manifest_out}")


if __name__ == "__main__":
    main()

