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

# Boilerplate sections are retained in the raw paper archive but should not
# become primary retrieval evidence in chunk-v2.  This list is v2-only so the
# historical chunk-v1 hash remains reproducible.
V2_SKIP_SECTION_KEYWORDS = SKIP_SECTION_KEYWORDS | {
    "funding",
    "conflicts of interest",
    "competing interests",
    "author contributions",
    "data availability",
    "ethics statement",
    "ethical approval",
    "informed consent",
    "publisher's note",
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


@dataclass(frozen=True)
class ChunkV2:
    """A structure-aware retrieval chunk.

    Offsets are relative to the normalized subsection text recorded in the
    same build.  They are intentionally not presented as offsets into the
    original PDF/XML because normalization and markdown parsing can change
    byte positions.
    """

    chunk_id: str
    pmcid: str
    pmid: str
    topic: str
    title: str
    year: str
    journal: str
    doi: str
    section: str
    subsection: str
    source_url: str
    text: str
    chunk_type: str
    paragraph_index: int
    paragraph_end_index: int
    sentence_start: int
    sentence_end: int
    char_start: int
    char_end: int
    offset_base: str


V2_VARIANTS = {
    "small": {"targetTokens": 256, "minTokens": 100, "maxTokens": 340},
    "medium": {"targetTokens": 512, "minTokens": 150, "maxTokens": 650},
    "large": {"targetTokens": 768, "minTokens": 220, "maxTokens": 960},
}


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
    parser.add_argument(
        "--version", choices=("v1", "v2"), default="v1",
        help="v1 preserves the original character splitter; v2 writes separate versioned assets",
    )
    parser.add_argument(
        "--variant", choices=tuple(V2_VARIANTS), default="medium",
        help="chunk-v2 target variant (small/medium/large)",
    )
    parser.add_argument(
        "--all-v2-variants", action="store_true",
        help="when --version v2, write all small/medium/large variants",
    )
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
        # A subset of imported XML records redundantly starts the body with
        # ``Abstract`` (sometimes glued to the first word as ``AbstractThe``).
        # The markdown heading already carries that structure, so remove only
        # this leading marker and never alter an occurrence inside the prose.
        abstract_text = re.sub(r"^abstract(?=[A-Z])", "", abstract_text, flags=re.I)
        abstract_text = re.sub(r"^abstract\s+", "", abstract_text, flags=re.I)

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


def _v2_token_count(text: str) -> int:
    """Count retrieval tokens without dropping stopwords used for sizing."""
    return len(TOKEN_RE.findall(text.lower()))


def split_into_structured_blocks(fulltext_text: str) -> List[Tuple[str, str, str]]:
    """Split normalized full text into (section, subsection, text) blocks.

    ``###`` is treated as a section and ``####`` as a subsection.  A
    subsection is never merged with the next subsection; this keeps the
    retrieval unit from silently crossing a scientific heading boundary.
    """
    if not fulltext_text:
        return []
    blocks: List[Tuple[str, str, str]] = []
    section = "正文"
    subsection = ""
    lines: List[str] = []

    def flush() -> None:
        nonlocal lines
        text = normalize_text("\n".join(lines))
        if text:
            blocks.append((section, subsection, text))
        lines = []

    for line in fulltext_text.splitlines():
        # Check the deeper heading first because #### also starts with ###.
        if line.startswith("#### "):
            flush()
            subsection = line[5:].strip() or ""
        elif line.startswith("### "):
            flush()
            section = line[4:].strip() or "正文"
            subsection = ""
        else:
            lines.append(line)
    flush()

    cleaned: List[Tuple[str, str, str]] = []
    for sec, sub, text in blocks:
        heading = f"{sec} {sub}".lower()
        if any(keyword in heading for keyword in V2_SKIP_SECTION_KEYWORDS):
            continue
        cleaned.append((sec, sub, text))
    return cleaned


def _paragraph_units(text: str) -> List[dict[str, object]]:
    """Return paragraph text and normalized-subsection character offsets."""
    units: List[dict[str, object]] = []
    sentence_index = 0
    pattern = re.compile(r"(?s)(.+?)(?:\n\s*\n+|\Z)")
    for paragraph_index, match in enumerate(pattern.finditer(text)):
        raw = match.group(1)
        paragraph = raw.strip()
        if not paragraph:
            continue
        leading = len(raw) - len(raw.lstrip())
        start = match.start(1) + leading
        end = start + len(paragraph)
        sentence_spans = [
            sentence_match
            for sentence_match in re.finditer(r"[^.!?。！？\n]+(?:[.!?。！？]|$)", paragraph)
            if sentence_match.group(0).strip()
        ]
        count = max(1, len(sentence_spans))
        units.append({
            "text": paragraph,
            "start": start,
            "end": end,
            "paragraphIndex": paragraph_index,
            "sentenceStart": sentence_index,
            "sentenceEnd": sentence_index + count,
            "sentenceSpans": sentence_spans,
            "chunkType": classify_chunk_type(paragraph),
        })
        sentence_index += count
    return units


def classify_chunk_type(text: str) -> str:
    """Keep captions/tables as independent retrieval records when detectable."""
    stripped = text.strip()
    lowered = stripped.lower()
    if re.match(r"^(?:figure|fig\.?|图)\s*\d+", lowered):
        return "figure_caption"
    if re.match(r"^table\s*\d+", lowered) or ("\n|" in stripped and stripped.startswith("|")):
        return "table"
    return "paragraph"


def _sentence_windows(
    unit: dict[str, object],
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
) -> List[dict[str, object]]:
    """Split one oversized paragraph by sentence, overlapping one sentence."""
    paragraph = str(unit["text"])
    spans = list(unit["sentenceSpans"])
    if not spans:
        spans = [re.match(r"(?s).+", paragraph)]  # type: ignore[list-item]
    sentences = [span.group(0).strip() for span in spans if span and span.group(0).strip()]
    if not sentences:
        return []
    output: List[dict[str, object]] = []
    index = 0
    while index < len(sentences):
        end = index
        count = 0
        while end < len(sentences):
            next_count = count + _v2_token_count(sentences[end])
            if end > index and next_count > max_tokens:
                break
            count = next_count
            end += 1
            if count >= target_tokens and end < len(sentences):
                lookahead = count + _v2_token_count(sentences[end])
                if lookahead > max_tokens:
                    break
        if end == index:
            # An individual sentence can exceed maxTokens.  Make a bounded
            # hard window only for this pathological case.
            words = paragraph.split()
            word_start = 0
            while word_start < len(words):
                word_end = min(len(words), word_start + max_tokens)
                piece = " ".join(words[word_start:word_end])
                while word_end > word_start + 1 and _v2_token_count(piece) > max_tokens:
                    word_end -= 1
                    piece = " ".join(words[word_start:word_end])
                output.append({
                    "text": piece,
                    "paragraphIndex": int(unit["paragraphIndex"]),
                    "sentenceStart": int(unit["sentenceStart"]),
                    "sentenceEnd": int(unit["sentenceStart"]) + 1,
                    "charStart": int(unit["start"]),
                    "charEnd": int(unit["end"]),
                    "chunkType": str(unit["chunkType"]),
                })
                if word_end >= len(words):
                    break
                word_start = max(word_start + 1, word_end - max_tokens // 10)
            break
        selected = sentences[index:end]
        first_span = spans[index]
        last_span = spans[end - 1]
        output.append({
            "text": " ".join(selected),
            "paragraphIndex": int(unit["paragraphIndex"]),
            "sentenceStart": int(unit["sentenceStart"]) + index,
            "sentenceEnd": int(unit["sentenceStart"]) + end,
            # Sentence-window offsets point to the selected evidence rather
            # than the complete oversized paragraph.  This makes downstream
            # graph/source highlighting auditable after v2 is rebuilt.
            "charStart": int(unit["start"]) + int(first_span.start()),
            "charEnd": int(unit["start"]) + int(last_span.end()),
            "chunkType": str(unit["chunkType"]) if str(unit["chunkType"]) != "paragraph" else "sentence_window",
        })
        if end >= len(sentences):
            break
        index = max(index + 1, end - 1)  # one-sentence overlap
    return output


def pack_structured_block(
    text: str,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
) -> List[dict[str, object]]:
    """Merge complete paragraphs until the soft target/max bound is reached."""
    units = _paragraph_units(text)
    packed: List[dict[str, object]] = []
    current: List[dict[str, object]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        value = "\n\n".join(str(unit["text"]) for unit in current).strip()
        tokens = _v2_token_count(value)
        if tokens >= min_tokens:
            packed.append({
                "text": value,
                "paragraphIndex": int(current[0]["paragraphIndex"]),
                "paragraphEndIndex": int(current[-1]["paragraphIndex"]) + 1,
                "sentenceStart": int(current[0]["sentenceStart"]),
                "sentenceEnd": int(current[-1]["sentenceEnd"]),
                "charStart": int(current[0]["start"]),
                "charEnd": int(current[-1]["end"]),
                "chunkType": (
                    str(current[0]["chunkType"])
                    if len({str(unit["chunkType"]) for unit in current}) == 1
                    and str(current[0]["chunkType"]) != "paragraph"
                    else "paragraph"
                ),
            })
        elif packed and _v2_token_count(str(packed[-1]["text"])) + tokens <= max_tokens:
            # A short tail is retained with the previous chunk.  This is
            # preferable to dropping a final result sentence as v1 did.
            packed[-1]["text"] = str(packed[-1]["text"]) + "\n\n" + value
            packed[-1]["paragraphEndIndex"] = int(current[-1]["paragraphIndex"]) + 1
            packed[-1]["sentenceEnd"] = int(current[-1]["sentenceEnd"])
            packed[-1]["charEnd"] = int(current[-1]["end"])
        elif str(current[0]["chunkType"]) in {"figure_caption", "table"}:
            # Captions/tables are valuable standalone evidence even when
            # shorter than the normal paragraph minimum.
            packed.append({
                "text": value,
                "paragraphIndex": int(current[0]["paragraphIndex"]),
                "paragraphEndIndex": int(current[-1]["paragraphIndex"]) + 1,
                "sentenceStart": int(current[0]["sentenceStart"]),
                "sentenceEnd": int(current[-1]["sentenceEnd"]),
                "charStart": int(current[0]["start"]),
                "charEnd": int(current[-1]["end"]),
                "chunkType": str(current[0]["chunkType"]),
            })
        current = []
        current_tokens = 0

    for unit in units:
        unit_tokens = _v2_token_count(str(unit["text"]))
        if unit_tokens > max_tokens:
            flush()
            packed.extend(_sentence_windows(unit, target_tokens, min_tokens, max_tokens))
            continue
        if current and current_tokens + unit_tokens > max_tokens:
            flush()
        current.append(unit)
        current_tokens += unit_tokens
        # Target is a soft boundary: keep adding the next whole paragraph when
        # it still fits, otherwise flush before that paragraph.
        if current_tokens >= target_tokens:
            continue
    flush()
    return packed


def build_chunks_v2(
    fulltext_dir: Path,
    fulltext_index: Dict[str, Dict[str, str]],
    variant: str,
) -> Tuple[List[ChunkV2], Dict[str, int]]:
    """Build a non-destructive structure-aware v2 variant."""
    if variant not in V2_VARIANTS:
        raise ValueError("unsupported chunk-v2 variant")
    params = V2_VARIANTS[variant]
    chunks: List[ChunkV2] = []
    topic_counter: Counter[str] = Counter()
    variant_code = variant[0].upper()

    for md_file in sorted(fulltext_dir.glob("PMCID-*.md")):
        pmcid = md_file.stem.replace("PMCID-", "")
        index_rec = fulltext_index.get(pmcid) or {}
        if index_rec.get("ingested") is False:
            continue
        title, meta, abstract_text, fulltext_text = parse_markdown_metadata(
            md_file.read_text(encoding="utf-8", errors="ignore")
        )
        pmid = meta.get("pmid") or index_rec.get("pmid") or ""
        topic = meta.get("topic") or index_rec.get("topic") or "unknown"
        year = meta.get("year", "")
        journal = meta.get("journal", "")
        doi = meta.get("doi", "")
        source_url = index_rec.get("source_url") or meta.get("europepmc", "")
        topic_counter[topic] += 1
        chunk_no = 0

        blocks: List[Tuple[str, str, str]] = []
        if abstract_text:
            blocks.append(("Abstract", "", abstract_text))
        blocks.extend(split_into_structured_blocks(fulltext_text))
        for section, subsection, block_text in blocks:
            # Abstracts are compact evidence units and must remain searchable
            # even when a source has fewer tokens than the normal variant
            # minimum.  The normal minimum still applies to body paragraphs.
            effective_min_tokens = 1 if section == "Abstract" else int(params["minTokens"])
            pieces = pack_structured_block(
                block_text,
                int(params["targetTokens"]),
                effective_min_tokens,
                int(params["maxTokens"]),
            )
            for piece in pieces:
                text = str(piece["text"]).strip()
                if not text:
                    continue
                chunk_no += 1
                kind = "abstract" if section == "Abstract" else str(piece["chunkType"])
                id_kind = "A" if section == "Abstract" else "F"
                chunk_id = f"{pmcid}-V2{variant_code}-{id_kind}{chunk_no:04d}"
                chunks.append(ChunkV2(
                    chunk_id=chunk_id,
                    pmcid=pmcid,
                    pmid=pmid,
                    topic=topic,
                    title=title,
                    year=year,
                    journal=journal,
                    doi=doi,
                    section=section,
                    subsection=subsection,
                    source_url=source_url,
                    text=text,
                    chunk_type=kind,
                    paragraph_index=int(piece["paragraphIndex"]),
                    paragraph_end_index=int(piece.get("paragraphEndIndex", piece["paragraphIndex"] + 1)),
                    sentence_start=int(piece["sentenceStart"]),
                    sentence_end=int(piece["sentenceEnd"]),
                    char_start=int(piece["charStart"]),
                    char_end=int(piece["charEnd"]),
                    offset_base="normalized_subsection",
                ))
    return chunks, dict(topic_counter)


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


def write_v2_variant(
    out_dir: Path,
    chunks: List[ChunkV2],
    topic_counter: Dict[str, int],
    variant: str,
    fulltext_dir: Path,
    fulltext_index_path: Path,
) -> tuple[Path, Path]:
    """Write chunk-v2 assets without touching any v1 file."""
    code = variant[0]
    chunk_out = out_dir / f"medical_chunks_v2_{variant}.jsonl"
    with chunk_out.open("w", encoding="utf-8") as fp:
        for chunk in chunks:
            embedding_text = "\n".join([
                f"Title: {chunk.title or 'none'}",
                f"Topic: {chunk.topic or 'unknown'}",
                f"Section: {chunk.section or '正文'}",
                f"Subsection: {chunk.subsection or 'none'}",
                chunk.text,
            ])
            fp.write(json.dumps({
                "chunkVersion": "chunk-v2",
                "variant": variant,
                "chunkId": chunk.chunk_id,
                "pmcid": chunk.pmcid,
                "pmid": chunk.pmid,
                "topic": chunk.topic,
                "title": chunk.title,
                "year": chunk.year,
                "journal": chunk.journal,
                "doi": chunk.doi,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "sourceUrl": chunk.source_url,
                "chunkType": chunk.chunk_type,
                "paragraphIndex": chunk.paragraph_index,
                "paragraphEndIndex": chunk.paragraph_end_index,
                "sentenceStart": chunk.sentence_start,
                "sentenceEnd": chunk.sentence_end,
                "charStart": chunk.char_start,
                "charEnd": chunk.char_end,
                "offsetBase": chunk.offset_base,
                "charCount": len(chunk.text),
                "tokenCount": _v2_token_count(chunk.text),
                "text": chunk.text,
                "embeddingText": embedding_text,
            }, ensure_ascii=False) + "\n")

    counts = Counter(chunk.chunk_type for chunk in chunks)
    char_counts = [len(chunk.text) for chunk in chunks]
    token_counts = [_v2_token_count(chunk.text) for chunk in chunks]
    manifest = {
        "chunkVersion": "chunk-v2",
        "variant": variant,
        "params": V2_VARIANTS[variant],
        "input": {
            "fulltextDir": str(fulltext_dir),
            "fulltextIndex": str(fulltext_index_path),
        },
        "output": {"chunks": str(chunk_out)},
        "summary": {
            "chunkCount": len(chunks),
            "paperCount": sum(topic_counter.values()),
            "topicPaperCount": topic_counter,
            "chunkTypeCounts": dict(sorted(counts.items())),
            "sections": dict(sorted(Counter(chunk.section for chunk in chunks).items())),
            "charStats": {
                "min": min(char_counts) if char_counts else 0,
                "mean": round(sum(char_counts) / len(char_counts), 3) if char_counts else 0,
                "median": int(sorted(char_counts)[len(char_counts) // 2]) if char_counts else 0,
                "max": max(char_counts) if char_counts else 0,
            },
            "tokenStats": {
                "min": min(token_counts) if token_counts else 0,
                "mean": round(sum(token_counts) / len(token_counts), 3) if token_counts else 0,
                "median": int(sorted(token_counts)[len(token_counts) // 2]) if token_counts else 0,
                "max": max(token_counts) if token_counts else 0,
            },
            "offsetBase": "normalized_subsection",
            "embeddingText": "title + topic + section + subsection + chunk text",
        },
    }
    manifest_out = out_dir / f"medical_rag_manifest_v2_{variant}.json"
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return chunk_out, manifest_out


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    fulltext_dir = (root / args.fulltext_dir).resolve()
    fulltext_index_path = (root / args.fulltext_index).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    index_records = load_fulltext_index(fulltext_index_path)
    if args.version == "v2":
        variants = tuple(V2_VARIANTS) if args.all_v2_variants else (args.variant,)
        for variant in variants:
            chunks_v2, topic_counter_v2 = build_chunks_v2(
                fulltext_dir, index_records, variant
            )
            chunk_out, manifest_out = write_v2_variant(
                out_dir, chunks_v2, topic_counter_v2, variant,
                fulltext_dir, fulltext_index_path,
            )
            print("RAG_CHUNK_V2_BUILD_DONE")
            print(f"variant={variant}")
            print(f"chunks={len(chunks_v2)}")
            print(f"papers={sum(topic_counter_v2.values())}")
            print(f"chunk_file={chunk_out}")
            print(f"manifest_file={manifest_out}")
        return

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

