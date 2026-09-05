from __future__ import annotations

"""Freeze provider-free input packets for an independent query set.

This script intentionally reads only chunk-v2 text and metadata.  It never
opens the GraphRAG assets, graph edges, or old provenance-derived qrels.  The
result is therefore a *query-generation work queue*, not qrels and not a
claim that questions have already been judged.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGETS = {
    "direct_evidence": 20,
    "entity_relation": 20,
    "mechanism": 20,
    "multi_hop": 15,
    "cross_paper_synthesis": 15,
    "conflicting_negative": 10,
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stable(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _section_priority(section: str) -> int:
    value = section.lower()
    if "abstract" in value:
        return 0
    if "result" in value:
        return 1
    if "discussion" in value or "conclusion" in value:
        return 2
    if "method" in value or "material" in value:
        return 4
    if "introduction" in value:
        return 3
    return 5


def _paper_context(rows: list[dict[str, Any]], max_chars: int = 5200) -> dict[str, Any]:
    rows = sorted(
        rows,
        key=lambda row: (
            _section_priority(str(row.get("section") or "")),
            int(row.get("paragraphIndex") or 0),
            str(row.get("chunkId")),
        ),
    )
    first = rows[0]
    selected: list[dict[str, Any]] = []
    used = 0
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        excerpt = text[:remaining]
        selected.append({
            "chunkId": row.get("chunkId"),
            "section": row.get("section"),
            "chunkType": row.get("chunkType"),
            "text": excerpt,
        })
        used += len(excerpt)
    return {
        "pmcid": first.get("pmcid"),
        "title": first.get("title"),
        "topic": first.get("topic"),
        "year": first.get("year"),
        "journal": first.get("journal"),
        "sourceUrl": first.get("sourceUrl"),
        "chunks": selected,
    }


def _prompt(category: str) -> str:
    instructions = {
        "direct_evidence": "Write one answerable question asking for a concrete finding, intervention, phenotype, or measured change explicitly reported in the supplied paper.",
        "entity_relation": "Write one question about whether two named biomedical entities are related in the supplied paper; require direction or comparison when the text supports it.",
        "mechanism": "Write one question asking how or through which pathway, mediator, barrier, metabolite, or signaling process the reported phenomenon may occur.",
        "multi_hop": "Write one question that requires connecting evidence from at least two sections of the same paper, without inventing a causal claim.",
        "cross_paper_synthesis": "Write one question that compares or synthesizes findings across the supplied distinct papers; do not ask for a fact present in only one paper.",
        "conflicting_negative": "Write one question that explicitly tests a null, negative, limitation, or conflicting finding in the supplied papers.",
    }
    return (
        "You are generating an independent biomedical retrieval query. "
        "Use only the supplied paper text; do not use graph edges, entity triples, or old qrels. "
        "Return one concise research question in English, and a short answerability note. "
        + instructions[category]
    )


def run(chunks_path: Path, output_path: Path) -> dict[str, Any]:
    rows = _jsonl(chunks_path)
    papers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        papers[str(row.get("pmcid"))].append(row)
    contexts = {
        pmcid: _paper_context(paper_rows)
        for pmcid, paper_rows in papers.items()
        if paper_rows
    }
    ordered = sorted(contexts.values(), key=lambda item: _stable(str(item["pmcid"])))
    topic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in ordered:
        topic_groups[str(context.get("topic") or "unknown")].append(context)

    packets: list[dict[str, Any]] = []
    # Single-paper categories rotate through the corpus to avoid concentrating
    # all questions on one topic.  Text is carried as source material for a
    # later LLM judge/generator, never converted into an automatic gold label.
    single_categories = [
        ("direct_evidence", TARGETS["direct_evidence"]),
        ("entity_relation", TARGETS["entity_relation"]),
        ("mechanism", TARGETS["mechanism"]),
        ("multi_hop", TARGETS["multi_hop"]),
        ("conflicting_negative", TARGETS["conflicting_negative"]),
    ]
    cursor = 0
    for category, count in single_categories:
        for _ in range(count):
            context = ordered[cursor % len(ordered)]
            cursor += 1
            packets.append({
                "queryId": f"IQ-{len(packets) + 1:03d}",
                "category": category,
                "status": "pending_llm_generation",
                "question": None,
                "answerabilityNote": None,
                "sourcePmcids": [context["pmcid"]],
                "generationPrompt": _prompt(category),
                "sourceContext": [context],
                "independenceGuard": {
                    "graphRead": False,
                    "oldQrelsRead": False,
                    "questionDerivedFromEdges": False,
                },
            })

    topic_order = sorted(topic_groups)
    for index in range(TARGETS["cross_paper_synthesis"]):
        topic = topic_order[index % len(topic_order)]
        group = topic_groups[topic]
        if len(group) < 2:
            # Fall back to two distinct papers while preserving the guard.
            selected = [ordered[index % len(ordered)], ordered[(index + 1) % len(ordered)]]
        else:
            start = index % len(group)
            selected = [group[start], group[(start + 1) % len(group)]]
        packets.append({
            "queryId": f"IQ-{len(packets) + 1:03d}",
            "category": "cross_paper_synthesis",
            "status": "pending_llm_generation",
            "question": None,
            "answerabilityNote": None,
            "sourcePmcids": [item["pmcid"] for item in selected],
            "generationPrompt": _prompt("cross_paper_synthesis"),
            "sourceContext": selected,
            "independenceGuard": {
                "graphRead": False,
                "oldQrelsRead": False,
                "questionDerivedFromEdges": False,
            },
        })

    result = {
        "reportVersion": "p2g-independent-query-freeze-v1",
        "status": "pending_llm_generation",
        "boundary": {
            "input": str(chunks_path),
            "graphAssetsRead": False,
            "oldQrelsRead": False,
            "questionCount": len(packets),
            "targetDistribution": TARGETS,
        },
        "annotationContract": {
            "queryOutput": ["question", "category", "sourcePmcids", "answerabilityNote"],
            "mustBeAnswerableFromSource": True,
            "mustNotUseGraphTriples": True,
            "mustNotTreatQuestionAsGold": True,
        },
        "counts": {category: sum(item["category"] == category for item in packets) for category in TARGETS},
        "packets": packets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.chunks, args.output)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
