from __future__ import annotations

"""Audit and freeze the graph-free query drafts.

The audit is intentionally conservative and provider-free.  It checks corpus
answerability, leakage/overlap, category shape, multi-hop evidence shape and
cross-paper feasibility.  A passed item is still a query candidate, not a
relevance judgment or a gold answer.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


TOPIC_CANONICAL = {
    "t2d": "type 2 diabetes", "t2dm": "type 2 diabetes", "crc": "colorectal cancer",
    "ad": "alzheimer disease", "early_ad": "alzheimer disease", "ibd": "inflammatory bowel disease",
    "ms": "multiple sclerosis", "cad": "coronary artery disease",
    "fatty_liver": "fatty liver disease", "healthy_baseline": "healthy control",
    "healthy_control": "healthy control",
}
ENTITY_PHRASES = (
    "type 2 diabetes", "healthy control", "cirrhosis", "colorectal cancer",
    "coronary artery disease", "alzheimer disease", "multiple sclerosis",
    "fatty liver", "inflammatory bowel disease", "obesity", "gut microbiome",
    "gut microbiota", "intestinal flora", "dysbiosis", "short-chain fatty acids",
    "bile acids", "lipopolysaccharide", "butyrate", "serotonin", "gut-brain axis",
    "gut-liver axis", "inflammation", "neuroinflammation", "tlr4",
)
STOPWORDS = {
    "what", "which", "how", "does", "do", "are", "is", "the", "a", "an", "and", "or",
    "of", "in", "on", "to", "from", "with", "about", "reported", "supplied", "study",
    "evidence", "findings", "concrete", "biological", "mechanisms", "pathways", "described",
    "across", "fit", "together", "regarding", "null", "negative", "limited", "conflicting",
    "studies", "study", "reported", "report",
}
MECHANISM_CUES = ("through", "via", "pathway", "mechanism", "mediated", "barrier", "signaling", "metabolite", "axis", "regulat")
POSITIVE_CUES = ("increased", "higher", "enriched", "elevated", "promot", "associated", "linked", "correlat")
NEGATIVE_CUES = ("decreased", "lower", "reduced", "depleted", "not associated", "no significant", "unclear", "lack", "failed", "did not", "contrary", "inconsistent")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _topic(value: str) -> str:
    raw = str(value or "").strip().lower()
    return TOPIC_CANONICAL.get(raw, raw.replace("_", " "))


def _text(rows: list[dict[str, Any]]) -> str:
    return " ".join(str(row.get("text") or "") for row in rows).lower()


def _entities(value: str) -> list[str]:
    lowered = value.lower()
    return [phrase for phrase in ENTITY_PHRASES if re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", lowered)]


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z][a-z0-9-]+", value.lower())
        if token not in STOPWORDS and len(token) > 2
    }


def _overlap_risk(question: str, source_rows: list[dict[str, Any]]) -> tuple[str, float]:
    # Topic/entity names must occur in a valid query, so do not mistake their
    # expected overlap for answer leakage.  Only compare residual wording.
    entity_tokens = _tokens(" ".join(ENTITY_PHRASES))
    query_tokens = _tokens(question) - entity_tokens
    if not query_tokens:
        return "low", 0.0
    max_ratio = 0.0
    for row in source_rows:
        for sentence in re.split(r"(?<=[.!?])\s+", str(row.get("text") or "")):
            sentence_tokens = _tokens(sentence)
            if not sentence_tokens:
                continue
            ratio = len(query_tokens & sentence_tokens) / max(1, len(query_tokens))
            max_ratio = max(max_ratio, ratio)
    if max_ratio >= 0.75:
        return "high", round(max_ratio, 8)
    if max_ratio >= 0.50:
        return "medium", round(max_ratio, 8)
    return "low", round(max_ratio, 8)


def _context_rows(papers: dict[str, list[dict[str, Any]]], pmcids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pmcid in pmcids:
        candidates = sorted(
            papers.get(str(pmcid), []),
            key=lambda row: (0 if "abstract" in str(row.get("section") or "").lower() else 1, str(row.get("section") or ""), int(row.get("paragraphIndex") or 0), str(row.get("chunkId"))),
        )
        rows.extend(candidates[:12])
    return rows


def _audit_item(item: dict[str, Any], papers: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    pmcids = [str(value) for value in item.get("sourcePmcids") or []]
    rows = _context_rows(papers, pmcids)
    all_text = _text(rows)
    topic = _topic(str((papers.get(pmcids[0], [{}])[0]).get("topic") if pmcids and papers.get(pmcids[0]) else ""))
    entities = _entities(all_text)
    if topic and topic not in entities:
        entities.insert(0, topic)
    entities = list(dict.fromkeys(entities))
    category = str(item.get("category"))
    sections = list(dict.fromkeys(str(row.get("section") or "unknown") for row in rows))
    mechanism_hits = [cue for cue in MECHANISM_CUES if cue in all_text]
    positive_hits = [cue for cue in POSITIVE_CUES if cue in all_text]
    negative_hits = [cue for cue in NEGATIVE_CUES if cue in all_text]
    if category == "direct_evidence":
        answerability = "supported" if topic and topic in all_text else "uncertain"
        evidence_shape = "single_chunk"
    elif category == "entity_relation":
        answerability = "supported" if len(entities) >= 2 else "uncertain"
        evidence_shape = "single_chunk" if len(rows) else "uncertain"
    elif category == "mechanism":
        answerability = "supported" if len(entities) >= 2 and mechanism_hits else "uncertain"
        evidence_shape = "multi_chunk_same_paper" if len(set(str(row.get("chunkId")) for row in rows)) >= 2 else "single_chunk"
    elif category == "multi_hop":
        answerability = "supported" if len(sections) >= 2 and len(rows) >= 2 and len(entities) >= 2 else "uncertain"
        evidence_shape = "multi_chunk_same_paper" if len(sections) >= 2 else "single_chunk"
    elif category == "cross_paper_synthesis":
        answerability = "supported" if len(set(pmcids)) >= 2 and sum(topic in _text(papers.get(pmcid, [])) for pmcid in set(pmcids)) >= 2 else "insufficient"
        evidence_shape = "multi_chunk_cross_paper"
    else:
        answerability = "supported" if len(set(pmcids)) >= 2 and positive_hits and negative_hits else "uncertain"
        evidence_shape = "multi_chunk_cross_paper" if len(set(pmcids)) >= 2 else "single_chunk"
    overlap, overlap_ratio = _overlap_risk(str(item.get("question") or ""), rows)
    category_correct = True
    if category == "entity_relation" and len(entities) < 2:
        category_correct = False
    if category == "mechanism" and not mechanism_hits:
        category_correct = False
    if category == "multi_hop" and evidence_shape == "single_chunk":
        category_correct = False
    if category == "cross_paper_synthesis" and len(set(pmcids)) < 2:
        category_correct = False
    if category == "conflicting_negative" and not negative_hits:
        category_correct = False
    status = "pass" if answerability == "supported" and overlap != "high" and category_correct else "rewrite"
    return {
        "answerability": answerability,
        "sourceOverlapRisk": overlap,
        "sourceOverlapRatio": overlap_ratio,
        "categoryCorrect": category_correct,
        "requiredEvidenceShape": evidence_shape,
        "sourcePaperCount": len(set(pmcids)),
        "detectedEntityCount": len(entities),
        "detectedEntities": entities[:16],
        "mechanismCueCount": len(mechanism_hits),
        "positiveCueCount": len(positive_hits),
        "negativeCueCount": len(negative_hits),
        "status": status,
    }


def _replacement_candidates(
    category: str,
    items: list[dict[str, Any]],
    papers: dict[str, list[dict[str, Any]]],
    used: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in items:
        if str(item.get("category")) != category or str(item.get("queryId")) in used:
            continue
        candidate = dict(item)
        candidate["question"] = _rewrite_question(category, papers, [str(value) for value in candidate.get("sourcePmcids") or []])
        audit = _audit_item(candidate, papers)
        if audit["status"] == "pass":
            candidate["quality"] = audit
            candidates.append(candidate)
    return candidates


def _conflict_replacement_candidates(
    items: list[dict[str, Any]],
    papers: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Pair same-topic papers with opposing/negative evidence cues."""
    by_topic: dict[str, list[tuple[str, str, bool, bool]]] = defaultdict(list)
    for pmcid, rows in sorted(papers.items()):
        text = _text(rows)
        if not rows:
            continue
        topic = _topic(str(rows[0].get("topic") or ""))
        positive = any(cue in text for cue in POSITIVE_CUES)
        negative = any(cue in text for cue in NEGATIVE_CUES)
        by_topic[topic].append((pmcid, topic, positive, negative))
    pairs_by_topic: dict[str, list[list[str]]] = defaultdict(list)
    for topic, candidates in sorted(by_topic.items()):
        positives = [row[0] for row in candidates if row[2]]
        negatives = [row[0] for row in candidates if row[3]]
        for left in positives:
            right = next((value for value in negatives if value != left), None)
            if right:
                pair = [left, right]
                if pair not in pairs_by_topic[topic]:
                    pairs_by_topic[topic].append(pair)
    # Spread the ten conflicting queries across topics first, then fill any
    # remaining slots with additional deterministic pairs.  This prevents all
    # questions from reusing one negative anchor paper.
    pairs: list[list[str]] = []
    for topic in sorted(pairs_by_topic):
        if pairs_by_topic[topic]:
            pairs.append(pairs_by_topic[topic][0])
    for topic in sorted(pairs_by_topic):
        for pair in pairs_by_topic[topic][1:]:
            if pair not in pairs:
                pairs.append(pair)
    output: list[dict[str, Any]] = []
    conflict_items = [item for item in items if str(item.get("category")) == "conflicting_negative"]
    for item, pair in zip(conflict_items, pairs):
        candidate = dict(item)
        candidate["sourcePmcids"] = pair
        candidate["question"] = _rewrite_question("conflicting_negative", papers, pair)
        candidate["quality"] = _audit_item(candidate, papers)
        if candidate["quality"]["status"] == "pass":
            output.append(candidate)
    return output


def _rewrite_question(category: str, papers: dict[str, list[dict[str, Any]]], pmcids: list[str]) -> str:
    rows = _context_rows(papers, pmcids)
    topic = _topic(str(rows[0].get("topic") or "the reported phenomenon")) if rows else "the reported phenomenon"
    entities = _entities(_text(rows))
    other = next((entity for entity in entities if entity != topic), "the related biological process")
    if category == "direct_evidence":
        return f"What concrete findings about {topic} are reported in the supplied study?"
    if category == "entity_relation":
        return f"What relationship between {topic} and {other} is described in the supplied study?"
    if category == "mechanism":
        return f"What biological mechanisms or pathways are proposed to connect {topic} with {other} in the supplied study?"
    if category == "multi_hop":
        sections = list(dict.fromkeys(str(row.get("section") or "the study") for row in rows))
        return f"How do the evidence in the {sections[0]} and the {sections[min(1, len(sections)-1)]} fit together regarding {topic}?"
    if category == "cross_paper_synthesis":
        return f"Across the supplied studies, what findings about {topic} are consistent, and where do they differ?"
    shared = set(_entities(_text(papers.get(pmcids[0], [])))) if pmcids else set()
    for pmcid in pmcids[1:]:
        shared &= set(_entities(_text(papers.get(pmcid, []))))
    other = next((entity for entity in sorted(shared) if entity != topic), None)
    if other:
        return f"What null, negative, limited, or conflicting findings about {topic} and {other} are reported across the supplied studies?"
    return f"What null, negative, limited, or conflicting findings about {topic} are reported across the supplied studies?"


def _unique_question(
    category: str,
    papers: dict[str, list[dict[str, Any]]],
    pmcids: list[str],
    occurrence: int,
) -> str:
    """Add a source-grounded focus so duplicate template questions diverge."""
    rows = _context_rows(papers, pmcids)
    if not rows:
        return _rewrite_question(category, papers, pmcids)
    topic = _topic(str(rows[0].get("topic") or "the reported phenomenon"))
    entities = [entity for entity in _entities(_text(rows)) if entity != topic]
    focus = entities[occurrence % len(entities)] if entities else None
    sections = list(dict.fromkeys(str(row.get("section") or "the study") for row in rows))
    section = sections[occurrence % len(sections)] if sections else "the reported study"
    scope = (
        "in the reported cohort",
        "in the mechanistic context",
        "in the study comparison",
        "in the reported discussion",
    )[occurrence % 4]
    if category == "direct_evidence" and focus:
        return f"What concrete findings about {topic}, particularly involving {focus}, are reported {scope} in the supplied study?"
    if category == "entity_relation" and focus:
        return f"What relationship involving {topic} and {focus} is described {scope} in the supplied study?"
    if category == "mechanism" and focus:
        return f"How might {focus} participate in the biological mechanisms or pathways connecting {topic} {scope}?"
    if category == "multi_hop" and focus:
        return f"How do evidence from the {section} and another section fit together regarding {topic} and {focus} {scope}?"
    if category == "cross_paper_synthesis" and focus:
        return f"Across the supplied studies, how do findings about {topic} in relation to {focus} agree or differ {scope}?"
    if category == "conflicting_negative" and focus:
        return f"What null, negative, limited, or conflicting findings about {topic} and {focus} are reported {scope} across the supplied studies?"
    return f"{_rewrite_question(category, papers, pmcids)} (focus: {section})"


def run(query_path: Path, freeze_path: Path, chunks_path: Path, output_path: Path) -> dict[str, Any]:
    query_payload = json.loads(query_path.read_text(encoding="utf-8"))
    freeze_payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    chunk_rows = _jsonl(chunks_path)
    papers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunk_rows:
        papers[str(row.get("pmcid"))].append(row)
    packet_by_id = {str(packet.get("queryId")): packet for packet in freeze_payload.get("packets") or []}
    items = [dict(item) for item in query_payload.get("items") or []]
    used = {str(item.get("queryId")) for item in items}
    audits: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    # First pass audits the frozen source assignment.
    for item in items:
        item["quality"] = _audit_item(item, papers)
    # Replace weak assignments using a same-category packet with supported
    # evidence shape.  The query id remains stable for traceability.
    conflict_candidates = {
        str(candidate.get("queryId")): candidate
        for candidate in _conflict_replacement_candidates(items, papers)
    }
    for index, item in enumerate(items):
        if item["quality"]["status"] == "pass":
            continue
        category = str(item.get("category"))
        candidates = (
            [conflict_candidates[str(item.get("queryId"))]]
            if category == "conflicting_negative"
            and str(item.get("queryId")) in conflict_candidates
            else _replacement_candidates(category, list(packet_by_id.values()), papers, {str(item.get("queryId"))})
        )
        if not candidates:
            continue
        candidate = candidates[0]
        previous = {"sourcePmcids": item.get("sourcePmcids"), "question": item.get("question")}
        item["sourcePmcids"] = candidate.get("sourcePmcids")
        item["sourcePaperCount"] = len(set(item.get("sourcePmcids") or []))
        item["question"] = candidate.get("question")
        item["answerability"] = "candidate_supported_by_source_pending_review"
        item["quality"] = _audit_item(item, papers)
        replacements.append({"queryId": item.get("queryId"), "category": category, "previous": previous, "replacement": {"sourcePmcids": item.get("sourcePmcids"), "question": item.get("question")}})
    # Template generation can produce identical wording for several papers in
    # the same topic.  Rewrite duplicate occurrences with a source-grounded
    # entity/section focus before declaring the set frozen.
    occurrences: dict[str, int] = {}
    for item in items:
        signature = re.sub(r"\s+", " ", str(item.get("question") or "").strip().lower())
        occurrence = occurrences.get(signature, 0)
        occurrences[signature] = occurrence + 1
        if occurrence == 0:
            continue
        previous_question = item.get("question")
        item["question"] = _unique_question(
            str(item.get("category")),
            papers,
            [str(value) for value in item.get("sourcePmcids") or []],
            occurrence,
        )
        item["quality"] = _audit_item(item, papers)
        replacements.append({
            "queryId": item.get("queryId"),
            "category": item.get("category"),
            "previous": {"question": previous_question},
            "replacement": {"question": item.get("question"), "reason": "duplicate_template_rewrite"},
        })
    for item in items:
        audits.append({"queryId": item.get("queryId"), "category": item.get("category"), **item.get("quality", {})})
    status_counts: dict[str, int] = {}
    for audit in audits:
        status_counts[audit["status"]] = status_counts.get(audit["status"], 0) + 1
    unique_count = len({str(item.get("question")) for item in items})
    duplicate_count = len(items) - unique_count
    frozen = status_counts.get("rewrite", 0) == 0 and duplicate_count == 0
    result = {
        "reportVersion": "p2g-independent-query-set-v1-frozen" if frozen else "p2g-independent-query-set-v1-audited",
        "querySetVersion": "query-set-v1" if frozen else None,
        "status": "frozen_evaluation_query_set" if frozen else "rewrite_required",
        "qrelsVersion": "qrels-v1" if frozen else None,
        "qrelsStatus": "not_built" if frozen else None,
        "sources": {"queryDrafts": str(query_path), "freezePackets": str(freeze_path), "chunks": str(chunks_path), "graphAssetsRead": False, "oldQrelsRead": False},
        "auditContract": {
            "requiredPass": ["answerability=supported", "sourceOverlapRisk!=high", "categoryCorrect=true"],
            "multiHopRequires": "multi_chunk_same_paper",
            "crossPaperRequires": "sourcePaperCount>=2",
            "conflictRequires": "positive_and_negative_evidence_cues_across_at_least_two_papers",
            "heuristicNotGold": True,
            "querySetFrozen": frozen,
            "qrelsNotPresent": True,
            "changePolicy": "do not edit unless an explicit answerability or semantic error is confirmed",
        },
        "counts": {
            "total": len(items),
            "status": status_counts,
            "replacements": len(replacements),
            "uniqueQuestionCount": unique_count,
            "duplicateQuestionCount": duplicate_count,
        },
        "replacements": replacements,
        "audit": audits,
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.queries, args.packets, args.chunks, args.output)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
