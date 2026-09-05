from __future__ import annotations

"""Provider-free sentence triage for r3 self-loops without replacements.

This is a review aid, not an LLM adjudication.  It applies the same extractor
to the saved evidence sentence and reports candidate categories; a human/LLM
judge must confirm the final semantic label.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mico_agent_runtime.knowledge.graph_v3 import extract_mentions, extract_relations


# The graph extractor intentionally only admits a small controlled vocabulary
# and conservative taxon candidates.  For an unresolved self-loop that is a
# useful safety property, but it also means that a sentence can contain a
# real biomedical entity which is invisible to the extractor.  The patterns
# below are therefore *surface-candidate* detectors only.  They never create
# graph nodes or edges; they make the review queue tell us whether the likely
# problem is entity normalization rather than a legitimately unsupported
# relation.
_ACRONYM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z][A-Za-z]{1,8}-[A-Za-z0-9]{1,8}|[A-Z]{2,}[A-Za-z0-9]*|[A-Za-z]{2,}[0-9]+|[a-z]-[a-z]{2,})(?![A-Za-z0-9])"
)
_TAXON_WORD_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z][a-z]{3,})(?![A-Za-z0-9])")
_TAXON_SUFFIXES = (
    "aceae", "obacteria", "microbia", "mycota", "phyta", "viridae", "oides",
    "ibacter", "bacter", "monas", "coccus", "cocci", "ella", "buria", "etes",
    "utes", "ium", "ensis", "ense", "inae", "zoa",
)
_SURFACE_STOPWORDS = {
    "AD", "AI", "API", "DNA", "RNA", "USA", "US", "UK", "EU", "I", "II", "III",
    "The", "This", "That", "These", "Those", "Though", "Our", "But", "On", "To",
    "If", "In", "As", "For", "Unlike", "Compared", "Several", "Data", "Day", "Liu",
}
_NON_ENTITY_HYPHEN_SUFFIXES = ("-based", "-related", "-associated", "-deplete", "-induced", "-derived")
_RELATION_TRIGGER_RE = re.compile(
    r"\b(?:increase(?:d|s)?|decrease(?:d|s)?|higher|lower|enriched|elevated|reduced|"
    r"depleted|associated with|correlated with|related to|linked to|interact(?:s|ed)? with|"
    r"cause(?:s|d)?|lead(?:s|ing)? to|drive(?:s|n)?|trigger(?:s|ed)?|promot(?:e|es|ed|ing)|"
    r"inhibit(?:s|ed|ing)?|suppress(?:es|ed|ing)?|mediate(?:s|d)?|through|via|"
    r"regulat(?:e|es|ed|ing)|predict(?:s|ed)?|marker(?:s)?|difference(?:s)?|"
    r"significant(?:ly)?\s+(?:higher|lower|different)|positive(?:ly)?\s+correlat(?:ed|es)?|"
    r"negative(?:ly)?\s+correlat(?:ed|es)?)\b",
    re.I,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _surface_entity_candidates(text: str, mentions: list[Any]) -> list[dict[str, Any]]:
    """Find conservative, non-graph surface entity candidates.

    A candidate is considered recognized only when its character span
    overlaps an extractor mention.  We deliberately keep the raw surface
    and detector so a reviewer can quickly decide whether a normalizer or
    alias should be added.  Ordinary title-case prose is excluded unless it
    has a taxonomic suffix; acronyms must contain a digit, a hyphen, or at
    least two uppercase letters.
    """
    candidates: list[dict[str, Any]] = []

    def add(match: re.Match[str], detector: str) -> None:
        surface = match.group(0)
        if surface in _SURFACE_STOPWORDS or len(surface) < 2:
            return
        if surface.lower().endswith(_NON_ENTITY_HYPHEN_SUFFIXES):
            return
        start, end = match.start(), match.end()
        if any(item["start"] == start and item["end"] == end for item in candidates):
            return
        recognized = any(mention.start < end and mention.end > start for mention in mentions)
        candidates.append({
            "surface": surface,
            "start": start,
            "end": end,
            "detector": detector,
            "recognizedByExtractor": recognized,
        })

    for match in _ACRONYM_RE.finditer(text):
        surface = match.group(0)
        # Require an unambiguous biomedical-looking form.  This avoids
        # treating sentence-initial ordinary words such as ``Our`` as entities.
        if not (any(char.isdigit() for char in surface) or "-" in surface or surface.isupper()):
            continue
        add(match, "acronym_or_symbol")
    for match in _TAXON_WORD_RE.finditer(text):
        if match.group(1).lower().endswith(_TAXON_SUFFIXES):
            add(match, "taxon_like_word")
    return sorted(candidates, key=lambda item: (item["start"], item["end"], item["surface"]))


def run(r3_rejected_path: Path, r4_graph_path: Path, output_path: Path) -> dict[str, Any]:
    old = [
        row for row in _jsonl(r3_rejected_path)
        if row.get("recordType") == "edge"
        and "SELF_LOOP_RELATION" in (row.get("rejectionIssueCodes") or [])
    ]
    r4 = [
        row for row in _jsonl(r4_graph_path)
        if row.get("recordType") == "edge"
        and row.get("relationClass") != "structural"
        and row.get("source") != row.get("target")
    ]
    by_span: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in r4:
        by_span.setdefault((row.get("evidenceChunkId"), row.get("evidenceStart"), row.get("evidenceEnd")), []).append(row)
    records: list[dict[str, Any]] = []
    for row in old:
        key = (row.get("evidenceChunkId"), row.get("evidenceStart"), row.get("evidenceEnd"))
        replacements = by_span.get(key, [])
        text = str(row.get("evidenceText") or "")
        mentions = extract_mentions(text)
        distinct_mentions: dict[str, Any] = {}
        for mention in mentions:
            distinct_mentions.setdefault(mention.node_id, mention)
        relations, _ = extract_relations(text)
        relation_shapes = sorted({(item.relation, item.assertion_status) for item in relations})
        surface_candidates = _surface_entity_candidates(text, mentions)
        unrecognized_candidates = [
            item for item in surface_candidates if not item["recognizedByExtractor"]
        ]
        distinct_surface_candidates = {
            _norm(item["surface"]): item for item in surface_candidates
        }
        distinct_unrecognized_candidates = {
            _norm(item["surface"]): item for item in unrecognized_candidates
        }
        trigger_hits = sorted({match.group(0).lower() for match in _RELATION_TRIGGER_RE.finditer(text)})
        if replacements:
            label = "recovered_distinct_relation"
        elif len(relation_shapes) > 1:
            label = "ambiguous"
        elif len(distinct_mentions) >= 2 and relations:
            label = "missed_distinct_relation"
        elif len(distinct_mentions) >= 2:
            label = "relation_trigger_error"
        elif len(distinct_unrecognized_candidates) >= 1 and len(distinct_surface_candidates) >= 2:
            # At least one plausible second entity was present but absent from
            # the controlled extractor.  This is a normalization/alias audit
            # candidate, not evidence that an edge should be restored.
            label = "entity_normalization_error"
        else:
            label = "correct_drop"
        records.append({
            "oldEdgeId": row.get("edgeId"),
            "evidenceChunkId": row.get("evidenceChunkId"),
            "oldRelation": row.get("relation"),
            "oldAssertionStatus": row.get("assertionStatus"),
            "oldEvidenceText": text,
            "r4ReplacementCount": len(replacements),
            "detectedDistinctMentionCount": len(distinct_mentions),
            "detectedMentions": [
                {"nodeId": mention.node_id, "entityType": mention.entity_type, "canonicalLabel": mention.canonical_label, "surface": mention.surface}
                for mention in distinct_mentions.values()
            ],
            "surfaceEntityCandidates": surface_candidates,
            "unrecognizedSurfaceCandidates": unrecognized_candidates,
            "distinctSurfaceCandidateCount": len(distinct_surface_candidates),
            "distinctUnrecognizedSurfaceCandidateCount": len(distinct_unrecognized_candidates),
            "relationTriggerHits": trigger_hits,
            "candidateRelations": [
                {"relation": item.relation, "relationClass": item.relation_class, "assertionStatus": item.assertion_status, "confidence": item.confidence}
                for item in relations
            ],
            "heuristicClass": label,
            "judge": {"decision": "pending_sentence_level_adjudication", "finalClass": None, "reason": None},
        })
    counts = Counter(record["heuristicClass"] for record in records)
    unresolved = [record for record in records if not record["r4ReplacementCount"]]
    result = {
        "reportVersion": "p2g-unresolved-selfloop-triage-v2-surface-audit",
        "status": "heuristic_review_aid",
        "classificationContract": {
            "correct_drop": "no second recognized or conservative surface entity candidate in the evidence sentence",
            "missed_distinct_relation": "two or more distinct entities and an explicit relation extracted, but no exact-span r4 replacement",
            "entity_normalization_error": "at least two surface candidates with one or more not recognized by the extractor; judge must verify the candidate is a real entity",
            "relation_trigger_error": "two distinct entities recognized but no relation candidate emitted",
            "ambiguous": "multiple competing relation/assertion interpretations",
            "heuristicNotGold": True,
        },
        "sources": {"r3Rejected": str(r3_rejected_path), "r4Graph": str(r4_graph_path)},
        "summary": {
            "totalSelfLoops": len(old),
            "counts": dict(sorted(counts.items())),
            "unresolvedWithoutExactSpanReplacement": len(unresolved),
            "unresolvedWithAtLeastTwoDistinctSurfaceCandidates": sum(
                record["distinctSurfaceCandidateCount"] >= 2 for record in unresolved
            ),
            "unresolvedWithUnrecognizedSurfaceCandidate": sum(
                record["distinctUnrecognizedSurfaceCandidateCount"] >= 1 for record in unresolved
            ),
            "unresolvedWithRelationTriggerHit": sum(
                bool(record["relationTriggerHits"]) for record in unresolved
            ),
            "heuristicOnly": True,
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-rejected", type=Path, required=True)
    parser.add_argument("--r4-graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.r3_rejected, args.r4_graph, args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
