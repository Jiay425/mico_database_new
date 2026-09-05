from __future__ import annotations

"""Materialize reviewable query drafts from frozen, graph-free packets.

No provider is called here.  The questions are conservative templates over
paper metadata/text and are explicitly marked ``heuristic_draft``.  They are
useful for exercising the pooling/quality workflow, but must be approved or
rewritten by a judge before becoming qrels-v2.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


ENTITY_PHRASES = (
    "type 2 diabetes", "healthy control", "cirrhosis", "colorectal cancer",
    "coronary artery disease", "alzheimer disease", "multiple sclerosis",
    "fatty liver", "inflammatory bowel disease", "obesity", "gut microbiome",
    "gut microbiota", "intestinal flora", "dysbiosis", "short-chain fatty acids",
    "bile acids", "lipopolysaccharide", "butyrate", "serotonin", "gut-brain axis",
    "gut-liver axis", "inflammation", "neuroinflammation", "tlr4",
)
LEAK_WORDS = {
    "increased", "decreased", "higher", "lower", "elevated", "reduced", "enriched",
    "depleted", "associated", "causes", "promotes", "inhibits", "mediates",
}
TOPIC_CANONICAL = {
    "t2d": "type 2 diabetes",
    "t2dm": "type 2 diabetes",
    "crc": "colorectal cancer",
    "ad": "alzheimer disease",
    "early_ad": "alzheimer disease",
    "ibd": "inflammatory bowel disease",
    "ms": "multiple sclerosis",
    "cad": "coronary artery disease",
    "fatty_liver": "fatty liver disease",
    "healthy_baseline": "healthy control",
    "healthy_control": "healthy control",
}


def _phrase_candidates(context: dict[str, Any]) -> list[str]:
    haystack = " ".join([
        str(context.get("topic") or ""),
        str(context.get("title") or ""),
        " ".join(str(chunk.get("text") or "") for chunk in context.get("chunks") or []),
    ]).lower()
    present = [phrase for phrase in ENTITY_PHRASES if re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", haystack)]
    raw_topic = str(context.get("topic") or "").strip().lower()
    topic = TOPIC_CANONICAL.get(raw_topic, raw_topic.replace("_", " "))
    ordered = [topic] + present
    return list(dict.fromkeys(item for item in ordered if item))


def _sections(context: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(chunk.get("section") or "unknown")
        for chunk in context.get("chunks") or []
        if chunk.get("section")
    ))


def _question(category: str, contexts: list[dict[str, Any]]) -> str:
    primary = contexts[0]
    phrases = _phrase_candidates(primary)
    topic = phrases[0] if phrases else "the reported biomedical phenomenon"
    entity = next((item for item in phrases[1:] if item != topic), "the related condition or biological process")
    if category == "direct_evidence":
        return f"What concrete findings about {topic} are reported in the supplied study?"
    if category == "entity_relation":
        return f"What relationship between {topic} and {entity} is described in the supplied study?"
    if category == "mechanism":
        return f"What biological mechanisms or pathways are proposed to connect {topic} with {entity} in the supplied study?"
    if category == "multi_hop":
        sections = _sections(primary)
        left, right = (sections + ["the results", "the discussion"])[:2]
        return f"How do the evidence in the {left} and the {right} of the supplied study fit together regarding {topic}?"
    if category == "cross_paper_synthesis":
        return f"Across the supplied studies, what findings about {topic} are consistent, and where do they differ?"
    return f"What null, negative, limited, or conflicting findings about {topic} are reported in the supplied study?"


def _quality(question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = set(re.findall(r"[a-z][a-z0-9-]+", question.lower()))
    title_tokens = set(re.findall(r"[a-z][a-z0-9-]+", str(contexts[0].get("title") or "").lower()))
    leakage = sorted(tokens & LEAK_WORDS)
    return {
        "answerLeakageRisk": bool(leakage),
        "leakageTerms": leakage,
        "titleCopyTokenRatio": round(len(tokens & title_tokens) / max(1, len(tokens)), 8),
        "tooGeneric": "the related condition or biological process" in question,
        "qualityStatus": "needs_review" if leakage or "the related condition or biological process" in question else "candidate",
    }


def run(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for packet in payload.get("packets") or []:
        contexts = packet.get("sourceContext") or []
        question = _question(str(packet.get("category")), contexts)
        quality = _quality(question, contexts)
        items.append({
            "queryId": packet.get("queryId"),
            "category": packet.get("category"),
            "status": "heuristic_draft",
            "question": question,
            "answerability": "candidate_supported_by_source_pending_review",
            "sourcePmcids": packet.get("sourcePmcids") or [],
            "sourcePaperCount": len(packet.get("sourcePmcids") or []),
            "rationale": "Template generated from chunk-v2 title/section/text only; rewrite if the supplied evidence cannot answer the question.",
            "quality": quality,
            "independenceGuard": packet.get("independenceGuard") or {},
        })
    counts: dict[str, int] = {}
    for item in items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    result = {
        "reportVersion": "p2g-independent-query-set-v1",
        "status": "draft_pending_review",
        "sources": {"frozenPackets": str(input_path), "graphAssetsRead": False, "oldQrelsRead": False},
        "qualityContract": {
            "heuristicDraftIsNotGold": True,
            "mustReviewAnswerability": True,
            "mustRejectAnswerLeakage": True,
            "mustCheckCorpusSufficiency": True,
        },
        "counts": {
            "total": len(items),
            "byCategory": counts,
            "needsReview": sum(item["quality"]["qualityStatus"] == "needs_review" for item in items),
            "candidate": sum(item["quality"]["qualityStatus"] == "candidate" for item in items),
        },
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
