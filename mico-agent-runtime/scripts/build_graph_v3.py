from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mico_agent_runtime.knowledge.graph_v3 import GRAPH_VERSION, build_v3_records


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    workspace_root = runtime_root.parent
    knowledge_dir = workspace_root / "mico_database_new" / "references" / "knowledge" / "medical" / "rag"
    chunks_path = knowledge_dir / "medical_chunks.jsonl"
    output_path = knowledge_dir / "medical_knowledge_graph_v3.jsonl"
    manifest_path = knowledge_dir / "medical_knowledge_graph_v3_manifest.json"
    rows = list(_jsonl(chunks_path))
    if not rows:
        raise SystemExit("FULLTEXT_CORPUS_EMPTY")
    records, counts = build_v3_records(rows)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "graphVersion": GRAPH_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "corpusScope": "fulltext_only",
        "evidenceTier": "fulltext",
        "paperCount": counts.get("papers", 0),
        "chunkCount": counts.get("chunks", 0),
        "nodeCount": counts.get("nodes", 0),
        "edgeCount": counts.get("edges", 0),
        "semanticRelationCount": counts.get("semantic_relations", 0),
        "semanticRelationChunks": counts.get("chunks_with_semantic_relations", 0),
        "lowConfidenceRelationsRejected": counts.get("low_confidence_relations_rejected", 0),
        "assertionCounts": {
            key.removeprefix("assertion_"): value
            for key, value in counts.items()
            if key.startswith("assertion_")
        },
        "relationCounts": {
            key.removeprefix("relation_"): value
            for key, value in counts.items()
            if key.startswith("relation_")
        },
        "output": output_path.name,
        "rules": {
            "entityNormalization": "stable v3 canonical IDs with aliases",
            "relationExtraction": "same-sentence explicit relation cue",
            "minimumRelationConfidence": 0.65,
            "evidenceRequirement": "every edge has evidenceChunkId and evidenceText",
            "candidateTaxonPolicy": "candidate_taxon normalization status remains speculative",
            "medicalClaimPolicy": "no clinical or causal conclusion is generated from extraction alone",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"graphVersion": GRAPH_VERSION, **counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

