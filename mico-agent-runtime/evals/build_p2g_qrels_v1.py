"""Build a fixed, source-bound graded GraphRAG evaluation set.

The generated labels are *expert-assisted provenance labels*, not claims of
human adjudication.  Grade 3 is reserved for the chunk explicitly bound to a
semantic edge; grade 2 for another direct edge for the same entity pair; grade
1 for source-bound contextual evidence incident to one requested entity.
Every label records why it exists so a domain reviewer can audit or replace it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "p2g-graded-qrels-v1"
CATEGORY_TARGETS = {
    "single_fact": 50,
    "relation": 60,
    "multi_hop": 50,
    "document_synthesis": 40,
}
SEMANTIC_CLASSES = {"association", "causal", "directional"}


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _chunk_id(value: Any) -> str | None:
    if not value:
        return None
    result = re.sub(r"^v[0-9]+:", "", str(value)).removeprefix("chunk:")
    return result or None


def _load_graph(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("recordType") == "node":
            nodes[str(row.get("nodeId"))] = row
        elif row.get("recordType") == "edge":
            edges.append(row)
    return nodes, edges


def _valid_edge(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> bool:
    if edge.get("relationClass") not in SEMANTIC_CLASSES:
        return False
    if str(edge.get("assertionStatus") or "asserted") != "asserted":
        return False
    if float(edge.get("confidence") or 0.0) < 0.72:
        return False
    source, target = str(edge.get("source")), str(edge.get("target"))
    if source == target or source not in nodes or target not in nodes or not _chunk_id(edge.get("evidenceChunkId")):
        return False
    return bool(str(nodes[source].get("label") or "").strip() and str(nodes[target].get("label") or "").strip())


def _case_id(category: str, index: int) -> str:
    return f"{category}-{index:03d}"


def _label(chunk_id: str, grade: int, rationale: str, edge_id: str | None = None) -> dict[str, Any]:
    return {
        "chunkId": chunk_id,
        "provenanceGrade": grade,
        "grade": grade,
        "rationale": rationale,
        "edgeId": edge_id,
        "labelMethod": "source_bound_expert_assisted_v1",
        # Kept separate from the provisional provenance grade.  A domain
        # reviewer must fill these fields before the file can tune or publish
        # retrieval weights.
        "humanGrade": None,
        "reviewerId": None,
        "adjudicationNote": None,
    }


def _edge_case(
    category: str,
    index: int,
    edge: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    pair_edges: dict[tuple[str, str], list[dict[str, Any]]],
    incident: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source, target = str(edge["source"]), str(edge["target"])
    left, right = str(nodes[source]["label"]), str(nodes[target]["label"])
    if category == "relation":
        query = f"What relationship is supported between {left} and {right}?"
    else:
        query = f"Find direct full-text evidence connecting {left} and {right}."
    labels: dict[str, dict[str, Any]] = {}
    primary = _chunk_id(edge.get("evidenceChunkId"))
    assert primary
    labels[primary] = _label(primary, 3, "explicit source chunk for the requested semantic edge", str(edge.get("edgeId")))
    key = tuple(sorted((source, target)))
    for related in pair_edges[key]:
        chunk = _chunk_id(related.get("evidenceChunkId"))
        if chunk and chunk not in labels:
            labels[chunk] = _label(chunk, 2, "another source-bound edge for the same entity pair", str(related.get("edgeId")))
    for node_id in (source, target):
        for related in incident[node_id][:3]:
            chunk = _chunk_id(related.get("evidenceChunkId"))
            if chunk and chunk not in labels:
                labels[chunk] = _label(chunk, 1, "source-bound contextual evidence incident to a requested entity", str(related.get("edgeId")))
    return {
        "caseId": _case_id(category, index),
        "category": category,
        "query": query,
        "entities": [left, right],
        "qrels": sorted(labels.values(), key=lambda item: (-item["grade"], item["chunkId"])),
    }


def _multi_hop_cases(
    edges: list[dict[str, Any]], nodes: dict[str, dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        incident[str(edge["source"])].append(edge)
        incident[str(edge["target"])].append(edge)
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for center in sorted(incident):
        options = sorted(incident[center], key=lambda edge: str(edge.get("edgeId")))
        for left_edge_index, left_edge in enumerate(options):
            left = str(left_edge["target"] if str(left_edge["source"]) == center else left_edge["source"])
            for right_edge in options[left_edge_index + 1 :]:
                right = str(right_edge["target"] if str(right_edge["source"]) == center else right_edge["source"])
                if left == right or left not in nodes or right not in nodes:
                    continue
                labels = [str(nodes[node].get("label") or "") for node in (left, center, right)]
                if any(not label for label in labels):
                    continue
                signature = tuple(sorted((_norm(labels[0]), _norm(labels[2]))) + [_norm(labels[1])])
                if signature in seen:
                    continue
                seen.add(signature)
                qrels: dict[str, dict[str, Any]] = {}
                for edge in (left_edge, right_edge):
                    chunk = _chunk_id(edge.get("evidenceChunkId"))
                    if chunk:
                        qrels[chunk] = _label(chunk, 3, "one of the two required source-bound mechanism hops", str(edge.get("edgeId")))
                if len(qrels) != 2:
                    continue
                cases.append({
                    "caseId": _case_id("multi_hop", len(cases) + 1),
                    "category": "multi_hop",
                    "query": f"Explain the supported mechanism linking {labels[0]}, {labels[1]}, and {labels[2]}.",
                    "entities": labels,
                    "qrels": sorted(qrels.values(), key=lambda item: item["chunkId"]),
                })
                if len(cases) == count:
                    return cases
    raise ValueError("INSUFFICIENT_MULTI_HOP_CASES")


def _synthesis_cases(
    edges: list[dict[str, Any]], nodes: dict[str, dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    by_node: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for edge in edges:
        chunk = _chunk_id(edge.get("evidenceChunkId"))
        if not chunk:
            continue
        document = chunk.split("-", 1)[0]
        by_node[str(edge["source"])][document].add(chunk)
        by_node[str(edge["target"])][document].add(chunk)
    cases: list[dict[str, Any]] = []
    for node_id in sorted(by_node):
        label = str(nodes[node_id].get("label") or "") if node_id in nodes else ""
        documents = by_node[node_id]
        # Cross-document synthesis requires at least two independent papers;
        # use a third when available rather than silently padding one paper.
        if not label or len(documents) < 2:
            continue
        qrels = []
        for document in sorted(documents)[:3]:
            chunk = sorted(documents[document])[0]
            qrels.append(_label(chunk, 3, "source-bound evidence from a distinct document for synthesis"))
        cases.append({
            "caseId": _case_id("document_synthesis", len(cases) + 1),
            "category": "document_synthesis",
            "query": f"Synthesize source-bound evidence across studies about {label}.",
            "entities": [label],
            "qrels": qrels,
        })
        if len(cases) == count:
            return cases
    raise ValueError("INSUFFICIENT_SYNTHESIS_CASES")


def build_qrels(graph_path: Path) -> dict[str, Any]:
    nodes, raw_edges = _load_graph(graph_path)
    edges = [edge for edge in raw_edges if _valid_edge(edge, nodes)]
    pair_edges: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        source, target = str(edge["source"]), str(edge["target"])
        pair_edges[tuple(sorted((source, target)))].append(edge)
        incident[source].append(edge)
        incident[target].append(edge)
    ordered = sorted(edges, key=lambda edge: (str(edge.get("edgeId")), str(edge.get("evidenceChunkId"))))
    single = [_edge_case("single_fact", index, edge, nodes, pair_edges, incident) for index, edge in enumerate(ordered[:50], 1)]
    relation = [_edge_case("relation", index, edge, nodes, pair_edges, incident) for index, edge in enumerate(ordered[50:110], 1)]
    if len(single) != CATEGORY_TARGETS["single_fact"] or len(relation) != CATEGORY_TARGETS["relation"]:
        raise ValueError("INSUFFICIENT_DIRECT_EDGE_CASES")
    cases = single + relation + _multi_hop_cases(edges, nodes, 50) + _synthesis_cases(edges, nodes, 40)
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schemaVersion": SCHEMA_VERSION,
        "labelingProtocol": "source_bound_expert_assisted_v1",
        "humanAdjudicationStatus": "pending_domain_review",
        "sourceGraph": graph_path.name,
        "caseCountsByCategory": CATEGORY_TARGETS,
        "caseCount": len(cases),
        "qrelsFingerprint": "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-bound graded GraphRAG qrels")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_qrels(args.graph)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("schemaVersion", "caseCount", "caseCountsByCategory", "qrelsFingerprint")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
