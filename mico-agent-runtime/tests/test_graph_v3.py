from __future__ import annotations

from mico_agent_runtime.knowledge.graph_v3 import (
    GRAPH_VERSION,
    build_v3_records,
    extract_mentions,
    extract_relations,
)


def test_v3_taxon_normalization_rejects_common_capitalized_phrases() -> None:
    mentions = extract_mentions("The pathogenesis and Current research describe gut microbiome changes.")
    assert all(mention.normalization_status != "candidate_taxon" for mention in mentions)


def test_v3_extracts_normalized_taxon_and_relation_assertion() -> None:
    mentions = extract_mentions(
        "Akkermansia muciniphila is associated with type 2 diabetes, although this may be speculative."
    )
    assert any(mention.entity_type == "Taxon" for mention in mentions)
    relations, rejected = extract_relations(
        "Akkermansia muciniphila is associated with type 2 diabetes, although this may be speculative."
    )
    assert rejected == 0
    assert relations
    assert relations[0].relation == "ASSOCIATED_WITH"
    assert relations[0].assertion_status == "speculative"
    assert relations[0].confidence >= 0.65


def test_v3_negated_and_causal_relations_are_distinct() -> None:
    relations, _ = extract_relations(
        "Gut microbiome promotes butyrate production. However, obesity was not associated with gut microbiome changes."
    )
    assert {relation.relation for relation in relations} >= {"PROMOTES", "ASSOCIATED_WITH"}
    assert any(relation.assertion_status == "negated" for relation in relations)
    assert any(relation.relation_class == "causal" for relation in relations)


def test_v3_does_not_pair_full_name_and_acronym_as_self_loop() -> None:
    relations, rejected = extract_relations(
        "If TLR4 is exposed to increased lipopolysaccharide (LPS) expression, "
        "the response may lead to chronic inflammatory liver disease."
    )
    assert rejected == 0
    assert relations
    assert all(item.source.node_id != item.target.node_id for item in relations)


def test_v3_records_are_source_bound_and_versioned() -> None:
    rows = [
        {
            "pmcid": "PMC-A",
            "chunkId": "PMC-A-A001",
            "title": "Gut microbiome and diabetes",
            "text": "Akkermansia muciniphila is associated with type 2 diabetes.",
            "topic": "t2d",
            "section": "Results",
            "sourceUrl": "https://example.invalid/a",
        },
        {
            "pmcid": "PMC-B",
            "chunkId": "PMC-B-A001",
            "title": "Microbiome and obesity",
            "text": "Akkermansia muciniphila is associated with obesity.",
            "topic": "obesity",
            "section": "Results",
            "sourceUrl": "https://example.invalid/b",
        },
    ]
    records, counts = build_v3_records(rows)
    assert counts["papers"] == 2
    assert counts["semantic_relations"] >= 2
    assert all(record["graphVersion"] == GRAPH_VERSION for record in records)
    edges = [record for record in records if record["recordType"] == "edge"]
    assert all(edge["evidenceChunkId"] for edge in edges)
    assert all(edge["confidence"] >= 0.65 for edge in edges)
    taxon_ids = {
        record["nodeId"]
        for record in records
        if record["recordType"] == "node" and record["entityType"] == "Taxon"
    }
    assert len(taxon_ids) == 1
