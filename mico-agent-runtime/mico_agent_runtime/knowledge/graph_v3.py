from __future__ import annotations

"""Versioned, source-bound graph extraction for the full-text corpus.

The v2 graph was intentionally permissive and therefore admitted ordinary
capitalized phrases as candidate taxa.  v3 keeps the graph useful for
GraphRAG by requiring a normalized entity pair and an explicit relation cue in
the same source sentence.  It is still an extraction graph, not a clinical
ontology: every semantic edge carries its source chunk, sentence span,
assertion state and confidence.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


GRAPH_VERSION = "fulltext-provenance-graphrag-v3"
MIN_RELATION_CONFIDENCE = 0.65

_BINOMIAL_RE = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z][a-z-]{3,})\b")
_SENTENCE_RE = re.compile(r"[^.!?。！？\n]+(?:[.!?。！？]|$)")
_CONTROLLED_ALIAS_RE = re.compile(r"\b[a-z0-9][a-z0-9 -]{1,80}\b", re.I)

# Common capitalized words which are not taxa.  This is a language guard, not
# a blacklist of scientific values; candidate taxa remain explicit candidates
# until a taxonomy aligner is added.
_NON_TAXON_HEADS = {
    "a", "an", "the", "this", "that", "these", "those", "current", "chronic",
    "under", "during", "after", "before", "from", "with", "without", "through",
    "intestinal", "oral", "human", "gut", "host", "major", "clinical", "dietary",
    "altered", "targeting", "role", "effects", "effect", "high", "low", "type",
    "multiple", "metabolic", "microbial", "bile", "leaky", "partners", "intestinal",
}
_NON_TAXON_TAILS = {
    "pathogenesis", "research", "interaction", "causes", "cause", "liver", "brain",
    "disease", "diseases", "microbiota", "microbiome", "flora", "study", "studies",
    "patients", "patient", "normal", "physiological", "persistent", "complications",
    "therapy", "therapies", "metabolism", "inflammation", "impairment", "transmission",
    "infection", "risk", "severity", "regulation", "disorders", "disorder",
    "previous", "file", "analysis", "phylum", "genus", "sepsis", "consensus",
    "anosim", "supporting", "this", "although", "additional", "our", "better",
    "against", "quantitative", "transcriptomic", "enrichment", "correlation",
    "meta-analysis",
}
_TAXON_LATIN_SUFFIXES = (
    "aceae", "oides", "ensis", "ense", "acidophilus", "muciniphila", "difficile",
    "faecalis", "nucleatum", "pneumoniae", "intestinalis", "prausnitzii", "longum",
    "um", "us", "is", "ae", "ii", "i", "ile",
)

ENTITY_ALIASES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "type_2_diabetes": ("Disease", "type 2 diabetes", ("t2d", "t2dm", "type 2 diabetes", "type 2 diabetes mellitus")),
    "healthy_control": ("Disease", "healthy control", ("healthy control", "healthy controls", "healthy subject", "healthy subjects")),
    "cirrhosis": ("Disease", "cirrhosis", ("cirrhosis",)),
    "colorectal_cancer": ("Disease", "colorectal cancer", ("colorectal cancer", "crc")),
    "coronary_artery_disease": ("Disease", "coronary artery disease", ("coronary artery disease", "cad")),
    "alzheimer_disease": ("Disease", "alzheimer disease", ("alzheimer disease", "alzheimer's disease", "ad")),
    "multiple_sclerosis": ("Disease", "multiple sclerosis", ("multiple sclerosis", "ms")),
    "fatty_liver": ("Disease", "fatty liver disease", ("fatty liver", "nafld", "nonalcoholic fatty liver")),
    "inflammatory_bowel_disease": ("Disease", "inflammatory bowel disease", ("inflammatory bowel disease", "ibd")),
    "obesity": ("Disease", "obesity", ("obesity",)),
    "gut_microbiome": ("Concept", "gut microbiome", ("gut microbiome", "gut microbiota", "intestinal microbiota", "intestinal flora")),
    "dysbiosis": ("Concept", "dysbiosis", ("dysbiosis", "microbial dysbiosis")),
    "short_chain_fatty_acids": ("Metabolite", "short-chain fatty acids", ("short-chain fatty acid", "short chain fatty acid", "scfa")),
    "bile_acid": ("Metabolite", "bile acid", ("bile acid", "bile acids")),
    "lipopolysaccharide": ("Metabolite", "lipopolysaccharide", ("lipopolysaccharide", "lps")),
    "butyrate": ("Metabolite", "butyrate", ("butyrate",)),
    "serotonin": ("Metabolite", "serotonin", ("serotonin",)),
    "gut_brain_axis": ("Pathway", "gut-brain axis", ("gut-brain axis", "microbiota-gut-brain axis")),
    "gut_liver_brain_axis": ("Pathway", "gut-liver-brain axis", ("gut-liver-brain axis",)),
    "bile_acid_metabolism": ("Pathway", "bile acid metabolism", ("bile acid metabolism",)),
    "inflammation": ("HostProcess", "inflammation", ("inflammation", "inflammatory")),
    "neuroinflammation": ("HostProcess", "neuroinflammation", ("neuroinflammation",)),
}

RELATION_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("CAUSES", "causal", r"\b(?:causally associated|causes?|leads? to|drives?|triggers?)\b"),
    ("MEDIATES", "causal", r"\b(?:mediates?|mediated|through|via)\b"),
    ("PROMOTES", "causal", r"\b(?:promotes?|promoted|contributes? to)\b"),
    ("INHIBITS", "causal", r"\b(?:inhibits?|inhibited|suppresses?)\b"),
    ("INCREASED_IN", "directional", r"\b(?:increased|higher|enriched|elevated)\b"),
    ("DECREASED_IN", "directional", r"\b(?:decreased|lower|depleted|reduced)\b"),
    ("ASSOCIATED_WITH", "association", r"\b(?:associated with|correlated with|related to|linked to|interact(?:s|ed)? with)\b"),
)
_COMPILED_RELATIONS = tuple((name, kind, re.compile(pattern, re.I)) for name, kind, pattern in RELATION_PATTERNS)
_NEGATION_RE = re.compile(r"\b(?:not|no|without|lack(?:s|ing|ed)?|absence|failed|did not|neither|never)\b", re.I)
_SPECULATIVE_RE = re.compile(r"\b(?:may|might|could|possibly|potential(?:ly)?|suggest(?:s|ed)?|appears?|likely|putative)\b", re.I)


@dataclass(frozen=True)
class EntityMention:
    node_id: str
    entity_type: str
    canonical_label: str
    surface: str
    start: int
    end: int
    normalization_status: str


@dataclass(frozen=True)
class RelationCandidate:
    source: EntityMention
    target: EntityMention
    relation: str
    relation_class: str
    assertion_status: str
    confidence: float
    sentence: str
    start: int
    end: int
    extraction_method: str


def _stable_id(prefix: str, value: str) -> str:
    normalized = value.strip().lower().encode("utf-8")
    return f"v3:{prefix}:" + hashlib.sha256(normalized).hexdigest()[:32]


def _norm(value: str) -> str:
    return re.sub(r"[_\-]+", " ", re.sub(r"\s+", " ", value.strip().lower()))


def _entity_for_controlled(canonical: str, start: int, end: int, surface: str) -> EntityMention:
    entity_type, label, _ = ENTITY_ALIASES[canonical]
    return EntityMention(
        node_id=_stable_id(entity_type.lower(), canonical),
        entity_type=entity_type,
        canonical_label=label,
        surface=surface,
        start=start,
        end=end,
        normalization_status="canonical",
    )


def _taxon_mentions(text: str) -> list[EntityMention]:
    mentions: list[EntityMention] = []
    seen: set[tuple[str, int]] = set()
    for match in _BINOMIAL_RE.finditer(text):
        genus, species = match.group(1), match.group(2)
        if genus.lower() in _NON_TAXON_HEADS or species.lower() in _NON_TAXON_TAILS:
            continue
        if species.lower() in {"acid", "based", "related", "derived", "associated", "induced"}:
            continue
        # A capitalized phrase is not enough to call something a taxon.  The
        # suffix check removes ordinary prose such as "Current research",
        # "Patients with", and "Prevotella plays" while retaining the common
        # Latinized endings used by the microbiome corpus.  Unrecognized
        # names remain outside the active graph until a taxonomy aligner is
        # introduced.
        if not species.lower().endswith(_TAXON_LATIN_SUFFIXES):
            continue
        label = f"{genus} {species}"
        key = (_norm(label), match.start())
        if key in seen:
            continue
        seen.add(key)
        mentions.append(EntityMention(
            node_id=_stable_id("taxon", _norm(label)),
            entity_type="Taxon",
            canonical_label=label,
            surface=label,
            start=match.start(),
            end=match.end(),
            normalization_status="candidate_taxon",
        ))
    return mentions


def extract_mentions(text: str) -> list[EntityMention]:
    mentions: list[EntityMention] = []
    lowered = text.lower()
    for canonical, (entity_type, _label, aliases) in ENTITY_ALIASES.items():
        for alias in aliases:
            pattern = re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", re.I)
            for match in pattern.finditer(lowered):
                mentions.append(_entity_for_controlled(canonical, match.start(), match.end(), text[match.start():match.end()]))
    mentions.extend(_taxon_mentions(text))
    unique: dict[tuple[str, int, int], EntityMention] = {}
    for mention in mentions:
        unique[(mention.node_id, mention.start, mention.end)] = mention
    return sorted(unique.values(), key=lambda item: (item.start, item.end, item.node_id))


def _assertion_status(sentence: str, relation_start: int) -> str:
    window = sentence[max(0, relation_start - 120):relation_start + 180]
    if _NEGATION_RE.search(window):
        return "negated"
    if _SPECULATIVE_RE.search(window):
        return "speculative"
    return "asserted"


def extract_relations(text: str) -> tuple[list[RelationCandidate], int]:
    relations: list[RelationCandidate] = []
    rejected_low_confidence = 0
    for sentence_match in _SENTENCE_RE.finditer(text):
        sentence = sentence_match.group(0).strip()
        if len(sentence) < 20:
            continue
        offset = sentence_match.start()
        mentions = extract_mentions(sentence)
        if len(mentions) < 2:
            continue
        relation_match: tuple[str, str, re.Match[str]] | None = None
        for name, kind, pattern in _COMPILED_RELATIONS:
            candidate = pattern.search(sentence)
            if candidate is not None:
                relation_match = (name, kind, candidate)
                break
        if relation_match is None:
            continue
        relation, relation_class, marker = relation_match
        before = [item for item in mentions if item.end <= marker.start()]
        after = [item for item in mentions if item.start >= marker.end()]
        if before and after:
            pairs = [(before[-1], after[0])]
        else:
            pairs = [(mentions[index], mentions[index + 1]) for index in range(min(len(mentions) - 1, 4))]
        for source, target in pairs:
            confidence = 0.86
            if source.normalization_status == "candidate_taxon" or target.normalization_status == "candidate_taxon":
                confidence -= 0.10
            if relation_class == "causal" and _SPECULATIVE_RE.search(sentence):
                confidence -= 0.04
            status = _assertion_status(sentence, marker.start())
            if status == "negated":
                confidence -= 0.03
            if confidence < MIN_RELATION_CONFIDENCE:
                rejected_low_confidence += 1
                continue
            relations.append(RelationCandidate(
                source=source,
                target=target,
                relation=relation,
                relation_class=relation_class,
                assertion_status=status,
                confidence=round(confidence, 4),
                sentence=re.sub(r"\s+", " ", sentence).strip()[:800],
                start=offset + marker.start(),
                end=offset + marker.end(),
                extraction_method="sentence_relation_pattern_v3",
            ))
    return relations, rejected_low_confidence


def build_v3_records(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    counters: Counter[str] = Counter()

    def add_node(node_id: str, entity_type: str, label: str, **extra: Any) -> None:
        if node_id in node_ids:
            return
        node_ids.add(node_id)
        records.append({
            "recordType": "node",
            "graphVersion": GRAPH_VERSION,
            "nodeId": node_id,
            "nodeType": entity_type.lower(),
            "entityType": entity_type,
            "label": label,
            "canonicalLabel": label,
            "evidenceTier": "fulltext",
            **extra,
        })

    def add_edge(source: str, target: str, relation: str, chunk_id: str, **extra: Any) -> None:
        raw = f"{GRAPH_VERSION}|{source}|{relation}|{target}|{chunk_id}|{extra.get('evidenceStart')}"
        edge_id = "v3-edge-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        if edge_id in edge_ids:
            return
        edge_ids.add(edge_id)
        records.append({
            "recordType": "edge",
            "graphVersion": GRAPH_VERSION,
            "edgeId": edge_id,
            "source": source,
            "relation": relation,
            "relationClass": extra.pop("relationClass", "structural"),
            "target": target,
            "evidenceTier": "fulltext",
            "evidenceChunkId": chunk_id,
            "confidence": float(extra.pop("confidence", 1.0)),
            "assertionStatus": extra.pop("assertionStatus", "asserted"),
            "extractionMethod": extra.pop("extractionMethod", "graph_schema_v3"),
            **extra,
        })

    for row in rows:
        pmcid = str(row["pmcid"])
        chunk_id = str(row["chunkId"])
        paper_id = _stable_id("paper", pmcid)
        # Chunk IDs are already opaque, source-stable identifiers used by the
        # relational full-text store.  Preserve that exact suffix so a graph
        # path can be joined back to its source chunk without a second lookup
        # table.  Other entity IDs remain hashed and opaque.
        chunk_node = f"v3:chunk:{chunk_id}"
        section = str(row.get("section") or "unknown")
        section_id = _stable_id("section", f"{pmcid}|{section}")
        topic = _norm(str(row.get("topic") or "unknown"))
        topic_id = _stable_id("topic", topic)
        add_node(paper_id, "Paper", str(row.get("title") or pmcid), pmcid=pmcid, pmid=row.get("pmid"), doi=row.get("doi"), year=row.get("year"), sourceUrl=row.get("sourceUrl"), normalizationStatus="source_identifier")
        add_node(chunk_node, "Chunk", chunk_id, pmcid=pmcid, section=section, sourceUrl=row.get("sourceUrl"), normalizationStatus="source_identifier")
        add_node(section_id, "Section", section, pmcid=pmcid, normalizationStatus="source_identifier")
        add_node(topic_id, "Topic", topic, normalizationStatus="controlled_topic")
        add_edge(chunk_node, paper_id, "PART_OF", chunk_id)
        add_edge(chunk_node, section_id, "IN_SECTION", chunk_id)
        add_edge(paper_id, topic_id, "HAS_TOPIC", chunk_id)

        text = f"{row.get('title', '')}. {row.get('text', '')}"
        mentions = extract_mentions(text)
        by_node: dict[str, EntityMention] = {}
        for mention in mentions:
            by_node.setdefault(mention.node_id, mention)
            add_node(
                mention.node_id,
                mention.entity_type,
                mention.canonical_label,
                aliases=[mention.surface],
                normalizationStatus=mention.normalization_status,
                extractionMethod="controlled_alias_v3" if mention.normalization_status == "canonical" else "binomial_candidate_v3",
            )
            add_edge(chunk_node, mention.node_id, "MENTIONS_ENTITY", chunk_id, confidence=0.98, relationClass="structural")
            counters["entity_mentions"] += 1

        relations, rejected = extract_relations(text)
        counters["low_confidence_relations_rejected"] += rejected
        for relation in relations:
            add_edge(
                relation.source.node_id,
                relation.target.node_id,
                relation.relation,
                chunk_id,
                relationClass=relation.relation_class,
                confidence=relation.confidence,
                assertionStatus=relation.assertion_status,
                extractionMethod=relation.extraction_method,
                evidenceText=relation.sentence,
                evidenceStart=relation.start,
                evidenceEnd=relation.end,
            )
            counters["semantic_relations"] += 1
            counters[f"relation_{relation.relation.lower()}"] += 1
            counters[f"assertion_{relation.assertion_status}"] += 1
        if relations:
            counters["chunks_with_semantic_relations"] += 1

    counters["nodes"] = sum(record["recordType"] == "node" for record in records)
    counters["edges"] = sum(record["recordType"] == "edge" for record in records)
    counters["chunks"] = sum(record.get("nodeType") == "chunk" for record in records if record["recordType"] == "node")
    counters["papers"] = sum(record.get("nodeType") == "paper" for record in records if record["recordType"] == "node")
    return records, dict(counters)
