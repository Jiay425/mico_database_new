# P2-G Chunk-v2 implementation report

Date: 2026-08-27

## 1. The correction that matters

The corpus was not unchunked.  The 65 papers already had 3,617 chunk-v1
records.  The defect was at the dense-index granularity: the historical
Gemini asset contains 65 paper vectors, one per PMCID, and uses a truncated
paper input.  A query therefore ranked papers first and only then mapped back
to a chunk.  The new work keeps that asset as a frozen baseline and creates
chunk-v2 assets beside it.

No Gemini request was made while generating or validating chunk-v2.  The new
chunk embedder is dry-run by default and requires an explicit `--execute`.

## 2. Source and version boundaries

```text
65 full-text markdown papers
  ├─ chunk-v1: medical_chunks.jsonl (3,617 records; unchanged)
  │    ├─ paper-dense-v1 (65 Gemini vectors; unchanged)
  │    └─ graph-v3 / old v4 (unchanged)
  └─ chunk-v2: structure-aware variants (new JSONL)
       ├─ v2-small: 2,305 records
       ├─ v2-medium: 1,308 records
       └─ v2-large: 953 records
            └─ graph-v4-v2m-r3: new staging graph (not published)
```

The v2 JSONL files and manifests are:

```text
references/knowledge/medical/rag/medical_chunks_v2_small.jsonl
references/knowledge/medical/rag/medical_chunks_v2_medium.jsonl
references/knowledge/medical/rag/medical_chunks_v2_large.jsonl
references/knowledge/medical/rag/medical_rag_manifest_v2_{small,medium,large}.json
```

## 3. How chunk-v2 is cut

The parser first separates the imported markdown into `Abstract`, `###`
sections and `####` subsections.  A subsection is never merged with another
subsection, so a retrieval record cannot silently cross a scientific heading.
Reference, funding, conflict-of-interest, author-contribution, data-
availability, ethics/consent and publisher-note boilerplate is excluded from
v2 retrieval assets; the original full text remains untouched.

Inside each structured block:

1. blank-line paragraphs are the base units;
2. whole paragraphs are merged until a soft target is reached;
3. a paragraph larger than the variant max is split on sentence boundaries;
4. adjacent sentence windows overlap by one sentence;
5. a pathological single sentence is hard-windowed only as a last resort;
6. a short final tail is merged into the previous chunk when it still fits;
7. a figure caption or table is retained as a standalone record when the
   imported source exposes it as a distinct paragraph/table.

The target is not a hard cut point:

| variant | target | normal minimum | hard maximum |
| --- | ---: | ---: | ---: |
| small | 256 tokens | 100 | 340 |
| medium | 512 tokens | 150 | 650 |
| large | 768 tokens | 220 | 960 |

The token counter is a deterministic sizing counter over English words,
alphanumeric scientific terms and contiguous CJK text.  It is not presented
as the provider tokenizer; the max bound is a conservative build-time guard.

## 4. Row shape

Each v2 record has this shape (values shortened here):

```json
{
  "chunkVersion": "chunk-v2",
  "variant": "medium",
  "chunkId": "PMC10034054-V2M-A0001",
  "pmcid": "PMC10034054",
  "pmid": "36968098",
  "topic": "cirrhosis",
  "title": "...",
  "year": "2023",
  "journal": "...",
  "doi": "...",
  "section": "Abstract",
  "subsection": "",
  "sourceUrl": "https://...",
  "chunkType": "abstract",
  "paragraphIndex": 0,
  "paragraphEndIndex": 1,
  "sentenceStart": 0,
  "sentenceEnd": 7,
  "charStart": 0,
  "charEnd": 1265,
  "offsetBase": "normalized_subsection",
  "charCount": 1265,
  "tokenCount": 184,
  "text": "source evidence text",
  "embeddingText": "Title: ...\\nTopic: ...\\nSection: ...\\nSubsection: ...\\nsource evidence text"
}
```

`text` is the evidence returned to the answer layer.  `embeddingText` is the
retrieval-only representation: the title/topic/section/subsection header is
prepended so that an otherwise ambiguous sentence (for example, “its
abundance increased”) keeps its paper context.  `charStart`/`charEnd` are
relative to normalized subsection text, not raw PDF/XML byte offsets.  For a
sentence window they now point to the selected sentence span, not the whole
oversized paragraph.

## 5. Current build statistics

All variants cover 65 papers and pass the independent validator.  There are
no duplicate IDs, invalid offsets, missing context headers or max-token
violations.

| variant | chunks | per-paper min / mean / median / max | token min / mean / median / max | types |
| --- | ---: | --- | --- | --- |
| small | 2,305 | 11 / 35.46 / 35 / 92 | 13 / 247.90 / 265 / 340 | abstract 67; paragraph 1,398; sentence_window 839; table 1 |
| medium | 1,308 | 4 / 20.12 / 20 / 48 | 13 / 421.96 / 426.5 / 650 | abstract 65; paragraph 900; sentence_window 342; table 1 |
| large | 953 | 4 / 14.66 / 14 / 36 | 13 / 550.84 / 499 / 960 | abstract 65; paragraph 669; sentence_window 218; table 1 |

The apparent decrease from 3,617 v1 rows is expected: v1 uses overlapping
character windows, while v2 removes boilerplate and preserves larger complete
paragraph units.  The validator reports short rows separately rather than
silently changing them: small 97, medium 40 and large 59 are below the normal
minimum, while all three variants have zero rows above their hard maximum.

The source markdown is not uniformly table/caption structured.  Therefore only
one table was detected as a standalone table record in this pass; the build
does not claim that every visually implied figure caption was recovered.

## 6. Graph-v4 staging result

The previous `medical_knowledge_graph_v4.jsonl` points to chunk-v1 and was not
overwritten.  A separate graph was built from v2-medium:

```text
graphVersion: fulltext-provenance-graphrag-v4-v2m-r3
inputAsset: medical_chunks_v2_medium.jsonl
paperCount: 65
chunkCount: 1,308
nodeCount: 2,584
edgeCount: 10,865
semanticRelationCount: 1,219
reviewRequiredRelationCount: 97
rejectedRecordCount: 379 (self-loop quality failures)
status: review_pending
```

This is an offline, source-bound graph build.  It has not been sent to Neo4j,
approved or made active.  The 379 self-loop rejects must be inspected before a
publication decision; they are not counted as successful evidence.

## 7. Dense embedding status and quota protection

The chunk-level builder is
`scripts/build_medical_chunk_embedding_index.py`.  Its output is versioned as
`medical_gemini_chunk_embedding_index_v2_{variant}.jsonl` plus a metadata file.
It embeds `embeddingText`, stores an input fingerprint per chunk and supports
resumption only when that fingerprint still matches.

Without `--execute`, the command performs no provider initialization and no
network call.  A v2-medium dry-run currently reports 1,308 possible calls and
0 calls made.  The script intentionally limits workers to four and stops
immediately on the known Gemini quota error.  This is the point at which the
next budgeted action would happen; it has not happened in this pass.

## 8. What is deliberately not claimed yet

- chunk-v2 has not yet been embedded, so chunk-level Dense Recall/nDCG cannot
  honestly be reported;
- Graph-v4-v2m-r3 has not been published, so it is not an online Graph result;
- the existing 200-case qrels remain provisional/source-bound and are not a
  human gold benchmark;
- no RRF/reranker improvement claim is made from this asset-generation pass.

The next valid experiment is: choose a v2 variant, approve the embedding
budget, build its chunk vectors, ingest it into a version-isolated store,
construct independent pooled qrels, and only then compare Dense/Sparse/Graph,
RRF, cross-encoder and constrained listwise ranking on Recall@50, nDCG@10,
MRR@10, Evidence Precision@5, path correctness, source binding and P50/P95.

## 9. Medium mainline execution checkpoint

Following the experiment order, v2-medium is now the mainline candidate. The
database schema has explicit `chunk_version`, `chunk_variant`,
`embedding_model`, `embedding_version` and `graph_version` columns. The 1,308
medium rows were inserted into the independent `mico_knowledge` database
without replacing chunk-v1:

```text
chunk-v1 / legacy: 3,617 rows, 0 embeddings
chunk-v2 / medium: 1,308 rows, 0 embeddings before the smoke batch
```

The first embedding batch was deliberately limited to 100 rows, single
worker, and completed successfully:

```text
vectorsRead: 100
rowsUpdated: 100
dimension: 3072
databaseEmbeddingNonNull: 100
databaseEmbeddingModelMatches: 100
databaseEmbeddingVersionMatches: 100
```

One real vector search was then executed against the configured
`chunk-v2/medium` slice. It returned five source-bound v2 chunks, with the top
results carrying the expected `PMC...-V2M-...` IDs. The remaining 1,208 medium
chunks have not been embedded; no small/large vectors were generated.

The database importer is `scripts/import_chunk_embeddings.py`. Its UPDATE
predicate includes both chunk version and variant, so a partial batch cannot
update v1 or another v2 variant. This checkpoint is a plumbing smoke test, not
a retrieval quality claim; the paper-level-versus-chunk-level comparison still
requires completing the medium index and running the same evaluation set.

The subsequent resume run processed 899 additional rows before the provider
returned `GEMINI_EMBEDDING_QUOTA_EXHAUSTED`.  The quota error is treated as
non-transient: no retry was issued.  The resulting safe checkpoint is:

```text
local chunk embedding records: 999 / 1,308
database v2-medium rows:      1,308
database non-null vectors:      999
remaining unembedded:           309
manifest embeddingStatus:    partial
```

The existing cached query vector was reused for a final no-provider smoke
query.  It returned five v2-medium source chunks, confirming that the
chunk-level pgvector path remains usable with the partial index.  No quality
metric is reported from this smoke query, and the 312 missing vectors must be
completed only after the provider quota resets or an explicitly approved
alternative embedding budget is available.
