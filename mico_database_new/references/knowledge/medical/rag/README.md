# Medical RAG Index

This folder stores searchable artifacts built from medical full-text papers.

## Artifacts

- `medical_chunks.jsonl`
  - One chunk per JSON line.
  - Includes source metadata (`pmcid`, `pmid`, `topic`, `title`, `section`, `sourceUrl`) and chunk text.
- `medical_bm25_stats.json`
  - BM25 corpus statistics (`n_docs`, `avgdl`, `df`).
- `medical_rag_manifest.json`
  - Build summary, paths, and parameters.
- `medical_vector_index.jsonl` and `medical_vector_meta.json`
  - Full-text-only deterministic TF-IDF cosine index (`fulltext-tfidf-cosine-v1`).
- `medical_knowledge_graph.jsonl`
  - Versioned GraphRAG graph for papers, chunks, topics, sections, controlled
    terms and candidate taxa. Every relation retains an evidence chunk.
- `medical_knowledge_manifest.json`
  - Full-text corpus scope, graph quality coverage and vector/graph build summary.

- `medical_gemini_paper_embedding_index.jsonl` and
  `medical_gemini_paper_embedding_meta.json`
  - Optional full-text-only `gemini-embedding-2` paper-level dense vectors
    (one vector per PMCID). Credentials are never written to the index.

## Build

Run from project root:

```bash
python scripts/build_medical_rag_index.py
```

After injecting the Gemini environment variables, build dense vectors with:

```bash
python scripts/build_medical_gemini_embedding_index.py --resume
```

Optional parameters:

- `--fulltext-dir` full-text markdown directory
- `--fulltext-index` full-text import index JSONL
- `--out-dir` output directory
- `--max-chars` max chars per chunk (default: 1800)
- `--overlap` chunk overlap chars (default: 250)
- `--min-chars` min chars per chunk (default: 240)

## Query Smoke Test

```bash
python scripts/query_medical_rag.py "t2d gut microbiota healthy controls dysbiosis" --top-k 5
```

## GraphRAG and hybrid retrieval

The current manifest is `fulltext-knowledge-index-v2` and the graph is
`fulltext-provenance-graphrag-v2`. LangGraph dynamically chooses vector,
graph or hybrid retrieval. The graph branch performs bounded bidirectional
traversal up to three hops and returns structured paths such as
`controlled_term -> full-text chunk -> candidate_taxon`; every hop carries
`evidenceChunkId`. The hybrid branch reranks dense relevance, graph support
and path coverage. Results preserve `evidenceTier=fulltext`, PMCID, chunk ID
and section. Candidate taxon edges are source-local retrieval hints, not
reviewed biological causality.

## Notes

- Reference-heavy sections (e.g. `References`) are skipped when chunking.
- The offline vector baseline is TF-IDF. The production embedding path is
  `gemini-embedding-2` with task-prefixed query/document inputs. The dense
  index is paper-level to keep the first 65-paper build bounded; GraphRAG and
  the existing full-text chunk index select the exact evidence chunk after
  paper ranking. Multi-hop paths are bounded, source-bound and exposed as
  structured evidence paths rather than hidden unsupported reasoning.
