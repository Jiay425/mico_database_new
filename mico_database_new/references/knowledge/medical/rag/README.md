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

## Build

Run from project root:

```bash
python scripts/build_medical_rag_index.py
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

## Notes

- Reference-heavy sections (e.g. `References`) are skipped when chunking.
- This is lexical retrieval baseline (BM25 style). It can be extended with embedding vector retrieval later.
