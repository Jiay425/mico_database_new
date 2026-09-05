"""Content-only listwise reranking for a compact (<=20) candidate pool."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from llm_listwise_rerank_v1 import ListwiseReranker, _stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-branch", default="rrf_top15")
    args = parser.parse_args()
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    chunks = {str(row.get("chunkId")): row for row in (json.loads(line) for line in args.chunks.read_text(encoding="utf-8").splitlines() if line.strip())}
    previous = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {"queries": []}
    rows = {str(row.get("queryId")): row for row in previous.get("queries") or []}
    reranker = ListwiseReranker()
    try:
        for index, case in enumerate(pool.get("cases") or [], 1):
            query_id = str(case["queryId"])
            if rows.get(query_id, {}).get("status") == "reranked":
                continue
            try:
                original = [str(row["chunkId"]) for row in (case.get("branches") or {}).get(args.candidate_branch) or []]
                if not original or len(original) > 20:
                    raise RuntimeError("COMPACT_CANDIDATE_COUNT_INVALID")
                shuffled = list(original)
                random.Random("compact:" + query_id).shuffle(shuffled)
                ranked = _stage(reranker, str(case["question"]), shuffled, chunks)
                rows[query_id] = {"queryId": query_id, "category": case.get("category"), "status": "reranked", "candidateBranch": args.candidate_branch, "inputCandidateCount": len(original), "top5": [{"chunkId": chunk_id, "rank": rank} for rank, chunk_id in enumerate(ranked[:5], 1)], "ranking": [{"chunkId": chunk_id, "rank": rank} for rank, chunk_id in enumerate(ranked, 1)]}
            except Exception as exc:
                rows[query_id] = {"queryId": query_id, "category": case.get("category"), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            report = {"reportVersion": "compact-listwise-v1", "status": "in_progress", "pool": str(args.pool), "rerankerModel": reranker.model, "method": "one content-only listwise ranking over shuffled RRF Top15; emit Top5", "judgeLeakageBoundary": "The reranker receives only randomized candidate numbers, title, section and text; no retrieval source/rank/RRF score or judge labels.", "queries": [rows[key] for key in sorted(rows)]}
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"reranked={index}/{len(pool.get('cases') or [])} queryId={query_id} status={rows[query_id]['status']}", flush=True)
    finally:
        reranker.close()
    complete = all(rows.get(str(case["queryId"]), {}).get("status") == "reranked" for case in pool.get("cases") or [])
    report["status"] = "READY_FOR_METRICS" if complete else "PENDING_RETRY"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "queryCount": len(pool.get("cases") or []), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
