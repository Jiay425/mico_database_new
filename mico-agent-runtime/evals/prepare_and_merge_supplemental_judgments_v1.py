"""Prepare only newly introduced candidates for judging, then merge labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(pool: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    labels = {
        str(row.get("queryId")): {str(label.get("chunkId")) for label in row.get("labels") or []}
        for row in base.get("queries") or []
    }
    cases = []
    for case in pool.get("cases") or []:
        query_id = str(case.get("queryId"))
        missing = [item for item in case.get("candidatePool") or [] if str(item.get("chunkId")) not in labels.get(query_id, set())]
        if missing:
            cases.append({
                "queryId": query_id, "category": case.get("category"), "question": case.get("question"),
                "candidatePool": missing, "candidatePoolSize": len(missing), "branches": {},
            })
    return {"reportVersion": "p2g-supplemental-judge-pool-v1", "status": "READY_FOR_SINGLE_JUDGING", "cases": cases, "counts": {"queryCount": len(cases), "candidateCount": sum(case["candidatePoolSize"] for case in cases)}}


def merge(pool: dict[str, Any], base: dict[str, Any], supplemental: dict[str, Any]) -> dict[str, Any]:
    base_rows = {str(row.get("queryId")): row for row in base.get("queries") or []}
    extra_rows = {str(row.get("queryId")): row for row in supplemental.get("queries") or []}
    merged = []
    for case in pool.get("cases") or []:
        query_id = str(case.get("queryId"))
        base_row, extra_row = base_rows.get(query_id, {}), extra_rows.get(query_id, {})
        if base_row.get("status") != "judged" or (extra_row and extra_row.get("status") != "judged"):
            raise RuntimeError(f"JUDGMENT_NOT_READY {query_id}")
        expected = {str(item.get("chunkId")) for item in case.get("candidatePool") or []}
        # The base judge may contain candidates from a broader earlier pool.
        # Only labels in the current strategy's candidate universe belong in
        # this comparison; supplemental labels then fill its newly introduced
        # graph evidence candidates.
        labels = {
            str(item.get("chunkId")): item
            for item in base_row.get("labels") or []
            if str(item.get("chunkId")) in expected
        }
        labels.update({str(item.get("chunkId")): item for item in extra_row.get("labels") or []})
        if set(labels) != expected:
            raise RuntimeError(f"LABEL_COVERAGE_MISMATCH {query_id}")
        merged.append({"queryId": query_id, "category": case.get("category"), "question": case.get("question"), "status": "judged", "candidateCount": len(expected), "labels": [labels[key] for key in sorted(labels)]})
    return {"reportVersion": "p2g-merged-single-judge-v1", "status": "READY_FOR_METRICS", "queries": merged}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "merge"), required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path)
    args = parser.parse_args()
    pool, base = _load(args.pool), _load(args.base)
    result = prepare(pool, base) if args.mode == "prepare" else merge(pool, base, _load(args.supplemental))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result.get("counts"), "queryCount": len(result.get("queries") or [])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
