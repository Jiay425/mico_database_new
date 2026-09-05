"""Fairly compare two graph-supplement strategies on one judged candidate union."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from score_final_main30_dense_hybrid_v1 import _aggregate, _score


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _labels(*reports: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for report in reports:
        for query in report.get("queries") or []:
            if query.get("status") != "judged":
                continue
            current = output.setdefault(str(query.get("queryId")), {})
            current.update({str(row.get("chunkId")): row for row in query.get("labels") or []})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-pool", type=Path, required=True)
    parser.add_argument("--new-pool", type=Path, required=True)
    parser.add_argument("--base-judge", type=Path, required=True)
    parser.add_argument("--old-supplemental", type=Path, required=True)
    parser.add_argument("--new-supplemental", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_pool, new_pool = _load(args.old_pool), _load(args.new_pool)
    labels = _labels(_load(args.base_judge), _load(args.old_supplemental), _load(args.new_supplemental))
    old_by_id = {str(row["queryId"]): row for row in old_pool["cases"]}
    new_by_id = {str(row["queryId"]): row for row in new_pool["cases"]}
    scores = {"dense": [], "old": [], "pathAware": []}
    per_query: list[dict[str, Any]] = []
    for query_id in sorted(new_by_id):
        old_case, new_case = old_by_id[query_id], new_by_id[query_id]
        candidates = {
            str(row["chunkId"])
            for case in (old_case, new_case)
            for row in case.get("candidatePool") or []
        }
        absent = candidates - set(labels.get(query_id, {}))
        if absent:
            raise RuntimeError(f"MISSING_LABELS {query_id}: {len(absent)}")
        relevant = {chunk_id for chunk_id in candidates if labels[query_id][chunk_id].get("relevant")}
        values = {
            "dense": _score(new_case["branches"]["dense"], relevant),
            "old": _score(old_case["branches"]["dense_preserving_graph"], relevant),
            "pathAware": _score(new_case["branches"]["dense_preserving_graph"], relevant),
        }
        for name, value in values.items():
            scores[name].append(value)
        per_query.append({"queryId": query_id, "category": new_case.get("category"), "candidateCount": len(candidates), "relevantEvidenceCount": len(relevant), **values})
    result = {
        "reportVersion": "p2g-graph-supplement-strategy-comparison-v1",
        "status": "FIXED_DEV30_SAME_CANDIDATE_UNION",
        "warning": "Development-only comparison; this set selected the strategy and is not an independent final result.",
        "queryCount": len(per_query),
        "systems": {name: _aggregate(values) for name, values in scores.items()},
        "pathAwareMinusOld": {
            key: round(_aggregate(scores["pathAware"])[key] - _aggregate(scores["old"])[key], 6)
            for key in _aggregate(scores["old"])
        },
        "perQuery": per_query,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"systems": result["systems"], "pathAwareMinusOld": result["pathAwareMinusOld"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
