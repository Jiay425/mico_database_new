"""Freeze blind LLM-assisted qrels and score the frozen Top-20 baselines.

The candidate pool contains only Top-20 rankings.  Therefore this evaluator
reports Recall@20, nDCG@10, MRR@10 and Evidence Precision@5.  It deliberately
does not manufacture Recall@50: a valid value needs judged ranks 21--50.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


BRANCHES = ("dense", "sparse", "graph", "rrf_standard")


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _dcg(grades: list[int]) -> float:
    return sum(((2 ** grade) - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _score_case(case: dict[str, Any], grades: dict[str, int], branch: str) -> dict[str, float | None]:
    ranked = list((case.get("branches") or {}).get(branch) or [])
    positives = [grade for grade in grades.values() if grade > 0]
    retrieved_grades = [int(grades.get(str(item.get("chunkId")), 0)) for item in ranked]
    recall20 = (sum(grade > 0 for grade in retrieved_grades[:20]) / len(positives)) if positives else None
    observed = retrieved_grades[:10]
    ideal = sorted(grades.values(), reverse=True)[:10]
    ideal_dcg = _dcg(ideal)
    ndcg10 = _dcg(observed) / ideal_dcg if ideal_dcg else None
    first_core = next((index + 1 for index, grade in enumerate(observed) if grade >= 2), None)
    mrr10 = 1.0 / first_core if first_core else 0.0
    evidence_p5 = sum(grade >= 2 for grade in retrieved_grades[:5]) / 5.0
    return {
        "recallAt20": recall20,
        "nDCGAt10": ndcg10,
        "mrrAt10": mrr10,
        "evidencePrecisionAt5": evidence_p5,
    }


def _aggregate(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    return {
        key: _mean([float(row[key]) for row in rows if row.get(key) is not None])
        for key in ("recallAt20", "nDCGAt10", "mrrAt10", "evidencePrecisionAt5")
    }


def _support_type(grade: int) -> str:
    return {
        3: "direct_core_evidence",
        2: "partial_direct_support",
        1: "background_only",
        0: "none",
    }[grade]


def run(pool_path: Path, judged_path: Path, qrels_path: Path, report_path: Path) -> dict[str, Any]:
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    judged = json.loads(judged_path.read_text(encoding="utf-8"))
    cases = {str(case.get("queryId")): case for case in pool.get("cases") or []}
    judgments = {str(row.get("queryId")): row for row in judged.get("queries") or []}
    missing = sorted(set(cases) - set(judgments))
    invalid = [
        query_id for query_id in cases
        if judgments[query_id].get("status") != "adjudicated" or judgments[query_id].get("unresolvedChunkIds")
    ]
    if missing or invalid:
        raise RuntimeError(
            f"QRELS_NOT_READY missingQueries={len(missing)} unresolvedOrFailedQueries={len(invalid)}"
        )

    qrel_cases: list[dict[str, Any]] = []
    by_branch: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    by_category: dict[str, dict[str, list[dict[str, float | None]]]] = defaultdict(lambda: defaultdict(list))
    for query_id, case in cases.items():
        row = judgments[query_id]
        labels = list(row.get("finalLabels") or [])
        if len(labels) != len(case.get("candidatePool") or []):
            raise RuntimeError(f"QRELS_INCOMPLETE_LABEL_SET queryId={query_id}")
        grades = {str(label.get("chunkId")): int(label.get("finalGrade")) for label in labels}
        category = str(case.get("category") or "unknown")
        qrel_cases.append({
            "queryId": query_id,
            "category": category,
            "question": case.get("question"),
            "judgments": [{
                "chunkId": label.get("chunkId"),
                "grade": int(label.get("finalGrade")),
                "evidenceSpan": label.get("evidenceSpan"),
                "supportType": _support_type(int(label.get("finalGrade"))),
                "confidence": label.get("confidence"),
            } for label in labels],
        })
        for branch in BRANCHES:
            score = _score_case(case, grades, branch)
            by_branch[branch].append(score)
            by_category[category][branch].append(score)

    qrels = {
        "qrelsVersion": "qrels-v1-llm-assisted-blind",
        "status": "FROZEN_LLM_ASSISTED_PENDING_DOMAIN_AUDIT",
        "annotationProtocol": "Judge A full coverage; independent Judge B for grade>=2, low-confidence, and deterministic 10% 0/1 audit sample; conflicts adjudicated; grade>=2 requires exact evidence span.",
        "pool": str(pool_path),
        "judgeReport": str(judged_path),
        "queryCount": len(qrel_cases),
        "judgmentCount": sum(len(item["judgments"]) for item in qrel_cases),
        "cases": qrel_cases,
    }
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    qrels_path.write_text(json.dumps(qrels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "reportVersion": "p2g-final-baseline-v1",
        "status": "LLM_ASSISTED_BASELINE_PENDING_DOMAIN_AUDIT",
        "qrels": str(qrels_path),
        "retrievalDepth": 20,
        "metricBoundary": {
            "recallAt50": None,
            "recallAt50Status": "NOT_COMPUTED",
            "reason": "Only Top-20 results were pooled and judged; ranks 21-50 are unjudged.",
        },
        "overall": {branch: _aggregate(by_branch[branch]) for branch in BRANCHES},
        "byCategory": {
            category: {branch: _aggregate(rows[branch]) for branch in BRANCHES}
            for category, rows in sorted(by_category.items())
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.pool, args.judged, args.qrels, args.report)
    print(json.dumps({"status": report["status"], "overall": report["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
