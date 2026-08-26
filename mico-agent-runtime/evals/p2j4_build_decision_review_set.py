"""Build a deduplicated, weighted owner-review set for Decision candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evals.p2j4_decision_dataset import AgentDecisionCandidate


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DECISION_REVIEW_SET_PAYLOAD_INVALID")
    return payload


def _signature(candidate: AgentDecisionCandidate) -> tuple[Any, ...]:
    return (
        candidate.state_summary,
        candidate.decision_reason,
        candidate.selected_action,
        tuple(candidate.alternative_actions),
        candidate.stop_reason,
        tuple(candidate.allowedActions),
    )


def _signature_hash(signature: tuple[Any, ...]) -> str:
    return hashlib.sha256(
        json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _source_index(paths: list[tuple[str, Path]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for label, path in paths:
        payload = _read(path)
        for score in payload.get("scores", []):
            result[score["traceId"]] = {
                "sourceRun": label,
                "caseId": score["caseId"],
            }
    return result


def _priority(candidate: AgentDecisionCandidate, hard_class: str | None) -> int:
    class_priority = {
        "confounder_trap": {
            "adjust_confounders": 100,
            "cross_project_validate": 90,
            "retrieve_evidence": 80,
            "compare_groups": 70,
        },
        "evidence_conflict": {
            "finish": 100,
            "retrieve_evidence": 90,
        },
        "premature_stop": {
            "cross_project_validate": 100,
            "retrieve_evidence": 90,
            "finish": 50,
        },
        "tool_selection": {
            "execute_read_query": 100,
            "inspect_cohort": 90,
        },
    }
    return class_priority.get(hard_class or "", {}).get(candidate.selected_action, 10)


def build(
    candidate_path: Path,
    source_paths: list[tuple[str, Path]],
    hard_spec_paths: list[Path],
) -> dict[str, Any]:
    payload = _read(candidate_path)
    candidates = [AgentDecisionCandidate.model_validate(item) for item in payload.get("candidates", [])]
    source_index = _source_index(source_paths)
    hard_classes: dict[str, str] = {}
    for path in hard_spec_paths:
        for item in _read(path).get("cases", []):
            hard_classes[item["caseId"]] = item["hardCaseClass"]

    groups: dict[tuple[Any, ...], list[AgentDecisionCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[_signature(candidate)].append(candidate)

    selected: dict[str, dict[str, Any]] = {}
    selected_source_traces: set[str] = set()

    def add_item(
        candidate: AgentDecisionCandidate,
        *,
        reason: str,
        hard_case_id: str | None = None,
    ) -> None:
        signature = _signature(candidate)
        signature_hash = _signature_hash(signature)
        existing = selected.get(signature_hash)
        if existing is None:
            group = groups[signature]
            sources = [source_index[item.sourceTraceId] for item in group if item.sourceTraceId in source_index]
            hard_cases = sorted({
                source["caseId"]
                for source in sources
                if source["caseId"] in hard_classes
            })
            selected[signature_hash] = {
                "candidate": candidate.model_dump(mode="json"),
                "signatureHash": signature_hash,
                "replicationCount": len(group),
                "sourceTraceCount": len({item.sourceTraceId for item in group}),
                "sourceRuns": sorted({source["sourceRun"] for source in sources}),
                "hardCaseIds": hard_cases,
                "hardClassCounts": dict(Counter(hard_classes[case_id] for case_id in hard_cases)),
                "reviewReasons": [reason],
            }
        elif reason not in existing["reviewReasons"]:
            existing["reviewReasons"].append(reason)
        if hard_case_id and hard_case_id not in selected[signature_hash]["hardCaseIds"]:
            selected[signature_hash]["hardCaseIds"].append(hard_case_id)
            selected[signature_hash]["hardCaseIds"].sort()
            selected[signature_hash]["hardClassCounts"] = dict(
                Counter(hard_classes[case_id] for case_id in selected[signature_hash]["hardCaseIds"])
            )
        selected_source_traces.add(candidate.sourceTraceId)

    # One representative per exact State/Reason/Action template. Prefer a Hard
    # source and, within a Hard class, the most informative action.
    for signature, group in groups.items():
        ranked = sorted(
            group,
            key=lambda item: (
                max(
                    (_priority(item, hard_classes.get(source_index[item.sourceTraceId]["caseId"]))
                     for _ in [0]
                     if item.sourceTraceId in source_index),
                    default=0,
                ),
                item.sourceTraceId,
            ),
            reverse=True,
        )
        add_item(ranked[0], reason="exact_signature_representative")

    # Add one informative candidate for every Hard case not already represented
    # by the exact-signature representatives.
    by_case: dict[str, list[AgentDecisionCandidate]] = defaultdict(list)
    for candidate in candidates:
        source = source_index.get(candidate.sourceTraceId)
        if source and source["caseId"] in hard_classes:
            by_case[source["caseId"]].append(candidate)
    represented_cases = {
        case_id
        for item in selected.values()
        for case_id in item["hardCaseIds"]
    }
    for case_id in sorted(hard_classes):
        if case_id in represented_cases:
            continue
        options = by_case.get(case_id, [])
        if not options:
            raise ValueError("DECISION_REVIEW_SET_HARD_CASE_MISSING:" + case_id)
        chosen = max(
            options,
            key=lambda item: (_priority(item, hard_classes[case_id]), item.sourceTraceId),
        )
        add_item(chosen, reason="hard_case_coverage", hard_case_id=case_id)

    items = sorted(
        selected.values(),
        key=lambda item: (
            0 if "hard_case_coverage" in item["reviewReasons"] else 1,
            item["candidate"]["selected_action"],
            item["signatureHash"],
        ),
    )
    covered_hard_cases = sorted({case_id for item in items for case_id in item["hardCaseIds"]})
    return {
        "schemaVersion": "p2j4-decision-review-set-v1",
        "sourceCandidateArtifact": candidate_path.name,
        "status": "OWNER_REVIEW_PENDING",
        "trainingStarted": False,
        "rawCandidateCount": len(candidates),
        "reviewItemCount": len(items),
        "exactSignatureCount": len(groups),
        "hardCaseCount": len(hard_classes),
        "hardCaseCoverageCount": len(covered_hard_cases),
        "hardCaseCoverageComplete": len(covered_hard_cases) == len(hard_classes),
        "candidateFieldRepairs": payload.get("candidateFieldRepairs", {}),
        "reviewPolicy": {
            "oneExactSignatureRepresentative": True,
            "oneInformativeRepresentativePerHardCase": True,
            "rawPoolRetainedAsAuditEvidence": True,
            "noTrainingStarted": True,
        },
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build P2-J4 Decision owner-review set")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--hard-spec", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(
            args.candidates,
            [(label, Path(path)) for label, path in args.source],
            args.hard_spec,
        )
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "errorCode": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": result["status"],
        "rawCandidateCount": result["rawCandidateCount"],
        "reviewItemCount": result["reviewItemCount"],
        "exactSignatureCount": result["exactSignatureCount"],
        "hardCaseCoverageCount": result["hardCaseCoverageCount"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
