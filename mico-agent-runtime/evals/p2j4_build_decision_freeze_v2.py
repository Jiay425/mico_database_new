"""Build the single canonical, de-identified Decision/DPO dataset freeze.

This replaces layered v1-v4 staging artifacts with a provenance-labelled
freeze.  It never reads questions, SQL, tool payloads, or held-out completions
into training.  Existing real Trace decisions are context-enriched; additional
state-difference records are closed-contract controlled cases, not paraphrases
of held-out prompts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "evals" / "p2j4-decision-review-set-context-v5-20260824.json"
SPLIT = ROOT / "evals" / "p2j4-decision-family-split-v2-20260824.json"
TEST70 = ROOT / "sft-data" / "decision-v4-staging" / "test.jsonl"
OOD30 = ROOT / "evals" / "p2j4-external-ood-v2" / "records.jsonl"
SFT_OUTPUT = ROOT / "sft-data" / "decision-freeze-v2-staging"
DPO_OUTPUT = ROOT / "sft-data" / "decision-dpo-v2-staging"

SYSTEM = (
    "You are Mico's Scientific Agent decision policy. Choose the next action "
    "from the candidate actions in the policy state. Use only the supplied "
    "structured state and action history. Return one JSON object and no Markdown. "
    "The JSON keys must be exactly: selected_action, decision_reason, "
    "alternative_actions, stop_reason. Use stop_reason only when selected_action "
    "is finish."
)
FORBIDDEN = ("question", "sql", "payload", "token", "password", "api_key", "sourceSampleId")
ALL_ACTIONS = {
    "execute_read_query", "inspect_cohort", "compare_groups", "stratified_analysis",
    "adjust_confounders", "cross_project_validate", "cross_disease_validate",
    "retrieve_evidence", "analyze_projection", "finish",
}
STOP = {"EVIDENCE_SUFFICIENT", "QUALITY_RISK", "NO_NEW_INFORMATION"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _policy_state(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(record["messages"][1]["content"])["policy_state"]


def _target(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(record["messages"][2]["content"])


def _signature(state: dict[str, Any]) -> str:
    safe = {
        key: state.get(key)
        for key in ("task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary")
    }
    return _hash(safe)[:20]


def _reason(action: str, state: dict[str, Any], *, quality_stop: bool = False) -> str:
    flags = set(state["observation_flags"])
    goal = state["goal_code"]
    if action == "inspect_cohort":
        return f"the {goal} objective has no validated cohort observation, so inspect metadata before analysis"
    if action == "execute_read_query":
        return f"the {goal} objective requires one bounded read after the available metadata context"
    if action == "compare_groups":
        return f"the cohort context is validated and the {goal} objective now requires a bounded group comparison"
    if action == "adjust_confounders":
        return "age, sex, project, or related cohort imbalance remains; adjust confounders before interpreting the comparison"
    if action == "stratified_analysis":
        return "the observed pattern needs a bounded subgroup check to determine whether it is stable across the flagged strata"
    if action == "cross_project_validate":
        return "the preliminary observation needs replication across projects before it can support a stable finding"
    if action == "cross_disease_validate":
        return "the observation is not yet disease-specific; validate it against the available disease comparison context"
    if action == "analyze_projection":
        return "a bounded comparison is available and must be summarized before the next evidence or stopping decision"
    if action == "retrieve_evidence":
        return "validated data evidence is available but external evidence is still required for the current research obligation"
    if quality_stop or "EVIDENCE_CONFLICT" in flags or "EVIDENCE_INCOMPLETE" in flags:
        return "the remaining evidence is conflicted or incomplete, so stop with an explicit quality-risk boundary"
    return "all required bounded observations and validation obligations are complete, so the exploration may stop"


def _record(
    *, record_id: str, provenance: str, family: str, hard_class: str | None,
    state: dict[str, Any], selected: str, rejected: str | None = None,
    stop_reason: str | None = None, source_trace_id: str | None = None,
) -> dict[str, Any]:
    if selected not in state["candidate_actions"]:
        raise ValueError("SELECTED_ACTION_NOT_CANDIDATE")
    if rejected is not None and rejected not in state["candidate_actions"]:
        raise ValueError("REJECTED_ACTION_NOT_CANDIDATE")
    if selected == "finish":
        stop_reason = stop_reason or "EVIDENCE_SUFFICIENT"
    elif stop_reason is not None:
        raise ValueError("NON_TERMINAL_STOP_REASON")
    target = {
        "selected_action": selected,
        "decision_reason": _reason(selected, state, quality_stop=stop_reason == "QUALITY_RISK"),
        "alternative_actions": [action for action in state["candidate_actions"] if action != selected],
        "stop_reason": stop_reason,
    }
    prompt = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _canonical({"policy_state": state})},
    ]
    return {
        "id": record_id,
        "messages": [*prompt, {"role": "assistant", "content": _canonical(target)}],
        "metadata": {
            "provenance": provenance,
            "task_family": family,
            "hard_case_class": hard_class,
            "state_signature": _signature(state),
            "dpo_rejected_action": rejected,
            "source_trace_id": source_trace_id or record_id,
        },
    }


def _state(kind: str, goal: str, flags: list[str], history: list[str], actions: list[str], context: str) -> dict[str, Any]:
    return {
        "task_kind": kind,
        "goal_code": goal,
        "observation_flags": flags,
        "history_actions": history,
        "candidate_actions": actions,
        "state_summary": (
            f"goal={goal}; context={context}; observation_flags={','.join(flags)}; "
            f"history_actions={','.join(history) if history else 'none'}; "
            f"candidate_actions={','.join(actions)}"
        ),
    }


def _real_policy_eligible(state: dict[str, Any], selected: str, stop_reason: str | None) -> bool:
    """Reject legacy Trace decisions whose old context cannot justify the label.

    A missing obligation flag is not repaired from the selected label: doing so
    would leak the answer into the policy state.  Such rows remain provenance,
    but are not admitted to Freeze v2.
    """
    flags = set(state["observation_flags"])
    if "NO_OBSERVATION" in flags and selected not in {"inspect_cohort", "execute_read_query"}:
        return False
    if selected in state["history_actions"] and selected != "finish":
        return False
    if "CONFOUNDER_PRESENT" in flags and "CONFOUNDER_ADJUSTED" not in flags and selected == "finish":
        return False
    if "CROSS_PROJECT_REQUIRED" in flags and "PROJECT_VALIDATED" not in flags and selected == "finish":
        return False
    if "CROSS_DISEASE_REQUIRED" in flags and "CROSS_DISEASE_VALIDATED" not in flags and selected == "finish":
        return False
    if "EVIDENCE_CONFLICT" in flags and selected == "finish" and stop_reason != "QUALITY_RISK":
        return False
    if "EVIDENCE_INCOMPLETE" in flags and selected == "finish" and stop_reason != "QUALITY_RISK":
        return False
    if selected == "adjust_confounders" and "CONFOUNDER_PRESENT" not in flags:
        return False
    if selected == "cross_project_validate" and "CROSS_PROJECT_REQUIRED" not in flags:
        return False
    if selected == "cross_disease_validate" and "CROSS_DISEASE_REQUIRED" not in flags:
        return False
    if selected == "retrieve_evidence" and not flags.intersection({"ANALYSIS_AVAILABLE", "PROJECT_VALIDATED", "CROSS_DISEASE_VALIDATED", "EVIDENCE_INCOMPLETE", "EVIDENCE_CONFLICT"}):
        return False
    if selected == "execute_read_query" and "BOUNDED_READ_REQUIRED" not in flags:
        return False
    return not (selected == "finish" and not stop_reason)


def _real_records() -> list[dict[str, Any]]:
    review = _read(REVIEW)
    split = _read(SPLIT)
    train_trace_ids = {
        item["candidate"]["sourceTraceId"]
        for item in split["splits"]["train"]
    }
    selected: dict[str, dict[str, Any]] = {}
    for item in review["items"]:
        candidate = item["candidate"]
        if candidate["sourceTraceId"] not in train_trace_ids:
            continue
        state = _state(
            candidate["task_kind"], candidate["goal_code"], candidate["observation_flags"],
            candidate["history_actions"], candidate["candidate_actions"],
            "real_trace_context",
        )
        if not _real_policy_eligible(state, candidate["selected_action"], candidate["stop_reason"]):
            continue
        signature = _signature(state)
        record = _record(
            record_id="real-" + signature,
            provenance="enriched_real_trace_v5",
            family="real:" + candidate["task_family"],
            hard_class=candidate.get("hard_case_class"),
            state=state,
            selected=candidate["selected_action"],
            rejected=(candidate["alternative_actions"][0] if candidate["alternative_actions"] else None),
            stop_reason=candidate["stop_reason"],
            source_trace_id=candidate["sourceTraceId"],
        )
        existing = selected.get(signature)
        if existing is None or record["metadata"]["hard_case_class"] and not existing["metadata"]["hard_case_class"]:
            selected[signature] = record
    # No real source family can dominate a controlled-policy freeze.
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected.values():
        by_family[item["metadata"]["task_family"]].append(item)
    capped: list[dict[str, Any]] = []
    for family in sorted(by_family):
        capped.extend(sorted(by_family[family], key=lambda item: item["id"])[:70])
    # Keep the real-trace component substantial but leave room for balanced
    # state-difference coverage; the raw pool remains intact as provenance.
    return sorted(capped, key=lambda item: item["id"])[:400]


def _controlled_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    contexts = [
        "age_imbalance", "sex_imbalance", "project_imbalance", "country_imbalance",
        "age_sex_imbalance", "age_project_imbalance", "sex_project_imbalance",
        "country_project_imbalance", "batch_project_imbalance", "single_project_risk",
        "partial_project_coverage", "replication_direction_gap", "effect_size_gap",
        "cross_country_gap", "cross_age_group_gap", "cross_sex_group_gap",
        "vector_evidence_gap", "graph_evidence_gap", "literature_direction_conflict", "evidence_scope_gap",
    ]

    def add(family: str, hard: str | None, index: int, kind: str, goal: str, flags: list[str], history: list[str], actions: list[str], selected: str, rejected: str, stop: str | None = None) -> None:
        # ``index`` advances phase-first.  Dividing by five gives every phase
        # all twenty distinct cohort/evidence contexts instead of repeating
        # four summaries per phase.
        context = f"phase={phase}; profile={contexts[(index // 5) % len(contexts)]}"
        record_id = f"controlled-{family}-{index:03d}"
        records.append(_record(
            record_id=record_id,
            provenance="controlled_state_difference_v2",
            # A block contains all five phases, so a family-level split does
            # not place a whole action type exclusively in validation.
            family=f"controlled:{family}:block{index // 25}",
            hard_class=hard,
            state=_state(kind, goal, flags, history, actions, context),
            selected=selected,
            rejected=rejected,
            stop_reason=stop,
        ))

    # 5 families x 100 records.  Each phase changes the next obligation; the
    # context is de-identified and changes a real state attribute, not wording.
    for index in range(100):
        phase = index % 5
        if phase == 0:
            add("premature_stop", "premature_stop", index, "open_exploration", "cross_project_stability", ["NO_OBSERVATION", "METADATA_FIRST", "CROSS_PROJECT_REQUIRED"], [], ["inspect_cohort", "finish"], "inspect_cohort", "finish")
        elif phase == 1:
            add("premature_stop", "premature_stop", index, "open_exploration", "cross_project_stability", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "CROSS_PROJECT_REQUIRED"], ["inspect_cohort"], ["compare_groups", "finish"], "compare_groups", "finish")
        elif phase == 2:
            add("premature_stop", "premature_stop", index, "open_exploration", "cross_project_stability", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE", "CROSS_PROJECT_REQUIRED"], ["inspect_cohort", "compare_groups"], ["cross_project_validate", "finish"], "cross_project_validate", "finish")
        elif phase == 3:
            add("premature_stop", "premature_stop", index, "open_exploration", "cross_project_stability", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "PROJECT_VALIDATED", "CROSS_PROJECT_REQUIRED"], ["inspect_cohort", "compare_groups", "cross_project_validate"], ["retrieve_evidence", "finish"], "retrieve_evidence", "finish")
        else:
            add("premature_stop", "premature_stop", index, "open_exploration", "cross_project_stability", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "PROJECT_VALIDATED", "EVIDENCE_RETRIEVED"], ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence"], ["finish", "retrieve_evidence"], "finish", "retrieve_evidence", "EVIDENCE_SUFFICIENT")

        if phase == 0:
            add("confounder", "confounder_trap", index, "focused_analysis", "confounder_adjustment", ["NO_OBSERVATION", "METADATA_FIRST", "CONFOUNDER_PRESENT"], [], ["inspect_cohort", "finish"], "inspect_cohort", "finish")
        elif phase == 1:
            add("confounder", "confounder_trap", index, "focused_analysis", "confounder_adjustment", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "CONFOUNDER_PRESENT"], ["inspect_cohort"], ["compare_groups", "finish"], "compare_groups", "finish")
        elif phase == 2:
            add("confounder", "confounder_trap", index, "focused_analysis", "confounder_adjustment", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE", "CONFOUNDER_PRESENT"], ["inspect_cohort", "compare_groups"], ["adjust_confounders", "stratified_analysis", "finish"], "adjust_confounders", "finish")
        elif phase == 3:
            add("confounder", "confounder_trap", index, "focused_analysis", "confounder_adjustment", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "CONFOUNDER_PRESENT", "ANALYSIS_AVAILABLE"], ["inspect_cohort", "compare_groups"], ["adjust_confounders", "stratified_analysis", "finish"], "adjust_confounders", "finish")
        else:
            add("confounder", "confounder_trap", index, "focused_analysis", "confounder_adjustment", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "CONFOUNDER_ADJUSTED", "ANALYSIS_AVAILABLE"], ["inspect_cohort", "compare_groups", "adjust_confounders"], ["stratified_analysis", "retrieve_evidence", "finish"], "stratified_analysis", "finish")

        if phase == 0:
            add("evidence_conflict", "evidence_conflict", index, "open_exploration", "evidence_conflict_resolution", ["NO_OBSERVATION", "METADATA_FIRST", "EVIDENCE_CONFLICT"], [], ["inspect_cohort", "finish"], "inspect_cohort", "finish")
        elif phase in {1, 2}:
            add("evidence_conflict", "evidence_conflict", index, "open_exploration", "evidence_conflict_resolution", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE", "EVIDENCE_CONFLICT", "EVIDENCE_INCOMPLETE"], ["inspect_cohort", "compare_groups"], ["retrieve_evidence", "finish"], "retrieve_evidence", "finish")
        else:
            add("evidence_conflict", "evidence_conflict", index, "open_exploration", "evidence_conflict_resolution", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "VECTOR_EVIDENCE", "GRAPH_EVIDENCE", "EVIDENCE_RETRIEVED", "EVIDENCE_CONFLICT", "EVIDENCE_INCOMPLETE"], ["inspect_cohort", "compare_groups", "retrieve_evidence"], ["finish", "retrieve_evidence"], "finish", "retrieve_evidence", "QUALITY_RISK")

        if phase == 0:
            add("tool_choice", "tool_selection", index, "data_fact", "bounded_tool_selection", ["NO_OBSERVATION", "METADATA_FIRST"], [], ["inspect_cohort", "execute_read_query", "finish"], "inspect_cohort", "execute_read_query")
        elif phase in {1, 2}:
            add("tool_choice", "tool_selection", index, "data_fact", "bounded_tool_selection", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "BOUNDED_READ_REQUIRED", "METADATA_FIRST"], ["inspect_cohort"], ["execute_read_query", "finish"], "execute_read_query", "finish")
        else:
            add("tool_choice", "tool_selection", index, "data_fact", "bounded_tool_selection", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED"], ["inspect_cohort", "execute_read_query"], ["finish", "execute_read_query"], "finish", "execute_read_query", "EVIDENCE_SUFFICIENT")

        if phase == 0:
            add("specificity_and_analysis", None, index, "open_exploration", "cross_disease_specificity", ["NO_OBSERVATION", "METADATA_FIRST", "CROSS_DISEASE_REQUIRED"], [], ["inspect_cohort", "finish"], "inspect_cohort", "finish")
        elif phase == 1:
            add("specificity_and_analysis", None, index, "open_exploration", "cross_disease_specificity", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "CROSS_DISEASE_REQUIRED"], ["inspect_cohort"], ["compare_groups", "finish"], "compare_groups", "finish")
        elif phase == 2:
            add("specificity_and_analysis", None, index, "open_exploration", "cross_disease_specificity", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE", "CROSS_DISEASE_REQUIRED"], ["inspect_cohort", "compare_groups"], ["analyze_projection", "cross_disease_validate", "finish"], "analyze_projection", "finish")
        elif phase == 3:
            add("specificity_and_analysis", None, index, "open_exploration", "cross_disease_specificity", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE", "CROSS_DISEASE_REQUIRED"], ["inspect_cohort", "compare_groups", "analyze_projection"], ["cross_disease_validate", "finish"], "cross_disease_validate", "finish")
        else:
            add("specificity_and_analysis", None, index, "open_exploration", "cross_disease_specificity", ["OBSERVATION_VALIDATED", "JAVA_OBSERVED", "ANALYSIS_AVAILABLE", "CROSS_DISEASE_REQUIRED"], ["inspect_cohort", "compare_groups", "analyze_projection"], ["cross_disease_validate", "retrieve_evidence", "finish"], "cross_disease_validate", "finish")
    return records


def _partition(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {"train": [], "validation": []}
    # Entire source family stays in one split.  Real and controlled source
    # families cannot share a normalized state signature across partitions.
    for record in records:
        family = record["metadata"]["task_family"]
        bucket = "validation" if int(_hash(family)[:8], 16) % 10 < 2 else "train"
        result[bucket].append(record)
    return result


def _dpo(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["metadata"]["provenance"] != "controlled_state_difference_v2":
            continue
        family = record["metadata"]["task_family"].split(":")[1]
        by_family[family].append(record)
    selected: list[dict[str, Any]] = []
    for family in sorted(by_family):
        # 60 per controlled family gives exactly 300 balanced, explicit pairs.
        selected.extend(sorted(by_family[family], key=lambda item: item["id"])[:60])
    pairs: list[dict[str, Any]] = []
    for record in selected:
        rejected_action = record["metadata"]["dpo_rejected_action"]
        if not rejected_action:
            continue
        state = _policy_state(record)
        target = _target(record)
        rejected = {
            "selected_action": rejected_action,
            "decision_reason": _reason(rejected_action, state, quality_stop=rejected_action == "finish"),
            "alternative_actions": [action for action in state["candidate_actions"] if action != rejected_action],
            "stop_reason": "EVIDENCE_SUFFICIENT" if rejected_action == "finish" else None,
        }
        if rejected == target:
            continue
        pairs.append({
            "id": "dpo-" + record["id"],
            "prompt": record["messages"][:2],
            "chosen": _canonical(target),
            "rejected": _canonical(rejected),
            "metadata": dict(record["metadata"]),
        })
    return pairs


def _validate(records: list[dict[str, Any]], external_signatures: set[str]) -> dict[str, Any]:
    signatures = [record["metadata"]["state_signature"] for record in records]
    actions = Counter(_target(record)["selected_action"] for record in records)
    families = Counter(record["metadata"]["task_family"].split(":")[0] for record in records)
    texts = _canonical(records).lower()
    return {
        "count": len(records),
        "uniqueStateCount": len(set(signatures)),
        "duplicateStateCount": len(signatures) - len(set(signatures)),
        "actionCounts": dict(actions),
        "familyCounts": dict(families),
        "externalExactStateOverlap": len(set(signatures) & external_signatures),
        "forbiddenTextAbsent": not any(term.lower() in texts for term in FORBIDDEN),
        "allActionsKnown": set(actions).issubset(ALL_ACTIONS),
        "allActionsMinimum30": all(actions.get(action, 0) >= 30 for action in ALL_ACTIONS),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.write_text("".join(_canonical(record) + "\n" for record in records), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    real = _real_records()
    controlled = _controlled_records()
    records = [*real, *controlled]
    # Deduplicate state signatures without losing a hard-case representative.
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        signature = record["metadata"]["state_signature"]
        prior = deduped.get(signature)
        if prior is None or (record["metadata"]["hard_case_class"] and not prior["metadata"]["hard_case_class"]):
            deduped[signature] = record
    records = sorted(deduped.values(), key=lambda record: record["id"])

    external = {
        _signature(_policy_state(record))
        for record in [*_jsonl(TEST70), *_jsonl(OOD30)]
    }
    partitions = _partition(records)
    train_signatures = {record["metadata"]["state_signature"] for record in partitions["train"]}
    validation_signatures = {record["metadata"]["state_signature"] for record in partitions["validation"]}
    if train_signatures & validation_signatures:
        raise ValueError("DECISION_FREEZE_SPLIT_SIGNATURE_LEAK")
    quality = {
        name: _validate(items, external)
        for name, items in partitions.items()
    }
    overall_quality = _validate(records, external)
    total = len(records)
    if not 600 <= total <= 900:
        raise ValueError("DECISION_FREEZE_COUNT_OUT_OF_RANGE")
    if not overall_quality["allActionsMinimum30"]:
        raise ValueError("DECISION_FREEZE_ACTION_COVERAGE_BELOW_30")
    if any(item["externalExactStateOverlap"] for item in quality.values()):
        raise ValueError("DECISION_FREEZE_EXTERNAL_EXACT_LEAK")

    SFT_OUTPUT.mkdir(parents=True, exist_ok=True)
    # Balance the preference pool once globally, then inherit the frozen SFT
    # state-family partition.  Balancing each split independently would double
    # some families and silently bias DPO.
    dpo_partitions = {"train": [], "validation": []}
    for pair in _dpo(records):
        bucket = "train" if pair["metadata"]["state_signature"] in train_signatures else "validation"
        dpo_partitions[bucket].append(pair)
    dpo_total = sum(len(items) for items in dpo_partitions.values())
    if not 250 <= dpo_total <= 400:
        raise ValueError("DECISION_DPO_PAIR_COUNT_OUT_OF_RANGE")
    DPO_OUTPUT.mkdir(parents=True, exist_ok=True)
    sft_hashes = {name: _write_jsonl(SFT_OUTPUT / f"{name}.jsonl", items) for name, items in partitions.items()}
    dpo_hashes = {name: _write_jsonl(DPO_OUTPUT / f"{name}.jsonl", items) for name, items in dpo_partitions.items()}
    manifest = {
        "schemaVersion": "p2j4-decision-freeze-v2",
        "trainingStarted": False,
        "source": {"realReview": REVIEW.name, "familySplit": SPLIT.name, "controlled": "controlled_state_difference_v2"},
        "counts": {"realEnriched": len(real), "controlled": len(controlled), "accepted": total, "splits": {name: len(items) for name, items in partitions.items()}},
        "quality": {"overall": overall_quality, **quality},
        "splitPolicy": "task-family hash plus disjoint normalized state signatures; Test70/OOD30 exact states excluded",
        "externalEvaluation": {"test70": TEST70.name, "ood30": str(OOD30.relative_to(ROOT))},
        "sft": {"hashes": sft_hashes},
        "dpo": {"counts": {name: len(items) for name, items in dpo_partitions.items()}, "hashes": dpo_hashes, "pairCount": sum(len(items) for items in dpo_partitions.values()), "pairPolicy": "same normalized state; chosen closed-contract action versus explicit inferior allow-listed action"},
        "releaseGate": "OWNER_REVIEW_REQUIRED",
    }
    for path in (SFT_OUTPUT / "manifest.json", DPO_OUTPUT / "manifest.json"):
        path.write_text(_canonical(manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    try:
        result = build()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(_canonical({"status": "FAILED", "errorCode": str(exc)}))
        return 1
    print(_canonical({"status": "PASS", "counts": result["counts"], "dpo": result["dpo"]["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
