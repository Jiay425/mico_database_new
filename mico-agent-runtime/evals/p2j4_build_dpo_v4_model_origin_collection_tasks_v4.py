"""Build another provenance-focused action-set variant without reusing case IDs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from evals import p2j4_build_dpo_v4_model_origin_collection_tasks_v3 as base

EXTRAS = [
    ("compare_groups",),
    ("cross_project_validate",),
    ("cross_disease_validate",),
    ("stratified_analysis",),
    ("adjust_confounders",),
    ("compare_groups", "retrieve_evidence"),
    ("cross_project_validate", "analyze_projection"),
    ("cross_disease_validate", "retrieve_evidence"),
    ("stratified_analysis", "retrieve_evidence"),
    ("adjust_confounders", "cross_project_validate"),
]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--task-output", type=Path, required=True); parser.add_argument("--preflight-output", type=Path, required=True); args = parser.parse_args()
    old = base.EXTRAS; base.EXTRAS = EXTRAS
    tasks, audit = base.build()
    for case in tasks["cases"]:
        case["caseId"] = case["caseId"].replace("model-origin-v3", "model-origin-v4")
    for case in audit["cases"]:
        case["caseId"] = case["caseId"].replace("model-origin-v3", "model-origin-v4")
    tasks["name"] = "DPO v4 model-origin collection v4"
    tasks["purpose"] = "provenance_complete_model_origin_collection_variant_v4"
    args.task_output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.preflight_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"caseCount": tasks["caseCount"], "semanticNone": audit["allNonInitialStatesSemanticNone"]}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
