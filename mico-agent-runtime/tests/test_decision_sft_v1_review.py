from __future__ import annotations

import json
from pathlib import Path

from mico_agent_runtime.datasets.decision_sft_v1 import _load_catalog, build_decision_sft_v1
from mico_agent_runtime.datasets.decision_sft_v1_review import review_decision_sft_v1


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "semantic_catalog.json"


def _jsonl_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_repository_agent_review_revalidates_and_freezes_final_dataset(tmp_path: Path) -> None:
    input_dir = tmp_path / "candidate"
    output_dir = tmp_path / "reviewed"
    build_decision_sft_v1(input_dir, catalog_path=CATALOG_PATH, manual_review_completed=False)

    audit = review_decision_sft_v1(
        input_dir,
        catalog_path=CATALOG_PATH,
        output_dir=output_dir,
    )

    assert audit["candidate_v1_count"] == 480
    assert audit["total_reviewed"] == 531
    assert audit["new_samples_added"] == 51
    assert audit["pass"] == 499
    assert audit["revise"] == 19
    assert audit["reject"] == 13
    assert audit["final_approved"] == 518
    assert audit["external_model_calls"] == 0
    assert audit["training_started"] is False
    assert audit["DECISION_SFT_V1_DATA_READY"] is True
    assert audit["training_eligible"] is True

    checks = audit["checks"]
    assert all(checks.values())
    assert audit["hard_case_distribution"]["availability_boundary"] >= 30
    assert audit["hard_case_distribution"]["redundant_action"] >= 20
    assert audit["hard_case_distribution"]["premature_finish"] >= 20
    assert set(audit["frozen_test_hard_cases"]) >= {
        "availability_boundary",
        "objective_action_confusion",
        "observation_sensitivity",
        "heterogeneity_conflict",
        "premature_finish",
        "redundant_action",
        "missed_finish",
    }
    assert audit["second_pass"]["hard_case_count"] == 225
    assert audit["second_pass"]["pass"] == 225
    assert audit["second_pass"]["reject"] == 0
    assert audit["near_duplicate_audit"]["cross_split_groups"] == {}
    assert audit["pair_trajectory_split_leakage"]["overlap_count"] == 0

    # The original 480 records remain available byte-for-byte at the record
    # level, while the reviewed output is a separate, final artifact set.
    assert _jsonl_count(output_dir / "candidate_v1.jsonl") == 480
    assert _jsonl_count(output_dir / "review_report.jsonl") == 531
    assert _jsonl_count(output_dir / "reviewed_v1.jsonl") == 518
    assert _jsonl_count(output_dir / "reviewed_train.jsonl") == 388
    assert _jsonl_count(output_dir / "reviewed_validation.jsonl") == 65
    assert _jsonl_count(output_dir / "reviewed_test.jsonl") == 65

    manifest = json.loads((output_dir / "review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["review_origin"] == "repository_agent"
    assert manifest["external_model_calls"] == 0
    assert manifest["training_started"] is False
    assert manifest["training_eligible"] is True
    assert manifest["DECISION_SFT_V1_DATA_READY"] is True

    model_record = json.loads(
        (output_dir / "reviewed_train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert set(model_record) == {"input", "output"}
    assert set(model_record["input"]) == {"decision_type", "state"}
    assert set(model_record["output"]) == {
        "selected_action",
        "decision_reason",
        "alternative_actions",
        "stop_reason",
    }


def test_review_report_contains_three_examples_for_every_action(tmp_path: Path) -> None:
    input_dir = tmp_path / "candidate"
    output_dir = tmp_path / "reviewed"
    build_decision_sft_v1(input_dir, catalog_path=CATALOG_PATH, manual_review_completed=False)
    review_decision_sft_v1(input_dir, catalog_path=CATALOG_PATH, output_dir=output_dir)
    audit = json.loads((output_dir / "review_audit.json").read_text(encoding="utf-8"))

    examples = audit["examples_by_action"]
    assert set(examples) == {
        "inspect_cohort",
        "execute_read_query",
        "compare_groups",
        "analyze_projection",
        "stratified_analysis",
        "adjust_confounders",
        "cross_project_validate",
        "cross_disease_validate",
        "retrieve_evidence",
        "finish",
    }
    assert all(len(values) == 3 for values in examples.values())
    assert all("decision_reason" in example for values in examples.values() for example in values)

    hard_examples = audit["examples_by_hard_case"]
    assert len(hard_examples["availability_boundary"]) == 3
    assert len(hard_examples["redundant_action"]) == 3
    assert len(hard_examples["premature_finish"]) == 3
