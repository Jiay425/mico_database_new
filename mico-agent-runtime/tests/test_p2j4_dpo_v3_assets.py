import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dpo_v3_audit_stages_balanced_leak_free_review_set() -> None:
    audit = json.loads((ROOT / "evals" / "p2j4-decision-dpo-v3-audit-20260825.json").read_text(encoding="utf-8"))
    assert audit["status"] == "REVIEW_REQUIRED"
    assert audit["candidateCount"] == 1100
    assert audit["selectedStagingCount"] == 750
    assert audit["trainCount"] == 600
    assert audit["validationCount"] == 150
    assert audit["dimensionCounts"] == {
        "efficiency": 350,
        "exploration_depth": 200,
        "reason_quality": 200,
    }
    assert audit["heldoutSignatureOverlap"] == 0
    assert audit["trainValidationFamilyOverlap"] == []
    assert audit["uniqueSelectedStateSignatures"] == 750


def test_preference_runtime_holdout_is_independent_and_policy_sensitive() -> None:
    base = ROOT / "evals" / "p2j4-preference-runtime-holdout-v1"
    tasks = json.loads((base / "task-set.json").read_text(encoding="utf-8"))
    preflight = json.loads((base / "semantic-preflight.json").read_text(encoding="utf-8"))
    reason_oracle = json.loads((base / "reason-oracle.json").read_text(encoding="utf-8"))
    assert tasks["caseCount"] == 30
    assert tasks["dimensionDistribution"] == {
        "efficiency": 10,
        "exploration_depth": 10,
        "reason_quality": 10,
    }
    assert preflight["allNonEmptyStatesSemanticNone"] is True
    assert len(reason_oracle["cases"]) == 10
    assert all(case["caseId"].startswith("p2j4-preference-") for case in tasks["cases"])


def test_dpo_v3_user_authorized_core_review_produced_a_hash_pinned_freeze() -> None:
    review = json.loads((ROOT / "evals" / "p2j4-decision-dpo-v3-core-review-20260825.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "evals" / "p2j4-decision-dpo-v3-freeze-manifest-20260825.json").read_text(encoding="utf-8"))
    assert review["status"] == "USER_AUTHORIZED_AGENT_AUDIT_PASS"
    assert review["reviewCount"] == 150
    assert review["rejectedCount"] == 0
    assert manifest["status"] == "FROZEN"
    assert manifest["counts"] == {"train": 600, "validation": 150, "total": 750}
    for item in manifest["files"].values():
        assert (ROOT / item["path"]).exists()
        assert len(item["sha256"]) == 64
