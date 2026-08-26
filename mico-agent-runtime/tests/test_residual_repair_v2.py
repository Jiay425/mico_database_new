import json
from pathlib import Path

from sft.p2j4_build_residual_repair_v2 import build


def test_residual_repair_v2_is_small_closed_and_not_training(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    manifest = build(
        root / "evals" / "p2j4-external-ood-v2" / "records.jsonl",
        tmp_path / "residual-repair-v2",
    )
    assert manifest["recordCount"] == 3
    assert manifest["trainingStarted"] is False
    assert manifest["audit"]["status"] == "PASS"
    rows = [
        json.loads(line)
        for line in (tmp_path / "residual-repair-v2" / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_id = {row["id"]: row for row in rows}
    for record_id, expected_flag in {
        "repair-residual-001": "CROSS_PROJECT_REQUIRED",
        "repair-residual-002": "EVIDENCE_REQUIRED",
        "repair-residual-003": "EVIDENCE_REQUIRED",
    }.items():
        state = json.loads(by_id[record_id]["messages"][1]["content"])["policy_state"]
        assert expected_flag in state["observation_flags"]
