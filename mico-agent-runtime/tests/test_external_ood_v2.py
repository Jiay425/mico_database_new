import json
from pathlib import Path

from sft.p2j4_build_external_ood_v2 import build


def test_external_ood_v2_preserves_gold_and_adds_runtime_obligations(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    manifest = build(
        root / "evals" / "p2j4-external-ood-v1",
        tmp_path / "external-ood-v2",
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "external-ood-v2" / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_id = {row["id"]: row for row in rows}
    assert manifest["recordCount"] == 30
    assert manifest["contractAudit"]["status"] == "PASS"
    assert "ANALYSIS_REQUIRED" in json.loads(
        by_id["ood-focused-09"]["messages"][1]["content"]
    )["policy_state"]["observation_flags"]
    assert "EVIDENCE_REQUIRED" in json.loads(
        by_id["ood-evidence-29"]["messages"][1]["content"]
    )["policy_state"]["observation_flags"]

    source_rows = [
        json.loads(line)
        for line in (root / "evals" / "p2j4-external-ood-v1" / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for source, refreshed in zip(source_rows, rows):
        assert source["id"] == refreshed["id"]
        assert source["ood_family"] == refreshed["ood_family"]
        assert source["messages"][2] == refreshed["messages"][2]
