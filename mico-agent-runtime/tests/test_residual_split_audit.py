import json
from pathlib import Path

from sft.p2j4_audit_residual_split import audit


def test_residual_candidates_do_not_overlap_frozen_splits(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    result = audit(
        root / "sft-data" / "decision-v3-staging",
        root / "evals" / "p2j4-state-obligation-residual-repair-v2",
        tmp_path / "split-audit.json",
    )
    assert result["status"] == "PASS"
    assert result["duplicateIds"] == []
    assert result["duplicateStateSignatures"] == []
    assert json.loads((tmp_path / "split-audit.json").read_text(encoding="utf-8"))["repairCount"] == 3
