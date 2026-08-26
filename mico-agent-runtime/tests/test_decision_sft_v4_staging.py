from pathlib import Path

from sft.p2j4_build_decision_sft_v4_staging import build


def test_decision_sft_v4_staging_keeps_frozen_eval_splits(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    manifest = build(
        root / "sft-data" / "decision-v3-staging",
        root / "evals" / "p2j4-state-obligation-residual-repair-v2",
        tmp_path / "decision-v4-staging",
    )
    assert manifest["trainingStarted"] is False
    assert manifest["splitCounts"] == {"train": 655, "validation": 32, "test": 70}
    assert manifest["validationUnchanged"] is True
    assert manifest["testUnchanged"] is True
