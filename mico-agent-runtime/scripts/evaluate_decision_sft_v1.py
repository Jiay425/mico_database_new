"""Evaluate a Decision SFT v1 prediction file on the frozen test contract.

The evaluator is intentionally separate from training.  It requires the
immutable ``reviewed_test.jsonl`` provenance and accepts only
``{"sample_id": ..., "prediction": {...}}`` JSONL.  Running without
``--predictions`` prints the contract schema/audit only; it does not run a
model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.training.decision_sft_v1_preparation import (
    DEFAULT_DATA_DIR,
    DatasetPreparationError,
    evaluate_frozen_test_contract,
    score_frozen_predictions,
    validate_frozen_dataset,
)


def _read_predictions(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise DatasetPreparationError(f"prediction_record_not_object:{line_number}")
        rows.append(value)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Decision SFT v1 Frozen Test predictions")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args(argv)
    try:
        # No-prediction invocation is a pre-training schema check and must
        # not open Frozen Test labels.  Supplying predictions is the explicit
        # post-training scoring boundary.
        freeze = validate_frozen_dataset(args.data_dir, inspect_test=args.predictions is not None)
        if args.predictions is None:
            result = evaluate_frozen_test_contract(args.data_dir, freeze, inspect_test=False)
        else:
            result = score_frozen_predictions(_read_predictions(args.predictions), args.data_dir, freeze)
    except (DatasetPreparationError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
