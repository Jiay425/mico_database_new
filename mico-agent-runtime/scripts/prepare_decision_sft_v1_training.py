"""Run the offline Decision SFT v1 preparation gate.

This command only audits frozen files and writes preparation reports.  It does
not call a model, start a service, connect to an A100, or train an adapter.
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
    DEFAULT_PREP_DIR,
    DEFAULT_TOKENIZER_DIR,
    DatasetPreparationError,
    prepare_decision_sft_v1,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Decision-State-native SFT v1 offline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREP_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    args = parser.parse_args(argv)
    try:
        report = prepare_decision_sft_v1(args.data_dir, args.output_dir, args.tokenizer_dir)
    except (DatasetPreparationError, OSError, ValueError, TypeError, ImportError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "PASS", "report": report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
