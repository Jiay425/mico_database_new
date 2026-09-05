"""Build and audit Decision-State-native SFT v1 without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.datasets.decision_sft_v1 import build_decision_sft_v1


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "semantic_catalog.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "decision_sft_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manual-review-completed",
        action="store_true",
        help="Legacy pre-review override; prefer scripts/review_decision_sft_v1.py; never starts training.",
    )
    args = parser.parse_args(argv)
    try:
        audit = build_decision_sft_v1(
            args.output_dir,
            catalog_path=args.catalog,
            manual_review_completed=args.manual_review_completed,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "READY" if audit["DECISION_SFT_V1_DATA_READY"] else "REVIEW_PENDING",
        "candidate_count": audit["candidate_count"],
        "approved_count": audit["approved_count"],
        "rejected_count": audit["rejected_count"],
        "split_distribution": audit["split_distribution"],
        "DECISION_SFT_V1_DATA_READY": audit["DECISION_SFT_V1_DATA_READY"],
        "training_started": False,
        "audit": str(args.output_dir / "audit_report.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
