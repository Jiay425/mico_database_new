"""Run the repository-agent semantic review for Decision SFT v1.

This command is offline only.  It never calls an LLM, starts a service, or
launches training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mico_agent_runtime.datasets.decision_sft_v1_review import review_decision_sft_v1


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "artifacts" / "gemini_dynamic_canary_4d1g_final3" / "semantic_catalog.json"
DEFAULT_INPUT = REPO_ROOT / "artifacts" / "decision_sft_v1"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "decision_sft_v1_reviewed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args(argv)
    try:
        report = review_decision_sft_v1(
            args.input_dir,
            catalog_path=args.catalog,
            output_dir=args.output_dir,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": "READY" if report["DECISION_SFT_V1_DATA_READY"] else "REVIEW_FAILED",
        "total_reviewed": report["total_reviewed"],
        "pass": report["pass"],
        "revise": report["revise"],
        "reject": report["reject"],
        "new_samples_added": report["new_samples_added"],
        "final_approved": report["final_approved"],
        "DECISION_SFT_V1_DATA_READY": report["DECISION_SFT_V1_DATA_READY"],
        "training_started": False,
        "audit": str(args.output_dir / "review_audit.json"),
    }, ensure_ascii=False, indent=2))
    return 0 if report["DECISION_SFT_V1_DATA_READY"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
