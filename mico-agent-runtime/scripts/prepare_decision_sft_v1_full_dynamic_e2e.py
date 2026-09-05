"""Prepare the Decision SFT v1 Full Dynamic E2E canary offline.

The default command only validates/snapshots the task set.  It does not open
network sockets, call Gemini/Qwen/DeepSeek, start Java/MySQL, or start an
A100.  The future live runner is a separate explicit ``--live`` command.
"""

from __future__ import annotations

# The script is also runnable directly from a checkout, so it inserts the
# repository root before importing the package under test.
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.e2e.decision_sft_v1_full_dynamic import (
    DEFAULT_FREEZE_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TASK_SET_PATH,
    TaskSetValidationError,
    prepare_task_set,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Decision SFT v1 Full Dynamic E2E without external calls"
    )
    parser.add_argument("--task-set", type=Path, default=DEFAULT_TASK_SET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Refuse: live execution belongs to scripts/run_decision_sft_v1_full_dynamic_e2e.py",
    )
    args = parser.parse_args(argv)
    if args.live:
        print(json.dumps({
            "status": "REFUSED",
            "error": "use scripts/run_decision_sft_v1_full_dynamic_e2e.py --live after operator confirmation",
            "model_calls": 0,
        }, ensure_ascii=False))
        return 2
    try:
        manifest = prepare_task_set(
            args.task_set,
            args.output_dir,
            args.freeze_manifest,
        )
    except (TaskSetValidationError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({
        "status": manifest["status"],
        "FULL_E2E_READY": manifest["FULL_E2E_READY"],
        "READY_TO_START_A100": manifest["READY_TO_START_A100"],
        "task_count": manifest["task_audit"]["task_count"],
        "task_set_sha256": manifest["task_set_sha256"],
        "model_calls": manifest["model_calls"],
        "no_remote_contact": manifest["no_remote_contact"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
