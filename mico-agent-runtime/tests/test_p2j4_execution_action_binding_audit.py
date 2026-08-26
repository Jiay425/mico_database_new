from __future__ import annotations

import json
from pathlib import Path

from evals.p2j4_audit_execution_action_binding import audit


def _artifact(*, bound: bool) -> dict[str, object]:
    return {
        "traces": [{
            "traceId": "trace-binding-test",
            "executionActionBindingVersion": "execution-action-v1" if bound else None,
            "decisions": [{"chosenAction": "compare_groups"}],
            "executedActions": ["compare_groups"],
            "events": [{
                "node": "execute_action",
                "actionName": "compare_groups" if bound else "analyze_projection",
            }],
        }],
    }


def test_binding_audit_accepts_explicit_high_level_action(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "audit.json"
    input_path.write_text(json.dumps(_artifact(bound=True)), encoding="utf-8")

    result = audit(input_path, output_path)

    assert result["status"] == "PASS"
    assert result["bindingPassCount"] == 1
    assert result["decisionExecutionAgreementCount"] == 1


def test_binding_audit_rejects_legacy_shared_tool_inference(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "audit.json"
    input_path.write_text(json.dumps(_artifact(bound=False)), encoding="utf-8")

    result = audit(input_path, output_path)

    assert result["status"] == "FAIL"
    assert result["bindingPassCount"] == 0
