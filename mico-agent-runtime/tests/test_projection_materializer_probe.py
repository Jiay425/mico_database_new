from __future__ import annotations

import json

from scripts.run_projection_materializer_probe import (
    _audit_transport,
    _validate_projection_response,
)


def _valid_response() -> dict[str, object]:
    return {
        "actionId": "action-" + "0" * 32,
        "actionName": "analyze_projection",
        "rationale": "summarize the validated projection",
        "arguments": {
            "actionName": "analyze_projection",
            "observationId": "observation-" + "e" * 32,
            "analysisGoal": "summarize the validated projection",
        },
    }


def test_projection_probe_accepts_singular_dispatch_contract() -> None:
    valid, reason = _validate_projection_response(_valid_response())

    assert valid is True
    assert reason is None


def test_projection_probe_rejects_plural_observation_ids() -> None:
    payload = _valid_response()
    arguments = dict(payload["arguments"])
    arguments.pop("observationId")
    arguments["observationIds"] = ["observation-" + "e" * 32]
    payload["arguments"] = arguments

    valid, reason = _validate_projection_response(payload)

    assert valid is False
    assert reason == "plural_observation_ids_for_projection"


def test_projection_probe_separates_transport_and_model_contract_counts() -> None:
    transport = type(
        "Transport",
        (),
        {
            "records": [
                {
                    "request_index": 1,
                    "response_status": None,
                    "response": None,
                    "error": "RemoteProtocolError",
                    "request": {"body": {"messages": [{}, {}]}},
                },
                {
                    "request_index": 2,
                    "response_status": 200,
                    "response": {
                        "choices": [{
                            "message": {"content": json.dumps(_valid_response())}
                        }]
                    },
                    "error": None,
                    "request": {"body": {"messages": [{}, {}]}},
                },
            ]
        },
    )()

    audit = _audit_transport(transport)

    assert audit["transport_attempts"] == 2
    assert audit["model_responses"] == 1
    assert audit["contract_repairs"] == 0
    assert audit["first_model_response_valid"] is True
