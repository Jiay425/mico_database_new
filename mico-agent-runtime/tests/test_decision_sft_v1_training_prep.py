from __future__ import annotations

import json
from pathlib import Path

import pytest

from mico_agent_runtime.ports.decision_policy import SCIENTIFIC_POLICY_SYSTEM_PROMPT
from mico_agent_runtime.training.decision_sft_v1_preparation import (
    DEFAULT_DATA_DIR,
    DEFAULT_TOKENIZER_DIR,
    DatasetPreparationError,
    QwenAssetTokenizer,
    audit_old_sft_configuration,
    build_training_messages,
    evaluate_frozen_test_contract,
    fingerprint_frozen_dataset,
    render_training_example,
    validate_frozen_dataset,
    validate_model_record,
)


def _test_record() -> dict:
    return json.loads((DEFAULT_DATA_DIR / "reviewed_train.jsonl").read_text(encoding="utf-8").splitlines()[0])


def test_frozen_dataset_counts_and_fingerprint() -> None:
    freeze = validate_frozen_dataset(DEFAULT_DATA_DIR, inspect_test=False)
    assert {key: len(value) for key, value in freeze.model_records.items()} == {
        "train": 388,
        "validation": 65,
        "test": 0,
    }
    assert freeze.test_inspected is False
    assert fingerprint_frozen_dataset(DEFAULT_DATA_DIR, freeze)["training_started"] is False


def test_model_record_is_serving_shaped_and_closed() -> None:
    normalized = validate_model_record(_test_record())
    assert tuple(normalized["input"]["state"]) == (
        "task", "data_state", "analysis_state", "evidence_state", "progress", "action_space"
    )
    invalid = _test_record()
    invalid["input"]["state"]["next_action_hint"] = "compare_groups"
    with pytest.raises(DatasetPreparationError):
        validate_model_record(invalid)


def test_qwen_asset_tokenizer_and_assistant_mask() -> None:
    tokenizer = QwenAssetTokenizer(DEFAULT_TOKENIZER_DIR)
    assert tokenizer.encode("<|im_start|>assistant\n")[0] == 151644
    feature = render_training_example(_test_record(), tokenizer, max_length=1280)
    assert feature["target_tokens"] > 0
    assert all(value == -100 for value in feature["labels"][: feature["prompt_tokens"]])
    assert tokenizer.eos_token_id in feature["labels"][feature["prompt_tokens"] :]


def test_messages_use_exact_serving_prompt_and_no_metadata() -> None:
    messages = build_training_messages(_test_record())
    assert messages["system"] == SCIENTIFIC_POLICY_SYSTEM_PROMPT
    assert "metadata" not in messages["user"]
    assert "<think>\n\n</think>" in messages["prompt_text"]


def test_old_sft_audit_does_not_reuse_dpo_learning_rate() -> None:
    audit = audit_old_sft_configuration()
    assert audit["old_training_arguments_observed"]["learning_rate"] == 2e-4
    assert audit["historical_dpo_excluded"]["learning_rate_observed"] == 1e-6
    assert audit["historical_dpo_excluded"]["used_for_new_sft"] is False


def test_frozen_test_evaluator_is_blind_by_default() -> None:
    # The ordinary preparation/test suite must not open Frozen Test labels.
    # Post-training scoring is exercised only by the explicit evaluator path.
    freeze = validate_frozen_dataset(DEFAULT_DATA_DIR, inspect_test=False)
    schema = evaluate_frozen_test_contract(DEFAULT_DATA_DIR, freeze, inspect_test=False)
    assert schema["sample_count"] == 65
    assert schema["test_content_inspected"] is False
    assert freeze.test_inspected is False
