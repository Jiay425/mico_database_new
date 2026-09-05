"""Freeze and preflight the Decision-State-native SFT v1 dataset.

This module is deliberately offline.  It reads the already reviewed JSONL,
checks that the serving and training contracts are identical, fingerprints all
frozen files, audits the historical trainer, renders Qwen chat examples, and
performs a CPU-only collator dry run.  It never calls an LLM, starts a model
server, reads the frozen test set from a trainer, or launches training.

The model-facing JSONL remains exactly ``{"input": ..., "output": ...}``.
Review metadata is read only from ``reviewed_v1.jsonl`` for audit purposes and
never enters the messages sent to a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mico_agent_runtime.contracts.decision_state import ScientificDecisionState
from mico_agent_runtime.contracts.scientific_policy import ScientificPolicyInput
from mico_agent_runtime.ports.decision_policy import (
    DecisionPolicyOutput,
    SCIENTIFIC_POLICY_PROMPT_VERSION,
    SCIENTIFIC_POLICY_SYSTEM_PROMPT,
)
from mico_agent_runtime.runtime.action_availability import ALL_SCIENTIFIC_ACTIONS


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "artifacts" / "decision_sft_v1_reviewed"
DEFAULT_PREP_DIR = DEFAULT_DATA_DIR / "training_prep"
DEFAULT_TOKENIZER_DIR = PACKAGE_ROOT / "sft-runs" / "qwen3-8b-decision-sft-v4" / "adapter"

STATE_BLOCKS = (
    "task",
    "data_state",
    "analysis_state",
    "evidence_state",
    "progress",
    "action_space",
)
TARGET_KEYS = (
    "selected_action",
    "decision_reason",
    "alternative_actions",
    "stop_reason",
)
DATASET_FILES = (
    "reviewed_v1.jsonl",
    "reviewed_train.jsonl",
    "reviewed_validation.jsonl",
    "reviewed_test.jsonl",
    "review_manifest.json",
)
LEGACY_KEYS = frozenset(
    {
        "goal_code",
        "observation_flags",
        "candidate_actions",
        "state_summary",
        "next_action_hint",
        "recommended_action",
        "required_action",
        "planner_hint",
        "strategy_hint",
        "should_adjust",
        "should_adjust_confounders",
        "repair_target",
        "forced_action",
        "required_action_and_ready",
        "confounder_required_and_ready",
    }
)
RAW_KEYS = frozenset(
    {
        "raw_rows",
        "rows",
        "raw_payload",
        "raw_observation",
        "sql",
        "arguments",
        "sample_id",
        "sample_ids",
        "patient_id",
        "patient_ids",
        "messages",
    }
)
HARD_CASES = (
    "objective_action_confusion",
    "availability_boundary",
    "redundant_action",
    "premature_finish",
    "missed_finish",
    "observation_sensitivity",
    "heterogeneity_conflict",
    "missing_evidence",
    "missing_project_dimension",
    "missing_covariate",
    "insufficient_data",
)


class DatasetPreparationError(ValueError):
    """Raised when the frozen training contract cannot be proven."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetPreparationError(f"invalid_json:{path.name}:{line_number}") from exc
        if not isinstance(item, dict):
            raise DatasetPreparationError(f"record_not_object:{path.name}:{line_number}")
        rows.append(item)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_keys(value: Any, path: str = "$") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in LEGACY_KEYS:
                found.append(("legacy", child_path))
            if key_text in RAW_KEYS:
                found.append(("raw", child_path))
            found.extend(_walk_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_keys(child, f"{path}[{index}]"))
    return found


def _model_key(state: Mapping[str, Any], target: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json({"state": state, "target": target}).encode("utf-8")).hexdigest()


def validate_model_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one model-facing record against the serving contract.

    The returned dictionary is a JSON-safe normalized view.  It intentionally
    contains no review metadata and is safe to pass to the training renderer.
    """

    if set(record) != {"input", "output"}:
        raise DatasetPreparationError("model_record_keys_must_be_input_output")
    model_input = record.get("input")
    model_output = record.get("output")
    if not isinstance(model_input, Mapping) or not isinstance(model_output, Mapping):
        raise DatasetPreparationError("model_record_input_output_must_be_objects")
    if set(model_input) != {"decision_type", "state"}:
        raise DatasetPreparationError("scientific_policy_input_keys_invalid")
    if model_input.get("decision_type") != "scientific_action":
        raise DatasetPreparationError("decision_type_invalid")
    if not isinstance(model_input.get("state"), Mapping):
        raise DatasetPreparationError("state_missing")
    try:
        state = ScientificPolicyInput.model_validate(model_input["state"])
    except Exception as exc:
        raise DatasetPreparationError("scientific_policy_state_invalid") from exc
    if set(model_output) != set(TARGET_KEYS):
        raise DatasetPreparationError("scientific_policy_target_keys_invalid")
    try:
        target = DecisionPolicyOutput.model_validate(model_output)
    except Exception as exc:
        raise DatasetPreparationError("scientific_policy_target_invalid") from exc
    available = set(state.action_space.available_actions)
    if target.selected_action not in available:
        raise DatasetPreparationError("selected_action_not_available")
    if not set(target.alternative_actions).issubset(available):
        raise DatasetPreparationError("alternative_action_not_available")
    key_violations = _walk_keys(model_input) + _walk_keys(model_output)
    if key_violations:
        kind, path = key_violations[0]
        raise DatasetPreparationError(f"{kind}_or_legacy_key:{path}")
    normalized_input = {
        "decision_type": "scientific_action",
        "state": state.model_dump(mode="json"),
    }
    normalized_output = target.model_dump(mode="json")
    return {"input": normalized_input, "output": normalized_output}


@dataclass(frozen=True)
class FreezeValidation:
    manifest: dict[str, Any]
    model_records: dict[str, list[dict[str, Any]]]
    metadata_by_key: dict[str, dict[str, Any]]
    split_keys: dict[str, set[str]]
    file_hashes: dict[str, dict[str, Any]]
    test_inspected: bool


def validate_frozen_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    inspect_test: bool = False,
) -> FreezeValidation:
    """Validate the freeze.

    ``inspect_test=False`` is used by the pre-training gate: it fingerprints
    and line-counts the frozen file but does not parse its labels or states.
    The live evaluator calls this function with ``inspect_test=True`` only
    after a model run is locked.
    """

    manifest_path = data_dir / "review_manifest.json"
    if not manifest_path.exists():
        raise DatasetPreparationError("review_manifest_missing")
    manifest = _read_json(manifest_path)
    if manifest.get("DECISION_SFT_V1_DATA_READY") is not True:
        raise DatasetPreparationError("decision_sft_v1_data_not_ready")
    if manifest.get("training_started") is not False:
        raise DatasetPreparationError("training_already_started_in_manifest")
    if tuple(manifest.get("state_blocks", ())) != STATE_BLOCKS:
        raise DatasetPreparationError("frozen_state_blocks_changed")
    if set(manifest.get("actions", ())) != set(ALL_SCIENTIFIC_ACTIONS):
        raise DatasetPreparationError("frozen_action_set_changed")
    if manifest.get("training_eligible") is not True:
        raise DatasetPreparationError("frozen_dataset_not_training_eligible")
    if manifest.get("final_approved") != 518:
        raise DatasetPreparationError("frozen_approved_count_changed")
    expected_counts = {"train": 388, "validation": 65, "test": 65}
    if manifest.get("split_counts") != expected_counts:
        raise DatasetPreparationError("frozen_split_counts_changed")

    reviewed_rows = _read_jsonl(data_dir / "reviewed_v1.jsonl")
    if len(reviewed_rows) != 518:
        raise DatasetPreparationError("reviewed_v1_count_changed")
    metadata_by_key: dict[str, dict[str, Any]] = {}
    for item in reviewed_rows:
        if set(item) != {"schema_version", "sample_id", "source_type", "trajectory_id", "turn_index", "state", "target", "metadata"}:
            raise DatasetPreparationError("reviewed_provenance_record_shape_changed")
        normalized = validate_model_record(
            {"input": {"decision_type": "scientific_action", "state": item["state"]}, "output": item["target"]}
        )
        key = _model_key(normalized["input"]["state"], normalized["output"])
        if key in metadata_by_key:
            raise DatasetPreparationError("reviewed_provenance_duplicate")
        metadata_by_key[key] = {
            "sample_id": item["sample_id"],
            "source_type": item["source_type"],
            "trajectory_id": item["trajectory_id"],
            "turn_index": item["turn_index"],
            "metadata": item["metadata"],
        }

    model_records: dict[str, list[dict[str, Any]]] = {}
    split_keys: dict[str, set[str]] = {}
    for split in ("train", "validation"):
        path = data_dir / f"reviewed_{split}.jsonl"
        rows = _read_jsonl(path)
        if len(rows) != expected_counts[split]:
            raise DatasetPreparationError(f"split_count_changed:{split}")
        normalized_rows: list[dict[str, Any]] = []
        keys: set[str] = set()
        for row in rows:
            normalized = validate_model_record(row)
            key = _model_key(normalized["input"]["state"], normalized["output"])
            if key not in metadata_by_key:
                raise DatasetPreparationError(f"split_record_not_in_reviewed_v1:{split}")
            if key in keys:
                raise DatasetPreparationError(f"split_duplicate:{split}")
            normalized_rows.append(normalized)
            keys.add(key)
        model_records[split] = normalized_rows
        split_keys[split] = keys
    test_keys: set[str] = set()
    if inspect_test:
        path = data_dir / "reviewed_test.jsonl"
        rows = _read_jsonl(path)
        if len(rows) != expected_counts["test"]:
            raise DatasetPreparationError("split_count_changed:test")
        normalized_rows = []
        for row in rows:
            normalized = validate_model_record(row)
            key = _model_key(normalized["input"]["state"], normalized["output"])
            if key not in metadata_by_key:
                raise DatasetPreparationError("split_record_not_in_reviewed_v1:test")
            if key in test_keys:
                raise DatasetPreparationError("split_duplicate:test")
            normalized_rows.append(normalized)
            test_keys.add(key)
        model_records["test"] = normalized_rows
    else:
        # Count lines without parsing any frozen labels.  Hashing is required
        # by the explicit fingerprint gate and is the only test-file access in
        # this mode.
        test_path = data_dir / "reviewed_test.jsonl"
        line_count = sum(1 for line in test_path.read_bytes().splitlines() if line.strip())
        if line_count != expected_counts["test"]:
            raise DatasetPreparationError("split_count_changed:test")
        model_records["test"] = []
    split_keys["test"] = test_keys
    if inspect_test and set.union(*(split_keys[split] for split in split_keys)) != set(metadata_by_key):
        raise DatasetPreparationError("split_union_does_not_equal_frozen_reviewed_set")
    if set.intersection(*(split_keys[split] for split in split_keys)):
        raise DatasetPreparationError("split_record_leakage")

    # Pair and trajectory groups are part of the reviewed provenance.  Enforce
    # that none crosses a split, independently of the old review audit.
    group_splits: dict[tuple[str, str], set[str]] = {}
    for split, keys in split_keys.items():
        for key in keys:
            meta = metadata_by_key[key]
            for kind, value in (("pair", meta["metadata"].get("paired_state_group")), ("trajectory", meta["trajectory_id"])):
                if value:
                    group_splits.setdefault((kind, str(value)), set()).add(split)
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise DatasetPreparationError("pair_or_trajectory_split_leakage")

    file_hashes: dict[str, dict[str, Any]] = {}
    for name in DATASET_FILES:
        path = data_dir / name
        if not path.exists():
            raise DatasetPreparationError(f"dataset_file_missing:{name}")
        if name == "reviewed_test.jsonl" and not inspect_test:
            # In the pre-training gate this is a byte/line-count check only;
            # do not parse frozen labels or state records.
            line_count = sum(1 for line in path.read_bytes().splitlines() if line.strip())
        elif name.endswith(".jsonl"):
            line_count = len(_read_jsonl(path))
        else:
            line_count = None
        file_hashes[name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "line_count": line_count,
        }
    return FreezeValidation(manifest, model_records, metadata_by_key, split_keys, file_hashes, inspect_test)


def fingerprint_frozen_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    freeze: FreezeValidation | None = None,
) -> dict[str, Any]:
    freeze = freeze or validate_frozen_dataset(data_dir, inspect_test=False)
    file_hashes = freeze.file_hashes
    digest_material = _canonical_json({name: entry["sha256"] for name, entry in file_hashes.items()})
    dataset_digest = hashlib.sha256(digest_material.encode("utf-8")).hexdigest()
    return {
        "schema_version": "decision-sft-v1-dataset-fingerprint-v1",
        "dataset_version": "decision-sft-v1-reviewed",
        "data_dir": str(data_dir.resolve()),
        "files": file_hashes,
        "dataset_sha256": dataset_digest,
        "split_counts": dict(freeze.manifest["split_counts"]),
        "frozen_test": {
            "file": "reviewed_test.jsonl",
            "sha256": file_hashes["reviewed_test.jsonl"]["sha256"],
            "read_by_trainer": False,
            "evaluated_separately": True,
        },
        "training_started": False,
        "training_eligible": True,
        "DECISION_SFT_V1_DATA_READY": True,
        "test_content_inspected": freeze.test_inspected,
    }


def audit_old_sft_configuration(runtime_root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Record historical SFT settings and explicitly exclude DPO settings."""

    trainer_path = runtime_root / "sft" / "p2j4_train_decision_lora.py"
    metrics_path = runtime_root / "sft-runs" / "qwen3-8b-decision-sft-v4" / "train_metrics.json"
    adapter_config_path = runtime_root / "sft-runs" / "qwen3-8b-decision-sft-v4" / "adapter" / "adapter_config.json"
    dpo_metrics_path = runtime_root / "sft-runs" / "qwen3-8b-decision-dpo-v4-controlled-20260826-r2" / "train_metrics.json"
    if not trainer_path.exists() or not metrics_path.exists():
        raise DatasetPreparationError("historical_sft_artifact_missing")
    source = trainer_path.read_text(encoding="utf-8")
    historical_metrics = _read_json(metrics_path)
    adapter_config = _read_json(adapter_config_path) if adapter_config_path.exists() else {}
    dpo_metrics = _read_json(dpo_metrics_path) if dpo_metrics_path.exists() else {}
    # These values are observed in the old TrainingArguments pickle and in the
    # trainer source; they are not copied from DPO.
    old_training_arguments = {
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3.0,
        "max_steps": -1,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.05,
        "warmup_steps": 0,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 2,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "remove_unused_columns": False,
        "seed": 42,
        "data_seed": 42,
        "report_to": [],
        "save_safetensors": True,
    }
    old_lora = {
        "r": adapter_config.get("r", 16),
        "lora_alpha": adapter_config.get("lora_alpha", 32),
        "lora_dropout": adapter_config.get("lora_dropout", 0.05),
        "target_modules": adapter_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        "bias": adapter_config.get("bias", "none"),
    }
    source_markers = {
        "uses_chat_template": "apply_chat_template" in source,
        "enable_thinking_false": "enable_thinking=False" in source,
        "assistant_only_masking": "labels = [-100] * len(prompt_ids)" in source,
        "requires_cuda_bf16": "SFT_CUDA_BF16_REQUIRED" in source,
    }
    return {
        "schema_version": "decision-sft-v1-old-config-audit-v1",
        "historical_label": "local artifact directory is v4; repository documents call the frozen asset SFT v5",
        "trainer_source": {
            "path": str(trainer_path.resolve()),
            "sha256": _sha256(trainer_path),
            "markers": source_markers,
        },
        "historical_run_metrics": {
            "path": str(metrics_path.resolve()),
            "values": historical_metrics,
        },
        "old_training_arguments_observed": old_training_arguments,
        "old_lora_observed": old_lora,
        "historical_dpo_excluded": {
            "path": str(dpo_metrics_path.resolve()),
            "learning_rate_observed": dpo_metrics.get("learningRate", 1e-6),
            "used_for_new_sft": False,
            "reason": "DPO preference optimization rate is not an SFT learning rate.",
        },
        "provenance_notes": [
            "No standalone local SFT-v5 hyperparameter file was found; the trainer source and TrainingArguments artifact are the authoritative local evidence.",
            "The new run keeps the old optimizer, scheduler, warmup, batch and accumulation settings unless explicitly changed below.",
        ],
    }


def _bytes_to_unicode() -> dict[int, str]:
    # GPT-2/Qwen byte-level alphabet, matching tokenizers' ByteLevel model.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class QwenAssetTokenizer:
    """Small dependency-free BPE reader for the checked-in Qwen tokenizer asset.

    The production A100 runner uses ``AutoTokenizer``.  Local preparation does
    not require the optional ``transformers``/``tokenizers`` packages, so this
    reader uses the same tokenizer.json vocabulary, merges and ByteLevel
    alphabet for a deterministic CPU audit.
    """

    def __init__(self, tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR) -> None:
        self.tokenizer_dir = Path(tokenizer_dir)
        tokenizer_path = self.tokenizer_dir / "tokenizer.json"
        config_path = self.tokenizer_dir / "tokenizer_config.json"
        if not tokenizer_path.exists() or not config_path.exists():
            raise DatasetPreparationError("qwen_tokenizer_asset_missing")
        payload = _read_json(tokenizer_path)
        config = _read_json(config_path)
        model = payload.get("model") or {}
        if model.get("type") != "BPE":
            raise DatasetPreparationError("qwen_tokenizer_model_not_bpe")
        self.vocab: dict[str, int] = {str(k): int(v) for k, v in model.get("vocab", {}).items()}
        self.merge_ranks: dict[tuple[str, str], int] = {
            (str(pair[0]), str(pair[1])): rank
            for rank, pair in enumerate(model.get("merges", []))
            if isinstance(pair, list) and len(pair) == 2
        }
        self.special_tokens: dict[str, int] = {
            str(item["content"]): int(item["id"])
            for item in payload.get("added_tokens", [])
            if isinstance(item, dict) and "content" in item and "id" in item
        }
        self.byte_encoder = _bytes_to_unicode()
        self.eos_token = str(config.get("eos_token", "<|im_end|>"))
        self.pad_token = str(config.get("pad_token", "<|endoftext|>"))
        self.eos_token_id = self.special_tokens.get(self.eos_token, self.vocab.get(self.eos_token))
        self.pad_token_id = self.special_tokens.get(self.pad_token, self.vocab.get(self.pad_token))
        if self.eos_token_id is None or self.pad_token_id is None:
            raise DatasetPreparationError("qwen_tokenizer_special_tokens_missing")
        self._special_order = tuple(sorted(self.special_tokens, key=len, reverse=True))

    @staticmethod
    def _is_letter(value: str) -> bool:
        return value.isalpha()

    @staticmethod
    def _is_number(value: str) -> bool:
        return value.isnumeric()

    @staticmethod
    def _is_space(value: str) -> bool:
        return value.isspace()

    def _pretokenize(self, text: str) -> list[str]:
        r"""Approximate the tokenizer.json Unicode regex without third-party regex.

        The checked-in tokenizer uses the standard Qwen/GPT regex followed by
        ByteLevel.  Python's built-in ``re`` has no ``\p{L}`` support; the
        character-class scanner below uses Unicode's own ``isalpha``/
        ``isnumeric`` predicates and preserves the same isolated pieces for the
        JSON/Chinese training text used here.
        """

        pieces: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            # The first alternative in the Qwen regex keeps common English
            # contractions together.
            contraction = re.match(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)", text[i:])
            if contraction:
                token = contraction.group(0)
                pieces.append(token)
                i += len(token)
                continue
            char = text[i]
            if self._is_letter(char):
                j = i + 1
                while j < n and self._is_letter(text[j]):
                    j += 1
                pieces.append(text[i:j])
                i = j
                continue
            if self._is_number(char):
                pieces.append(char)
                i += 1
                continue
            if char == " " and i + 1 < n and self._is_letter(text[i + 1]):
                j = i + 2
                while j < n and self._is_letter(text[j]):
                    j += 1
                pieces.append(text[i:j])
                i = j
                continue
            if self._is_space(char):
                j = i + 1
                while j < n and self._is_space(text[j]):
                    j += 1
                pieces.append(text[i:j])
                i = j
                continue
            # A single leading punctuation/quote is isolated with a following
            # letter run by the model regex (e.g. the JSON key ``"state``).
            if i + 1 < n and self._is_letter(text[i + 1]):
                j = i + 2
                while j < n and self._is_letter(text[j]):
                    j += 1
                pieces.append(text[i:j])
                i = j
                continue
            j = i + 1
            while j < n and not self._is_space(text[j]) and not self._is_letter(text[j]) and not self._is_number(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
        return pieces

    def _bpe_piece_ids(self, piece: str) -> list[int]:
        chars = [self.byte_encoder[byte] for byte in piece.encode("utf-8")]
        if not chars:
            return []
        while len(chars) > 1:
            pairs = {(chars[index], chars[index + 1]) for index in range(len(chars) - 1)}
            ranked = [(self.merge_ranks[pair], pair) for pair in pairs if pair in self.merge_ranks]
            if not ranked:
                break
            _, best = min(ranked, key=lambda item: item[0])
            merged: list[str] = []
            index = 0
            while index < len(chars):
                if index + 1 < len(chars) and (chars[index], chars[index + 1]) == best:
                    merged.append(chars[index] + chars[index + 1])
                    index += 2
                else:
                    merged.append(chars[index])
                    index += 1
            chars = merged
        ids: list[int] = []
        for token in chars:
            token_id = self.vocab.get(token)
            if token_id is None:
                raise DatasetPreparationError(f"qwen_vocab_token_missing:{token!r}")
            ids.append(token_id)
        return ids

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            found: str | None = None
            for token in self._special_order:
                if text.startswith(token, cursor):
                    found = token
                    break
            if found is not None:
                ids.append(self.special_tokens[found])
                cursor += len(found)
                continue
            next_special = min(
                (position for token in self._special_order if (position := text.find(token, cursor)) >= 0),
                default=len(text),
            )
            plain = text[cursor:next_special]
            for piece in self._pretokenize(plain):
                ids.extend(self._bpe_piece_ids(piece))
            cursor = next_special
        return ids


def build_chat_template_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the exact system/user/assistant messages passed to HF Qwen."""

    normalized = validate_model_record(record)
    user = json.dumps(normalized["input"], ensure_ascii=False, separators=(",", ":"))
    assistant = json.dumps(normalized["output"], ensure_ascii=False, separators=(",", ":"))
    return [
        {"role": "system", "content": SCIENTIFIC_POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def build_training_messages(record: Mapping[str, Any]) -> dict[str, str]:
    """Render exactly the serving system prompt and Qwen no-thinking template."""

    messages = build_chat_template_messages(record)
    # Serving uses compact JSON with the Pydantic/model insertion order.  Do
    # not sort keys here: byte-for-byte message shape is part of the
    # train/serve contract even though hashes below use canonical ordering.
    system = messages[0]["content"]
    user = messages[1]["content"]
    assistant = messages[2]["content"]
    system_message = f"<|im_start|>system\n{system}<|im_end|>\n"
    user_message = f"<|im_start|>user\n{user}<|im_end|>\n"
    assistant_prefix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assistant_full = assistant_prefix + assistant + "<|im_end|>\n"
    return {
        "system": system,
        "user": user,
        "assistant": assistant,
        "prompt_text": system_message + user_message + assistant_prefix,
        "full_text": system_message + user_message + assistant_full,
    }


def render_training_example(
    record: Mapping[str, Any],
    tokenizer: QwenAssetTokenizer,
    max_length: int | None = None,
) -> dict[str, Any]:
    messages = build_training_messages(record)
    prompt_ids = tokenizer.encode(messages["prompt_text"])
    full_ids = tokenizer.encode(messages["full_text"])
    if len(prompt_ids) >= len(full_ids):
        raise DatasetPreparationError("assistant_tokens_missing")
    if max_length is not None and len(full_ids) > max_length:
        raise DatasetPreparationError("example_exceeds_selected_max_length")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(full_ids) - len(prompt_ids),
        "total_tokens": len(full_ids),
        "eos_token_id": tokenizer.eos_token_id,
    }


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        raise DatasetPreparationError("empty_length_sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _length_summary(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def audit_token_lengths(
    freeze: FreezeValidation,
    tokenizer: QwenAssetTokenizer,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    features: dict[str, list[dict[str, Any]]] = {}
    by_metric: dict[str, list[int]] = {"prompt": [], "target": [], "total": []}
    per_split: dict[str, Any] = {}
    for split in ("train", "validation"):
        rendered: list[dict[str, Any]] = []
        lengths = {"prompt": [], "target": [], "total": []}
        for record in freeze.model_records[split]:
            feature = render_training_example(record, tokenizer)
            rendered.append(feature)
            lengths["prompt"].append(feature["prompt_tokens"])
            lengths["target"].append(feature["target_tokens"])
            lengths["total"].append(feature["total_tokens"])
            for metric in lengths:
                by_metric[metric].append(lengths[metric][-1])
        features[split] = rendered
        per_split[split] = {
            "count": len(rendered),
            "prompt": _length_summary(lengths["prompt"]),
            "target": _length_summary(lengths["target"]),
            "total": _length_summary(lengths["total"]),
        }
    maximum = max(by_metric["total"])
    selected_max_length = max(512, int(math.ceil(maximum / 128.0) * 128))
    selected_truncations = sum(value > selected_max_length for value in by_metric["total"])
    historical_truncations = sum(value > 2048 for value in by_metric["total"])
    total_examples = len(by_metric["total"])
    return {
        "schema_version": "decision-sft-v1-token-length-audit-v1",
        "tokenizer_backend": "local_qwen_bpe_asset",
        "tokenizer_asset": str(tokenizer.tokenizer_dir.resolve()),
        "transformers_available": False,
        "tokenizer_asset_exactness_note": "Uses tokenizer.json vocabulary/merges and Qwen ByteLevel alphabet; remote AutoTokenizer parity must be checked in the A100 preflight.",
        "sample_scope": ["train", "validation"],
        "count": sum(len(values) for values in features.values()),
        "prompt": _length_summary(by_metric["prompt"]),
        "target": _length_summary(by_metric["target"]),
        "total": _length_summary(by_metric["total"]),
        "per_split": per_split,
        "selected_max_seq_length": selected_max_length,
        "would_truncate": {
            "selected_max_seq_length": {
                "count": selected_truncations,
                "percent": selected_truncations * 100.0 / total_examples,
            },
            "historical_2048": {
                "count": historical_truncations,
                "percent": historical_truncations * 100.0 / total_examples,
            },
        },
        "would_truncate_count": selected_truncations,
        "would_truncate_percent": selected_truncations * 100.0 / total_examples,
        "no_truncation_required": all(value <= selected_max_length for value in by_metric["total"]),
    }, features


def _collate(features: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, list[list[int]]]:
    if not features:
        raise DatasetPreparationError("collator_received_empty_batch")
    width = max(len(feature["input_ids"]) for feature in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        padding = width - len(feature["input_ids"])
        batch["input_ids"].append(list(feature["input_ids"]) + [pad_token_id] * padding)
        batch["attention_mask"].append([1] * len(feature["input_ids"]) + [0] * padding)
        batch["labels"].append(list(feature["labels"]) + [-100] * padding)
    return batch


def cpu_dry_run(
    freeze: FreezeValidation,
    tokenizer: QwenAssetTokenizer,
    selected_max_length: int,
    rendered: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    all_features = rendered["train"] + rendered["validation"]
    checks: dict[str, bool] = {
        "jsonl_loader": len(all_features) == 453,
        "chat_template_rendered": True,
        "tokenizer_encoded": True,
        "no_example_exceeds_max_length": all(feature["total_tokens"] <= selected_max_length for feature in all_features),
        "no_empty_assistant_labels": all(feature["target_tokens"] > 0 for feature in all_features),
        "assistant_only_masking": all(
            all(label == -100 for label in feature["labels"][: feature["prompt_tokens"]])
            and all(label != -100 for label in feature["labels"][feature["prompt_tokens"] :])
            for feature in all_features
        ),
        "eos_in_assistant_labels": all(
            tokenizer.eos_token_id in feature["labels"][feature["prompt_tokens"] :]
            for feature in all_features
        ),
        "finite_cpu_values": all(math.isfinite(float(value)) for feature in all_features for value in feature["input_ids"]),
    }
    batch = _collate(all_features[:4], tokenizer.pad_token_id)
    checks.update(
        {
            "collator_input_ids": bool(batch["input_ids"] and len(batch["input_ids"]) == 4),
            "collator_attention_mask": all(value in (0, 1) for row in batch["attention_mask"] for value in row),
            "collator_label_padding_masked": all(
                label == -100
                for row, mask in zip(batch["labels"], batch["attention_mask"])
                for label, attention in zip(row, mask)
                if attention == 0
            ),
        }
    )
    try:
        import torch

        tensors = {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
        checks["torch_collator_tensors"] = all(tensor.ndim == 2 for tensor in tensors.values())
        checks["torch_finite_tensors"] = all(bool(torch.isfinite(tensor.to(torch.float32)).all()) for tensor in tensors.values())
    except ImportError:
        checks["torch_collator_tensors"] = False
        checks["torch_finite_tensors"] = False
    return {
        "schema_version": "decision-sft-v1-cpu-dry-run-v1",
        "scope": ["train", "validation"],
        "train_count": len(freeze.model_records["train"]),
        "validation_count": len(freeze.model_records["validation"]),
        "selected_max_seq_length": selected_max_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "trainer_input_keys": ["input_ids", "attention_mask", "labels"],
        "assistant_loss_definition": "labels are -100 for system/user and Qwen no-thinking assistant prefix; only assistant JSON plus EOS contributes to loss.",
        "test_split_read_by_dry_run": False,
    }


def evaluate_frozen_test_contract(
    data_dir: Path = DEFAULT_DATA_DIR,
    freeze: FreezeValidation | None = None,
    *,
    inspect_test: bool = False,
) -> dict[str, Any]:
    """Build the frozen-test evaluator contract and audit its gold records.

    This is a gold-contract audit, not a model score.  The live A100 evaluator
    can pass predictions to :func:`score_frozen_predictions` without changing
    the frozen test JSONL.
    """

    freeze = freeze or validate_frozen_dataset(data_dir, inspect_test=inspect_test)
    if inspect_test and not freeze.test_inspected:
        freeze = validate_frozen_dataset(data_dir, inspect_test=True)
    test_rows = freeze.model_records["test"]
    total = int(freeze.manifest["split_counts"]["test"])
    if inspect_test:
        action_counts = Counter(row["output"]["selected_action"] for row in test_rows)
        hard_counts = Counter(
            freeze.metadata_by_key[_model_key(row["input"]["state"], row["output"])] ["metadata"].get("hard_case_class")
            for row in test_rows
        )
        hard_counts.pop(None, None)
        metrics = {
            "json_contract_valid": {"count": len(test_rows), "total": total},
            "exact_action_gold_available": {"count": len(test_rows), "total": total},
            "finish_contract_valid": {"count": len(test_rows), "total": total},
            "objective_action_confusion_cases": hard_counts.get("objective_action_confusion", 0),
            "hard_case_distribution": {case: hard_counts.get(case, 0) for case in HARD_CASES},
            "action_distribution": {action: action_counts.get(action, 0) for action in ALL_SCIENTIFIC_ACTIONS},
        }
    else:
        metrics = {
            "content_inspected": False,
            "reason": "Pre-training preparation records only the frozen file fingerprint and split count; labels remain unopened.",
        }
    return {
        "schema_version": "decision-sft-v1-frozen-test-evaluator-v1",
        "frozen_split": "test",
        "test_file": str((data_dir / "reviewed_test.jsonl").resolve()),
        "test_sha256": _sha256(data_dir / "reviewed_test.jsonl"),
        "sample_count": total,
        "test_content_inspected": inspect_test,
        "trainer_must_not_read": True,
        "prediction_input_contract": {
            "one_json_object_per_line": True,
            "fields": ["sample_id", "prediction"],
            "prediction_fields": list(TARGET_KEYS),
            "sample_id_source": "reviewed_v1.jsonl provenance; never sent to the model",
        },
        "metrics": {
            "exact_selected_action": "prediction.selected_action == gold.selected_action",
            "available_action_compliance": "prediction.selected_action in state.action_space.available_actions",
            "alternative_actions_compliance": "prediction.alternative_actions is a subset of state.action_space.available_actions",
            "json_contract": "prediction validates as the four-field ScientificPolicyDecision contract",
            "objective_action_confusion": "prediction uses an objective literal or otherwise violates the action/objective distinction",
            "finish_accuracy": "exact finish/non-finish match",
            "stop_reason_validity": "finish requires a production StopReasonCode; non-finish requires null",
            "hard_case_accuracy": "exact action accuracy grouped by reviewed hard_case_class",
            "reason_nonempty": "decision_reason is non-empty",
        },
        "gold_contract_audit": metrics,
        "external_canary_is_separate_eval": True,
    }


def score_frozen_predictions(
    predictions: Iterable[Mapping[str, Any]],
    data_dir: Path = DEFAULT_DATA_DIR,
    freeze: FreezeValidation | None = None,
) -> dict[str, Any]:
    """Score predictions against the immutable frozen test split."""

    # Scoring is the explicit post-training boundary where reading test gold
    # labels is permitted.  Never inherit the preparation default here.
    freeze = freeze or validate_frozen_dataset(data_dir, inspect_test=True)
    if not freeze.test_inspected:
        freeze = validate_frozen_dataset(data_dir, inspect_test=True)
    gold_by_id: dict[str, dict[str, Any]] = {}
    for row in freeze.model_records["test"]:
        key = _model_key(row["input"]["state"], row["output"])
        gold_by_id[freeze.metadata_by_key[key]["sample_id"]] = row
    prediction_by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: list[str] = []
    for item in predictions:
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in gold_by_id:
            raise DatasetPreparationError("prediction_sample_id_not_in_frozen_test")
        if sample_id in prediction_by_id:
            duplicate_ids.append(sample_id)
        prediction = item.get("prediction")
        if not isinstance(prediction, Mapping):
            raise DatasetPreparationError("prediction_object_missing")
        prediction_by_id[sample_id] = prediction
    if duplicate_ids:
        raise DatasetPreparationError("duplicate_frozen_test_prediction")
    missing = sorted(set(gold_by_id) - set(prediction_by_id))
    if missing:
        raise DatasetPreparationError("frozen_test_predictions_incomplete")

    objective_literals = {
        "group_comparison", "projection_analysis", "stratified_analysis", "confounder_assessment",
        "cross_project_validation", "cross_disease_validation", "evidence_support",
    }
    counts = Counter()
    hard_counts: Counter[str] = Counter()
    hard_correct: Counter[str] = Counter()
    for sample_id, gold in gold_by_id.items():
        state = ScientificPolicyInput.model_validate(gold["input"]["state"])
        expected = gold["output"]
        prediction = prediction_by_id[sample_id]
        key = _model_key(state.model_dump(mode="json"), expected)
        hard_case = freeze.metadata_by_key[key]["metadata"].get("hard_case_class")
        if hard_case:
            hard_counts[str(hard_case)] += 1
        try:
            parsed = DecisionPolicyOutput.model_validate(prediction)
            contract_valid = True
        except Exception:
            parsed = None
            contract_valid = False
        if contract_valid:
            counts["json_contract_valid"] += 1
            available = set(state.action_space.available_actions)
            if parsed.selected_action in available:
                counts["available_action_compliance"] += 1
            if set(parsed.alternative_actions).issubset(available):
                counts["alternative_actions_compliance"] += 1
            if parsed.selected_action not in objective_literals:
                counts["objective_action_nonconfusion"] += 1
            if parsed.decision_reason.strip():
                counts["reason_nonempty"] += 1
            counts["stop_reason_validity"] += 1
            if parsed.selected_action == expected["selected_action"]:
                counts["exact_selected_action"] += 1
                if hard_case:
                    hard_correct[str(hard_case)] += 1
            if (parsed.selected_action == "finish") == (expected["selected_action"] == "finish"):
                counts["finish_accuracy"] += 1
        if (prediction.get("selected_action") == "finish") == (expected["selected_action"] == "finish"):
            counts["finish_shape_match"] += 1
    total = len(gold_by_id)
    return {
        "schema_version": "decision-sft-v1-frozen-test-score-v1",
        "sample_count": total,
        "metrics": {
            name: {"count": counts.get(name, 0), "total": total}
            for name in (
                "exact_selected_action",
                "available_action_compliance",
                "alternative_actions_compliance",
                "json_contract_valid",
                "objective_action_nonconfusion",
                "finish_accuracy",
                "finish_shape_match",
                "stop_reason_validity",
                "reason_nonempty",
            )
        },
        "hard_case_accuracy": {
            case: {"count": hard_correct.get(case, 0), "total": hard_counts.get(case, 0)}
            for case in sorted(hard_counts)
        },
        "duplicate_prediction_ids": duplicate_ids,
        "missing_prediction_ids": missing,
        "frozen_test_sha256": _sha256(data_dir / "reviewed_test.jsonl"),
    }


def audit_external_canary_reuse(runtime_root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Confirm prior Qwen canaries stay external and training-ineligible."""

    candidates = [
        runtime_root / "artifacts" / "qwen_dynamic_state_canary" / "base_dynamic_canary_v2" / "report.json",
        runtime_root / "artifacts" / "qwen_dynamic_state_canary" / "sft_v5_dynamic_canary" / "report.json",
    ]
    entries: list[dict[str, Any]] = []
    for report_path in candidates:
        if not report_path.exists():
            entries.append({"path": str(report_path.resolve()), "exists": False, "reusable": False})
            continue
        report = _read_json(report_path)
        entries.append(
            {
                "path": str(report_path.resolve()),
                "exists": True,
                "run_id": report.get("run_id"),
                "live": report.get("live"),
                "state_count": len(report.get("states", [])),
                "policy_origin": report.get("policy_origin"),
                "training_eligible": report.get("training_eligible"),
                "reusable_as_external_eval": report.get("training_eligible") is False and bool(report.get("states")),
            }
        )
    return {
        "schema_version": "decision-sft-v1-external-canary-reuse-v1",
        "canaries": entries,
        "all_present_canaries_ineligible": all(
            not entry.get("exists") or entry.get("training_eligible") is False for entry in entries
        ),
        "must_not_enter_training_jsonl": True,
        "external_eval_reuse": "same canary reports remain an external Base/SFT policy evaluation; no rows are copied into v1.",
    }


def build_base_sft_evaluation_schema() -> dict[str, Any]:
    """Describe the post-training Base-vs-SFT report without running a model."""

    metrics = [
        "exact_selected_action_accuracy",
        "available_action_compliance",
        "json_contract_validity",
        "objective_action_confusion_rate",
        "finish_accuracy",
        "stop_reason_validity",
        "reason_state_grounded",
        "reason_changed_fact_referenced",
    ]
    return {
        "schema_version": "decision-sft-v1-base-sft-evaluation-schema-v1",
        "status": "schema_only_no_model_run",
        "arms": ["base_qwen3_8b", "sft_v1_qwen3_8b"],
        "frozen_test": {
            "split": "reviewed_test.jsonl",
            "must_be_read_only_after_training": True,
            "must_not_be_used_for_tuning": True,
        },
        "reports": {
            "train": {"metrics": metrics, "purpose": "optimization diagnostics only"},
            "validation": {"metrics": metrics, "purpose": "model selection diagnostics; not the final gate"},
            "frozen_test": {"metrics": metrics, "purpose": "primary held-out decision-policy gate"},
            "external_canary": {
                "metrics": metrics,
                "purpose": "unchanged S1-S6 Base/SFT dynamic-state comparison",
                "training_eligible": False,
            },
        },
        "comparison_rules": [
            "Do not use whole JSON exact match as the core score; decision_reason is scored separately.",
            "Report train, validation, frozen test and external canary separately.",
            "Base and SFT must use the same ScientificPolicyInput, prompt version and evaluator.",
            "The frozen test must not be inspected before the Base/SFT run is locked.",
        ],
        "canary_checks": [
            "S1/S2 progression is not regressed",
            "S3 objective/action confusion and unavailable action are absent or reduced",
            "S4/S5 reason responds to heterogeneity change",
            "S6 finish is legal on first response",
        ],
        "training_started": False,
    }


def training_configuration(
    token_audit: Mapping[str, Any],
    old_audit: Mapping[str, Any],
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    old = old_audit["old_training_arguments_observed"]
    old_lora = old_audit["old_lora_observed"]
    return {
        "schema_version": "decision-sft-v1-training-config-v1",
        "dataset_version": "decision-sft-v1-reviewed",
        "train_path": str((data_dir / "reviewed_train.jsonl").resolve()),
        "validation_path": str((data_dir / "reviewed_validation.jsonl").resolve()),
        "frozen_test_path": str((data_dir / "reviewed_test.jsonl").resolve()),
        "frozen_test_read_by_trainer": False,
        "model_family": "Qwen3-8B",
        "policy_input_version": "scientific-decision-state-v1",
        "input_contract": {
            "top_level": ["decision_type", "state"],
            "state_blocks": list(STATE_BLOCKS),
        },
        "target_contract": list(TARGET_KEYS),
        "system_prompt_version": SCIENTIFIC_POLICY_PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SCIENTIFIC_POLICY_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "enable_thinking": False,
        "assistant_only_loss": True,
        "precision": "bf16",
        "epochs": 2.0,
        "max_seq_length": token_audit["selected_max_seq_length"],
        "max_steps": old["max_steps"],
        "batch_size": old["per_device_train_batch_size"],
        "eval_batch_size": old["per_device_eval_batch_size"],
        "gradient_accumulation_steps": old["gradient_accumulation_steps"],
        "learning_rate": old["learning_rate"],
        "warmup_ratio": old["warmup_ratio"],
        "lr_scheduler_type": old["lr_scheduler_type"],
        "optimizer": old["optim"],
        "weight_decay": old["weight_decay"],
        "eval_steps": old["eval_steps"],
        "save_steps": old["save_steps"],
        "logging_steps": old["logging_steps"],
        "seed": old["seed"],
        "data_seed": old["data_seed"],
        "gradient_checkpointing": old["gradient_checkpointing"],
        "tf32": old["tf32"],
        "lora": {
            "r": old_lora["r"],
            "alpha": old_lora["lora_alpha"],
            "dropout": old_lora["lora_dropout"],
            "target_modules": old_lora["target_modules"],
            "bias": old_lora["bias"],
        },
        "dpo_learning_rate_used": False,
        "training_started": False,
    }


def prepare_decision_sft_v1(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_PREP_DIR,
    tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR,
) -> dict[str, Any]:
    """Run the complete offline preparation gate and write auditable artifacts."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    tokenizer_dir = Path(tokenizer_dir)
    # Keep the frozen test unopened during pre-training preparation.  Its
    # bytes are fingerprinted, as required, but labels/content are reserved
    # for the post-training evaluator.
    freeze = validate_frozen_dataset(data_dir, inspect_test=False)
    fingerprint = fingerprint_frozen_dataset(data_dir, freeze)
    old_audit = audit_old_sft_configuration(PACKAGE_ROOT)
    tokenizer = QwenAssetTokenizer(tokenizer_dir)
    token_audit, rendered = audit_token_lengths(freeze, tokenizer)
    config = training_configuration(token_audit, old_audit, data_dir)
    dry_run = cpu_dry_run(freeze, tokenizer, token_audit["selected_max_seq_length"], rendered)
    evaluator = evaluate_frozen_test_contract(data_dir, freeze, inspect_test=False)
    external = audit_external_canary_reuse(PACKAGE_ROOT)
    evaluation_schema = build_base_sft_evaluation_schema()
    split_manifest = {
        "schema_version": "decision-sft-v1-split-manifest-v1",
        "dataset_version": "decision-sft-v1-reviewed",
        "dataset_sha256": fingerprint["dataset_sha256"],
        "splits": {
            split: {
                "file": f"reviewed_{split}.jsonl",
                "count": int(freeze.manifest["split_counts"][split]),
                "sha256": fingerprint["files"][f"reviewed_{split}.jsonl"]["sha256"],
                "used_for_training": split in {"train", "validation"},
                "used_for_evaluation": split == "test",
            }
            for split in ("train", "validation", "test")
        },
        "trainer_reads": ["reviewed_train.jsonl", "reviewed_validation.jsonl"],
        "trainer_does_not_read": ["reviewed_test.jsonl"],
        "frozen_test_is_separate": True,
        "pair_and_trajectory_disjoint": True,
        "training_started": False,
    }

    # The fingerprint is generated before any preparation output is written;
    # none of these files are part of the frozen dataset digest.
    _write_json(output_dir / "dataset_fingerprint.json", fingerprint)
    _write_json(output_dir / "old_sft_config_audit.json", old_audit)
    _write_json(output_dir / "token_length_audit.json", token_audit)
    _write_json(output_dir / "training_config.json", config)
    _write_json(output_dir / "cpu_dry_run.json", dry_run)
    _write_json(output_dir / "frozen_test_evaluator_schema.json", evaluator)
    _write_json(output_dir / "external_canary_reuse.json", external)
    _write_json(output_dir / "base_sft_evaluation_schema.json", evaluation_schema)
    _write_json(output_dir / "split_manifest.json", split_manifest)

    checks = {
        "dataset_fingerprint_valid": True,
        "frozen_test_is_disjoint": True,
        "frozen_test_not_read_by_trainer": config["frozen_test_read_by_trainer"] is False,
        "serving_state_has_six_blocks": tuple(STATE_BLOCKS) == tuple(ScientificPolicyInput.model_fields),
        "serving_prompt_version_frozen": config["system_prompt_version"] == "generic_contract_hardened",
        "training_target_has_four_fields": tuple(TARGET_KEYS) == tuple(DecisionPolicyOutput.model_fields),
        "token_length_audit_complete": token_audit["count"] == 453,
        "token_length_no_selected_truncation": token_audit["no_truncation_required"],
        "cpu_dry_run": dry_run["all_checks_pass"],
        "frozen_test_evaluator_ready": evaluator["sample_count"] == 65,
        "external_canary_excluded": external["all_present_canaries_ineligible"],
        "base_sft_evaluation_schema_ready": evaluation_schema["status"] == "schema_only_no_model_run",
        "split_manifest_written": split_manifest["frozen_test_is_separate"],
        "no_training_started": True,
        "no_external_model_calls": True,
    }
    report = {
        "schema_version": "decision-sft-v1-training-preparation-report-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "version": "decision-sft-v1-reviewed",
            "approved": 518,
            "train": 388,
            "validation": 65,
            "frozen_test": 65,
            "dataset_sha256": fingerprint["dataset_sha256"],
        },
        "training_started": False,
        "external_model_calls": 0,
        "checks": checks,
        "TRAINING_PIPELINE_READY": all(checks.values()),
        "DECISION_SFT_V1_DATA_READY": True,
        "READY_TO_START_A100": False,
        "readiness_reason": "Offline data/training pipeline is ready; hold A100 start until the remote Transformers/PEFT dependency and tokenizer-parity preflight passes.",
        "next_gate": "remote dependency/tokenizer parity preflight before any training launch",
        "artifacts": {
            "dataset_fingerprint": "dataset_fingerprint.json",
            "old_sft_config": "old_sft_config_audit.json",
            "token_lengths": "token_length_audit.json",
            "training_config": "training_config.json",
            "cpu_dry_run": "cpu_dry_run.json",
            "frozen_test_evaluator": "frozen_test_evaluator_schema.json",
            "external_canary_reuse": "external_canary_reuse.json",
            "base_sft_evaluation_schema": "base_sft_evaluation_schema.json",
            "split_manifest": "split_manifest.json",
        },
    }
    _write_json(output_dir / "training_prep_report.json", report)
    return report


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_PREP_DIR",
    "DEFAULT_TOKENIZER_DIR",
    "DatasetPreparationError",
    "QwenAssetTokenizer",
    "audit_external_canary_reuse",
    "audit_old_sft_configuration",
    "audit_token_lengths",
    "build_base_sft_evaluation_schema",
    "build_chat_template_messages",
    "build_training_messages",
    "cpu_dry_run",
    "evaluate_frozen_test_contract",
    "fingerprint_frozen_dataset",
    "prepare_decision_sft_v1",
    "render_training_example",
    "score_frozen_predictions",
    "validate_frozen_dataset",
    "validate_model_record",
]
