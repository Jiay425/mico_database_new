"""Repair a frozen DPO split to use the exact SFT-v5 input prompt contract.

This is an offline metadata-only repair.  It replaces only the DPO system
message with the unique system template from Decision SFT Freeze v2; state,
chosen/rejected completions, provenance, split membership, and all pair fields
must remain byte-equivalent to their source rows.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _sft_system(path: Path) -> str:
    rows = _rows(path)
    templates = {
        row["messages"][0]["content"]
        for row in rows
        if len(row.get("messages", [])) == 3 and row["messages"][0].get("role") == "system"
    }
    if len(templates) != 1 or len(templates) == 0:
        raise SystemExit("DPO_V4_SFT_SYSTEM_TEMPLATE_NOT_UNIQUE")
    return next(iter(templates))


def _repair(row: dict[str, Any], system: str) -> dict[str, Any]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 2 or [message.get("role") for message in prompt] != ["system", "user"]:
        raise SystemExit("DPO_V4_PROMPT_SHAPE_INVALID:" + str(row.get("id")))
    try:
        state = json.loads(prompt[1]["content"])["policy_state"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit("DPO_V4_PROMPT_STATE_INVALID:" + str(row.get("id"))) from exc
    fields = ("task_kind", "goal_code", "observation_flags", "history_actions", "candidate_actions", "state_summary")
    if any(state.get(field) != row.get(field) for field in fields):
        raise SystemExit("DPO_V4_PROMPT_STATE_MISMATCH:" + str(row.get("id")))
    repaired = copy.deepcopy(row)
    repaired["prompt"][0]["content"] = system
    if any(repaired[key] != row[key] for key in row if key != "prompt") or repaired["prompt"][1] != row["prompt"][1]:
        raise SystemExit("DPO_V4_REPAIR_MUTATED_NON_PROMPT_FIELD:" + str(row.get("id")))
    return repaired


def _manifest(source: Path, output: Path, sft_train: Path, system: str, train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schemaVersion": "p2j4-dpo-v4-freeze-manifest-v2",
        "status": "PROMPT_CONTRACT_REPAIRED_PENDING_AUDIT",
        "trainingStarted": False,
        "recordCount": len(train) + len(validation),
        "trainCount": len(train),
        "validationCount": len(validation),
        "sourceFreeze": str(source),
        "sourceTrainSha256": _sha(source / "train.jsonl"),
        "sourceValidationSha256": _sha(source / "validation.jsonl"),
        "sftPromptContract": {
            "sourceSftTrain": str(sft_train),
            "sourceSftTrainSha256": _sha(sft_train),
            "systemTemplateSha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "promptShape": ["system", "user"],
            "stateCarrier": "prompt[1].content.policy_state",
            "nonPromptPairFieldsChanged": False,
        },
        "files": {
            "train": str(output / "train.jsonl"),
            "validation": str(output / "validation.jsonl"),
            "all": str(output / "all.jsonl"),
            "trainSha256": _sha(output / "train.jsonl"),
            "validationSha256": _sha(output / "validation.jsonl"),
            "allSha256": _sha(output / "all.jsonl"),
        },
        "pairAudit": None,
        "leakageAudit": None,
        "referenceLogprobManifest": None,
        "a100Gate": "PAIR_AND_LEAKAGE_AUDIT_PLUS_REFERENCE_LOGPROB_AND_DRY_RUN_REQUIRED",
    }


def repair(source: Path, output: Path, sft_train: Path) -> None:
    system = _sft_system(sft_train)
    train = [_repair(row, system) for row in _rows(source / "train.jsonl")]
    validation = [_repair(row, system) for row in _rows(source / "validation.jsonl")]
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "train.jsonl", train)
    _write(output / "validation.jsonl", validation)
    _write(output / "all.jsonl", [*train, *validation])
    (output / "manifest.json").write_text(json.dumps(_manifest(source, output, sft_train, system, train, validation), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finalize(output: Path, pair_audit: Path, leakage_audit: Path) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair = json.loads(pair_audit.read_text(encoding="utf-8"))
    leakage = json.loads(leakage_audit.read_text(encoding="utf-8"))
    if pair.get("status") != "PASS" or leakage.get("status") != "PASS":
        raise SystemExit("DPO_V4_PROMPT_CONTRACT_REPAIR_AUDIT_FAILED")
    manifest["status"] = "FROZEN"
    manifest["pairAudit"] = {"path": str(pair_audit), "sha256": _sha(pair_audit), "status": pair.get("status")}
    manifest["leakageAudit"] = {"path": str(leakage_audit), "sha256": _sha(leakage_audit), "status": leakage.get("status")}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sft-train", type=Path)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--pair-audit", type=Path)
    parser.add_argument("--leakage-audit", type=Path)
    args = parser.parse_args()
    if args.finalize:
        if args.pair_audit is None or args.leakage_audit is None:
            raise SystemExit("DPO_V4_PROMPT_CONTRACT_FINALIZE_AUDITS_REQUIRED")
        finalize(args.output, args.pair_audit, args.leakage_audit)
    else:
        if args.source is None or args.sft_train is None:
            raise SystemExit("DPO_V4_PROMPT_CONTRACT_SOURCE_AND_SFT_REQUIRED")
        repair(args.source, args.output, args.sft_train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
