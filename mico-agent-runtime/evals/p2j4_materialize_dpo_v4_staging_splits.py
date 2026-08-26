"""Materialize train/validation files from a reviewed-but-not-frozen staging file."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("DPO_V4_STAGING_SPLIT_OUTPUT_NOT_EMPTY")
    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "val"]
    train_states = {row["state_signature"] for row in train}
    validation_states = {row["state_signature"] for row in validation}
    if train_states & validation_states:
        raise SystemExit("DPO_V4_STAGING_SPLIT_STATE_LEAKAGE")
    train_families = {row["task_family"] for row in train}
    validation_families = {row["task_family"] for row in validation}
    if train_families & validation_families:
        raise SystemExit("DPO_V4_STAGING_SPLIT_FAMILY_LEAKAGE")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    train_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in train) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in validation) + "\n", encoding="utf-8")
    manifest = {
        "schemaVersion": "p2j4-dpo-v4-staging-split-manifest-v1",
        "status": "STAGING_REVIEW_REQUIRED",
        "trainingStarted": False,
        "sourceStaging": str(args.input),
        "sourceStagingSha256": _sha(args.input),
        "trainCount": len(train),
        "validationCount": len(validation),
        "trainStateCount": len(train_states),
        "validationStateCount": len(validation_states),
        "trainFamilyCount": len(train_families),
        "validationFamilyCount": len(validation_families),
        "splitCounts": dict(Counter(row.get("split") for row in rows)),
        "files": {
            "train": str(train_path),
            "validation": str(validation_path),
            "trainSha256": _sha(train_path),
            "validationSha256": _sha(validation_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "trainCount", "validationCount", "trainStateCount", "validationStateCount", "trainFamilyCount", "validationFamilyCount")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
