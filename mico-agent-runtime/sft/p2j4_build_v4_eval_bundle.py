"""Build one combined, non-training evaluation bundle for Decision SFT v4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(test_file: Path, ood_file: Path, repair_file: Path, output_dir: Path) -> dict[str, Any]:
    groups = {
        "test70": _read_jsonl(test_file),
        "ood_v2": _read_jsonl(ood_file),
        "residual_repair": _read_jsonl(repair_file),
    }
    records = [record for values in groups.values() for record in values]
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("V4_EVAL_IDS_NOT_UNIQUE")
    if len(groups["test70"]) != 70 or len(groups["ood_v2"]) != 30 or len(groups["residual_repair"]) != 3:
        raise ValueError("V4_EVAL_COUNTS_CHANGED")
    for record in records:
        if len(record.get("messages", [])) != 3:
            raise ValueError(f"V4_EVAL_RECORD_INVALID:{record.get('id')}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": "p2j4-decision-sft-v4-eval-bundle-v1",
        "trainingStarted": False,
        "recordCount": len(records),
        "groups": {name: len(values) for name, values in groups.items()},
        "sourceFiles": {
            "test70": str(test_file),
            "ood_v2": str(ood_file),
            "residual_repair": str(repair_file),
        },
        "recordsFile": records_path.name,
        "sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "modelRuns": 1,
        "notes": "One Base/SFT v4 inference pass; split metrics locally after completion.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("sft-data/decision-v4-staging/test.jsonl"))
    parser.add_argument("--ood", type=Path, default=Path("evals/p2j4-external-ood-v2/records.jsonl"))
    parser.add_argument("--repair", type=Path, default=Path("evals/p2j4-state-obligation-residual-repair-v2/records.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evals/p2j4-v4-eval-bundle"))
    args = parser.parse_args()
    manifest = build(args.test, args.ood, args.repair, args.output_dir)
    print(json.dumps({
        "status": "PASS",
        "recordCount": manifest["recordCount"],
        "groups": manifest["groups"],
        "sha256": manifest["sha256"],
        "outputDir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
