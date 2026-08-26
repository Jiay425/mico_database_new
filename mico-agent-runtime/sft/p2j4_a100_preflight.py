"""Fail-closed A100 preflight for the one approved Decision-DPO session."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path,
                        help="Required only after the Freeze v2 SFT v5 adapter is trained.")
    parser.add_argument("--sft-train", type=Path, required=True)
    parser.add_argument("--sft-validation", type=Path, required=True)
    parser.add_argument("--dpo-train", type=Path, required=True)
    parser.add_argument("--dpo-validation", type=Path, required=True)
    parser.add_argument("--dpo-audit", type=Path,
                        help="Required for DPO v3: an approved freeze audit, not a REVIEW_REQUIRED staging audit.")
    parser.add_argument("--min-free-gb", type=float, default=22.0)
    parser.add_argument("--expected-hash", action="append", default=[], metavar="PATH=SHA256")
    args = parser.parse_args()
    errors: list[str] = []
    required_paths = [args.model, args.sft_train, args.sft_validation, args.dpo_train, args.dpo_validation]
    if args.sft_adapter is not None:
        required_paths.append(args.sft_adapter)
    for path in required_paths:
        if not path.exists():
            errors.append("MISSING:" + str(path))
    dpo_audit_status = None
    if args.dpo_audit is not None:
        if not args.dpo_audit.exists():
            errors.append("MISSING_DPO_AUDIT:" + str(args.dpo_audit))
        else:
            try:
                dpo_audit_status = json.loads(args.dpo_audit.read_text(encoding="utf-8")).get("status")
                if dpo_audit_status != "FROZEN":
                    errors.append("DPO_AUDIT_NOT_FROZEN:" + str(dpo_audit_status))
            except (OSError, json.JSONDecodeError):
                errors.append("DPO_AUDIT_INVALID:" + str(args.dpo_audit))
    for value in args.expected_hash:
        try:
            raw_path, expected = value.rsplit("=", 1)
            path = Path(raw_path)
            if not path.exists() or _sha256(path) != expected.lower():
                errors.append("HASH_MISMATCH:" + str(path))
        except ValueError:
            errors.append("EXPECTED_HASH_FORMAT_INVALID")
    try:
        import torch
        import transformers
        import peft
        cuda = bool(torch.cuda.is_available())
        bf16 = bool(cuda and torch.cuda.is_bf16_supported())
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
        gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if cuda else 0.0
        if not cuda:
            errors.append("CUDA_UNAVAILABLE")
        if not bf16:
            errors.append("BF16_UNAVAILABLE")
        if gpu_memory_gb < 38:
            errors.append("GPU_MEMORY_BELOW_38GB")
    except Exception:
        cuda = bf16 = False
        gpu_name = None
        gpu_memory_gb = 0.0
        transformers = peft = None
        errors.append("PYTORCH_OR_PEFT_UNAVAILABLE")
    disk = shutil.disk_usage(args.model)
    free_gb = round(disk.free / 1024**3, 2)
    if free_gb < args.min_free_gb:
        errors.append("DISK_FREE_BELOW_MINIMUM")
    output = {
        "status": "PASS" if not errors else "FAIL",
        "python": sys.version.split()[0],
        "gpu": {"name": gpu_name, "memoryGb": gpu_memory_gb, "cuda": cuda, "bf16": bf16},
        "packages": {
            "transformers": getattr(transformers, "__version__", None),
            "peft": getattr(peft, "__version__", None),
        },
        "diskFreeGb": free_gb,
        "inputs": {
            "model": str(args.model), "sftAdapter": str(args.sft_adapter) if args.sft_adapter else None,
            "sftTrainHash": _sha256(args.sft_train) if args.sft_train.exists() else None,
            "sftValidationHash": _sha256(args.sft_validation) if args.sft_validation.exists() else None,
            "dpoTrainHash": _sha256(args.dpo_train) if args.dpo_train.exists() else None,
            "dpoValidationHash": _sha256(args.dpo_validation) if args.dpo_validation.exists() else None,
            "dpoAudit": str(args.dpo_audit) if args.dpo_audit else None,
            "dpoAuditStatus": dpo_audit_status,
        },
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
