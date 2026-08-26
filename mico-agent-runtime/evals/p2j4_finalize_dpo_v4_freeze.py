"""Finalize DPO v4 only when all local data audits are PASS."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--staging",type=Path,required=True); parser.add_argument("--pair-audit",type=Path,required=True); parser.add_argument("--leakage-audit",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args()
    pair=json.loads(args.pair_audit.read_text(encoding="utf-8")); leakage=json.loads(args.leakage_audit.read_text(encoding="utf-8"))
    if pair.get("status") != "PASS" or leakage.get("status") != "PASS": raise SystemExit("DPO_V4_FREEZE_AUDITS_NOT_PASS")
    rows=[json.loads(line) for line in args.staging.read_text(encoding="utf-8").splitlines() if line.strip()]
    train=[row for row in rows if row.get("split")=="train"]; validation=[row for row in rows if row.get("split")=="val"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_path=args.output_dir/"train.jsonl"; val_path=args.output_dir/"validation.jsonl"; manifest_path=args.output_dir/"manifest.json"
    train_path.write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in train)+"\n",encoding="utf-8"); val_path.write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in validation)+"\n",encoding="utf-8")
    manifest={"schemaVersion":"p2j4-dpo-v4-freeze-manifest-v1","status":"FROZEN","trainingStarted":False,"pairConstructionStarted":True,"recordCount":len(rows),"trainCount":len(train),"validationCount":len(validation),"sourceStaging":str(args.staging),"sourceStagingSha256":sha(args.staging),"pairAudit":str(args.pair_audit),"leakageAudit":str(args.leakage_audit),"pairAuditStatus":pair["status"],"leakageAuditStatus":leakage["status"],"files":{"train":str(train_path),"validation":str(val_path),"trainSha256":sha(train_path),"validationSha256":sha(val_path)},"referenceLogprobManifest":None,"a100Gate":"REFERENCE_LOGPROB_AND_DRY_FORWARD_BACKWARD_REQUIRED"}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":manifest["status"],"recordCount":len(rows),"trainCount":len(train),"validationCount":len(validation)},ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
