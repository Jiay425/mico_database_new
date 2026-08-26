"""Check normalized DPO staging states against frozen Test70/OOD30 states."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_AUDITS = (
    ROOT / "evals" / "p2j4-v4-eval-bundle" / "test70-v4-reason-audit.json",
    ROOT / "evals" / "p2j4-v4-eval-bundle" / "ood_v2-v4-reason-audit.json",
)

def _norm(value: Any) -> str:
    if isinstance(value, dict):
        value = {k: value[k] for k in ("candidate_actions", "history_actions", "observation_flags", "state_summary", "task_kind") if k in value}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def audit(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_signatures: set[str] = set()
    eval_counts: dict[str, int] = {}
    for audit_path in EVAL_AUDITS:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        for case in payload.get("cases", []):
            sig = _norm(case.get("state", {})); eval_signatures.add(sig); eval_counts[audit_path.name] = eval_counts.get(audit_path.name, 0) + 1
    dpo_signatures = set()
    overlap = 0
    for row in rows:
        state = json.loads(row["prompt"][1]["content"])["policy_state"]
        sig = _norm(state); dpo_signatures.add(sig); overlap += sig in eval_signatures
    result = {"schemaVersion": "p2j4-dpo-v4-eval-leakage-audit-v1", "status": "PASS" if overlap == 0 else "FAIL", "dpoRecordCount": len(rows), "dpoUniqueNormalizedStateCount": len(dpo_signatures), "evaluationStateCounts": eval_counts, "normalizedOverlapCount": overlap, "test70Ood30Leakage": overlap, "errors": [] if overlap == 0 else ["NORMALIZED_STATE_OVERLAP_WITH_TEST70_OR_OOD30"]}
    return result

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--input",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); result=audit(args.input); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,ensure_ascii=False)); return 0 if result["status"]=="PASS" else 1
if __name__ == "__main__": raise SystemExit(main())
