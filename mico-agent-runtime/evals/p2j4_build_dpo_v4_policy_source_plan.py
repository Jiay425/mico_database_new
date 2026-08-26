"""Build a five-batch, success-preserving plan for model-policy Trace sources."""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "p2j4-dpo-v4-policy-source-task-set-v1.json"
CANARY = ROOT / "p2j4-dpo-v4-policy-source-runs" / "canary-compact-001.json"

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    payload=json.loads(TASKS.read_text(encoding='utf-8'))
    done=[]
    if CANARY.exists():
        run=json.loads(CANARY.read_text(encoding='utf-8'))
        if run.get('servicesObserved',{}).get('plannerModelUsed') is True and run.get('scores',[{}])[0].get('status')=='PASS': done=['p2j4-dpo-v4-policy-compact-001']
    pending=[case['caseId'] for case in payload['cases'] if case['caseId'] not in done]
    batches=[pending[i::5] for i in range(5)]
    out={'schemaVersion':'p2j4-dpo-v4-policy-source-plan-v1','status':'READY_NOT_EXECUTED','caseCount':60,'preservedCompletedCaseIds':done,'pendingCaseCount':len(pending),'sourceGate':{'plannerModelUsedRequired':True,'successNeverRerun':True,'failedOnlyRepair':True},'batches':[{'batchId':f'policy-source-{i+1:02d}','caseIds':cases,'checkpoint':f'evals/p2j4-dpo-v4-policy-source-runs/batch-{i+1:02d}.json'} for i,cases in enumerate(batches)]}
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'pending':len(pending),'batchSizes':[len(x) for x in batches]},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
