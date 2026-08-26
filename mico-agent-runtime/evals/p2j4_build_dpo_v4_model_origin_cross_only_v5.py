from __future__ import annotations
import argparse, json
from pathlib import Path
from mico_agent_runtime.contracts.research import ScientificObservationSummary, ScientificPlannerContext
from mico_agent_runtime.ports.scientific_planner import semantic_scientific_next_action

VERSION = "p2j4-dpo-v4-targeted-weak-actions-v2"
BASE = ["inspect_cohort", "compare_groups", "cross_disease_validate", "finish"]
EXTRAS = [
    (), ("analyze_projection",), ("retrieve_evidence",), ("adjust_confounders",),
    ("cross_project_validate",), ("stratified_analysis",),
    ("analyze_projection", "retrieve_evidence"), ("adjust_confounders", "analyze_projection"),
    ("cross_project_validate", "retrieve_evidence"), ("stratified_analysis", "analyze_projection"),
    ("analyze_projection", "cross_project_validate"), ("retrieve_evidence", "stratified_analysis"),
    ("adjust_confounders", "cross_project_validate"), ("cross_project_validate", "stratified_analysis"),
    ("analyze_projection", "adjust_confounders"), ("retrieve_evidence", "cross_project_validate"),
    ("stratified_analysis", "adjust_confounders"), ("cross_project_validate", "analyze_projection"),
    ("analyze_projection", "retrieve_evidence", "adjust_confounders"), ("retrieve_evidence", "cross_project_validate", "stratified_analysis"),
    ("adjust_confounders", "cross_project_validate", "analyze_projection"), ("stratified_analysis", "analyze_projection", "retrieve_evidence"),
    ("cross_project_validate", "analyze_projection", "retrieve_evidence"), ("adjust_confounders", "stratified_analysis", "retrieve_evidence"),
]

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument('--task-output',type=Path,required=True); parser.add_argument('--preflight-output',type=Path,required=True); args=parser.parse_args()
    cases=[]; pf=[]
    for i,extra in enumerate(EXTRAS,1):
        allowed=list(dict.fromkeys([*BASE,*extra])); cid=f'p2j4-dpo-v4-model-origin-v5-cross-disease-{i:03d}'
        q='Use the current validated state and approved bounded actions. Preserve observational conclusions and stop after the evidence obligation is satisfied without exposing raw records.'
        cases.append({'schemaVersion':VERSION,'caseId':cid,'kind':'open_exploration','question':q,'expectedStatus':'COMPLETED','requiredSources':['java'],'allowedActions':allowed,'requiredActions':BASE,'forbiddenActions':['direct_mysql','unbounded_export','diagnosis','causal_claim'],'minEvidenceBindings':2,'maxActionCount':4,'expectedStopReason':'EVIDENCE_SUFFICIENT','requiresNonDiagnostic':True,'allowedActionPaths':[BASE]})
        states=[]
        for j in range(1,len(BASE)):
            hist=BASE[:j]; obs=[ScientificObservationSummary(observationId=f'observation-{k:032x}',actionName=a,status='VALIDATED',source='java_controlled_read' if a in {'inspect_cohort','compare_groups'} else 'python_bounded_analysis',rowCount=1) for k,a in enumerate(hist,1)]
            sem=semantic_scientific_next_action(ScientificPlannerContext(questionSummary=q,intent='scientific_exploration',approvedActions=allowed,remainingActionBudget=len(BASE)-j,observations=obs,schemaCatalog=None))
            states.append({'history':hist,'semanticAction':sem.actionName if sem else None})
        pf.append({'caseId':cid,'states':states})
    task={'schemaVersion':VERSION,'name':'DPO v4 model-origin cross-disease collection v5','purpose':'provenance_complete_model_origin_collection','caseCount':len(cases),'trainingStarted':False,'reviewStatus':'SEMANTIC_PREFLIGHT_REQUIRED','cases':cases}
    audit={'schemaVersion':VERSION+'-cross-v5-preflight','caseCount':len(cases),'allNonInitialStatesSemanticNone':all(s['semanticAction'] is None for c in pf for s in c['states']),'cases':pf}
    args.task_output.write_text(json.dumps(task,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); args.preflight_output.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'caseCount':len(cases),'semanticNone':audit['allNonInitialStatesSemanticNone']},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
