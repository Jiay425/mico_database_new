"""Build the high-value Hard Eval variants used to reach the 800-candidate gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "p2j4-hard-variant-task-set-v1.json"
SPEC_OUTPUT = ROOT / "p2j4-hard-variant-spec-v1.json"
VERSION = "p2j4-hard-variant-task-set-v1"


def _task(
    *, case_id: str, kind: str, question: str, allowed: list[str],
    required: list[str], paths: list[list[str]], sources: list[str],
    min_bindings: int, stop_reason: str = "EVIDENCE_SUFFICIENT",
) -> dict[str, Any]:
    return {
        "schemaVersion": VERSION,
        "caseId": case_id,
        "kind": kind,
        "question": question,
        "expectedStatus": "COMPLETED",
        "requiredSources": sources,
        "allowedActions": allowed,
        "requiredActions": required,
        "forbiddenActions": ["direct_mysql", "unbounded_export", "causal_claim", "diagnosis"],
        "minEvidenceBindings": min_bindings,
        "maxActionCount": 8,
        "expectedStopReason": stop_reason,
        "requiresNonDiagnostic": True,
        "allowedActionPaths": paths,
    }


def _build() -> tuple[dict[str, Any], dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []

    def add(case_id: str, hard_class: str, task: dict[str, Any], assertions: list[str]) -> None:
        tasks.append(task)
        specs.append({
            "caseId": case_id,
            "hardCaseClass": hard_class,
            "assertions": assertions,
            "requiredSupportStatuses": ["conflicted"] if hard_class == "evidence_conflict" else [],
            "reviewStatus": "UNREVIEWED",
        })

    # 27 confounder variants, focused-analysis kind.
    diseases = ["T2D", "CRC", "IBD", "Obesity", "ASD", "Cirrhosis", "NAFLD", "Metabolic syndrome"]
    dimensions = [
        "年龄和性别", "年龄和 project", "性别和 project", "年龄、性别及 project",
        "年龄和国家", "性别和国家", "project 和批次", "年龄、国家及批次",
    ]
    for index in range(27):
        disease = diseases[index % len(diseases)]
        dimension = dimensions[index % len(dimensions)]
        if index % 3 == 0:
            question = f"分析 {disease} 与 Healthy 的候选菌差异，{dimension} 存在混杂，请控制后再补充证据。"
            required = ["inspect_cohort", "compare_groups", "adjust_confounders", "retrieve_evidence", "finish"]
            allowed = ["inspect_cohort", "compare_groups", "adjust_confounders", "retrieve_evidence", "finish"]
        else:
            question = f"探索 {disease} 的稳定微生态特征，{dimension} 不平衡；请先做混杂校正，再验证跨 project 稳定性，并补充文献证据后再停止。"
            required = ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "retrieve_evidence", "finish"]
            allowed = ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "retrieve_evidence", "finish"]
        add(
            f"p2j4-hard-variant-confounder-{index + 1:02d}",
            "confounder_trap",
            _task(
                case_id=f"p2j4-hard-variant-confounder-{index + 1:02d}",
                kind="focused_analysis", question=question, allowed=allowed,
                required=required, paths=[required], sources=["java", "vector"], min_bindings=2,
            ),
            ["INSPECT_BEFORE_ANALYSIS", "REQUIRES_CONFOUNDER_ADJUSTMENT", "NO_CAUSAL_ESCALATION"],
        )

    # 18 tool-selection variants, data-fact kind. The first half must remain
    # metadata-only; the second half exercises the bounded aggregate adapter.
    for index in range(9):
        disease = diseases[index % len(diseases)]
        case_id = f"p2j4-hard-variant-tool-metadata-{index + 1:02d}"
        required = ["inspect_cohort", "finish"]
        add(
            case_id, "tool_selection",
            _task(
                case_id=case_id, kind="data_fact",
                question=f"当前元数据中 {disease} 有多少样本记录？只检查 metadata，不要读取丰度矩阵。",
                allowed=required, required=required, paths=[required], sources=["java"], min_bindings=1,
            ),
            ["METADATA_FIRST", "NO_UNBOUNDED_ABUNDANCE_SCAN"],
        )
    for index in range(9):
        case_id = f"p2j4-hard-variant-tool-bounded-{index + 1:02d}"
        required = ["execute_read_query", "finish"]
        add(
            case_id, "tool_selection",
            _task(
                case_id=case_id, kind="data_fact",
                question="统计当前数据中独立 sample key 的有界数量，不要导出全部丰度记录。",
                allowed=required, required=required, paths=[required], sources=["java"], min_bindings=1,
            ),
            ["BOUNDED_READ_ONLY", "NO_UNBOUNDED_ABUNDANCE_SCAN"],
        )

    # 15 premature-stop variants, open-exploration kind.
    for index in range(15):
        disease = diseases[(index + 2) % len(diseases)]
        case_id = f"p2j4-hard-variant-premature-stop-{index + 1:02d}"
        required = ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence", "finish"]
        add(
            case_id, "premature_stop",
            _task(
                case_id=case_id, kind="open_exploration",
                question=f"探索 {disease} 的相关微生态现象，即使初步显著也不能停止；请比较后验证跨 project 稳定性并补充文献证据。",
                allowed=required, required=required, paths=[required], sources=["java", "vector"], min_bindings=2,
            ),
            ["INSPECT_BEFORE_ANALYSIS", "NO_PREMATURE_FINISH", "CROSS_PROJECT_BEFORE_FINISH", "EVIDENCE_BEFORE_FINISH"],
        )

    # 15 controlled conflict variants, open-exploration kind. The controlled
    # knowledge fixture is selected by the shared evidence-conflict prefix.
    for index in range(15):
        disease = diseases[(index + 3) % len(diseases)]
        case_id = f"p2j4-hard-evidence-conflict-variant-{index + 1:02d}"
        required = ["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"]
        add(
            case_id, "evidence_conflict",
            _task(
                case_id=case_id, kind="open_exploration",
                question=f"核查 {disease} 候选菌的证据：Graph 支持但论文给出相反方向，请保留 conflicted 状态，不要宣布 biomarker。",
                allowed=required, required=required, paths=[required], sources=["java", "vector", "graph"], min_bindings=2,
                stop_reason="QUALITY_RISK",
            ),
            ["RETRIEVE_GRAPH_AND_VECTOR", "REQUIRES_CONFLICTED_STATUS", "NO_SUPPORT_ESCALATION", "NO_BIOMARKER_CERTAINTY"],
        )

    task_payload = {
        "schemaVersion": VERSION,
        "name": "Mico Scientific Agent Hard Eval Variants v1",
        "reviewStatus": "STRUCTURE_READY_HARD_ORACLE_PENDING",
        "sourceTaskSet": "p2j4-hard-task-set-v1",
        "caseCount": len(tasks),
        "hardCaseDistribution": {
            "confounder_trap": 27,
            "tool_selection": 18,
            "premature_stop": 15,
            "evidence_conflict": 15,
        },
        "kindDistribution": {
            "focused_analysis": 27,
            "data_fact": 18,
            "open_exploration": 30,
        },
        "cases": tasks,
    }
    spec_payload = {
        "schemaVersion": "p2j4-hard-eval-spec-v1-variants",
        "taskSet": VERSION,
        "caseCount": len(specs),
        "cases": specs,
        "reviewGates": [
            "all variant cases retain closed allow-listed actions",
            "metadata cases cannot use unbounded abundance export",
            "confounder cases require adjustment before evidence conclusion",
            "conflict cases require conflicted status and QUALITY_RISK stop",
            "variant traces remain separate from frozen v1 and baseline assets",
        ],
    }
    return task_payload, spec_payload


if __name__ == "__main__":
    task_payload, spec_payload = _build()
    OUTPUT.write_text(json.dumps(task_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SPEC_OUTPUT.write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "READY",
        "caseCount": task_payload["caseCount"],
        "hardCaseDistribution": task_payload["hardCaseDistribution"],
        "kindDistribution": task_payload["kindDistribution"],
        "taskOutput": str(OUTPUT),
        "specOutput": str(SPEC_OUTPUT),
    }, ensure_ascii=False))
