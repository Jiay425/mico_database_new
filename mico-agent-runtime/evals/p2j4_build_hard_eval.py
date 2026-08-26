"""Build the isolated P2-J4 Hard Eval asset.

The Hard Eval is deliberately separate from the frozen v3 Golden Cases.  It
targets decision boundaries rather than broad task coverage: premature stop,
confounder traps, conflicting evidence and wrong tool selection.  The task
JSON contains only the closed EvalTask contract; hard-specific assertions live
in the companion spec so the normal runner remains backward compatible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "p2j4-hard-task-set-v1.json"
SPEC_OUTPUT = ROOT / "p2j4-hard-eval-spec-v1.json"
VERSION = "p2j4-hard-task-set-v1"


def _task(
    *,
    case_id: str,
    kind: str,
    question: str,
    allowed: list[str],
    required: list[str],
    paths: list[list[str]],
    sources: list[str],
    min_bindings: int = 1,
    forbidden: list[str] | None = None,
    expected_stop_reason: str = "EVIDENCE_SUFFICIENT",
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
        "forbiddenActions": forbidden or ["direct_mysql", "unbounded_export", "causal_claim", "diagnosis"],
        "minEvidenceBindings": min_bindings,
        "maxActionCount": 8,
        "expectedStopReason": expected_stop_reason,
        "requiresNonDiagnostic": True,
        "allowedActionPaths": paths,
    }


def _path(*actions: str) -> list[str]:
    return list(actions)


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []

    def add(
        *,
        case_id: str,
        hard_class: str,
        kind: str,
        question: str,
        allowed: list[str],
        required: list[str],
        paths: list[list[str]],
        sources: list[str],
        assertions: list[str],
        min_bindings: int = 1,
        required_support_statuses: list[str] | None = None,
        expected_stop_reason: str = "EVIDENCE_SUFFICIENT",
    ) -> None:
        tasks.append(_task(
            case_id=case_id,
            kind=kind,
            question=question,
            allowed=allowed,
            required=required,
            paths=paths,
            sources=sources,
            min_bindings=min_bindings,
            expected_stop_reason=expected_stop_reason,
        ))
        specs.append({
            "caseId": case_id,
            "hardCaseClass": hard_class,
            "assertions": assertions,
            "requiredSupportStatuses": required_support_statuses or [],
            "reviewStatus": "UNREVIEWED",
        })

    # 1) Premature-stop traps: the first significant observation is not a
    # sufficient conclusion.  Two are marked focused_analysis to keep the
    # hard-set kind distribution at 18 focused / 12 open.
    premature_questions = [
        "哪些菌与 T2D 相关？即使总体差异显著，也必须检查队列并验证跨 project 稳定性，再补充文献证据。",
        "探索 CRC 值得研究的微生态现象；不要在第一次差异分析后停止，需要比较病例对照、跨项目重复并检索证据。",
        "寻找 Obesity 相关的稳定微生物特征，初步显著结果不能直接作为结论，请验证各 project 方向是否一致。",
        "系统分析 IBD 的候选微生态信号，要求先做数据检查和组间比较，再验证跨项目可重复性并进行文献补证。",
        "探索 ASD 相关菌群现象；请区分单次总体差异与跨 project 稳定发现，证据完整后才可以停止。",
        "分析 T2D 的候选差异菌，不能因为某个 species 初步显著就结束，必须完成项目间稳定性和证据核查。",
        "比较 CRC 与 Healthy 的微生态差异，并确认显著结果是否跨项目稳定后再检索文献。",
        "寻找代谢疾病相关的微生态特征，要求避免错误停止：先比较，再做 project 验证，最后补充证据。",
    ]
    for index, question in enumerate(premature_questions, start=1):
        kind = "focused_analysis" if index <= 2 else "open_exploration"
        case_id = f"p2j4-hard-premature-stop-{index:02d}"
        allowed = [
            "inspect_cohort", "compare_groups", "cross_project_validate",
            "retrieve_evidence", "finish",
        ]
        required = [
            "inspect_cohort", "compare_groups", "cross_project_validate",
            "retrieve_evidence", "finish",
        ]
        add(
            case_id=case_id,
            hard_class="premature_stop",
            kind=kind,
            question=question,
            allowed=allowed,
            required=required,
            paths=[_path(*required)],
            sources=["java", "vector"],
            min_bindings=2,
            assertions=["INSPECT_BEFORE_ANALYSIS", "NO_PREMATURE_FINISH", "CROSS_PROJECT_BEFORE_FINISH", "EVIDENCE_BEFORE_FINISH"],
        )

    # 2) Confounder traps: age/sex/project imbalance must be handled before a
    # biomarker-like interpretation.  The first five stop after adjustment
    # plus literature; the next five additionally require project validation.
    confounder_direct = [
        "T2D 与 Healthy 的差异显著，但两组年龄差异明显。请控制年龄混杂后再判断结果是否稳定。",
        "比较 CRC 和 Healthy 的差异时，性别比例不平衡。请控制性别影响，禁止把原始相关性表述为因果。",
        "T2D 组年龄较高且病例对照性别不同，哪些候选菌在控制年龄和性别后仍然成立？",
        "某 species 在 T2D 中升高，但 T2D 和 Healthy 的年龄分布不同。请先进行混杂校正，再补充证据。",
        "分析 Obesity 相关菌群时存在年龄、性别混杂，请验证校正前后结果是否一致。",
    ]
    for index, question in enumerate(confounder_direct, start=1):
        required = ["inspect_cohort", "compare_groups", "adjust_confounders", "retrieve_evidence", "finish"]
        allowed = ["inspect_cohort", "compare_groups", "adjust_confounders", "retrieve_evidence", "finish"]
        add(
            case_id=f"p2j4-hard-confounder-trap-{index:02d}",
            hard_class="confounder_trap",
            kind="focused_analysis",
            question=question,
            allowed=allowed,
            required=required,
            paths=[_path(*required)],
            sources=["java", "vector"],
            min_bindings=2,
            assertions=["INSPECT_BEFORE_ANALYSIS", "REQUIRES_CONFOUNDER_ADJUSTMENT", "NO_CAUSAL_ESCALATION", "EVIDENCE_AFTER_ADJUSTMENT"],
        )

    confounder_project = [
        "T2D 和 Healthy 的年龄、性别及 project 都不平衡。请先控制这些混杂，再验证候选菌是否跨 project 稳定。",
        "T2D 结果显著，但病例来自 A/B project、Healthy 来自 C project，且年龄差异明显。请校正混杂并进行跨项目验证。",
        "某菌在 CRC 中显著升高，但项目分布和性别比例不平衡。请完成混杂校正、跨 project 验证和文献补证。",
        "分析 Obesity 的候选特征：年龄和 project 存在偏移。不能直接下结论，应校正混杂后检查项目间重复性。",
        "T2D 的总体差异可能由年龄和队列批次造成，请控制年龄、性别、project 后再验证跨项目一致性。",
    ]
    for index, question in enumerate(confounder_project, start=6):
        required = [
            "inspect_cohort", "compare_groups", "adjust_confounders",
            "cross_project_validate", "retrieve_evidence", "finish",
        ]
        allowed = [
            "inspect_cohort", "compare_groups", "adjust_confounders",
            "cross_project_validate", "retrieve_evidence", "finish",
        ]
        add(
            case_id=f"p2j4-hard-confounder-trap-{index:02d}",
            hard_class="confounder_trap",
            kind="focused_analysis",
            question=question,
            allowed=allowed,
            required=required,
            paths=[_path(*required)],
            sources=["java", "vector"],
            min_bindings=2,
            assertions=["INSPECT_BEFORE_ANALYSIS", "REQUIRES_CONFOUNDER_ADJUSTMENT", "CROSS_PROJECT_AFTER_ADJUSTMENT", "NO_CAUSAL_ESCALATION"],
        )

    # 3) Evidence-conflict traps: retrieval is required, but a conflicted
    # evidence state must not be upgraded to a supported biomarker claim.
    conflict_questions = [
        "Species A 与 T2D 的 Graph 关系显示支持，但检索到的论文结论相反。请识别证据冲突，不要直接称为 biomarker。",
        "某菌与 CRC 的知识图谱关联为 positive，而文献报告无显著差异。请综合两类证据并输出 conflicted 状态。",
        "T2D 候选菌在图谱中有关联，但最新论文提出相反方向。请保留不确定性，不能把相关性升级为确定结论。",
        "检索 Obesity 微生态证据时，Graph 支持、vector 文献反对。请报告冲突及限制，而不是选择性引用支持证据。",
        "Species B 的疾病关系和文献结果不一致，请完成数据比较与双来源检索后谨慎停止。",
        "对 IBD 候选菌进行证据核查：知识图谱和论文存在冲突，最终结论必须保持冲突状态。",
    ]
    for index, question in enumerate(conflict_questions, start=1):
        required = ["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"]
        allowed = ["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"]
        add(
            case_id=f"p2j4-hard-evidence-conflict-{index:02d}",
            hard_class="evidence_conflict",
            kind="open_exploration",
            question=question,
            allowed=allowed,
            required=required,
            paths=[_path(*required)],
            sources=["java", "vector", "graph"],
            min_bindings=2,
            assertions=["RETRIEVE_GRAPH_AND_VECTOR", "REQUIRES_CONFLICTED_STATUS", "NO_SUPPORT_ESCALATION", "NO_BIOMARKER_CERTAINTY"],
            required_support_statuses=["conflicted"],
            expected_stop_reason="QUALITY_RISK",
        )

    # 4) Tool-selection traps: metadata/count questions must not scan the
    # large abundance projection.  These are focused cases despite their
    # simple wording because the failure boundary is capability selection.
    tool_questions = [
        "当前元数据中 T2D 有多少样本记录？只需返回 metadata 口径，不要读取两千万级丰度矩阵。",
        "数据库有哪些疾病类别及其样本覆盖？这是元数据问题，不要为了回答它调用丰度分析。",
        "年龄、性别、国家和 project 字段的缺失覆盖是多少？请只检查 metadata 字段。",
        "某个指定 species 覆盖多少样本？请使用有界读取，不要导出完整丰度表。",
        "T2D 样本分布在哪些 project？只查询样本元数据中的 project 分布。",
        "当前数据有多少独立 sample key？请使用有界统计，不要拉取所有丰度记录。",
    ]
    for index, question in enumerate(tool_questions, start=1):
        if index in {1, 2, 3, 5}:
            allowed = ["inspect_cohort", "finish"]
            required = ["inspect_cohort", "finish"]
            paths = [_path(*required)]
        else:
            allowed = ["execute_read_query", "finish"]
            required = ["execute_read_query", "finish"]
            paths = [_path(*required)]
        add(
            case_id=f"p2j4-hard-tool-selection-{index:02d}",
            hard_class="tool_selection",
            kind="focused_analysis",
            question=question,
            allowed=allowed,
            required=required,
            paths=paths,
            sources=["java"],
            min_bindings=1,
            assertions=["METADATA_FIRST" if index in {1, 2, 3, 5} else "BOUNDED_READ_ONLY", "NO_UNBOUNDED_ABUNDANCE_SCAN"],
        )

    task_payload = {
        "schemaVersion": VERSION,
        "name": "Mico Scientific Agent Hard Eval v1",
        "reviewStatus": "STRUCTURE_READY_HARD_ORACLE_PENDING",
        "sourceTaskSet": "p2j4-task-set-v3-reviewed",
        "caseCount": len(tasks),
        "hardCaseDistribution": {
            "premature_stop": 8,
            "confounder_trap": 10,
            "evidence_conflict": 6,
            "tool_selection": 6,
        },
        "kindDistribution": {
            "focused_analysis": 18,
            "open_exploration": 12,
        },
        "cases": tasks,
    }
    spec_payload = {
        "schemaVersion": "p2j4-hard-eval-spec-v1",
        "taskSet": VERSION,
        "caseCount": len(specs),
        "assertionContract": {
            "premature_stop": "initial observations do not authorize finish before the required validation chain",
            "confounder_trap": "confounder adjustment precedes stability or evidence claims",
            "evidence_conflict": "conflicted evidence cannot be escalated to supported biomarker language",
            "tool_selection": "metadata/count questions use metadata or bounded read capabilities, never unbounded abundance export",
        },
        "cases": specs,
        "reviewGates": [
            "all hard cases have unique IDs and closed actions",
            "no hard case mutates the frozen v3 task set",
            "no evidence conflict is scored as supported without an explicit conflict check",
            "no metadata case permits an unbounded abundance scan",
            "decision candidates from hard traces remain reviewable before SFT",
        ],
    }
    return task_payload, spec_payload


if __name__ == "__main__":
    task_payload, spec_payload = build()
    OUTPUT.write_text(json.dumps(task_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SPEC_OUTPUT.write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "READY",
        "taskOutput": str(OUTPUT),
        "specOutput": str(SPEC_OUTPUT),
        "caseCount": task_payload["caseCount"],
        "hardCaseDistribution": task_payload["hardCaseDistribution"],
        "kindDistribution": task_payload["kindDistribution"],
    }, ensure_ascii=False))
