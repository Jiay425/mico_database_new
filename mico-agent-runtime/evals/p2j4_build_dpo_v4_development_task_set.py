"""Build development-only Trace collection tasks for DPO v4.

This is deliberately not an evaluation suite.  Its successful real traces are
only candidate sources and must pass the later pair audit.  The task families,
case IDs and source label are disjoint from every frozen Golden, Hard, Test,
OOD and Runtime paired set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "p2j4-dpo-v4-development-v1"
FORBIDDEN = ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"]


def _case(
    number: int,
    family: str,
    kind: str,
    question: str,
    required: list[str],
    candidates: list[str],
    *,
    sources: list[str] | None = None,
    stop_reason: str = "EVIDENCE_SUFFICIENT",
    allowed_paths: list[list[str]] | None = None,
) -> dict[str, Any]:
    if not set(required).issubset(candidates):
        raise ValueError("DPO_V4_DEVELOPMENT_REQUIRED_NOT_CANDIDATE")
    return {
        "schemaVersion": VERSION,
        "caseId": f"p2j4-dpo-v4-dev-{number:03d}",
        "kind": kind,
        "question": question,
        "expectedStatus": "COMPLETED",
        "requiredSources": sources or ["java"],
        # Wider than the required path on purpose: these are Runtime-shaped
        # decision states, not one-option demonstration prompts.
        "allowedActions": candidates,
        "requiredActions": required,
        "forbiddenActions": FORBIDDEN,
        "minEvidenceBindings": 1 if kind == "data_fact" else 2,
        "maxActionCount": len(candidates),
        "expectedStopReason": stop_reason,
        "requiresNonDiagnostic": True,
        "allowedActionPaths": [required] if allowed_paths is None else allowed_paths,
    }


def build() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    number = 1

    # The three-option bounded-read cases are an explicit, small exception to
    # the later 4--7 candidate-width target.  A bounded factual read naturally
    # has a narrower legal action space and is needed for action coverage.
    bounded_topics = (
        "疾病标签的受限计数口径", "project 覆盖的受限计数口径",
        "country 覆盖的受限计数口径", "年龄字段缺失的受限计数口径",
        "性别字段缺失的受限计数口径", "病例对照共有 project 的受限计数口径",
        "候选 feature 非零记录的受限计数口径", "元数据与丰度样本键连接的受限计数口径",
        "指定 disease 的样本键覆盖口径", "指定 project 的病例对照覆盖口径",
        "taxonomy 版本字段的受限计数口径", "丰度来源批次字段的受限计数口径",
        "连续变量可用性的受限计数口径", "疾病与 country 共现的受限计数口径",
        "疾病与 sex 共现的受限计数口径", "疾病与年龄层共现的受限计数口径",
        "feature 覆盖阈值的受限计数口径", "样本元数据完整性的受限计数口径",
    )
    for topic in bounded_topics:
        cases.append(_case(
            number, "bounded_read", "data_fact",
            f"仅完成 {topic}；先执行一个有界只读查询，再给出汇总并停止，不输出原始行或样本标识。",
            ["execute_read_query", "finish"],
            ["inspect_cohort", "execute_read_query", "finish"],
            allowed_paths=[
                ["execute_read_query", "finish"],
                ["inspect_cohort", "execute_read_query", "finish"],
            ],
        ))
        number += 1

    stratified_topics = (
        "按年龄层检查组间 feature 变化", "按性别检查组间 feature 变化",
        "按 country 检查组间 feature 变化", "按 project 检查组间 feature 变化",
        "按缺失模式检查组间 feature 变化", "按 BMI 分层检查组间 feature 变化",
        "按年龄层总结多 feature 投影", "按性别总结多 feature 投影",
        "按 country 总结多 feature 投影", "按 project 总结多 feature 投影",
        "按年龄与性别交叉分层检查变化", "按 country 与 project 交叉分层检查变化",
        "按数据来源批次分层检查变化", "按连续变量分位数分层检查变化",
        "按 metadata 完整性分层检查变化", "按疾病亚组分层检查变化",
    )
    for topic in stratified_topics:
        cases.append(_case(
            number, "stratified_state", "focused_analysis",
            f"对批准的研究组{topic}。先核对 cohort 并完成有界组间比较；在存在可分层 metadata 时进行分层分析和投影总结，不作诊断或因果表述。",
            ["inspect_cohort", "compare_groups", "finish"],
            ["inspect_cohort", "compare_groups", "stratified_analysis", "adjust_confounders", "analyze_projection", "retrieve_evidence", "finish"],
            allowed_paths=[],
        ))
        number += 1

    confounder_topics = (
        "年龄分布不均", "性别分布不均", "project 构成不均", "country 构成不均",
        "年龄与 sex 同时不均", "country 与 project 同时不均",
        "缺失 metadata 模式不均", "BMI 与年龄同时不均", "来源批次与 project 同时不均",
        "病例和对照的年龄层不重叠风险", "病例和对照的 sex 构成不重叠风险",
        "地区构成与疾病标签耦合风险", "project 构成与疾病标签耦合风险",
        "连续协变量分布偏移", "多个协变量共同偏移", "调整后需要有界投影总结",
        "调整前后方向需要对照", "调整后仅允许观察性结论",
    )
    for topic in confounder_topics:
        cases.append(_case(
            number, "confounder_state", "open_exploration",
            f"探索批准研究组的微生态关联；当前 cohort 检查后发现{topic}。必须完成组间比较并控制相应混杂，再做有界投影总结；证据不足时不得把相关性表述为因果。",
            ["inspect_cohort", "compare_groups", "finish"],
            ["inspect_cohort", "compare_groups", "stratified_analysis", "adjust_confounders", "cross_project_validate", "analyze_projection", "finish"],
            stop_reason=None,
            allowed_paths=[],
        ))
        number += 1

    disease_topics = (
        "区分代谢相关共性与目标疾病特异性", "检查候选 feature 是否在多个疾病中同向变化",
        "检查候选 feature 是否仅见于目标疾病", "比较目标疾病与肥胖相关队列",
        "比较目标疾病与炎症相关队列", "比较目标疾病与肿瘤相关队列",
        "在多个疾病标签间检查 feature 的方向", "跨疾病后再总结有界投影",
        "跨疾病后再补充有限知识证据", "目标疾病与相邻表型的差异",
        "疾病特异性与 project 稳定性同时检查", "疾病特异性与年龄混杂同时检查",
        "疾病特异性与 sex 混杂同时检查", "疾病特异性不足时降级结论",
        "跨疾病方向不一致时保留不确定性", "跨疾病方向一致时仍避免因果结论",
    )
    for topic in disease_topics:
        cases.append(_case(
            number, "cross_disease_state", "open_exploration",
            f"对批准研究组完成初步差异分析后，{topic}。完成跨疾病验证与有限证据检索，再以观察性证据边界结束。",
            ["inspect_cohort", "compare_groups", "finish"],
            ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "cross_disease_validate", "retrieve_evidence", "finish"],
            sources=["java", "vector"],
            stop_reason=None,
            allowed_paths=[],
        ))
        number += 1

    combined_topics = (
        "先验证跨 project 方向，再进行有界投影", "比较完成后先进行有界投影，再补充有限证据",
        "project 可比时仅做必要稳定性验证", "多个 feature 共同变化后进行有界投影",
        "跨 project 一致但证据不足时仅补充一次有限检索", "结果不稳定时先检查 project 再调整混杂",
        "project 稳定与年龄调整都完成后停止扩展", "数据观察已充分但文献证据冲突时降级结论",
    )
    for topic in combined_topics:
        cases.append(_case(
            number, "combined_legal_choice", "open_exploration",
            f"对批准研究组进行有界探索：{topic}。只做当前证据义务所需的步骤，避免重复调用或无目标扩展。",
            ["inspect_cohort", "compare_groups", "finish"],
            ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "analyze_projection", "retrieve_evidence", "finish"],
            sources=["java", "vector"],
            allowed_paths=[],
        ))
        number += 1

    justified_stops = (
        "当前问题仅要求完成有界组间比较，且没有不平衡或冲突提示",
        "目标 feature 覆盖不足，继续扩展无法产生新信息",
        "已完成指定分层分析和投影总结，未要求外部证据补充",
        "已完成跨疾病验证，结果不支持疾病特异性，应以限制性结论停止",
    )
    for condition in justified_stops:
        cases.append(_case(
            number, "justified_stop", "focused_analysis",
            f"完成批准研究组的有界分析。{condition}；应给出观察性限制并正确停止，不进行额外无目标工具调用。",
            ["inspect_cohort", "compare_groups", "finish"],
            ["inspect_cohort", "compare_groups", "analyze_projection", "retrieve_evidence", "finish"],
            stop_reason=None,
            allowed_paths=[],
        ))
        number += 1

    distribution = dict(Counter(item["kind"] for item in cases))
    family_by_case = {
        case["caseId"]: {
            "collectionOnly": True,
            "dpoV4Family": family,
            "sourceLabel": "real_training_trace",
            "evaluationEligible": False,
        }
        for case, family in zip(
            cases,
            ["bounded_read"] * len(bounded_topics)
            + ["stratified_state"] * len(stratified_topics)
            + ["confounder_state"] * len(confounder_topics)
            + ["cross_disease_state"] * len(disease_topics)
            + ["combined_legal_choice"] * len(combined_topics)
            + ["justified_stop"] * len(justified_stops),
            strict=True,
        )
    }
    family_distribution = dict(Counter(item["dpoV4Family"] for item in family_by_case.values()))
    return {
        "schemaVersion": VERSION,
        "name": "Mico DPO v4 development Trace collection v1",
        "purpose": "development_trace_collection_only_not_evaluation",
        "reviewStatus": "STRUCTURE_READY_NOT_EXECUTED",
        "caseCount": len(cases),
        "kindDistribution": distribution,
        "familyDistribution": family_distribution,
        "developmentCaseMetadata": family_by_case,
        "frozenEvaluationExclusions": ["Golden50", "Hard30", "Test70", "OOD30", "Runtime20"],
        "trainingStarted": False,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build()
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "READY_NOT_EXECUTED", "caseCount": payload["caseCount"],
        "kindDistribution": payload["kindDistribution"],
        "familyDistribution": payload["familyDistribution"], "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
