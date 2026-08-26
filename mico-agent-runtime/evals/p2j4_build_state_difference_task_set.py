"""Build a new, non-overlapping state-difference task pool.

The pool is deliberately separate from the frozen Golden Case sets.  It is
made from distinct research-state requirements and action paths; it is not a
row-level copy or paraphrase of the existing 100 cases.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "p2j4-decision-state-difference-v1"
FORBIDDEN = ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"]


def _task(
    number: int,
    kind: str,
    question: str,
    path: list[str],
    *,
    sources: list[str] | None = None,
    allowed_paths: list[list[str]] | None = None,
    expected_stop_reason: str | None = None,
) -> dict[str, Any]:
    case_id = f"p2j4-state-diff-{number:03d}"
    required_sources = sources or (["java", "vector"] if "retrieve_evidence" in path else ["java"])
    paths = allowed_paths or [list(path)]
    allowed_actions = list(dict.fromkeys(action for candidate in paths for action in candidate))
    return {
        "schemaVersion": VERSION,
        "caseId": case_id,
        "kind": kind,
        "question": question,
        "expectedStatus": "COMPLETED",
        "requiredSources": required_sources,
        "allowedActions": allowed_actions,
        "requiredActions": list(path),
        "forbiddenActions": FORBIDDEN,
        "minEvidenceBindings": 1 if kind == "data_fact" else 2,
        "maxActionCount": len(path),
        "expectedStopReason": expected_stop_reason or "EVIDENCE_SUFFICIENT",
        "requiresNonDiagnostic": True,
        "allowedActionPaths": paths,
    }


def build() -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    number = 1

    data_variants = [
        ("metadata 字段可分组、过滤和排序的能力", ["inspect_cohort", "finish"]),
        ("当前可用疾病标签的记录覆盖", ["execute_read_query", "finish"]),
        ("样本键与丰度记录的覆盖口径", ["execute_read_query", "finish"]),
        ("project 与疾病标签的共现覆盖", ["inspect_cohort", "execute_read_query", "finish"]),
        ("年龄、性别和 country 的缺失覆盖", ["inspect_cohort", "finish"]),
        ("丰度版本、taxonomy 版本和来源批次字段", ["inspect_cohort", "finish"]),
        ("同时具有病例和对照记录的 project 覆盖", ["inspect_cohort", "finish"]),
        ("一个候选 feature 的非零样本键覆盖", ["execute_read_query", "finish"]),
    ]
    for label, path in data_variants:
        tasks.append(_task(
            number,
            "data_fact",
            f"只核对 {label}，说明内部记录数与样本键数口径，不输出原始行。",
            path,
        ))
        number += 1

    focused_specs = [
        ("候选 feature 的覆盖与丰度摘要", ["inspect_cohort", "compare_groups", "analyze_projection", "finish"]),
        ("候选 feature 的年龄分层变化", ["inspect_cohort", "stratified_analysis", "analyze_projection", "finish"]),
        ("候选 feature 的性别分层变化", ["inspect_cohort", "stratified_analysis", "analyze_projection", "finish"]),
        ("候选 feature 的 country 分层变化", ["inspect_cohort", "stratified_analysis", "analyze_projection", "finish"]),
        ("缺失 metadata 条件下的分层差异", ["inspect_cohort", "stratified_analysis", "analyze_projection", "finish"]),
        ("控制年龄混杂后的组间差异", ["inspect_cohort", "compare_groups", "adjust_confounders", "analyze_projection", "finish"]),
        ("控制性别混杂后的组间差异", ["inspect_cohort", "compare_groups", "adjust_confounders", "analyze_projection", "finish"]),
        ("控制 project 批次差异后的组间差异", ["inspect_cohort", "compare_groups", "adjust_confounders", "analyze_projection", "finish"]),
        ("跨 project 的方向一致性", ["inspect_cohort", "compare_groups", "cross_project_validate", "analyze_projection", "finish"]),
        ("跨 project 验证后再控制 country 差异", ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "analyze_projection", "finish"]),
        ("跨 disease 的特异性", ["inspect_cohort", "compare_groups", "cross_disease_validate", "analyze_projection", "finish"]),
        ("跨 project 稳定性验证后的文献补证", ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence", "finish"]),
        ("控制年龄性别后再做文献补证", ["inspect_cohort", "compare_groups", "adjust_confounders", "retrieve_evidence", "finish"]),
        ("分层结果的文献补证", ["inspect_cohort", "stratified_analysis", "analyze_projection", "retrieve_evidence", "finish"]),
        ("跨 disease 结果的文献补证", ["inspect_cohort", "compare_groups", "cross_disease_validate", "retrieve_evidence", "finish"]),
        ("跨 project 稳定性与文献补证", ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence", "finish"]),
        ("组合 feature 的受限投影", ["inspect_cohort", "compare_groups", "analyze_projection", "finish"]),
        ("project 与性别共同混杂的受限投影", ["inspect_cohort", "compare_groups", "adjust_confounders", "analyze_projection", "finish"]),
        ("country 与 project 共同分层的受限投影", ["inspect_cohort", "stratified_analysis", "analyze_projection", "finish"]),
        ("跨 project、跨 disease 的受限投影", ["inspect_cohort", "compare_groups", "cross_project_validate", "cross_disease_validate", "analyze_projection", "finish"]),
    ]
    for index, (label, path) in enumerate(focused_specs):
        dimension = ["年龄", "性别", "country", "project", "缺失模式"][index % 5]
        question = (
            f"对两个已批准研究组分析{label}，重点检查{dimension}维度；"
            "只使用有界投影和受限分析，不输出诊断或因果结论。"
        )
        if "retrieve_evidence" in path:
            question += " 在数据观察完成后补充有限文献证据。"
        allowed_paths = [list(path)]
        if label == "跨 project 验证后再控制 country 差异":
            # The two checks are independent after the group comparison.  The
            # primary path follows the conservative confounder-first policy,
            # but the validated project-first order is also legal.
            allowed_paths.append([
                "inspect_cohort",
                "compare_groups",
                "cross_project_validate",
                "adjust_confounders",
                "analyze_projection",
                "finish",
            ])
        tasks.append(_task(number, "focused_analysis", question, path, allowed_paths=allowed_paths))
        number += 1

    open_specs = [
        ("跨 project 稳定性", ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence", "finish"], "project 分布均衡"),
        ("project 不平衡下的稳定性", ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "retrieve_evidence", "finish"], "project 分布不平衡"),
        ("年龄性别混杂后的稳定性", ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "retrieve_evidence", "finish"], "年龄和性别存在混杂"),
        ("country 与 project 双维度稳定性", ["inspect_cohort", "stratified_analysis", "cross_project_validate", "retrieve_evidence", "finish"], "country 与 project 需要同时检查"),
        ("缺失 metadata 对稳定性的影响", ["inspect_cohort", "compare_groups", "stratified_analysis", "retrieve_evidence", "finish"], "缺失 metadata 需要保留限制"),
        ("跨 disease 特异性", ["inspect_cohort", "compare_groups", "cross_disease_validate", "retrieve_evidence", "finish"], "跨 disease 需要验证特异性"),
        ("跨 project 与跨 disease 联合验证", ["inspect_cohort", "compare_groups", "cross_project_validate", "cross_disease_validate", "analyze_projection", "retrieve_evidence", "finish"], "project 稳定性和 disease 特异性都必须验证"),
        ("跨 project、混杂与跨 disease 联合验证", ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "cross_disease_validate", "analyze_projection", "retrieve_evidence", "finish"], "混杂、project 稳定性和 disease 特异性都必须验证"),
        ("多 feature 共同变化", ["inspect_cohort", "compare_groups", "analyze_projection", "retrieve_evidence", "finish"], "多个 feature 的共同变化需要投影分析"),
        ("多 feature 跨 project 稳定性", ["inspect_cohort", "compare_groups", "cross_project_validate", "analyze_projection", "retrieve_evidence", "finish"], "多个 feature 共同变化并验证跨 project 稳定性"),
        ("证据冲突下的质量风险", ["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"], "数据和知识证据可能冲突，结论必须降级"),
        ("信息不足时的有界停止", ["inspect_cohort", "compare_groups", "finish"], "当前信息不足，不允许无依据扩展结论"),
    ]
    for block_index, (label, path, condition) in enumerate(open_specs):
        for replicate in range(1, 4):
            question = (
                f"开放探索第 {block_index + 1} 组变体 {replicate}：{label}。"
                f"当前条件是{condition}；请先检查数据状态，再按证据充分性决定继续验证、补证或停止。"
            )
            if "retrieve_evidence" in path:
                question += " 文献证据只能在数据观察之后调用。"
            allowed_paths = [list(path)]
            if label == "跨 project、混杂与跨 disease 联合验证":
                # Project consistency and approved confounder adjustment are
                # conditionally independent checks after the initial group
                # comparison.  Both orders are scientifically valid; the
                # oracle must not turn this into an arbitrary ordering test.
                allowed_paths.append([
                    "inspect_cohort",
                    "compare_groups",
                    "cross_project_validate",
                    "adjust_confounders",
                    "cross_disease_validate",
                    "analyze_projection",
                    "retrieve_evidence",
                    "finish",
                ])
            stop_reason = None
            if label == "证据冲突下的质量风险":
                stop_reason = "QUALITY_RISK"
            elif label == "信息不足时的有界停止":
                stop_reason = "NO_NEW_INFORMATION"
            tasks.append(_task(
                number,
                "open_exploration",
                question,
                path,
                allowed_paths=allowed_paths,
                expected_stop_reason=stop_reason,
            ))
            number += 1

    distribution = dict(Counter(item["kind"] for item in tasks))
    return {
        "schemaVersion": VERSION,
        "name": "Mico Decision state-difference task pool v1",
        "purpose": "new_state_difference_only",
        "caseCount": len(tasks),
        "kindDistribution": distribution,
        "baseTaskSets": [
            "p2j4-task-set-v2",
            "p2j4-task-set-v3-reviewed",
            "p2j4-hard-task-set-v1",
            "p2j4-hard-variant-task-set-v1",
        ],
        "trainingStarted": False,
        "cases": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build P2-J4 Decision state-difference task pool")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "READY",
        "schemaVersion": result["schemaVersion"],
        "caseCount": result["caseCount"],
        "kindDistribution": result["kindDistribution"],
        "trainingStarted": result["trainingStarted"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
