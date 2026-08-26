"""Build the second, non-overlapping Decision state-difference task pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "p2j4-decision-state-difference-v2"
FORBIDDEN = ["direct_mysql", "unbounded_export", "diagnosis", "causal_claim"]


def _alternate_path(path: list[str], variant: int) -> list[str] | None:
    """Return a legal, non-primary route for a state/action-set variant."""

    if "cross_project_validate" in path \
            and "adjust_confounders" in path:
        adjust_index = path.index("adjust_confounders")
        project_index = path.index("cross_project_validate")
        if adjust_index < project_index:
            return (
                path[:adjust_index]
                + ["cross_project_validate", "adjust_confounders"]
                + path[adjust_index + 1:project_index]
                + path[project_index + 1:]
            )
    if variant == 1 and "cross_project_validate" in path \
            and "adjust_confounders" not in path:
        index = path.index("cross_project_validate")
        return path[:index] + ["adjust_confounders"] + path[index:]
    if variant == 2 and "compare_groups" in path \
            and "stratified_analysis" not in path:
        index = next(
            (path.index(action) for action in ("analyze_projection", "retrieve_evidence", "finish")
             if action in path),
            len(path) - 1,
        )
        return path[:index] + ["stratified_analysis"] + path[index:]
    if variant == 2 and "analyze_projection" in path \
            and "retrieve_evidence" not in path:
        index = path.index("finish")
        return path[:index] + ["retrieve_evidence"] + path[index:]
    return None


def _task(
    number: int,
    kind: str,
    question: str,
    path: list[str],
    *,
    variant: int = 0,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    case_id = f"p2j4-state-diff-v2-{number:03d}"
    paths = [list(path)]
    alternate = _alternate_path(path, variant)
    if alternate is not None and alternate not in paths:
        paths.append(alternate)
    allowed_actions = list(dict.fromkeys(action for candidate in paths for action in candidate))
    return {
        "schemaVersion": VERSION,
        "caseId": case_id,
        "kind": kind,
        "question": question,
        "expectedStatus": "COMPLETED",
        "requiredSources": ["java", "vector"] if "retrieve_evidence" in path else ["java"],
        "allowedActions": allowed_actions,
        "requiredActions": list(path),
        "forbiddenActions": FORBIDDEN,
        "minEvidenceBindings": 1 if kind == "data_fact" else 2,
        "maxActionCount": max(len(candidate) for candidate in paths),
        "expectedStopReason": stop_reason or "EVIDENCE_SUFFICIENT",
        "requiresNonDiagnostic": True,
        "allowedActionPaths": paths,
    }


def build() -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    number = 1

    data_specs = [
        ("metadata 字段的可用分组维度", ["inspect_cohort", "finish"]),
        ("疾病标签的记录数量", ["execute_read_query", "finish"]),
        ("project 标签的记录数量", ["execute_read_query", "finish"]),
        ("country 标签的记录数量", ["execute_read_query", "finish"]),
        ("年龄字段的缺失覆盖", ["inspect_cohort", "finish"]),
        ("性别字段的缺失覆盖", ["inspect_cohort", "finish"]),
        ("病例和对照的样本键覆盖", ["inspect_cohort", "execute_read_query", "finish"]),
        ("疾病和 project 的交叉覆盖", ["inspect_cohort", "execute_read_query", "finish"]),
        ("country 和 project 的交叉覆盖", ["inspect_cohort", "execute_read_query", "finish"]),
        ("一个候选 feature 的非零记录覆盖", ["execute_read_query", "finish"]),
        ("丰度版本与来源批次的字段覆盖", ["inspect_cohort", "finish"]),
        ("样本键与元数据记录的连接覆盖", ["inspect_cohort", "execute_read_query", "finish"]),
    ]
    for label, path in data_specs:
        tasks.append(_task(
            number,
            "data_fact",
            f"只核对 {label}，输出有界汇总，不输出原始行或样本标识。",
            path,
        ))
        number += 1

    focused_specs = [
        ("候选 feature 的组间差异与有界投影", ["inspect_cohort", "compare_groups", "analyze_projection", "finish"]),
        ("候选 feature 的年龄分层差异", ["inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "finish"]),
        ("候选 feature 的性别分层差异", ["inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "finish"]),
        ("候选 feature 的 country 分层差异", ["inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "finish"]),
        ("控制年龄混杂后的组间差异", ["inspect_cohort", "compare_groups", "adjust_confounders", "analyze_projection", "finish"]),
        ("控制性别混杂后的组间差异", ["inspect_cohort", "compare_groups", "adjust_confounders", "analyze_projection", "finish"]),
        ("控制 country 与 project 后的组间差异", ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "analyze_projection", "finish"]),
        ("跨 project 方向一致性", ["inspect_cohort", "compare_groups", "cross_project_validate", "analyze_projection", "finish"]),
        ("跨 disease 特异性", ["inspect_cohort", "compare_groups", "cross_disease_validate", "analyze_projection", "finish"]),
        ("跨 project 稳定性后的文献补证", ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence", "finish"]),
        ("控制混杂后的文献补证", ["inspect_cohort", "compare_groups", "adjust_confounders", "retrieve_evidence", "finish"]),
        ("按年龄和性别分层结果的文献补证", ["inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "retrieve_evidence", "finish"]),
        ("跨 disease 结果的文献补证", ["inspect_cohort", "compare_groups", "cross_disease_validate", "retrieve_evidence", "finish"]),
        ("跨 project 与跨 disease 的联合投影", ["inspect_cohort", "compare_groups", "cross_project_validate", "cross_disease_validate", "analyze_projection", "finish"]),
        ("多 feature 跨 project 稳定性验证后的有界投影", ["inspect_cohort", "compare_groups", "cross_project_validate", "analyze_projection", "retrieve_evidence", "finish"]),
        ("比较结果后的有界证据检索", ["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"]),
    ]
    for spec_index, (label, path) in enumerate(focused_specs):
        for variant in range(3):
            tasks.append(_task(
                number,
                "focused_analysis",
                f"对两个已批准研究组完成{label}；这是状态差异变体 {variant + 1}，只使用已批准字段和有界投影，不输出诊断或因果结论。",
                path,
                variant=variant,
            ))
            number += 1

    open_specs = [
        ("跨 project 稳定性", ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence", "finish"], "project 分布已可比较"),
        ("project 不平衡下的稳定性", ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "retrieve_evidence", "finish"], "project 分布明显不平衡"),
        ("年龄性别混杂后的稳定性", ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "retrieve_evidence", "finish"], "年龄和性别存在混杂"),
        ("country 与 project 的联合稳定性", ["inspect_cohort", "compare_groups", "stratified_analysis", "cross_project_validate", "retrieve_evidence", "finish"], "country 与 project 需要同时检查"),
        ("缺失 metadata 对稳定性的影响", ["inspect_cohort", "compare_groups", "stratified_analysis", "retrieve_evidence", "finish"], "缺失 metadata 需要保留限制"),
        ("跨 disease 特异性", ["inspect_cohort", "compare_groups", "cross_disease_validate", "retrieve_evidence", "finish"], "跨 disease 需要验证特异性"),
        ("跨 project 与跨 disease 联合验证", ["inspect_cohort", "compare_groups", "cross_project_validate", "cross_disease_validate", "analyze_projection", "retrieve_evidence", "finish"], "project 稳定性和 disease 特异性都必须验证"),
        ("跨 project、混杂与跨 disease 联合验证", ["inspect_cohort", "compare_groups", "adjust_confounders", "cross_project_validate", "cross_disease_validate", "analyze_projection", "retrieve_evidence", "finish"], "混杂、project 稳定性和 disease 特异性都必须验证"),
        ("多 feature 的共同变化", ["inspect_cohort", "compare_groups", "analyze_projection", "retrieve_evidence", "finish"], "多个 feature 共同变化需要投影分析"),
        ("多 feature 跨 project 稳定性", ["inspect_cohort", "compare_groups", "cross_project_validate", "analyze_projection", "retrieve_evidence", "finish"], "多个 feature 共同变化并验证跨 project 稳定性"),
        ("证据冲突下的质量风险", ["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"], "数据和知识证据可能冲突，结论必须降级", "QUALITY_RISK"),
        ("信息不足时的有界停止", ["inspect_cohort", "compare_groups", "finish"], "当前信息不足，不允许无依据扩展结论", "NO_NEW_INFORMATION"),
        ("控制混杂后的微生态现象探索", ["inspect_cohort", "compare_groups", "adjust_confounders", "analyze_projection", "retrieve_evidence", "finish"], "年龄或 country 混杂需要先控制"),
        ("按年龄和性别分层后寻找稳定现象", ["inspect_cohort", "compare_groups", "stratified_analysis", "analyze_projection", "retrieve_evidence", "finish"], "年龄和性别分层结果需要再次投影"),
        ("跨 project 后寻找候选 feature", ["inspect_cohort", "compare_groups", "cross_project_validate", "analyze_projection", "retrieve_evidence", "finish"], "跨 project 验证后需要总结候选 feature"),
        ("跨 disease 后寻找候选 feature", ["inspect_cohort", "compare_groups", "cross_disease_validate", "analyze_projection", "retrieve_evidence", "finish"], "跨 disease 验证后需要总结候选 feature"),
        ("可比 cohort 的稳定性探索", ["inspect_cohort", "compare_groups", "cross_project_validate", "retrieve_evidence", "finish"], "病例和对照 project 可比但仍需验证稳定性"),
        ("country 分层的受限探索", ["inspect_cohort", "stratified_analysis", "analyze_projection", "finish"], "country 分层足以回答当前有界问题"),
        ("比较后补充有限文献证据", ["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"], "数据观察完成后才允许补充证据"),
        ("冲突证据下的最小化结论", ["inspect_cohort", "compare_groups", "retrieve_evidence", "finish"], "冲突证据只允许输出降级结论", "QUALITY_RISK"),
    ]
    for spec_index, spec in enumerate(open_specs):
        label, path, condition, *stop = spec
        for variant in range(3):
            question = (
                f"开放探索状态差异第 {spec_index + 1} 组变体 {variant + 1}：{label}。"
                f"当前条件是{condition}；请先检查数据状态，再按证据充分性决定继续验证、补证或停止。"
            )
            if "retrieve_evidence" in path:
                question += " 文献证据只能在数据观察之后调用。"
            tasks.append(_task(
                number,
                "open_exploration",
                question,
                path,
                variant=variant,
                stop_reason=stop[0] if stop else None,
            ))
            number += 1

    return {
        "schemaVersion": VERSION,
        "name": "Mico Decision state-difference task pool v2",
        "purpose": "new_state_difference_only",
        "caseCount": len(tasks),
        "kindDistribution": dict(Counter(item["kind"] for item in tasks)),
        "baseTaskSets": [
            "p2j4-decision-state-difference-v1",
            "p2j4-task-set-v3-reviewed",
            "p2j4-hard-task-set-v1",
            "p2j4-hard-variant-task-set-v1",
        ],
        "trainingStarted": False,
        "cases": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "READY",
        "schemaVersion": result["schemaVersion"],
        "caseCount": result["caseCount"],
        "kindDistribution": result["kindDistribution"],
        "trainingStarted": False,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
