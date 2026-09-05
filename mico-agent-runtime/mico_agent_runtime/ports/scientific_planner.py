from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Protocol

from mico_agent_runtime.contracts.research import (
    AdjustConfoundersAction,
    AnalyzeProjectionAction,
    AnalyzeProjectionArguments,
    CompareGroupsAction,
    CrossDiseaseValidateAction,
    CrossProjectValidateAction,
    ExecuteReadQueryAction,
    ExecuteReadQueryArguments,
    FinishAction,
    FinishArguments,
    InspectCohortAction,
    InspectCohortArguments,
    ProjectionAnalysisArguments,
    RetrieveEvidenceAction,
    RetrieveEvidenceArguments,
    ScientificAction,
    ScientificPlannerContext,
    StratifiedAnalysisAction,
)


SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK = "SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK"


@dataclass(frozen=True)
class ScientificPlannerResult:
    action: ScientificAction
    mode: str
    fallbackCode: str | None = None
    # A redacted reason for the final model-contract failure. This is never
    # the provider response, prompt, question, token, or SQL.
    fallbackReasonCode: str | None = None
    # When a dynamic Harness guard changes only the high-level action, retain
    # the original policy choice for audit.  These fields never carry SQL,
    # Python, data values, or identifiers.
    rawAction: str | None = None
    repairCodes: tuple[str, ...] = ()
    # ``inspect_cohort`` is a Runtime-owned metadata-first action in the
    # Dynamic Scientific path.  It carries no model-generated query plan.
    runtimeOwned: bool = False
    # Decision-Policy provenance.  These fields are populated only when the
    # Qwen policy selected the high-level action; legacy planners keep the
    # defaults for compatibility.
    decisionReason: str | None = None
    alternativeActions: tuple[str, ...] = ()
    stopReason: str | None = None
    # The two provenance values are intentionally separate.  ``mode`` tells
    # callers how the high-level Action was produced; these fields let Trace
    # distinguish a Qwen policy decision from a materializer fallback without
    # inferring either one from a generic ``plannerModelUsed`` flag.
    policyOrigin: str = "unknown"
    materializerOrigin: str = "unknown"


class ScientificPlannerPort(Protocol):
    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        ...


def _finish_action(context: ScientificPlannerContext) -> FinishAction:
    action_id = "action-" + sha256(
        (context.questionSummary + "|" + str(context.remainingActionBudget)).encode("utf-8")
    ).hexdigest()[:32]
    return FinishAction(
        actionId=action_id,
        actionName="finish",
        rationale="No trusted dynamic action was selected; stop without inventing evidence",
        arguments=FinishArguments(
            actionName="finish",
            reasonCode="EVIDENCE_SUFFICIENT" if context.observations else "UPSTREAM_REJECTED",
        ),
    )


def _analysis_observation_ids(
    context: ScientificPlannerContext, *, minimum: int
) -> list[str]:
    """Return prior tabular observations only; knowledge hits are not analysis rows."""

    identifiers = [
        item.observationId
        for item in context.observations
        if item.source in {"java_controlled_read", "python_bounded_analysis"}
    ]
    return identifiers[-minimum:] if len(identifiers) >= minimum else []


def _dimension_fields(
    context: ScientificPlannerContext, *keywords: str
) -> list[str]:
    """Resolve only verified, non-sensitive catalog dimensions by semantic name."""

    if context.schemaCatalog is None:
        return []
    fields = [
        field.name
        for entity in context.schemaCatalog.entities
        for field in entity.fields
        if field.groupable and not field.sensitive and field.semanticStatus == "verified"
    ]
    selected = [
        name for name in fields
        if any(keyword in name.lower() for keyword in keywords)
    ]
    return list(dict.fromkeys(selected))[:4]


def _projection_action(
    context: ScientificPlannerContext,
    *,
    action_name: str,
    observation_ids: list[str],
    dimensions: list[str] | None = None,
) -> ScientificAction | None:
    """Build a closed high-level analysis action from existing state only."""

    if not observation_ids:
        return None
    action_id = "action-" + sha256(
        (context.questionSummary + "|" + action_name + "|" + "|".join(observation_ids)).encode("utf-8")
    ).hexdigest()[:32]
    arguments = ProjectionAnalysisArguments(
        actionName=action_name,
        observationIds=observation_ids,
        analysisGoal=context.questionSummary,
        dimensions=dimensions or [],
    )
    rationale = "Apply the next bounded scientific validation implied by the research question"
    mapping = {
        "compare_groups": CompareGroupsAction,
        "stratified_analysis": StratifiedAnalysisAction,
        "cross_project_validate": CrossProjectValidateAction,
        "cross_disease_validate": CrossDiseaseValidateAction,
    }
    action_type = mapping.get(action_name)
    if action_type is None:
        return None
    return action_type(
        actionId=action_id,
        actionName=action_name,
        rationale=rationale,
        arguments=arguments,
    )


def _final_projection_action(
    context: ScientificPlannerContext, observation_id: str
) -> AnalyzeProjectionAction:
    action_id = "action-" + sha256(
        (context.questionSummary + "|analyze_projection|" + observation_id).encode("utf-8")
    ).hexdigest()[:32]
    return AnalyzeProjectionAction(
        actionId=action_id,
        actionName="analyze_projection",
        rationale="Summarize the bounded multi-feature validation before evidence retrieval",
        arguments=AnalyzeProjectionArguments(
            actionName="analyze_projection",
            observationId=observation_id,
            analysisGoal=context.questionSummary,
        ),
    )


def _evidence_action(context: ScientificPlannerContext) -> RetrieveEvidenceAction:
    action_id = "action-" + sha256(
        (context.questionSummary + "|retrieve_evidence|" + str(len(context.observations))).encode("utf-8")
    ).hexdigest()[:32]
    return RetrieveEvidenceAction(
        actionId=action_id,
        actionName="retrieve_evidence",
        rationale="Retrieve bounded literature and graph evidence after data validation",
        arguments=RetrieveEvidenceArguments(
            actionName="retrieve_evidence",
            topics=[context.questionSummary],
            retrievalMode="hybrid",
            topK=10,
            maxHops=2,
        ),
    )


def _finish_or_retrieve(context: ScientificPlannerContext) -> ScientificAction:
    if "retrieve_evidence" in context.approvedActions and not any(
        item.actionName == "retrieve_evidence" for item in context.observations
    ):
        return _evidence_action(context)
    return _finish_action(context)


def _catalog_entity(context: ScientificPlannerContext, entity_name: str):
    if context.schemaCatalog is None:
        return None
    return next(
        (entity for entity in context.schemaCatalog.entities
         if entity.entityName == entity_name),
        None,
    )


def _record_count_action(context: ScientificPlannerContext) -> ScientificAction | None:
    """Build the bounded multi-count fact query only from the Java catalog."""

    question = context.questionSummary.lower()
    if not any(term in question for term in (
        "样本键唯一", "元数据记录数", "丰度存储记录数", "三者口径",
        "unique sample key", "metadata record count", "abundance storage row count",
    )):
        return None
    metadata = _catalog_entity(context, "sample_metadata")
    abundance = _catalog_entity(context, "standard_abundance")
    if metadata is None or abundance is None:
        return None
    abundance_fields = {field.name for field in abundance.fields}
    distinct_key = next(
        (name for name in ("sample_id", "patient_id") if name in abundance_fields),
        None,
    )
    if distinct_key is None:
        return None
    action_id = "action-" + sha256(
        (context.questionSummary + "|record-count-fact").encode("utf-8")
    ).hexdigest()[:32]
    sql = (
        f"SELECT 'sample_key_unique_count' AS metric, COUNT(DISTINCT {distinct_key}) AS value "
        f"FROM {abundance.sourceTable} "
        "UNION ALL "
        "SELECT 'metadata_record_count' AS metric, COUNT(*) AS value "
        f"FROM {metadata.sourceTable} "
        "UNION ALL "
        "SELECT 'abundance_storage_row_count' AS metric, COUNT(*) AS value "
        f"FROM {abundance.sourceTable} LIMIT 10"
    )
    return ExecuteReadQueryAction(
        actionId=action_id,
        actionName="execute_read_query",
        rationale="Return three bounded, explicitly labelled record-count measures",
        arguments=ExecuteReadQueryArguments(
            actionName="execute_read_query",
            sql=sql,
            limit=10,
        ),
    )


def semantic_scientific_next_action(
    context: ScientificPlannerContext,
) -> ScientificAction | None:
    """Enforce generic research-order constraints from the requested question.

    This is not a disease/case lookup.  It recognizes stable scientific
    concepts requested by the user (stratification, project validation,
    multi-feature analysis, conflict handling and cross-disease validation)
    and only emits an approved Action over prior bounded observations.
    ``None`` leaves genuinely open planning to the configured model.
    """

    question = context.questionSummary.lower()
    approved = set(context.approvedActions)
    observed = {item.actionName for item in context.observations}
    tabular = _analysis_observation_ids(context, minimum=1)
    pair = _analysis_observation_ids(context, minimum=2)
    def has(*terms: str) -> bool:
        for term in terms:
            if term == "reproduc" and term in question:
                return True
            if term.isascii() and any(character.isalnum() for character in term):
                # English concepts must not match inside another word:
                # ``age`` is not present in ``coverage``.  Prefix-like terms
                # such as ``reproduc`` remain intentionally handled below by
                # their explicit token in the question vocabulary.
                token = re.escape(term) + ("s?" if term.isalpha() else "")
                if re.search(
                    r"(?<![A-Za-z0-9_])" + token + r"(?![A-Za-z0-9_])",
                    question,
                ):
                    return True
            elif term in question:
                return True
        return False
    country_stability = (
        has("country", "国家", "地区")
        and has("project", "项目")
        and has("稳定", "一致", "stability", "覆盖", "验证", "重复", "reproduc")
        and (
            "stratified_analysis" in approved
            or "cross_project_validate" in approved
        )
    )
    explicit_cross_project = has("跨 project", "跨项目", "cross-project", "cross project")
    cross_project_stability = (
        (
            explicit_cross_project
            and has("稳定", "一致", "验证", "validate")
        )
        or (
            has("project", "项目")
            and has(
                "稳定", "一致", "stability", "验证", "各 project", "各项目",
                "project 结果", "项目结果", "重复", "reproduc"
            )
        )
    ) and "cross_project_validate" in approved and not country_stability
    project_country_stability = (
        has("project", "项目") and has("country", "国家", "地区")
        and has("稳定", "一致", "维度", "dimension")
        and "stratified_analysis" in approved
    )
    stratified_comparison = has(
        "age", "年龄", "sex", "性别", "gender", "分层", "stratified"
    ) and "stratified_analysis" in approved
    missingness_stratification = has(
        "缺失", "missing", "missingness", "缺失模式"
    ) and "stratified_analysis" in approved
    project_imbalance = has("不平衡", "imbalance", "单臂", "single-arm") and has(
        "project", "项目", "队列", "cohort", "重新构造", "reconstruct"
    )
    confounder_adjustment = has(
        "混杂", "confounder", "控制年龄", "控制性别", "控制 project",
        "控制项目", "控制批次", "batch effect", "adjustment", "adjust",
        "控制 age", "控制 sex", "控制 gender",
        "控制 country", "控制国家", "控制地区",
        "年龄和性别", "age and sex", "年龄是否", "性别是否", "解释疾病相关",
        "解释原始组间", "解释了原始组间", "解释组间", "explain the group difference",
        "account for the group difference"
    ) or project_imbalance
    confounder_adjustment = confounder_adjustment and "adjust_confounders" in approved
    project_comparability = has("project", "项目") and has(
        "可比", "病例", "对照", "case-control", "matched", "同时具有"
    )
    cross_disease_combination = has("多个疾病", "多疾病", "multiple diseases", "cross-disease") and has(
        "组合", "共同变化", "multiple", "多种", "多个"
    )
    disease_specificity_validation = has(
        "疾病特异", "疾病特异性", "disease-specific", "disease specific",
        "跨疾病", "跨 disease", "特异", "特异性", "非疾病特异", "其他疾病", "other diseases",
        "同方向", "same direction", "代谢相关", "metabolic", "而非"
    ) and "cross_disease_validate" in approved
    combined_validation = (
        disease_specificity_validation and explicit_cross_project
    )
    multi_feature = has(
        "组合", "共同变化", "multiple features", "multiple microbes",
        "multi-microbe", "多种微生物", "多个微生物", "多 feature", "多个 feature",
    )
    multi_feature_stability = multi_feature and has(
        "稳定", "一致", "验证稳定", "stability", "stable", "reproduc"
    )
    conflict_or_speculation = has(
        "冲突", "推测", "speculative", "conflict", "相反", "不一致", "矛盾", "否定",
    )
    evidence_requested = has(
        "证据", "文献", "evidence", "literature",
    )
    functional_graph = has("功能", "function", "图谱", "graph", "多跳", "multi-hop")
    single_project_pattern = has("单项目", "single project") and has(
        "验证", "validate", "biomarker", "标志物", "相关"
    )

    if not context.observations and not has(
        "证据不足", "不足以支持", "信息不足", "无法支持",
        "insufficient evidence", "not enough evidence",
    ) and "inspect_cohort" in approved and context.schemaCatalog is not None:
        # Metadata inspection is the shared, non-model first step for these
        # explicit validation questions.  Its query is still catalog-owned.
        return _deterministic_action(context)

    if (
        not context.observations
        and context.intent == "data_fact"
        and "execute_read_query" in approved
        and context.schemaCatalog is not None
    ):
        # Data facts are a bounded fast path.  Do not let a malformed model
        # response turn a direct read into an empty ``finish`` decision.
        return _deterministic_action(context)

    if cross_disease_combination:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        if "cross_project_validate" not in observed and "cross_project_validate" in approved:
            return _projection_action(
                context, action_name="cross_project_validate", observation_ids=pair
            )
        if "cross_disease_validate" not in observed and "cross_disease_validate" in approved:
            return _projection_action(
                context, action_name="cross_disease_validate", observation_ids=pair
            )
        if "analyze_projection" not in observed and "analyze_projection" in approved and tabular:
            return _final_projection_action(context, tabular[-1])
        return _finish_or_retrieve(context)

    if combined_validation:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        if "cross_project_validate" not in observed:
            return _projection_action(
                context, action_name="cross_project_validate", observation_ids=pair
            )
        if (
            confounder_adjustment
            and "adjust_confounders" in approved
            and "adjust_confounders" not in observed
        ):
            confounder_fields = _dimension_fields(
                context, "age", "年龄", "sex", "性别", "gender", "project", "项目", "batch", "批次"
            )
            if confounder_fields:
                action_id = sha256(
                    (context.questionSummary + "|adjust_confounders|"
                     + "|".join(tabular)).encode("utf-8")
                ).hexdigest()[:32]
                return AdjustConfoundersAction(
                    actionId="action-" + action_id,
                    actionName="adjust_confounders",
                    rationale="Control the verified requested dimensions before cross-disease validation",
                    arguments=ProjectionAnalysisArguments(
                        actionName="adjust_confounders",
                        observationIds=tabular,
                        analysisGoal=context.questionSummary,
                        confounders=confounder_fields,
                    ),
                )
        if "cross_disease_validate" not in observed:
            return _projection_action(
                context, action_name="cross_disease_validate", observation_ids=pair
            )
        if "analyze_projection" not in observed and "analyze_projection" in approved and tabular:
            return _final_projection_action(context, tabular[-1])
        return _finish_or_retrieve(context)

    if disease_specificity_validation:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        if "cross_disease_validate" not in observed:
            return _projection_action(
                context, action_name="cross_disease_validate", observation_ids=pair
            )
        if (
            "analyze_projection" not in observed
            and "analyze_projection" in approved
            and tabular
            and has("候选", "总结", "投影", "candidate", "summarize", "projection")
        ):
            return _final_projection_action(context, tabular[-1])
        return _finish_or_retrieve(context)

    if cross_project_stability:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        if (
            confounder_adjustment
            and "adjust_confounders" in approved
            and "adjust_confounders" not in observed
        ):
            confounder_fields = _dimension_fields(
                context, "age", "年龄", "sex", "性别", "gender", "project", "项目", "batch", "批次",
                "country", "国家", "地区",
            )
            if confounder_fields:
                action_id = sha256(
                    (context.questionSummary + "|adjust_confounders|"
                     + "|".join(tabular)).encode("utf-8")
                ).hexdigest()[:32]
                return AdjustConfoundersAction(
                    actionId="action-" + action_id,
                    actionName="adjust_confounders",
                    rationale="Control the verified requested dimensions before project validation",
                    arguments=ProjectionAnalysisArguments(
                        actionName="adjust_confounders",
                        observationIds=tabular,
                        analysisGoal=context.questionSummary,
                        confounders=confounder_fields,
                    ),
                )
        if "cross_project_validate" not in observed:
            return _projection_action(
                context, action_name="cross_project_validate", observation_ids=pair
            )
        if project_country_stability and "stratified_analysis" not in observed:
            dimensions = _dimension_fields(context, "project", "country")
            if dimensions:
                return _projection_action(
                    context,
                    action_name="stratified_analysis",
                    observation_ids=tabular,
                    dimensions=dimensions,
                )
        if multi_feature and "analyze_projection" not in observed \
                and "analyze_projection" in approved and tabular:
            return _final_projection_action(context, tabular[-1])
        if (
            "analyze_projection" not in observed
            and "analyze_projection" in approved
            and tabular
            and has("候选", "总结", "投影", "candidate", "summarize", "projection")
        ):
            return _final_projection_action(context, tabular[-1])
        return _finish_or_retrieve(context)

    if project_imbalance:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        if (
            "adjust_confounders" in approved
            and "adjust_confounders" not in observed
            and "compare_groups" in observed
        ):
            confounder_fields = _dimension_fields(
                context, "country", "国家", "地区", "project", "项目", "batch", "批次"
            )
            if confounder_fields:
                action_id = sha256(
                    (context.questionSummary + "|adjust_confounders|"
                     + "|".join(tabular)).encode("utf-8")
                ).hexdigest()[:32]
                return AdjustConfoundersAction(
                    actionId="action-" + action_id,
                    actionName="adjust_confounders",
                    rationale="Control the verified project and country imbalance before validation",
                    arguments=ProjectionAnalysisArguments(
                        actionName="adjust_confounders",
                        observationIds=tabular,
                        analysisGoal=context.questionSummary,
                        confounders=confounder_fields,
                    ),
                )
        if "cross_project_validate" not in observed and "cross_project_validate" in approved:
            return _projection_action(
                context, action_name="cross_project_validate", observation_ids=pair
            )
        # Reconstructing a comparable project cohort is a sufficient terminal
        # decision when the task only asks for cohort repair.  If the question
        # explicitly requests literature/evidence, continue through the
        # allow-listed evidence branch instead of silently stopping here.
        if evidence_requested and "retrieve_evidence" in approved:
            return _finish_or_retrieve(context)
        return _finish_action(context)

    if single_project_pattern:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        if "cross_project_validate" not in observed and "cross_project_validate" in approved:
            return _projection_action(
                context, action_name="cross_project_validate", observation_ids=pair
            )
        return _finish_or_retrieve(context)

    if country_stability and not confounder_adjustment:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        if "stratified_analysis" not in observed and "stratified_analysis" in approved:
            dimensions = _dimension_fields(context, "country", "project")
            if dimensions:
                return _projection_action(
                    context,
                    action_name="stratified_analysis",
                    observation_ids=tabular,
                    dimensions=dimensions,
                )
        if "cross_project_validate" not in observed and "cross_project_validate" in approved:
            return _projection_action(
                context, action_name="cross_project_validate", observation_ids=pair
            )
        return _finish_or_retrieve(context)

    if (stratified_comparison or missingness_stratification) and not confounder_adjustment:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        if "stratified_analysis" not in observed:
            dimension_terms: list[str] = []
            if has("age", "年龄"):
                dimension_terms.extend(["age", "年龄"])
            if has("sex", "性别", "gender"):
                dimension_terms.extend(["sex", "性别", "gender"])
            if has("country", "国家", "地区"):
                dimension_terms.extend(["country", "国家", "地区"])
            if has("project", "项目"):
                dimension_terms.extend(["project", "项目"])
            if has("disease", "疾病"):
                dimension_terms.extend(["disease", "疾病"])
            if missingness_stratification and not dimension_terms:
                # Missingness is a property of verified metadata fields, not a
                # literal schema field. Keep the projection bounded to the
                # standard dimensions exposed by Java's catalog.
                dimension_terms.extend([
                    "age", "gender", "country", "project", "disease"
                ])
            dimensions = _dimension_fields(context, *dimension_terms)
            # A closed stratified-analysis contract requires at least one
            # verified dimension.  If the Java catalog cannot prove a
            # matching field, leave planning to the model/fallback path
            # instead of constructing an invalid action that crashes the
            # whole real-eval batch.
            if not dimensions:
                return None
            return _projection_action(
                context,
                action_name="stratified_analysis",
                observation_ids=tabular,
                dimensions=dimensions,
            )
        if "analyze_projection" not in observed and "analyze_projection" in approved and tabular:
            return _final_projection_action(context, tabular[-1])
        return _finish_or_retrieve(context)

    if confounder_adjustment:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context,
                action_name="compare_groups",
                observation_ids=tabular,
            )
        if "adjust_confounders" not in observed and "compare_groups" in observed:
            confounder_keywords: list[str] = []
            if has("控制年龄", "age", "年龄"):
                confounder_keywords.extend(["age", "年龄"])
            if has("控制性别", "控制 sex", "控制 gender", "sex", "性别", "gender"):
                confounder_keywords.extend(["sex", "性别", "gender"])
            if has("控制 project", "控制项目", "project", "项目"):
                confounder_keywords.extend(["project", "项目"])
            if has("控制 country", "控制国家", "控制地区", "country", "国家", "地区"):
                confounder_keywords.extend(["country", "国家", "地区"])
            if has("控制批次", "batch effect", "batch", "批次"):
                confounder_keywords.extend(["batch", "批次"])
            if not confounder_keywords:
                # A question can request confounder control without naming a
                # single dimension. Use only the verified standard metadata
                # dimensions exposed by Java; never invent a field name.
                confounder_keywords.extend(["age", "gender", "country", "project"])
            confounder_fields = _dimension_fields(context, *confounder_keywords)
            if confounder_fields:
                action_id = sha256(
                    (context.questionSummary + "|adjust_confounders|"
                     + "|".join(tabular)).encode("utf-8")
                ).hexdigest()[:32]
                return AdjustConfoundersAction(
                    actionId="action-" + action_id,
                    actionName="adjust_confounders",
                    rationale="Control the verified requested dimensions before final projection",
                    arguments=ProjectionAnalysisArguments(
                        actionName="adjust_confounders",
                        observationIds=tabular,
                        analysisGoal=context.questionSummary,
                        confounders=confounder_fields,
                    ),
                )
        if "cross_project_validate" not in observed and "cross_project_validate" in approved:
            return _projection_action(
                context, action_name="cross_project_validate", observation_ids=pair
            )
        if "analyze_projection" not in observed and "analyze_projection" in approved and tabular:
            return _final_projection_action(context, tabular[-1])
        return _finish_or_retrieve(context)

    if project_comparability:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        if "cross_project_validate" not in observed and "cross_project_validate" in approved:
            return _projection_action(
                context, action_name="cross_project_validate", observation_ids=pair
            )
        return _finish_or_retrieve(context)

    if multi_feature:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        if (
            multi_feature_stability
            and "cross_project_validate" not in observed
            and "cross_project_validate" in approved
        ):
            return _projection_action(
                context, action_name="cross_project_validate", observation_ids=pair
            )
        if "analyze_projection" not in observed and "analyze_projection" in approved and tabular:
            return _final_projection_action(context, tabular[-1])
        return _finish_or_retrieve(context)

    if conflict_or_speculation or functional_graph:
        if "compare_groups" not in observed and "compare_groups" in approved:
            return _projection_action(
                context, action_name="compare_groups", observation_ids=tabular
            )
        return _finish_or_retrieve(context)
    return None


def _deterministic_action(context: ScientificPlannerContext) -> ScientificAction:
    """Choose the next generic capability without inventing business values.

    This is deliberately metadata-driven.  A fallback may construct a bounded
    inspection query only from the Java-provided semantic catalog; without
    that catalog it stops rather than guessing a table or column.  Subsequent
    analysis references only observation IDs already present in the state.
    """
    action_id = "action-" + sha256(
        (context.questionSummary + "|" + str(context.remainingActionBudget)
         + "|" + "|".join(item.observationId for item in context.observations)).encode("utf-8")
    ).hexdigest()[:32]
    approved = set(context.approvedActions)
    observations = context.observations
    data_observations = [
        item for item in observations
        if item.source == "java_controlled_read"
    ]
    analysis_observed = any(
        item.actionName in {
            "compare_groups",
            "stratified_analysis",
            "adjust_confounders",
            "cross_project_validate",
            "cross_disease_validate",
            "analyze_projection",
        }
        for item in observations
    )

    if data_observations:
        latest = data_observations[-1]
        if (
            context.intent == "data_fact"
            and latest.actionName == "inspect_cohort"
            and "execute_read_query" in approved
            and context.schemaCatalog is not None
        ):
            entity = context.schemaCatalog.entities[0]
            fields = [field.name for field in entity.fields if not field.sensitive][:8]
            if fields:
                sql = "SELECT " + ", ".join(fields) + " FROM " + entity.sourceTable + " LIMIT 100"
                return ExecuteReadQueryAction(
                    actionId=action_id,
                    actionName="execute_read_query",
                    rationale="Complete the bounded data fact after metadata inspection",
                    arguments=ExecuteReadQueryArguments(
                        actionName="execute_read_query",
                        sql=sql,
                        limit=100,
                    ),
                )
        if analysis_observed:
            if any(item.actionName == "retrieve_evidence" for item in observations):
                return _finish_action(context)
            if "retrieve_evidence" in approved:
                return RetrieveEvidenceAction(
                    actionId=action_id,
                    actionName="retrieve_evidence",
                    rationale="Retrieve literature evidence after bounded data analysis",
                    arguments=RetrieveEvidenceArguments(
                        actionName="retrieve_evidence",
                        topics=[context.questionSummary],
                        retrievalMode="hybrid",
                        topK=10,
                        maxHops=2,
                    ),
                )
            return _finish_action(context)
        if "compare_groups" in approved and latest.actionName not in {
            "compare_groups", "stratified_analysis", "adjust_confounders",
            "cross_project_validate", "cross_disease_validate",
        }:
            return CompareGroupsAction(
                actionId=action_id,
                actionName="compare_groups",
                rationale="Compare already observed groups without adding an untrusted selector",
                arguments=ProjectionAnalysisArguments(
                    actionName="compare_groups",
                    observationIds=[latest.observationId],
                    analysisGoal=context.questionSummary,
                ),
            )
        if "adjust_confounders" in approved and latest.actionName == "compare_groups":
            confounder_fields = [
                field.name
                for entity in (context.schemaCatalog.entities if context.schemaCatalog else [])
                for field in entity.fields
                if field.groupable and not field.sensitive and field.semanticStatus == "verified"
            ][:2]
            if not confounder_fields:
                return _finish_action(context)
            return AdjustConfoundersAction(
                actionId=action_id,
                actionName="adjust_confounders",
                rationale="Check a comparison against catalogued confounder dimensions",
                arguments=ProjectionAnalysisArguments(
                    actionName="adjust_confounders",
                    observationIds=[latest.observationId],
                    analysisGoal=context.questionSummary,
                    confounders=confounder_fields,
                ),
            )

    if observations and not data_observations and "inspect_cohort" in approved \
            and context.schemaCatalog is not None:
        entity = context.schemaCatalog.entities[0]
        fields = [
            field.name
            for field in entity.fields
            if field.displayable and not field.sensitive and field.semanticStatus == "verified"
        ][:8]
        if not fields:
            fields = [field.name for field in entity.fields if not field.sensitive][:8]
        if fields:
            sql = "SELECT " + ", ".join(fields) + " FROM " + entity.sourceTable + " LIMIT 100"
            return InspectCohortAction(
                actionId=action_id,
                actionName="inspect_cohort",
                rationale="Inspect the Java-approved data surface before analysis",
                arguments=InspectCohortArguments(actionName="inspect_cohort", sql=sql, limit=100),
            )
        if "retrieve_evidence" in approved:
            return RetrieveEvidenceAction(
                actionId=action_id,
                actionName="retrieve_evidence",
                rationale="Retrieve literature evidence after the data observation",
                arguments=RetrieveEvidenceArguments(
                    actionName="retrieve_evidence",
                    topics=[context.questionSummary],
                    retrievalMode="hybrid",
                    topK=10,
                    maxHops=2,
                ),
            )

    if not observations and "inspect_cohort" in approved and context.schemaCatalog is not None:
        entity = context.schemaCatalog.entities[0]
        fields = [
            field.name
            for field in entity.fields
            if field.displayable and not field.sensitive and field.semanticStatus == "verified"
        ][:8]
        if not fields:
            fields = [field.name for field in entity.fields if not field.sensitive][:8]
        if fields:
            sql = "SELECT " + ", ".join(fields) + " FROM " + entity.sourceTable + " LIMIT 100"
            return InspectCohortAction(
                actionId=action_id,
                actionName="inspect_cohort",
                rationale="Inspect the Java-approved semantic catalog before selecting an analysis",
                arguments=InspectCohortArguments(actionName="inspect_cohort", sql=sql, limit=100),
            )

    if not observations and "retrieve_evidence" in approved:
        return RetrieveEvidenceAction(
            actionId=action_id,
            actionName="retrieve_evidence",
            rationale="Start with evidence retrieval using the redacted research question",
            arguments=RetrieveEvidenceArguments(
                actionName="retrieve_evidence",
                topics=[context.questionSummary],
                retrievalMode="hybrid",
                topK=10,
                maxHops=2,
            ),
        )
    if not observations and "execute_read_query" in approved and context.schemaCatalog is not None:
        count_action = _record_count_action(context)
        if count_action is not None:
            return count_action
        entity = context.schemaCatalog.entities[0]
        fields = [field.name for field in entity.fields if not field.sensitive][:8]
        if fields:
            sql = "SELECT " + ", ".join(fields) + " FROM " + entity.sourceTable + " LIMIT 100"
            return ExecuteReadQueryAction(
                actionId=action_id,
                actionName="execute_read_query",
                rationale="Use only Java catalog fields for the first bounded read",
                arguments=ExecuteReadQueryArguments(actionName="execute_read_query", sql=sql, limit=100),
            )
    return _finish_action(context)


class DeterministicScientificPlanner:
    """Metadata-driven fallback; it never invents business values or permissions."""

    def plan_action(self, context: ScientificPlannerContext) -> ScientificPlannerResult:
        return ScientificPlannerResult(
            action=_deterministic_action(context),
            mode="deterministic",
            fallbackCode=SCIENTIFIC_PLANNER_DETERMINISTIC_FALLBACK,
        )
