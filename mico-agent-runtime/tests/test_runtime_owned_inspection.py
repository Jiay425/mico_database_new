from datetime import datetime, timezone

from mico_agent_runtime.contracts.research import ScientificPlannerContext
from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.ports.decision_policy import DecisionPolicyOutput
from mico_agent_runtime.ports.research_planner import HybridIntentPlannerPort


def _catalog() -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=datetime(2026, 8, 27, tzinfo=timezone.utc),
        entities=[
            SchemaEntitySemantics(
                entityId="sample",
                entityName="sample",
                sourceTable="sample",
                fields=[
                    SchemaFieldSemantics(
                        fieldId="sample.project",
                        name="project",
                        dataType="string",
                        nullable=True,
                        semanticStatus="verified",
                        displayable=True,
                        description="redacted project field",
                    ),
                    SchemaFieldSemantics(
                        fieldId="sample.value",
                        name="value",
                        dataType="number",
                        nullable=True,
                        semanticStatus="verified",
                        displayable=True,
                        description="redacted numeric field",
                    ),
                ],
            )
        ],
        queryRules=[
            "select_or_with_only",
            "explicit_columns_only",
            "no_cross_database_reference",
            "bounded_limit_required",
            "java_final_validation",
        ],
    )


class _DecisionPlanner:
    def select_action(self, _context: ScientificPlannerContext) -> DecisionPolicyOutput:
        return DecisionPolicyOutput(
            selected_action="inspect_cohort",
            decision_reason="inspect metadata before the first data read",
        )


class _ExplodingMaterializer:
    def plan_action(self, _context: ScientificPlannerContext):
        raise AssertionError("inspect_cohort must not call the materializer")


def test_dynamic_inspection_is_runtime_owned_and_typed() -> None:
    context = ScientificPlannerContext(
        questionSummary="inspect available cohort metadata",
        intent="scientific_exploration",
        approvedActions=["inspect_cohort", "finish"],
        remainingActionBudget=4,
        schemaCatalog=_catalog(),
    )

    result = HybridIntentPlannerPort(
        _ExplodingMaterializer(),
        _DecisionPlanner(),
    ).plan_action(context)

    assert result.mode == "sft_policy"
    assert result.runtimeOwned is True
    assert result.action.actionName == "inspect_cohort"
    assert result.action.arguments.queryPlan is not None
    assert result.action.arguments.queryPlan.select_fields == [
        "sample.project",
        "sample.value",
    ]


def test_dynamic_inspection_bounds_long_policy_reason_for_action_contract() -> None:
    context = ScientificPlannerContext(
        questionSummary="inspect available cohort metadata",
        intent="scientific_exploration",
        approvedActions=["inspect_cohort"],
        remainingActionBudget=4,
        schemaCatalog=_catalog(),
    )
    long_reason = "reason " * 100

    class LongReasonPlanner:
        def select_action(self, _context: ScientificPlannerContext) -> DecisionPolicyOutput:
            return DecisionPolicyOutput(
                selected_action="inspect_cohort",
                decision_reason=long_reason[:512],
            )

    result = HybridIntentPlannerPort(
        _ExplodingMaterializer(),
        LongReasonPlanner(),
    ).plan_action(context)

    assert len(result.action.rationale) == 256
    assert result.decisionReason == long_reason[:512]
