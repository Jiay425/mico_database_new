"""Canonical Scientific Action → Materializer contract manifest.

The manifest is deliberately descriptive.  It does not route a scientific
task or construct a plan; it records the one canonical wire contract that a
selected Action is allowed to materialize into, together with the current
runtime dispatch envelope retained for compatibility.  The static audit and
the provider canary both consume this table so contract drift is visible
before a long Agent Loop is started.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .research import ScientificActionName


MaterializerContractKind = Literal[
    "query_plan",
    "analysis_plan_v2",
    "retrieval_arguments",
    "none",
]
MaterializerProducer = Literal["model", "runtime_owned"]


@dataclass(frozen=True)
class MaterializerContract:
    """One canonical materialization boundary for one Scientific Action."""

    action: ScientificActionName
    materializer_required: bool
    producer: MaterializerProducer
    contract_kind: MaterializerContractKind
    schema_name: str | None
    discriminator: str | None
    materializer_methods: tuple[str, ...]
    runtime_dispatch_envelope: str | None
    prompt_entrypoints: tuple[str, ...] = ()
    response_parsers: tuple[str, ...] = ()
    domain_validators: tuple[str, ...] = ()
    legacy_contracts: tuple[str, ...] = ()
    notes: str = ""


# Keep this tuple exhaustive and ordered like the closed Scientific Action
# union.  Analysis Action names intentionally retain their public names while
# ``analysis_type`` remains the canonical execution discriminator.
MATERIALIZER_CONTRACTS: tuple[MaterializerContract, ...] = (
    MaterializerContract(
        action="inspect_cohort",
        materializer_required=False,
        producer="runtime_owned",
        contract_kind="query_plan",
        schema_name="QueryPlan",
        discriminator="actionName=inspect_cohort",
        materializer_methods=("_runtime_owned_inspection_action",),
        runtime_dispatch_envelope="InspectCohortAction.arguments.queryPlan",
        response_parsers=("InspectCohortArguments", "validate_scientific_action"),
        domain_validators=("validate_query_plan_catalog", "Java QueryPlan compiler"),
        legacy_contracts=("InspectCohortArguments.sql",),
        notes="Metadata-first catalog probe; no model call in the dynamic path.",
    ),
    MaterializerContract(
        action="execute_read_query",
        materializer_required=True,
        producer="model",
        contract_kind="query_plan",
        schema_name="QueryPlan",
        discriminator="actionName=execute_read_query",
        materializer_methods=("HttpResearchPlannerPort.plan_action",),
        runtime_dispatch_envelope="ExecuteReadQueryAction.arguments.queryPlan",
        prompt_entrypoints=("HttpResearchPlannerPort.plan_action",),
        response_parsers=("ExecuteReadQueryArguments", "QueryPlan.model_validate", "validate_scientific_action"),
        domain_validators=("validate_query_plan_catalog", "Java QueryPlan compiler"),
        legacy_contracts=("ExecuteReadQueryArguments.sql", "sqlDraft"),
        notes="Canonical plan is a catalog-bound QueryPlan; the ScientificAction envelope is dispatch-only.",
    ),
    MaterializerContract(
        action="compare_groups",
        materializer_required=True,
        producer="model",
        contract_kind="analysis_plan_v2",
        schema_name="AnalysisPlan",
        discriminator="analysis_type=group_comparison",
        materializer_methods=(
            "HttpResearchPlannerPort.generate_typed_analysis",
        ),
        runtime_dispatch_envelope="CompareGroupsAction.arguments",
        prompt_entrypoints=("HttpResearchPlannerPort.generate_typed_analysis",),
        response_parsers=("TypedAnalysisPlannerResult", "AnalysisPlan.model_validate"),
        domain_validators=("match_analysis_capability", "_ACTION_ANALYSIS_TYPE"),
        legacy_contracts=("GeneratedAnalysisPlan", "HttpResearchPlannerPort.generate_analysis"),
    ),
    MaterializerContract(
        action="analyze_projection",
        materializer_required=True,
        producer="model",
        contract_kind="analysis_plan_v2",
        schema_name="AnalysisPlan",
        discriminator="analysis_type=projection",
        materializer_methods=(
            "HttpResearchPlannerPort.generate_typed_analysis",
        ),
        runtime_dispatch_envelope="AnalyzeProjectionAction.arguments",
        prompt_entrypoints=("HttpResearchPlannerPort.generate_typed_analysis",),
        response_parsers=("TypedAnalysisPlannerResult", "AnalysisPlan.model_validate"),
        domain_validators=("match_analysis_capability", "_ACTION_ANALYSIS_TYPE"),
        legacy_contracts=(
            "AnalyzeProjectionArguments.observationId",
            "HttpResearchPlannerPort.generate_analysis",
        ),
        notes="The dispatch envelope uses singular observationId; the canonical v2 plan uses source_observation_ids.",
    ),
    MaterializerContract(
        action="stratified_analysis",
        materializer_required=True,
        producer="model",
        contract_kind="analysis_plan_v2",
        schema_name="AnalysisPlan",
        discriminator="analysis_type=stratified_comparison",
        materializer_methods=(
            "HttpResearchPlannerPort.generate_typed_analysis",
        ),
        runtime_dispatch_envelope="StratifiedAnalysisAction.arguments",
        prompt_entrypoints=("HttpResearchPlannerPort.generate_typed_analysis",),
        response_parsers=("TypedAnalysisPlannerResult", "AnalysisPlan.model_validate"),
        domain_validators=("match_analysis_capability", "_ACTION_ANALYSIS_TYPE"),
        legacy_contracts=("GeneratedAnalysisPlan", "HttpResearchPlannerPort.generate_analysis"),
        notes=(
            "Public Action name is stratified_analysis; canonical AnalysisPlan discriminator "
            "is stratified_comparison. Numeric stratifiers require the closed "
            "NumericStratificationSpec for typed execution."
        ),
    ),
    MaterializerContract(
        action="adjust_confounders",
        materializer_required=True,
        producer="model",
        contract_kind="analysis_plan_v2",
        schema_name="AnalysisPlan",
        discriminator="analysis_type=confounder_adjustment",
        materializer_methods=(
            "HttpResearchPlannerPort.generate_typed_analysis",
        ),
        runtime_dispatch_envelope="AdjustConfoundersAction.arguments",
        prompt_entrypoints=("HttpResearchPlannerPort.generate_typed_analysis",),
        response_parsers=("TypedAnalysisPlannerResult", "AnalysisPlan.model_validate"),
        domain_validators=("match_analysis_capability", "_ACTION_ANALYSIS_TYPE"),
        legacy_contracts=("GeneratedAnalysisPlan", "HttpResearchPlannerPort.generate_analysis"),
    ),
    MaterializerContract(
        action="cross_project_validate",
        materializer_required=True,
        producer="model",
        contract_kind="analysis_plan_v2",
        schema_name="AnalysisPlan",
        discriminator="analysis_type=cross_project_validation",
        materializer_methods=(
            "HttpResearchPlannerPort.generate_typed_analysis",
        ),
        runtime_dispatch_envelope="CrossProjectValidateAction.arguments",
        prompt_entrypoints=("HttpResearchPlannerPort.generate_typed_analysis",),
        response_parsers=("TypedAnalysisPlannerResult", "AnalysisPlan.model_validate"),
        domain_validators=("match_analysis_capability", "_ACTION_ANALYSIS_TYPE"),
        legacy_contracts=("GeneratedAnalysisPlan", "HttpResearchPlannerPort.generate_analysis"),
        notes="validation_field must be an independent verified dimension with at least two observed values.",
    ),
    MaterializerContract(
        action="cross_disease_validate",
        materializer_required=True,
        producer="model",
        contract_kind="analysis_plan_v2",
        schema_name="AnalysisPlan",
        discriminator="analysis_type=cross_disease_validation",
        materializer_methods=(
            "HttpResearchPlannerPort.generate_typed_analysis",
        ),
        runtime_dispatch_envelope="CrossDiseaseValidateAction.arguments",
        prompt_entrypoints=("HttpResearchPlannerPort.generate_typed_analysis",),
        response_parsers=("TypedAnalysisPlannerResult", "AnalysisPlan.model_validate"),
        domain_validators=("match_analysis_capability", "_ACTION_ANALYSIS_TYPE"),
        legacy_contracts=("GeneratedAnalysisPlan", "HttpResearchPlannerPort.generate_analysis"),
        notes="validation_field must be an independent verified dimension with at least two observed values.",
    ),
    MaterializerContract(
        action="retrieve_evidence",
        materializer_required=True,
        producer="model",
        contract_kind="retrieval_arguments",
        schema_name="RetrieveEvidenceArguments",
        discriminator="actionName=retrieve_evidence",
        materializer_methods=("HttpResearchPlannerPort.plan_action",),
        runtime_dispatch_envelope="RetrieveEvidenceAction.arguments",
        prompt_entrypoints=("HttpResearchPlannerPort.plan_action",),
        response_parsers=("RetrieveEvidenceArguments", "validate_scientific_action"),
        domain_validators=("EvidenceQuery", "KnowledgeSearchPort.search_parallel"),
        legacy_contracts=("RetrievalPlan (intent compatibility path)",),
        notes="Scientific workflow currently materializes bounded retrieval arguments; the Knowledge port consumes EvidenceQuery.",
    ),
    MaterializerContract(
        action="finish",
        materializer_required=False,
        producer="runtime_owned",
        contract_kind="none",
        schema_name="FinishArguments",
        discriminator="actionName=finish",
        materializer_methods=("_finish_action",),
        runtime_dispatch_envelope="FinishAction.arguments",
        response_parsers=("FinishArguments", "validate_scientific_action"),
        domain_validators=("hard finish guard",),
        notes="No Materializer call; Runtime validates the closed stop reason.",
    ),
)


MATERIALIZER_CONTRACT_BY_ACTION: dict[ScientificActionName, MaterializerContract] = {
    item.action: item for item in MATERIALIZER_CONTRACTS
}


def materializer_contract_for_action(action: ScientificActionName) -> MaterializerContract:
    """Return the exhaustive canonical contract for ``action``."""

    return MATERIALIZER_CONTRACT_BY_ACTION[action]


__all__ = [
    "MaterializerContract",
    "MaterializerContractKind",
    "MaterializerProducer",
    "MATERIALIZER_CONTRACTS",
    "MATERIALIZER_CONTRACT_BY_ACTION",
    "materializer_contract_for_action",
]
