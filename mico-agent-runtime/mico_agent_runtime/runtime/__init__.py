"""Runtime facade for the dynamic analysis path and retained infrastructure.

The public service names remain available, but are loaded lazily.  Small
observation adapters (such as ``decision_state_builder``) must be importable
without constructing a LangGraph service or requiring the optional graph
runtime during contract/unit tests.
"""

from importlib import import_module


__all__ = [
    "IntentRuntime",
    "EvidenceRuntime",
    "ApprovalCoordinator",
    "RunControlCoordinator",
    "AnalysisCapabilityContext",
    "AnalysisCapabilityMatch",
    "AnalysisCapabilityRegistry",
    "match_analysis_capability",
    "ActionAvailabilityContext",
    "ActionAvailabilityResult",
    "compute_available_actions",
    "evaluate_action_availability",
    "ObjectiveResolution",
    "ObjectiveResolutionSummary",
    "active_remaining_objectives",
    "completion_semantics_from_resolution",
    "limitation_code_for_resolution",
    "limitation_codes_from_resolution",
    "persist_objective_resolution",
    "resolve_objective_lifecycle",
    "DataRequirementError",
    "DataRequirementSet",
    "augment_query_plan_with_fields",
    "derive_data_requirements",
]
_MODULES = {
    "IntentRuntime": ".intent_service",
    "EvidenceRuntime": ".evidence_service",
    "ApprovalCoordinator": ".approval",
    "RunControlCoordinator": ".control",
    "AnalysisCapabilityContext": ".analysis_capability_registry",
    "AnalysisCapabilityMatch": ".analysis_capability_registry",
    "AnalysisCapabilityRegistry": ".analysis_capability_registry",
    "match_analysis_capability": ".analysis_capability_registry",
    "ActionAvailabilityContext": ".action_availability",
    "ActionAvailabilityResult": ".action_availability",
    "compute_available_actions": ".action_availability",
    "evaluate_action_availability": ".action_availability",
    "ObjectiveResolution": ".objective_resolution",
    "ObjectiveResolutionSummary": ".objective_resolution",
    "active_remaining_objectives": ".objective_resolution",
    "completion_semantics_from_resolution": ".objective_resolution",
    "limitation_code_for_resolution": ".objective_resolution",
    "limitation_codes_from_resolution": ".objective_resolution",
    "persist_objective_resolution": ".objective_resolution",
    "resolve_objective_lifecycle": ".objective_resolution",
    "DataRequirementError": ".data_requirements",
    "DataRequirementSet": ".data_requirements",
    "augment_query_plan_with_fields": ".data_requirements",
    "derive_data_requirements": ".data_requirements",
}


def __getattr__(name: str):
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
