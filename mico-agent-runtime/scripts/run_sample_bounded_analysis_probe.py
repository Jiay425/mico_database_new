"""No-quota probe for the 4D-1G sample-bounded analysis contract.

This command intentionally does not construct a Gemini/DeepSeek provider.  It
first runs the same remote MySQL + Java preflight used by the dynamic canary,
then executes one Runtime-owned sample-bounded QueryPlan against the real Java
tool and feeds its result to the real typed operators.  Raw rows and opaque
sample keys stay in memory; the report contains only aggregate diagnostics.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "sample_bounded_analysis_probe"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mico_agent_runtime.contracts.materialization import AnalysisPlan, QueryPlan
from mico_agent_runtime.contracts.tools import (
    ExecuteReadQueryArguments,
    ExecuteReadQueryJavaToolCall,
)
from mico_agent_runtime.graph.generated_analysis import GeneratedAnalysisError, execute_typed_analysis
from mico_agent_runtime.ports.java_agent import HttpJavaAgentToolPort
from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort
from scripts.check_scientific_chain_services import run_preflight


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _distinct_counts(rows: list[dict[str, Any]], value_key: str) -> dict[str, int]:
    counts: dict[str, set[object]] = {}
    for row in rows:
        label = row.get(value_key)
        sample_key = row.get("a_analysis_sample_key")
        if label is None or sample_key is None:
            continue
        counts.setdefault(str(label), set()).add(sample_key)
    return {label: len(keys) for label, keys in sorted(counts.items())}


def _query_summary(plan: QueryPlan, response: Any) -> dict[str, Any]:
    data = response.data if isinstance(getattr(response, "data", None), dict) else {}
    rows = [row for row in data.get("rows", []) if isinstance(row, dict)]
    columns = [item for item in data.get("columns", []) if isinstance(item, str)]
    feature_values = {
        str(row.get("a_abundance_feature"))
        for row in rows
        if row.get("a_abundance_feature") is not None
    }
    return {
        "status": getattr(response, "status", None),
        "row_count": getattr(response, "rowCount", len(rows)),
        "returned_rows": len(rows),
        "columns": columns,
        "unique_samples": len({
            row.get("a_analysis_sample_key")
            for row in rows
            if row.get("a_analysis_sample_key") is not None
        }),
        "unique_samples_by_disease": _distinct_counts(rows, "a_sample_disease"),
        "distinct_features": len(feature_values),
        "sample_bound": {
            "sample_limit_per_group": plan.sample_limit_per_group,
            "sample_limit_group_field": plan.sample_limit_group_field,
            "limit": plan.limit,
        },
        "opaque_sample_key_returned": "a_analysis_sample_key" in columns,
    }


def _analysis_summary(result: Any) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    # Preserve the complete bounded metrics and only a small feature preview;
    # the result object itself remains available to the caller in memory.
    return {
        "status": payload.get("status"),
        "analysis_type": payload.get("analysis_type"),
        "execution_mode": payload.get("execution_mode"),
        "method_used": payload.get("method_used"),
        "metrics": payload.get("metrics", {}),
        "ranking_method": payload.get("ranking_method"),
        "adjusted_covariates": payload.get("adjusted_covariates", []),
        "feature_results_preview": payload.get("feature_results", [])[:5],
        "feature_results_count": len(payload.get("feature_results", [])),
    }


def run_probe(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = asyncio.run(run_preflight())
    _dump(output_dir / "preflight.json", preflight)
    if preflight.get("status") != "pass":
        report = {
            "status": "failed",
            "error": "CHAIN_PREFLIGHT_FAILED",
            "gemini_calls_made": 0,
            "deepseek_calls_made": 0,
            "preflight": preflight,
        }
        _dump(output_dir / "report.json", report)
        return report

    base_url = os.environ.get("MICO_JAVA_AGENT_TOOL_BASE_URL", "").strip()
    token = os.environ.get("MICO_AGENT_INTERNAL_TOKEN", "")
    if not base_url or not token:
        report = {
            "status": "failed",
            "error": "JAVA_TOOL_CONFIGURATION_MISSING",
            "gemini_calls_made": 0,
            "deepseek_calls_made": 0,
            "preflight": preflight,
        }
        _dump(output_dir / "report.json", report)
        return report

    java = HttpJavaAgentToolPort(base_url, token)
    try:
        seed = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        run_id = "probe-" + hashlib.sha256(seed.encode()).hexdigest()[:24]
        task_id = "task-" + hashlib.sha256((seed + "task").encode()).hexdigest()[:32]
        call_id = "call-" + hashlib.sha256((seed + "call").encode()).hexdigest()[:32]
        catalog = JavaSchemaCatalogPort(java).load(run_id=run_id, task_id=task_id)
        plan = QueryPlan(
            root_entity="sample",
            relation_path=["sample_to_abundance"],
            select_fields=[
                "sample.disease",
                "sample.age",
                "abundance.feature",
                "abundance.value",
            ],
            filters=[{
                "field": "sample.disease",
                "operator": "in",
                "value": ["T2D", "healthy"],
            }],
            limit=20_000,
            sample_limit_per_group=50,
            sample_limit_group_field="sample.disease",
        )
        call = ExecuteReadQueryJavaToolCall(
            toolName="execute_read_query",
            runId=run_id,
            toolCallId=call_id,
            arguments=ExecuteReadQueryArguments(
                queryPlan=plan,
                includeAnalysisSampleKey=True,
            ),
        )
        response = java.execute(call)
        query_summary = _query_summary(plan, response)
        _dump(output_dir / "query_summary.json", query_summary)
        if response.status != "COMPLETED" or not isinstance(response.data, dict):
            report = {
                "status": "failed",
                "error": getattr(getattr(response, "error", None), "code", response.status),
                "gemini_calls_made": 0,
                "deepseek_calls_made": 0,
                "preflight": preflight,
                "query": query_summary,
            }
            _dump(output_dir / "report.json", report)
            return report

        rows = [row for row in response.data.get("rows", []) if isinstance(row, dict)]
        observation_id = "observation-" + ("a" * 32)
        compare_plan = AnalysisPlan(
            analysis_type="group_comparison",
            source_observation_ids=[observation_id],
            outcome="abundance.value",
            feature_field="abundance.feature",
            group_field="sample.disease",
            metrics=["effect_size", "p_value"],
        )
        adjust_plan = AnalysisPlan(
            analysis_type="confounder_adjustment",
            source_observation_ids=[observation_id],
            outcome="abundance.value",
            feature_field="abundance.feature",
            group_field="sample.disease",
            covariates=["sample.age"],
            metrics=["effect_size", "p_value"],
        )
        compare_result = execute_typed_analysis(compare_plan, rows, response.rowCount)
        adjust_error: str | None = None
        adjust_result: Any | None = None
        try:
            adjust_result = execute_typed_analysis(adjust_plan, rows, response.rowCount)
        except (GeneratedAnalysisError, ValidationError, ValueError) as exc:
            adjust_error = str(exc)
        _dump(output_dir / "compare_result.json", _analysis_summary(compare_result))
        if adjust_result is not None:
            _dump(output_dir / "adjust_result.json", _analysis_summary(adjust_result))
        report = {
            "status": "pass" if adjust_result is not None else "partial",
            "gemini_calls_made": 0,
            "deepseek_calls_made": 0,
            "preflight": preflight,
            "catalog": {
                "schema_version": catalog.schemaVersion,
                "entity_count": len(catalog.entities),
                "internal_analysis_fields": list(catalog.internalAnalysisFields),
            },
            "query": query_summary,
            "compare_groups": _analysis_summary(compare_result),
            "adjust_confounders": (
                _analysis_summary(adjust_result)
                if adjust_result is not None
                else {"error": adjust_error}
            ),
        }
        _dump(output_dir / "report.json", report)
        return report
    finally:
        java.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the no-quota sample-bounded typed analysis probe")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_probe(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
