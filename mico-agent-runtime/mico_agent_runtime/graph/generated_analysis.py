from __future__ import annotations

import ast
import base64
import json
import math
import os
import re
import subprocess
import sys
from pydantic import ValidationError

from mico_agent_runtime.contracts.generated_analysis import (
    GeneratedAnalysisPlan,
    GeneratedAnalysisResult,
)
from mico_agent_runtime.contracts.materialization import (
    AnalysisPlan as TypedAnalysisPlan,
    NumericStratificationSpec,
)


# Keep this equal to the Java typed QueryPlan result cap.  The generated-code
# compatibility path remains separately bounded to its historical preview
# size; only approved typed operators consume the full sample-bounded read.
MAX_TYPED_ANALYSIS_ROWS = 20_000


class GeneratedAnalysisError(ValueError):
    """Safe, stable error for rejected or failed generated analysis code."""

    def __init__(
        self,
        code: str = "ANALYSIS_CODE_REJECTED",
        reason_code: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.reasonCode = reason_code


# These are unsupported execution shapes, not evidence/data insufficiency.
# They may be handed to the separately sandboxed Python materializer. Missing
# numeric outcomes and cross-group coverage remain fail-closed.
TYPED_ANALYSIS_SANDBOX_FALLBACK_CODES = frozenset({
    "ANALYSIS_TYPED_OPERATOR_UNSUPPORTED",
    "ANALYSIS_TYPED_OPERATOR_GROUP_REQUIRED",
    # A typed plan may be structurally valid but refer to a semantic field
    # absent from the returned projection.  Preserve the established
    # typed-first/generated-fallback contract: the bounded generated path
    # receives only the safe preview columns and may still produce a summary.
    "ANALYSIS_TYPED_PLAN_FIELD_REJECTED",
})


_SENSITIVE_NAME = re.compile(
    r"(?i)(patient|subject|sample|accession|source|record|metadata|disease|raw|token|password|auth|url|path|sql|query|locator|payload|cohort)"
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)https?://|file://|bearer\s|sourcesampleid\s*=|internalrecordid\s*=|cohortcondition|\b(?:srr|err|drr)\d+\b|\bmv_[a-z0-9_-]+\b"
)


def build_analysis_preview(
    data: object,
    *,
    allowed_columns: set[str] | None = None,
) -> tuple[list[str], list[dict[str, object]]]:
    """Return only safe column names and the first 20 scalar rows.

    The Java dynamic compiler returns opaque aliases such as
    ``a_sample_age``. They are safe only after the Runtime has matched them
    to a non-sensitive, verified Catalog field. ``allowed_columns`` carries
    that proof for the Dynamic Scientific Runtime; historical callers keep
    the conservative name filter when it is omitted.
    """

    if not isinstance(data, dict):
        return [], []
    columns = data.get("columns")
    rows = data.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return [], []
    kept: list[str] = []
    for value in columns[:64]:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value):
            continue
        if allowed_columns is None and _SENSITIVE_NAME.search(value):
            continue
        if allowed_columns is not None and value not in allowed_columns:
            continue
        kept.append(value)
    preview: list[dict[str, object]] = []
    for raw_row in rows[:20]:
        if not isinstance(raw_row, dict):
            continue
        safe_row: dict[str, object] = {}
        for column in kept:
            value = raw_row.get(column)
            if isinstance(value, bool) or isinstance(value, (int, float)):
                safe_row[column] = value
            elif isinstance(value, str) and len(value) <= 256 and not _SENSITIVE_VALUE.search(value):
                safe_row[column] = value
        if safe_row:
            preview.append(safe_row)
    return kept, preview


_SAFE_BUILTINS = {
    "abs": abs,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "sorted": sorted,
    "sum": sum,
    "enumerate": enumerate,
    "range": range,
    "str": str,
}


def _sandbox_process_environment() -> dict[str, str]:
    """Return the smallest environment that can launch Python on this host.

    POSIX ``execve`` accepts an empty environment, but Windows requires a
    system-root variable when ``CreateProcess`` receives an explicit
    environment block.  Keep the generated-code sandbox isolated while
    retaining only that platform runtime prerequisite.  The child still gets
    no API keys, database credentials, proxy settings, or user environment.
    """

    if os.name != "nt":
        return {}
    system_root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT")
    return {"SystemRoot": system_root} if system_root else {}


class _CodePolicy(ast.NodeVisitor):
    _allowed_nodes = {
        ast.Module, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr,
        ast.For, ast.If, ast.Pass, ast.Name, ast.Load, ast.Store,
        ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript,
        ast.Call,
        ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
        ast.comprehension, ast.IfExp, ast.JoinedStr, ast.FormattedValue,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub,
        ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
        ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    }

    def __init__(self) -> None:
        self.assigned: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in self._allowed_nodes:
            raise GeneratedAnalysisError(
                reason_code=f"ANALYSIS_CODE_NODE_{type(node).__name__.upper()}"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise GeneratedAnalysisError(reason_code="ANALYSIS_CODE_PRIVATE_NAME")
        if isinstance(node.ctx, ast.Store):
            self.assigned.add(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_BUILTINS:
            raise GeneratedAnalysisError(reason_code="ANALYSIS_CODE_CALL_POLICY")
        if node.keywords:
            raise GeneratedAnalysisError(reason_code="ANALYSIS_CODE_KEYWORD_POLICY")
        self.generic_visit(node)


def _validate_code(code: str) -> None:
    if len(code) > 12000:
        raise GeneratedAnalysisError(reason_code="ANALYSIS_CODE_LENGTH_POLICY")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise GeneratedAnalysisError(reason_code="ANALYSIS_CODE_SYNTAX") from exc
    if len(list(ast.walk(tree))) > 400:
        raise GeneratedAnalysisError(reason_code="ANALYSIS_CODE_AST_SIZE_POLICY")
    policy = _CodePolicy()
    policy.visit(tree)
    if "result" not in policy.assigned:
        raise GeneratedAnalysisError(reason_code="ANALYSIS_CODE_RESULT_ASSIGNMENT")


def execute_generated_analysis(plan: GeneratedAnalysisPlan, rows: list[dict[str, object]],
                               row_count: int,
                               planner_mode: str = "model") -> GeneratedAnalysisResult:
    _validate_code(plan.code)
    payload = json.dumps(rows[:1000], ensure_ascii=False, allow_nan=False)
    encoded_code = base64.urlsafe_b64encode(plan.code.encode("utf-8")).decode("ascii")
    wrapper = (
        "import base64,json,sys\n"
        "rows=json.loads(sys.stdin.read())\n"
        f"code=base64.urlsafe_b64decode({encoded_code!r}).decode('utf-8')\n"
        "safe={'abs':abs,'float':float,'int':int,'len':len,'max':max,'min':min,"
        "'round':round,'sorted':sorted,'sum':sum,'enumerate':enumerate,'range':range,'str':str}\n"
        "scope={'__builtins__': safe, 'rows': rows}\n"
        "exec(compile(code,'<generated-analysis>','exec'),scope,scope)\n"
        "print(json.dumps(scope.get('result'),ensure_ascii=False,allow_nan=False))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", wrapper],
            input=payload,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            env=_sandbox_process_environment(),
        )
        if completed.returncode != 0 or len(completed.stdout) > 20000:
            raise GeneratedAnalysisError(
                "ANALYSIS_CODE_EXECUTION_FAILED",
                "ANALYSIS_CODE_SUBPROCESS",
            )
        value = json.loads(completed.stdout)
        if not isinstance(value, dict):
            raise GeneratedAnalysisError(
                reason_code="ANALYSIS_CODE_RESULT_SHAPE"
            )
        result = GeneratedAnalysisResult.model_validate({
            "status": "COMPLETED",
            "analysisType": plan.analysisType,
            "execution_mode": "generated",
            "method_used": "generated_python",
            "plannerMode": planner_mode,
            "codeVersion": "sandbox-python-v1",
            "rowCount": row_count,
            "metrics": value.get("metrics", {}),
            "group_results": value.get("group_results", []),
            "stratum_results": value.get("stratum_results", []),
            "validation_results": value.get("validation_results", []),
            "feature_results": value.get("feature_results", []),
            "ranking_method": value.get("ranking_method"),
            "adjusted_covariates": value.get("adjusted_covariates", []),
            "used_row_count": value.get("used_row_count", row_count),
            "dropped_row_count": value.get("dropped_row_count", 0),
            "topFeatures": value.get("topFeatures", []),
            "limitations": [
                "generated_code_was_sandbox_validated",
                "preview_was_redacted_before_model_access",
                "snapshot_is_transient_and_not_replayable",
                "analysis_is_not_a_clinical_conclusion",
            ],
        })
        return result
    except (GeneratedAnalysisError, ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, GeneratedAnalysisError):
            raise
        raise GeneratedAnalysisError(
            "ANALYSIS_CODE_EXECUTION_FAILED",
            "ANALYSIS_CODE_RESULT_VALIDATION",
        ) from exc


def _typed_column_candidates(field_id: str) -> list[str]:
    """Resolve the Java compiler's stable result aliases without exposing SQL."""

    entity, field = field_id.split(".", 1)
    candidates = [
        field_id,
        f"a_{entity}_{field}",
        f"{entity}_{field}",
        field,
    ]
    # A grouped QueryPlan may return an aggregate alias rather than the raw
    # field alias. The suffix set is closed by the QueryPlan contract.
    for operation in ("count", "mean", "min", "max", "sum"):
        candidates.append(f"a_{entity}_{field}_{operation}")
    return candidates


def _typed_raw_column_candidates(field_id: str) -> list[str]:
    """Resolve only raw projection aliases (never SQL aggregate aliases)."""

    entity, field = field_id.split(".", 1)
    return [field_id, f"a_{entity}_{field}", f"{entity}_{field}", field]


def _find_typed_column(
    rows: list[dict[str, object]], field_id: str, *, raw_only: bool = False
) -> str | None:
    candidates = (
        _typed_raw_column_candidates(field_id)
        if raw_only else _typed_column_candidates(field_id)
    )
    for candidate in candidates:
        if any(candidate in row for row in rows[:MAX_TYPED_ANALYSIS_ROWS]):
            return candidate
    return None


def _prepare_feature_aware_rows(
    plan: TypedAnalysisPlan,
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int, int, int, list[str]]:
    """Bind abundance rows to sample-level observations before statistics.

    Java returns one row per ``sample × feature``.  A typed analysis must not
    treat those rows as independent observations.  When the projection carries
    a feature dimension, this helper requires an explicit ``feature_field`` and
    a sample identity column, then averages duplicate rows within each
    ``sample × feature`` key.  Legacy synthetic tests without either column
    retain their historical row-level behaviour.
    """

    bounded = [row for row in rows[:MAX_TYPED_ANALYSIS_ROWS] if isinstance(row, dict)]
    raw_count = len(bounded)
    if plan.outcome is None:
        return bounded, raw_count, 0, 0, []

    outcome_key = _find_typed_column(bounded, plan.outcome, raw_only=True)
    if outcome_key is None:
        # Aggregate aliases such as a_abundance_value_mean are deliberately
        # rejected: they no longer identify independent sample observations.
        if _find_typed_column(bounded, plan.outcome, raw_only=False) is not None:
            raise GeneratedAnalysisError("ANALYSIS_TYPED_SAMPLE_LEVEL_REQUIRED")
        return bounded, raw_count, 0, 0, []

    detected_feature_field = plan.feature_field
    feature_key = (
        _find_typed_column(bounded, detected_feature_field, raw_only=True)
        if detected_feature_field is not None else None
    )
    if detected_feature_field is None:
        for candidate_field in ("abundance.feature", "sample.feature"):
            candidate = _find_typed_column(bounded, candidate_field, raw_only=True)
            if candidate is not None:
                detected_feature_field = candidate_field
                feature_key = candidate
                break
    if feature_key is not None and plan.feature_field is None:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_FEATURE_FIELD_REQUIRED")
    if plan.feature_field is not None and feature_key is None:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_FEATURE_FIELD_REQUIRED")

    # Only abundance-like projections are expected to carry a sample×feature
    # fan-out.  For ordinary synthetic/non-abundance rows, preserve the old
    # operator semantics unless an explicit feature field was supplied.
    if feature_key is None and plan.feature_field is None:
        return bounded, raw_count, 0, 0, []

    sample_field_candidates = (
        "analysis.sample_key",
        "abundance.sample_key",
        "abundance.sample_id",
        "abundance.patient_id",
        "sample.sample_key",
        "sample.sample_id",
        "sample.patient_id",
    )
    sample_key: str | None = None
    for sample_field in sample_field_candidates:
        sample_key = _find_typed_column(bounded, sample_field, raw_only=True)
        if sample_key is not None:
            break
    if sample_key is None:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_SAMPLE_KEY_REQUIRED")

    # A feature-aware result is explicitly sample-level.  Rows with a numeric
    # outcome but a missing identity cannot be assigned to a sample and are
    # therefore rejected instead of silently becoming pseudo-replicates.
    grouped: dict[tuple[str, str], tuple[dict[str, object], list[float]]] = {}
    feature_values: list[str] = []
    sample_tokens: set[str] = set()
    for row in bounded:
        sample_value = row.get(sample_key)
        feature_value = row.get(feature_key) if feature_key is not None else "__single__"
        outcome_value = row.get(outcome_key)
        if sample_value is None or feature_value is None:
            if isinstance(outcome_value, (int, float)) and not isinstance(outcome_value, bool):
                raise GeneratedAnalysisError("ANALYSIS_TYPED_SAMPLE_KEY_REQUIRED")
            continue
        sample_token = _scalar_token(sample_value)
        feature_token = _scalar_token(feature_value)
        if sample_token is None or feature_token is None:
            raise GeneratedAnalysisError("ANALYSIS_TYPED_SAMPLE_KEY_REQUIRED")
        sample_tokens.add(sample_token)
        if feature_token not in feature_values:
            feature_values.append(feature_token)
        key = (sample_token, feature_token)
        if key not in grouped:
            grouped[key] = (dict(row), [])
        if isinstance(outcome_value, (int, float)) and not isinstance(outcome_value, bool):
            grouped[key][1].append(float(outcome_value))

    prepared: list[dict[str, object]] = []
    for base_row, values in grouped.values():
        if values:
            base_row[outcome_key] = sum(values) / len(values)
        else:
            base_row[outcome_key] = None
        prepared.append(base_row)
    return (
        prepared,
        raw_count,
        len(sample_tokens),
        max(0, raw_count - len(prepared)),
        feature_values,
    )


def _typed_values(rows: list[dict[str, object]], field_id: str) -> list[object]:
    candidates = _typed_column_candidates(field_id)
    values: list[object] = []
    for row in rows[:MAX_TYPED_ANALYSIS_ROWS]:
        key = next((candidate for candidate in candidates if candidate in row), None)
        if key is not None and row[key] is not None:
            values.append(row[key])
    return values


def _typed_value(row: dict[str, object], field_id: str) -> object | None:
    key = next((candidate for candidate in _typed_column_candidates(field_id) if candidate in row), None)
    return row[key] if key is not None else None


def _numeric_values(rows: list[dict[str, object]], field_id: str) -> list[float]:
    values: list[float] = []
    for value in _typed_values(rows, field_id):
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _grouped_numeric_values(
    rows: list[dict[str, object]],
    outcome_field: str,
    grouping_fields: list[str],
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows[:MAX_TYPED_ANALYSIS_ROWS]:
        group_values = [_typed_value(row, field_id) for field_id in grouping_fields]
        outcome_value = _typed_value(row, outcome_field)
        if any(value is None for value in group_values) \
                or isinstance(outcome_value, bool) \
                or not isinstance(outcome_value, (int, float)):
            continue
        group_key = "|".join(str(value) for value in group_values)
        grouped.setdefault(group_key, []).append(float(outcome_value))
    return grouped


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _group_comparison_stats(
    rows: list[dict[str, object]],
    outcome_field: str,
    grouping_fields: list[str],
) -> tuple[float, float]:
    """Return bounded normal-approximation p-value and 95% CI width.

    The Dynamic Runtime intentionally has no SciPy dependency.  For the
    typed operator we therefore expose an explicitly bounded two-sided normal
    approximation over the lowest/highest group means.  It is a descriptive
    screening metric, not a causal or clinical inference, and remains
    accompanied by the standard non-diagnostic report limitations.
    """

    grouped = _grouped_numeric_values(rows, outcome_field, grouping_fields)
    ranked = sorted(
        (values for values in grouped.values() if values),
        key=lambda values: sum(values) / len(values),
    )
    if len(ranked) < 2:
        return 1.0, 0.0
    low, high = ranked[0], ranked[-1]
    mean_difference = (sum(high) / len(high)) - (sum(low) / len(low))
    standard_error = math.sqrt(
        _sample_variance(low) / len(low) + _sample_variance(high) / len(high)
    )
    if standard_error == 0.0:
        return (0.0 if mean_difference != 0.0 else 1.0), 0.0
    z_score = abs(mean_difference) / standard_error
    p_value = 1.0 - math.erf(z_score / math.sqrt(2.0))
    confidence_interval_width = 2.0 * 1.96 * standard_error
    return max(0.0, min(1.0, p_value)), max(0.0, confidence_interval_width)


def _comparison_summary(
    grouped: dict[str, list[float]],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Summarize exactly two groups using a deterministic normal approximation."""

    if len(grouped) != 2:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_OPERATOR_TWO_GROUPS_REQUIRED")
    (group_a, values_a), (group_b, values_b) = list(grouped.items())
    if not values_a or not values_b:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_INSUFFICIENT_DATA")
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    mean_difference = mean_b - mean_a
    standard_error = math.sqrt(
        _sample_variance(values_a) / len(values_a)
        + _sample_variance(values_b) / len(values_b)
    )
    if standard_error == 0.0:
        p_value = 0.0 if mean_difference != 0.0 else 1.0
        half_width = 0.0
    else:
        z_score = abs(mean_difference) / standard_error
        p_value = 1.0 - math.erf(z_score / math.sqrt(2.0))
        half_width = 1.96 * standard_error
    metrics = {
        "mean_difference": float(mean_difference),
        "p_value": max(0.0, min(1.0, float(p_value))),
        "confidence_interval_low": float(mean_difference - half_width),
        "confidence_interval_high": float(mean_difference + half_width),
        # Compatibility for the retained v1 report consumers.  New State
        # mapping uses mean_difference and never depends on this alias.
        "effect_size": float(mean_difference),
        # AnalysisPlan v2 permits the shorter ``effect`` spelling.  It is an
        # alias of the same typed group contrast, not a second calculation.
        "effect": float(mean_difference),
        "confidence_interval": float(2.0 * half_width),
    }
    return [
        {"group": group_a, "n": len(values_a), "mean": float(mean_a)},
        {"group": group_b, "n": len(values_b), "mean": float(mean_b)},
    ], metrics


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH-FDR q-values in the input order.

    The implementation is intentionally dependency-free and deterministic.
    P-values are already bounded by ``_comparison_summary``; ties retain
    their input order so feature selection remains auditable.
    """

    count = len(p_values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * count
    running = 1.0
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        candidate = max(0.0, min(1.0, float(p_values[index]))) * count / rank
        running = min(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def _feature_sample_count_metrics(
    candidates: list[dict[str, object]],
) -> dict[str, float]:
    """Summarize the independent sample-n distribution for all eligible features."""

    if not candidates:
        return {}
    group_a = [
        int(candidate["group_results"][0]["n"])  # type: ignore[index]
        for candidate in candidates
    ]
    group_b = [
        int(candidate["group_results"][1]["n"])  # type: ignore[index]
        for candidate in candidates
    ]

    def summary(values: list[int], prefix: str) -> dict[str, float]:
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            float(ordered[middle])
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        return {
            f"{prefix}_sample_count_min": float(min(ordered)),
            f"{prefix}_sample_count_median": float(median),
            f"{prefix}_sample_count_max": float(max(ordered)),
            f"{prefix}_sample_count_eq_1": float(sum(value == 1 for value in values)),
            f"{prefix}_sample_count_eq_2": float(sum(value == 2 for value in values)),
            f"{prefix}_sample_count_eq_3": float(sum(value == 3 for value in values)),
            f"{prefix}_sample_count_lt_3": float(sum(value < 3 for value in values)),
        }

    return {
        **summary(group_a, "group_a"),
        **summary(group_b, "group_b"),
    }


def _direction(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "neutral"


def _numeric_stratification_cut_points(
    values: list[float], spec: NumericStratificationSpec,
) -> list[float]:
    """Derive deterministic numeric bin boundaries from sample observations."""

    if spec.strategy in {"fixed_bins", "custom_cut_points"}:
        return [float(value) for value in spec.cut_points]
    if not values:
        raise GeneratedAnalysisError(
            "ANALYSIS_TYPED_NUMERIC_STRATIFICATION_INSUFFICIENT_SAMPLE"
        )
    assert spec.bin_count is not None
    ordered = sorted(float(value) for value in values)
    boundaries: list[float] = []
    # Linear-interpolated empirical quantiles are deterministic and do not
    # require SciPy. Duplicate boundaries are retained; empty bins are
    # counted as insufficient rather than silently merged into another bin.
    for index in range(1, spec.bin_count):
        position = (len(ordered) - 1) * index / spec.bin_count
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            boundary = ordered[lower]
        else:
            fraction = position - lower
            boundary = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
        boundaries.append(float(boundary))
    return boundaries


def _numeric_stratum_label(
    value: object,
    spec: NumericStratificationSpec,
    cut_points: list[float],
) -> str | None:
    if value is None:
        return "missing" if spec.missing_value_policy == "separate_stratum" else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    # The intervals are (-inf, c1], (c1, c2], ..., (cN, +inf).  This
    # convention is stable at exact cut points and produces bounded labels
    # without exposing model-authored values in the Decision State.
    bin_index = 1
    for boundary in cut_points:
        if numeric > boundary:
            bin_index += 1
        else:
            break
    return f"bin_{bin_index}"


def _numeric_stratified_results(
    plan: TypedAnalysisPlan,
    rows: list[dict[str, object]],
    feature_values: list[str],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, float],
    str,
]:
    """Run sample-level numeric strata and BH-FDR across all valid tests."""

    spec = plan.numeric_stratification
    if spec is None or len(plan.stratify_by) != 1:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_NUMERIC_STRATIFICATION_SPEC_REQUIRED")
    stratifier = spec.stratifier
    numeric_values = [
        float(value) for value in (_typed_value(row, stratifier) for row in rows)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    cut_points = _numeric_stratification_cut_points(numeric_values, spec)
    feature_names = feature_values or ["__single__"]
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    total_strata: dict[str, int] = {feature: 0 for feature in feature_names}
    for row in rows[:MAX_TYPED_ANALYSIS_ROWS]:
        outcome = _typed_value(row, plan.outcome)
        group = _scalar_token(_typed_value(row, plan.group_field or ""))
        stratum = _numeric_stratum_label(_typed_value(row, stratifier), spec, cut_points)
        feature = (
            _scalar_token(_typed_value(row, plan.feature_field or ""))
            if feature_values else "__single__"
        )
        if (
            feature is None or group is None or stratum is None
            or isinstance(outcome, bool) or not isinstance(outcome, (int, float))
            or not math.isfinite(float(outcome))
        ):
            continue
        grouped.setdefault(feature, {}).setdefault(stratum, {}).setdefault(group, []).append(float(outcome))

    accepted: list[dict[str, object]] = []
    insufficient_by_feature: dict[str, int] = {feature: 0 for feature in feature_names}
    for feature in feature_names:
        strata = grouped.get(feature, {})
        total_strata[feature] = len(strata)
        for stratum, groups in sorted(strata.items()):
            if len(groups) != 2 or any(
                len(values) < spec.min_samples_per_group for values in groups.values()
            ):
                insufficient_by_feature[feature] += 1
                continue
            group_results, metrics = _comparison_summary(groups)
            raw_p = float(metrics["p_value"])
            accepted.append({
                "feature_name": None if feature == "__single__" else feature,
                "stratum": stratum,
                "group_results": group_results,
                "metrics": metrics,
                "raw_p_value": raw_p,
            })
    if not accepted:
        raise GeneratedAnalysisError(
            "ANALYSIS_TYPED_NUMERIC_STRATIFICATION_INSUFFICIENT_SAMPLE"
        )

    q_values = _benjamini_hochberg([
        float(item["raw_p_value"]) for item in accepted
    ])
    for item, q_value in zip(accepted, q_values):
        metrics = item["metrics"]
        metrics["raw_p_value"] = float(item["raw_p_value"])
        metrics["adjusted_p_value"] = float(q_value)
        metrics["q_value"] = float(q_value)
        item["q_value"] = float(q_value)
    accepted.sort(key=lambda item: (
        float(item["q_value"]),
        float(item["raw_p_value"]),
        -abs(float(item["metrics"]["mean_difference"])),
        str(item["feature_name"] or ""),
        str(item["stratum"]),
    ))

    public_strata: list[dict[str, object]] = []
    for item in accepted:
        metrics = item["metrics"]
        public_strata.append({
            "stratum": str(item["stratum"]),
            "feature_name": item["feature_name"],
            "group_a_n": item["group_results"][0]["n"],
            "group_b_n": item["group_results"][1]["n"],
            "mean_difference": float(metrics["mean_difference"]),
            "p_value": float(item["raw_p_value"]),
            "raw_p_value": float(item["raw_p_value"]),
            "adjusted_p_value": float(item["q_value"]),
            "q_value": float(item["q_value"]),
            "direction": _direction(float(metrics["mean_difference"])),
        })

    feature_results: list[dict[str, object]] = []
    for feature in feature_names:
        feature_strata = [
            item for item in public_strata
            if (item["feature_name"] or "__single__") == feature
        ]
        feature_metrics: dict[str, float] = {
            "total_tested_strata": float(total_strata.get(feature, 0)),
            "eligible_strata": float(len(feature_strata)),
            "insufficient_strata": float(insufficient_by_feature.get(feature, 0)),
        }
        if feature_strata:
            differences = [float(item["mean_difference"]) for item in feature_strata]
            feature_metrics.update({
                "effect_min": float(min(differences)),
                "effect_max": float(max(differences)),
            })
        else:
            feature_metrics["insufficient_sample"] = 1.0
        feature_results.append({
            "featureName": feature,
            "status": "supported" if feature_strata else "insufficient_data",
            "metrics": feature_metrics,
            "group_results": [],
            "stratum_results": feature_strata,
        })

    metrics = {
        "total_tested_strata": float(sum(total_strata.values())),
        "eligible_strata": float(len(public_strata)),
        "insufficient_strata": float(sum(insufficient_by_feature.values())),
        "returned_strata": float(min(64, len(public_strata))),
        "truncated_strata": float(max(0, len(public_strata) - 64)),
        "numeric_bin_count": float((len(cut_points) + 1)),
        "min_samples_per_group": float(spec.min_samples_per_group),
    }
    if public_strata:
        differences = [float(item["mean_difference"]) for item in public_strata]
        metrics.update({
            "positive_stratum_count": float(sum(value > 0 for value in differences)),
            "negative_stratum_count": float(sum(value < 0 for value in differences)),
            "neutral_stratum_count": float(sum(value == 0 for value in differences)),
            "effect_min": float(min(differences)),
            "effect_max": float(max(differences)),
        })
    return (
        public_strata,
        feature_results,
        metrics,
        "numeric_stratification_bh_fdr_then_q_value_then_raw_p_value_then_abs_effect_desc_then_feature_name_then_stratum",
    )


def _scalar_token(value: object) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    return str(value)


def _complete_case_rows(
    rows: list[dict[str, object]],
    field_ids: list[str],
) -> tuple[list[dict[str, object]], int]:
    complete: list[dict[str, object]] = []
    for row in rows[:MAX_TYPED_ANALYSIS_ROWS]:
        if all(_typed_value(row, field_id) is not None for field_id in field_ids):
            complete.append(row)
    return complete, max(0, len(rows[:MAX_TYPED_ANALYSIS_ROWS]) - len(complete))


def _matrix_solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    augmented = [list(row) + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda index: abs(augmented[index][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row_index in range(n):
            if row_index == column:
                continue
            factor = augmented[row_index][column]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                left - factor * right
                for left, right in zip(augmented[row_index], augmented[column])
            ]
    return [augmented[index][-1] for index in range(n)]


def _matrix_inverse(matrix: list[list[float]]) -> list[list[float]] | None:
    n = len(matrix)
    augmented = [
        list(row) + [1.0 if row_index == col else 0.0 for col in range(n)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(n):
        pivot = max(range(column, n), key=lambda index: abs(augmented[index][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row_index in range(n):
            if row_index == column:
                continue
            factor = augmented[row_index][column]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                left - factor * right
                for left, right in zip(augmented[row_index], augmented[column])
            ]
    return [row[n:] for row in augmented]


def _linear_adjustment_summary(
    rows: list[dict[str, object]],
    outcome_field: str,
    group_field: str,
    covariates: list[str],
    *,
    require_independent_samples: bool = False,
) -> tuple[dict[str, float], int, int]:
    fields = [outcome_field, group_field, *covariates]
    complete_rows, dropped = _complete_case_rows(rows, fields)
    if not complete_rows:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_INSUFFICIENT_DATA")
    observed_group_labels: list[str] = []
    for row in complete_rows:
        label = _scalar_token(_typed_value(row, group_field))
        if label is not None:
            observed_group_labels.append(label)
    group_labels = list(dict.fromkeys(observed_group_labels))
    if len(group_labels) != 2:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_OPERATOR_TWO_GROUPS_REQUIRED")
    # A two-group adjustment needs at least two independent samples in each
    # group.  This also prevents a saturated/near-saturated design from
    # producing a misleading p-value when one feature is observed in only one
    # sample per disease group.
    if require_independent_samples and any(
        observed_group_labels.count(label) < 2 for label in group_labels
    ):
        raise GeneratedAnalysisError("ANALYSIS_TYPED_INSUFFICIENT_DATA")

    # Build a deterministic design matrix: intercept, binary group indicator,
    # raw numeric covariates, and one-hot columns for categorical covariates.
    categorical_levels: dict[str, list[str]] = {}
    numeric_covariates: set[str] = set()
    for covariate in covariates:
        values = [_typed_value(row, covariate) for row in complete_rows]
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            numeric_covariates.add(covariate)
        else:
            levels: list[str] = []
            for value in values:
                token = _scalar_token(value)
                if token is None:
                    raise GeneratedAnalysisError("ANALYSIS_TYPED_COVARIATE_ENCODING_FAILED")
                if token not in levels:
                    levels.append(token)
            if not levels:
                raise GeneratedAnalysisError("ANALYSIS_TYPED_COVARIATE_ENCODING_FAILED")
            categorical_levels[covariate] = levels

    def design_row(row: dict[str, object]) -> list[float]:
        group = _scalar_token(_typed_value(row, group_field))
        values = [1.0, 1.0 if group == group_labels[1] else 0.0]
        for covariate in covariates:
            value = _typed_value(row, covariate)
            if covariate in numeric_covariates:
                assert isinstance(value, (int, float)) and not isinstance(value, bool)
                values.append(float(value))
            else:
                levels = categorical_levels[covariate]
                token = _scalar_token(value)
                values.extend(1.0 if token == level else 0.0 for level in levels[1:])
        return values

    design = [design_row(row) for row in complete_rows]
    outcomes = [float(_typed_value(row, outcome_field)) for row in complete_rows]  # type: ignore[arg-type]
    column_count = len(design[0])
    # Require a positive residual degree of freedom.  Returning a coefficient
    # from a saturated design (n == p) is deterministic but not an inferential
    # adjustment, so that feature is explicitly marked insufficient instead.
    if require_independent_samples and len(design) <= column_count:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_INSUFFICIENT_DATA")
    xtx = [
        [sum(row[left] * row[right] for row in design) for right in range(column_count)]
        for left in range(column_count)
    ]
    xty = [
        sum(row[column] * outcome for row, outcome in zip(design, outcomes))
        for column in range(column_count)
    ]
    coefficients = _matrix_solve(xtx, xty)
    inverse = _matrix_inverse(xtx)
    if coefficients is None or inverse is None:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_COVARIATE_ENCODING_FAILED")
    residuals = [
        outcome - sum(value * coefficient for value, coefficient in zip(row, coefficients))
        for row, outcome in zip(design, outcomes)
    ]
    degrees_of_freedom = len(design) - column_count
    residual_variance = (
        sum(value * value for value in residuals) / degrees_of_freedom
        if degrees_of_freedom > 0
        else 0.0
    )
    standard_error = math.sqrt(max(0.0, residual_variance * inverse[1][1]))
    effect = coefficients[1]
    if standard_error == 0.0:
        p_value = 0.0 if effect != 0.0 else 1.0
    else:
        p_value = 1.0 - math.erf(abs(effect / standard_error) / math.sqrt(2.0))
    return {
        "adjusted_group_effect": float(effect),
        "adjusted_p_value": max(0.0, min(1.0, float(p_value))),
        # The group coefficient's inferential p-value is the raw input to the
        # multi-feature BH correction.  Keep the explicit alias so consumers
        # never mistake the adjusted effect for a corrected p-value.
        "raw_p_value": max(0.0, min(1.0, float(p_value))),
    }, len(complete_rows), dropped


def execute_typed_analysis(
    plan: TypedAnalysisPlan,
    rows: list[dict[str, object]],
    row_count: int,
    *,
    planner_mode: str = "model",
) -> GeneratedAnalysisResult:
    """Execute the currently approved deterministic typed operator set."""

    if row_count < 0 or row_count > 1_000_000:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_PLAN_ROW_BOUND_REJECTED")
    if planner_mode not in {"model", "deterministic"}:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_PLAN_MODE_REJECTED")
    if plan.outcome is None:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_OPERATOR_NO_NUMERIC_OUTCOME")
    analysis_rows, raw_row_count, sample_count, collapsed_row_count, feature_values = (
        _prepare_feature_aware_rows(plan, rows)
    )
    outcome_values = _numeric_values(analysis_rows, plan.outcome)
    if not outcome_values:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_OPERATOR_NO_NUMERIC_OUTCOME")

    result_metrics: dict[str, float] = {
        "count": float(sample_count or len(analysis_rows))
    } if "count" in plan.metrics else {}
    if feature_values or sample_count:
        result_metrics.update({
            "raw_row_count": float(raw_row_count),
            "sample_count": float(sample_count),
            "collapsed_row_count": float(collapsed_row_count),
            "feature_count": float(len(feature_values) or 1),
        })
    group_results: list[dict[str, object]] = []
    stratum_results: list[dict[str, object]] = []
    validation_results: list[dict[str, object]] = []
    feature_results: list[dict[str, object]] = []
    method_used = "typed_descriptive_summary"
    ranking_method: str | None = None
    adjusted_covariates: list[str] = []
    used_row_count = len(analysis_rows)
    dropped_row_count = max(0, row_count - used_row_count)
    if plan.analysis_type == "group_comparison":
        if plan.group_field is None:
            raise GeneratedAnalysisError("ANALYSIS_TYPED_OPERATOR_GROUP_REQUIRED")
        if len(feature_values) > 1:
            skipped_features = 0
            eligible_feature_results: list[dict[str, object]] = []
            for feature in feature_values:
                feature_rows = [
                    row for row in analysis_rows
                    if _scalar_token(_typed_value(row, plan.feature_field or "abundance.feature"))
                    == feature
                ]
                grouped = _grouped_numeric_values(feature_rows, plan.outcome, [plan.group_field])
                # A bounded raw projection may contain a feature in only one
                # disease group (because LIMIT is applied to joined rows).
                # That feature is not a valid two-group comparison, but it
                # must not invalidate other features that have both groups.
                # Skip it and report the count; fail closed only when no
                # feature has a complete comparison.
                if len(grouped) != 2 or any(not values for values in grouped.values()):
                    skipped_features += 1
                    continue
                feature_groups, comparison_metrics = _comparison_summary(grouped)
                feature_metrics = dict(comparison_metrics)
                feature_metrics["raw_p_value"] = float(comparison_metrics["p_value"])
                feature_metrics["group_a_sample_count"] = float(feature_groups[0]["n"])
                feature_metrics["group_b_sample_count"] = float(feature_groups[1]["n"])
                eligible_feature_results.append({
                    "featureName": feature,
                    "metrics": feature_metrics,
                    "group_results": feature_groups,
                })
            # Never expose a pooled mean difference/p-value for multiple
            # features.  Consumers must inspect feature_results instead.
            if not eligible_feature_results:
                raise GeneratedAnalysisError("ANALYSIS_TYPED_INSUFFICIENT_DATA")
            q_values = _benjamini_hochberg([
                float(item["metrics"]["raw_p_value"])  # type: ignore[index]
                for item in eligible_feature_results
            ])
            for item, q_value in zip(eligible_feature_results, q_values):
                metrics = item["metrics"]  # type: ignore[assignment]
                metrics["adjusted_p_value"] = float(q_value)  # type: ignore[index]
                metrics["q_value"] = float(q_value)  # type: ignore[index]
            # All eligible features receive FDR correction before the bounded
            # public result is selected.  The order is deterministic and does
            # not depend on database row/feature insertion order.
            eligible_feature_results.sort(key=lambda item: (
                float(item["metrics"]["adjusted_p_value"]),  # type: ignore[index]
                float(item["metrics"]["raw_p_value"]),  # type: ignore[index]
                -abs(float(item["metrics"]["effect_size"])),  # type: ignore[index]
                str(item["featureName"]),
            ))
            feature_results = eligible_feature_results
            result_metrics.update({
                "total_tested_features": float(len(feature_values)),
                "eligible_features": float(len(eligible_feature_results)),
                "returned_features": float(min(64, len(eligible_feature_results))),
                "truncated_features": float(max(0, len(eligible_feature_results) - 64)),
                **_feature_sample_count_metrics(eligible_feature_results),
            })
            if skipped_features:
                result_metrics["skipped_feature_count"] = float(skipped_features)
            ranking_method = (
                "bh_fdr_adjusted_p_value_then_raw_p_value_then_abs_effect_desc_then_feature_name"
            )
        else:
            grouped = _grouped_numeric_values(analysis_rows, plan.outcome, [plan.group_field])
            group_results, comparison_metrics = _comparison_summary(grouped)
            result_metrics.update(comparison_metrics)
            if feature_values:
                feature_results.append({
                    "featureName": feature_values[0],
                    "metrics": comparison_metrics,
                    "group_results": group_results,
                })
            ranking_method = None
        # Preserve the legacy aggregate metric contract when explicitly
        # requested by an existing AnalysisPlan. The typed operator's
        # canonical comparison outputs remain mean_difference and the two
        # group summaries above; these aliases are compatibility-only.
        if len(feature_values) <= 1 and "mean" in plan.metrics:
            result_metrics["mean"] = float(sum(outcome_values) / len(outcome_values))
        if len(feature_values) <= 1 and "median" in plan.metrics:
            ordered = sorted(outcome_values)
            middle = len(ordered) // 2
            result_metrics["median"] = float(
                ordered[middle]
                if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2.0
            )
        method_used = "two_group_normal_approximation"
    elif plan.analysis_type == "confounder_adjustment":
        if plan.group_field is None or not plan.covariates:
            raise GeneratedAnalysisError("ANALYSIS_TYPED_OPERATOR_GROUP_REQUIRED")
        if len(feature_values) > 1:
            skipped_features = 0
            successful_feature_results: list[dict[str, object]] = []
            insufficient_feature_results: list[dict[str, object]] = []
            for feature in feature_values:
                feature_rows = [
                    row for row in analysis_rows
                    if _scalar_token(_typed_value(row, plan.feature_field or "abundance.feature"))
                    == feature
                ]
                complete_feature_rows, feature_dropped = _complete_case_rows(
                    feature_rows,
                    [plan.outcome, plan.group_field, *plan.covariates],
                )
                try:
                    adjusted_metrics, feature_used, _feature_dropped = _linear_adjustment_summary(
                        feature_rows,
                        plan.outcome,
                        plan.group_field,
                        list(plan.covariates),
                        require_independent_samples=True,
                    )
                except GeneratedAnalysisError as exc:
                    if exc.code in {
                        "ANALYSIS_TYPED_OPERATOR_TWO_GROUPS_REQUIRED",
                        "ANALYSIS_TYPED_INSUFFICIENT_DATA",
                        "ANALYSIS_TYPED_COVARIATE_ENCODING_FAILED",
                    }:
                        skipped_features += 1
                        insufficient_metrics: dict[str, float] = {
                            "insufficient_sample": 1.0,
                            "used_sample_count": float(len(complete_feature_rows)),
                            "dropped_sample_count": float(feature_dropped),
                        }
                        insufficient_feature_results.append({
                            "featureName": feature,
                            "status": "insufficient_data",
                            "metrics": insufficient_metrics,
                            "adjusted_covariates": list(plan.covariates),
                        })
                        continue
                    raise
                adjusted_metrics = dict(adjusted_metrics)
                adjusted_metrics["used_sample_count"] = float(feature_used)
                adjusted_metrics["dropped_sample_count"] = float(_feature_dropped)
                successful_feature_results.append({
                    "featureName": feature,
                    "metrics": adjusted_metrics,
                    "adjusted_covariates": list(plan.covariates),
                })
            if not successful_feature_results and not insufficient_feature_results:
                raise GeneratedAnalysisError("ANALYSIS_TYPED_INSUFFICIENT_DATA")
            if successful_feature_results:
                q_values = _benjamini_hochberg([
                    float(item["metrics"]["raw_p_value"])  # type: ignore[index]
                    for item in successful_feature_results
                ])
                for item, q_value in zip(successful_feature_results, q_values):
                    metrics = item["metrics"]  # type: ignore[assignment]
                    metrics["adjusted_p_value"] = float(q_value)  # type: ignore[index]
                    metrics["q_value"] = float(q_value)  # type: ignore[index]
                successful_feature_results.sort(key=lambda item: (
                    float(item["metrics"]["adjusted_p_value"]),  # type: ignore[index]
                    float(item["metrics"]["raw_p_value"]),  # type: ignore[index]
                    -abs(float(item["metrics"]["adjusted_group_effect"])),  # type: ignore[index]
                    str(item["featureName"]),
                ))
            # Supported features are ranked first; feature-level insufficiency
            # remains explicit and is never silently promoted to a completed
            # scientific result.
            insufficient_feature_results.sort(key=lambda item: str(item["featureName"]))
            feature_results = successful_feature_results + insufficient_feature_results
            result_metrics.update({
                "total_tested_features": float(len(feature_values)),
                "eligible_features": float(len(successful_feature_results)),
                "insufficient_features": float(len(insufficient_feature_results)),
                "returned_features": float(min(64, len(feature_results))),
                "truncated_features": float(max(0, len(feature_results) - 64)),
            })
            if skipped_features:
                result_metrics["skipped_feature_count"] = float(skipped_features)
            ranking_method = (
                "bh_fdr_adjusted_p_value_then_raw_p_value_then_abs_adjusted_effect_desc_then_feature_name"
            )
            used_row_count = len(analysis_rows)
        else:
            adjusted_metrics, used_row_count, dropped = _linear_adjustment_summary(
                analysis_rows,
                plan.outcome,
                plan.group_field,
                list(plan.covariates),
                require_independent_samples=plan.feature_field is not None,
            )
            result_metrics.update(adjusted_metrics)
            if feature_values:
                feature_results.append({
                    "featureName": feature_values[0],
                    "metrics": adjusted_metrics,
                    "adjusted_covariates": list(plan.covariates),
                })
        adjusted_covariates = list(plan.covariates)
        dropped_row_count = max(0, row_count - used_row_count)
        if dropped_row_count == 0 and len(feature_values) <= 1:
            dropped_row_count = dropped
        method_used = "ordinary_least_squares"
    elif plan.analysis_type == "stratified_comparison":
        if plan.group_field is None or not plan.stratify_by:
            raise GeneratedAnalysisError("ANALYSIS_TYPED_OPERATOR_GROUP_REQUIRED")
        if plan.numeric_stratification is not None:
            (
                numeric_strata,
                numeric_feature_results,
                numeric_metrics,
                numeric_ranking,
            ) = _numeric_stratified_results(plan, analysis_rows, feature_values)
            stratum_results.extend(numeric_strata[:64])
            feature_results.extend(numeric_feature_results)
            result_metrics.update(numeric_metrics)
            result_metrics["stratum_count"] = float(len(numeric_strata))
            ranking_method = numeric_ranking
            method_used = "numeric_stratified_two_group_normal_approximation"
        else:
            if len(feature_values) > 1:
                raise GeneratedAnalysisError("ANALYSIS_TYPED_FEATURE_AWARE_SHAPE_REQUIRED")
            grouped_by_stratum: dict[str, dict[str, list[float]]] = {}
            for row in analysis_rows:
                outcome = _typed_value(row, plan.outcome)
                group = _scalar_token(_typed_value(row, plan.group_field))
                strata = [_scalar_token(_typed_value(row, field)) for field in plan.stratify_by]
                if (
                    isinstance(outcome, bool) or not isinstance(outcome, (int, float))
                    or group is None or any(value is None for value in strata)
                ):
                    continue
                stratum = "|".join(value for value in strata if value is not None)
                grouped_by_stratum.setdefault(stratum, {}).setdefault(group, []).append(float(outcome))
            skipped = 0
            for stratum, grouped in grouped_by_stratum.items():
                if len(grouped) != 2:
                    skipped += 1
                    continue
                _groups, metrics = _comparison_summary(grouped)
                difference = metrics["mean_difference"]
                stratum_results.append({
                    "stratum": stratum,
                    "group_a_n": _groups[0]["n"],
                    "group_b_n": _groups[1]["n"],
                    "mean_difference": difference,
                    "p_value": metrics["p_value"],
                    "direction": _direction(difference),
                })
            if not stratum_results:
                raise GeneratedAnalysisError("ANALYSIS_TYPED_INSUFFICIENT_DATA")
            differences = [item["mean_difference"] for item in stratum_results]
            result_metrics.update({
                "stratum_count": float(len(stratum_results)),
                "positive_stratum_count": float(sum(value > 0 for value in differences)),
                "negative_stratum_count": float(sum(value < 0 for value in differences)),
                "neutral_stratum_count": float(sum(value == 0 for value in differences)),
                "effect_min": float(min(differences)),
                "effect_max": float(max(differences)),
            })
            if skipped:
                result_metrics["skipped_stratum_count"] = float(skipped)
            method_used = "stratified_two_group_normal_approximation"
    elif plan.analysis_type == "cross_project_validation":
        if plan.group_field is None or plan.validation_field is None:
            raise GeneratedAnalysisError("ANALYSIS_TYPED_VALIDATION_FIELD_REQUIRED")
        if len(feature_values) > 1:
            raise GeneratedAnalysisError("ANALYSIS_TYPED_FEATURE_AWARE_SHAPE_REQUIRED")
        grouped_by_validation: dict[str, dict[str, list[float]]] = {}
        for row in analysis_rows:
            outcome = _typed_value(row, plan.outcome)
            group = _scalar_token(_typed_value(row, plan.group_field))
            validation = _scalar_token(_typed_value(row, plan.validation_field))
            if (
                isinstance(outcome, bool) or not isinstance(outcome, (int, float))
                or group is None or validation is None
            ):
                continue
            grouped_by_validation.setdefault(validation, {}).setdefault(group, []).append(float(outcome))
        skipped = 0
        for validation_value, grouped in grouped_by_validation.items():
            if len(grouped) != 2:
                skipped += 1
                continue
            _groups, metrics = _comparison_summary(grouped)
            difference = metrics["mean_difference"]
            validation_results.append({
                "validation_value": validation_value,
                "group_a_n": _groups[0]["n"],
                "group_b_n": _groups[1]["n"],
                "mean_difference": difference,
                "p_value": metrics["p_value"],
                "direction": _direction(difference),
            })
        if not validation_results:
            # Keep both the domain-specific coverage diagnosis and the
            # generic insufficiency marker in the public error text. Older
            # callers match the former; observation builders classify the
            # latter as insufficient data.
            raise GeneratedAnalysisError(
                "ANALYSIS_TYPED_GROUP_COVERAGE_REQUIRED_ANALYSIS_TYPED_INSUFFICIENT_DATA"
            )
        differences = [item["mean_difference"] for item in validation_results]
        result_metrics.update({
            "project_count": float(len(validation_results)),
            "positive_project_count": float(sum(value > 0 for value in differences)),
            "negative_project_count": float(sum(value < 0 for value in differences)),
            "neutral_project_count": float(sum(value == 0 for value in differences)),
            "effect_min": float(min(differences)),
            "effect_max": float(max(differences)),
            # Compatibility alias for v1 report consumers. New State
            # mapping uses effect_min/effect_max and per-project results.
            "effect_size": float(max(differences)),
        })
        if skipped:
            result_metrics["skipped_project_count"] = float(skipped)
        method_used = "per_project_two_group_normal_approximation"
    else:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_OPERATOR_UNSUPPORTED")

    top_features = []
    for index, item in enumerate(group_results[:20], start=1):
        top_features.append({
            "featureName": f"group_{index}",
            "metricName": "group_present",
            "metricValue": 1.0,
            "group": str(item["group"]),
        })

    # The public AnalysisResult contract intentionally keeps the compressed
    # feature summary bounded.  Multi-feature comparison has already applied
    # BH-FDR to every eligible feature and sorted by the documented ranking;
    # only then is the public list capped at 64.  Other action families retain
    # their historical bounded list behavior.
    if len(feature_results) > 64:
        result_metrics.setdefault("feature_results_total_count", float(len(feature_results)))
        result_metrics.setdefault("feature_results_truncated_count", float(len(feature_results) - 64))
        feature_results = feature_results[:64]

    try:
        return GeneratedAnalysisResult.model_validate({
            "status": "COMPLETED",
            "analysisType": plan.analysis_type,
            "execution_mode": "typed",
            "method_used": method_used,
            "plannerMode": planner_mode,
            "codeVersion": "typed-analysis-operator-v1",
            "rowCount": row_count,
            "metrics": result_metrics,
            "group_results": group_results,
            "stratum_results": stratum_results,
            "validation_results": validation_results,
            "feature_results": feature_results,
            "ranking_method": ranking_method,
            "adjusted_covariates": adjusted_covariates,
            "used_row_count": used_row_count,
            "dropped_row_count": dropped_row_count,
            "topFeatures": top_features,
            "limitations": [
                "typed_plan_executed_by_approved_operator",
                "preview_was_redacted_before_model_access",
                "snapshot_is_transient_and_not_replayable",
                "analysis_is_not_a_clinical_conclusion",
            ],
        })
    except (ValidationError, ValueError, TypeError) as exc:
        raise GeneratedAnalysisError("ANALYSIS_TYPED_RESULT_INVALID") from exc
