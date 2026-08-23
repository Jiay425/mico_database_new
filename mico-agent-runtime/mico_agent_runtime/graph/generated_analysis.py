from __future__ import annotations

import ast
import base64
import json
import re
import subprocess
import sys
from typing import Any

from pydantic import ValidationError

from mico_agent_runtime.contracts.generated_analysis import (
    GeneratedAnalysisPlan,
    GeneratedAnalysisResult,
)


class GeneratedAnalysisError(ValueError):
    """Safe, stable error for rejected or failed generated analysis code."""

    def __init__(self, code: str = "ANALYSIS_CODE_REJECTED") -> None:
        super().__init__(code)
        self.code = code


_SENSITIVE_NAME = re.compile(
    r"(?i)(patient|subject|sample|accession|source|record|metadata|disease|raw|token|password|auth|url|path|sql|query|locator|payload|cohort)"
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)https?://|file://|bearer\s|sourcesampleid\s*=|internalrecordid\s*=|cohortcondition|\b(?:srr|err|drr)\d+\b|\bmv_[a-z0-9_-]+\b"
)


def build_analysis_preview(data: object) -> tuple[list[str], list[dict[str, object]]]:
    """Return only safe column names and the first 20 scalar rows."""

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
        if _SENSITIVE_NAME.search(value):
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


class _CodePolicy(ast.NodeVisitor):
    _allowed_nodes = {
        ast.Module, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr,
        ast.For, ast.If, ast.Pass, ast.Name, ast.Load, ast.Store,
        ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript,
        ast.Call,
        ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
        ast.comprehension, ast.IfExp, ast.JoinedStr, ast.FormattedValue,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.USub,
        ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
        ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    }

    def __init__(self) -> None:
        self.assigned: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in self._allowed_nodes:
            raise GeneratedAnalysisError()
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise GeneratedAnalysisError()
        if isinstance(node.ctx, ast.Store):
            self.assigned.add(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_BUILTINS:
            raise GeneratedAnalysisError()
        if node.keywords:
            raise GeneratedAnalysisError()
        self.generic_visit(node)


def _validate_code(code: str) -> None:
    if len(code) > 12000:
        raise GeneratedAnalysisError()
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise GeneratedAnalysisError() from exc
    if len(list(ast.walk(tree))) > 400:
        raise GeneratedAnalysisError()
    policy = _CodePolicy()
    policy.visit(tree)
    if "result" not in policy.assigned:
        raise GeneratedAnalysisError()


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
            env={},
        )
        if completed.returncode != 0 or len(completed.stdout) > 20000:
            raise GeneratedAnalysisError("ANALYSIS_CODE_EXECUTION_FAILED")
        value = json.loads(completed.stdout)
        if not isinstance(value, dict):
            raise GeneratedAnalysisError()
        result = GeneratedAnalysisResult.model_validate({
            "status": "COMPLETED",
            "analysisType": plan.analysisType,
            "plannerMode": planner_mode,
            "codeVersion": "sandbox-python-v1",
            "rowCount": row_count,
            "metrics": value.get("metrics", {}),
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
        raise GeneratedAnalysisError("ANALYSIS_CODE_EXECUTION_FAILED") from exc
