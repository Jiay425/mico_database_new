"""Small, dependency-free static boundary audit for the P6 release asset.

This is intentionally conservative: it checks that the Runtime application
layers do not import database/SSH clients or read Java configuration. The
storage package is the only allowed home for the MySQL adapter.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "mico_agent_runtime"
FORBIDDEN_MODULES = {
    "asyncmy",
    "aiomysql",
    "mysql",
    "mysql.connector",
    "pymysql",
    "paramiko",
    "sshtunnel",
}
FORBIDDEN_TEXT = (
    "application.properties",
    "application.yml",
    "application.yaml",
    "known_hosts",
    ".ssh",
    "patient_data_manager",
)


def _module_name(node: ast.ImportFrom | ast.Import) -> Iterable[str]:
    if isinstance(node, ast.Import):
        return (alias.name for alias in node.names)
    return (node.module or "",)


def find_violations(package_root: Path = PACKAGE_ROOT) -> list[str]:
    violations: list[str] = []
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root).as_posix()
        storage_layer = relative == "storage" or relative.startswith("storage/")
        text = source.read_text(encoding="utf-8")
        for marker in FORBIDDEN_TEXT:
            if marker in text and not storage_layer:
                violations.append(f"{relative}: forbidden configuration/data marker")
                break
        try:
            tree = ast.parse(text, filename=str(source))
        except SyntaxError:
            violations.append(f"{relative}: syntax error")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for module in _module_name(node):
                    if module in FORBIDDEN_MODULES and not storage_layer:
                        violations.append(f"{relative}: forbidden data-access import")
    return sorted(set(violations))


def main() -> int:
    violations = find_violations()
    result = {
        "status": "PASS" if not violations else "FAIL",
        "scope": "mico_agent_runtime_non_storage_layers",
        "violationCount": len(violations),
        "violations": violations,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
