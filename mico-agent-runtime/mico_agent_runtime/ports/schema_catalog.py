from __future__ import annotations

from hashlib import sha256
from typing import Protocol

from pydantic import ValidationError

from mico_agent_runtime.contracts.schema_catalog import SchemaSemanticCatalog
from mico_agent_runtime.contracts.tools import (
    DescribeReadSchemaArguments,
    DescribeReadSchemaJavaToolCall,
    JavaToolResponse,
)
from mico_agent_runtime.ports.java_agent import (
    JavaAgentToolPort,
    JavaPortContractError,
    JavaPortTransportError,
)


class SchemaCatalogPort(Protocol):
    def load(self, *, run_id: str, task_id: str) -> SchemaSemanticCatalog:
        ...


class JavaSchemaCatalogPort:
    """Loads only Java's metadata catalog; it never opens a DB connection."""

    def __init__(self, java_port: JavaAgentToolPort) -> None:
        self._java_port = java_port

    def load(self, *, run_id: str, task_id: str) -> SchemaSemanticCatalog:
        call_id = "call-" + sha256(
            f"{run_id}|{task_id}|describe_read_schema".encode("utf-8")
        ).hexdigest()[:32]
        call = DescribeReadSchemaJavaToolCall(
            toolName="describe_read_schema",
            runId=run_id,
            toolCallId=call_id,
            arguments=DescribeReadSchemaArguments(),
        )
        try:
            response: JavaToolResponse = self._java_port.execute(call)
        except JavaPortContractError as exc:
            raise JavaPortContractError("JAVA_SCHEMA_CATALOG_CALL_INVALID") from exc
        except JavaPortTransportError as exc:
            raise JavaPortTransportError("JAVA_SCHEMA_CATALOG_UNAVAILABLE") from exc
        except Exception as exc:
            raise JavaPortTransportError("JAVA_SCHEMA_CATALOG_UNAVAILABLE") from exc
        if response.runId != call.runId or response.toolCallId != call.toolCallId:
            raise JavaPortTransportError("JAVA_TOOL_RESPONSE_MISMATCH")
        if response.status != "COMPLETED" or response.data is None or response.dataSnapshot is not None:
            raise JavaPortTransportError("JAVA_SCHEMA_CATALOG_RESPONSE_INVALID")
        try:
            return SchemaSemanticCatalog.model_validate(response.data)
        except (ValidationError, TypeError, ValueError) as exc:
            raise JavaPortTransportError("JAVA_SCHEMA_CATALOG_RESPONSE_INVALID") from exc
