from __future__ import annotations

from datetime import datetime, timezone

from mico_agent_runtime.contracts.tools import (
    ExecuteReadQueryArguments,
    ExecuteReadQueryJavaToolCall,
    JavaDataSnapshot,
    JavaQualitySummary,
    JavaToolResponse,
)


def make_call(
    *,
    run_id: str = "run-test-00000000000000000000000000000001",
    sql: str = "SELECT 1 AS value LIMIT 10",
    tool_call_id: str = "call-00000000000000000000000000000001",
) -> ExecuteReadQueryJavaToolCall:
    return ExecuteReadQueryJavaToolCall(
        toolName="execute_read_query",
        runId=run_id,
        toolCallId=tool_call_id,
        arguments=ExecuteReadQueryArguments(sql=sql, limit=10),
    )


def completed_response(
    call: ExecuteReadQueryJavaToolCall,
    *,
    snapshot: JavaDataSnapshot | None = None,
) -> JavaToolResponse:
    snapshot = snapshot or JavaDataSnapshot(
        dataSnapshotId="transient-550e8400-e29b-41d4-a716-446655440000",
        dataSource="java_agent_read_model",
        queryHash="sha256:" + "a" * 64,
        rowCount=1,
        generatedAt=datetime(2026, 8, 21, tzinfo=timezone.utc),
        snapshotPersistence="transient",
    )
    return JavaToolResponse(
        toolCallId=call.toolCallId,
        runId=call.runId,
        status="COMPLETED",
        source="java_agent_read_model",
        rowCount=snapshot.rowCount,
        schemaVersion="p1b1-java-read-model-v1",
        generatedAt=snapshot.generatedAt,
        dataSnapshot=snapshot,
        qualitySummary=JavaQualitySummary(subjectLinkStatus="unverified"),
        data={"columns": ["value"], "rows": [{"value": 1}]},
    )


class FakeJavaPort:
    def __init__(self, response_factory=completed_response) -> None:
        self.calls = 0
        self.last_call: ExecuteReadQueryJavaToolCall | None = None
        self.response_factory = response_factory

    def execute(self, call: ExecuteReadQueryJavaToolCall) -> JavaToolResponse:
        self.calls += 1
        self.last_call = call
        return self.response_factory(call)
