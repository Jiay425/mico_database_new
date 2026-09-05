from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, TypeAdapter, model_validator

from .base import ClosedModel, Identifier, NonEmptyText
from .materialization import QueryPlan


# Java tool calls are intentionally reduced to the one dynamic read tool. The
# literature audit label is retained only for the separate evidence port; it is
# not a Java executable tool.
AllowedToolName = Literal["execute_read_query", "describe_read_schema", "literature_evidence"]
AllowedScope = Literal["mico:query:read", "mico:evidence:read"]
ToolStatus = Literal[
    "COMPLETED", "REJECTED", "FAILED", "NOT_IMPLEMENTED", "WAITING_APPROVAL"
]
TransientPersistence = Literal["transient"]
JavaTransientSnapshotId = Annotated[
    str,
    StringConstraints(
        pattern=r"^transient-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
]
TransientSnapshotId = JavaTransientSnapshotId
ToolCallIdentifier = Annotated[str, StringConstraints(pattern=r"^call-[0-9a-f]{32}$")]
DynamicSqlText = Annotated[str, Field(min_length=1, max_length=16000)]
# Legacy SQL remains bounded by Java's legacy 1,000-row policy.  Typed
# QueryPlans can use the wider finite cap required by sample-bounded
# abundance reads (the Java compiler applies the same bound).
Limit1000 = Annotated[int, Field(strict=True, ge=1, le=1000)]
Limit20000 = Annotated[int, Field(strict=True, ge=1, le=20_000)]
SourceSampleText = Annotated[str, Field(min_length=1, max_length=4096)]
StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]


class RecordProfileLocator(ClosedModel):
    """Retained typed locator for future controlled tools; never a Subject ID."""

    internalRecordId: StrictPositiveInt
    sourceSampleId: SourceSampleText


class ExecuteReadQueryArguments(ClosedModel):
    """Java read call; typed QueryPlan is the new path, SQL is legacy only.

    ``includeAnalysisSampleKey`` is a Runtime-owned technical projection.  It
    never comes from the model's QueryPlan and asks Java to append an opaque,
    run-scoped sample identity for sample-level analysis.  Raw sample/patient
    identifiers remain outside the response contract.
    """

    queryPlan: QueryPlan | None = None
    sql: DynamicSqlText | None = None
    limit: Limit20000 | None = None
    includeAnalysisSampleKey: bool = False

    @model_validator(mode="after")
    def require_one_representation(self) -> "ExecuteReadQueryArguments":
        if (self.queryPlan is None) == (self.sql is None):
            raise ValueError("Java read call requires exactly one queryPlan or legacy sql")
        if self.includeAnalysisSampleKey and self.queryPlan is None:
            raise ValueError("analysis sample key requires a typed QueryPlan")
        if self.queryPlan is not None and self.limit is not None \
                and self.limit != self.queryPlan.limit:
            raise ValueError("query plan limit and read limit must match")
        return self


class ExecuteReadQueryInput(ClosedModel):
    toolName: Literal["execute_read_query"]
    arguments: ExecuteReadQueryArguments


class JavaToolCallBase(ClosedModel):
    runId: Identifier
    toolCallId: ToolCallIdentifier

    def to_java_payload(self) -> dict[str, Any]:
        return {
            "toolName": self.toolName,
            "runId": self.runId,
            "toolCallId": self.toolCallId,
            "arguments": self.arguments.model_dump(exclude_none=True),
        }


class ExecuteReadQueryJavaToolCall(JavaToolCallBase):
    toolName: Literal["execute_read_query"]
    arguments: ExecuteReadQueryArguments


class DescribeReadSchemaArguments(ClosedModel):
    """The Java schema catalog has no business-data arguments."""


class DescribeReadSchemaJavaToolCall(JavaToolCallBase):
    toolName: Literal["describe_read_schema"]
    arguments: DescribeReadSchemaArguments


JavaToolCall = ExecuteReadQueryJavaToolCall | DescribeReadSchemaJavaToolCall
JavaToolCallAdapter = TypeAdapter(JavaToolCall)


def validate_java_tool_call(value: object) -> JavaToolCall:
    return JavaToolCallAdapter.validate_python(value)


class JavaDataSnapshot(ClosedModel):
    dataSnapshotId: JavaTransientSnapshotId
    dataSource: NonEmptyText
    importBatch: NonEmptyText | None = None
    diseaseMappingVersion: NonEmptyText | None = None
    taxonomyVersion: NonEmptyText | None = None
    featureVersion: NonEmptyText | None = None
    sourceBatch: NonEmptyText | None = None
    cohortCondition: NonEmptyText | None = None
    queryHash: NonEmptyText
    rowCount: Annotated[int, Field(strict=True, ge=0)]
    generatedAt: datetime
    snapshotPersistence: TransientPersistence


class JavaQualitySummary(ClosedModel):
    subjectLinkStatus: Literal["unverified"] | None = None
    missingCounts: dict[str, int] = Field(default_factory=dict)
    duplicateCount: int | None = None
    orphanCount: int | None = None
    warnings: list[NonEmptyText] = Field(default_factory=list)


class JavaToolError(ClosedModel):
    code: NonEmptyText
    message: NonEmptyText
    field: NonEmptyText | None = None
    retryable: bool = False
    policyReason: NonEmptyText | None = None


class JavaToolResponse(ClosedModel):
    toolCallId: ToolCallIdentifier | None = None
    runId: Identifier | None = None
    status: ToolStatus
    source: NonEmptyText | None = None
    rowCount: int | None = None
    schemaVersion: NonEmptyText | None = None
    generatedAt: datetime
    dataSnapshot: JavaDataSnapshot | None = None
    qualitySummary: JavaQualitySummary | None = None
    data: Any = None
    error: JavaToolError | None = None
