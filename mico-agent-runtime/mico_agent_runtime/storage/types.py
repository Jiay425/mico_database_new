"""Field-specific storage types for opaque runtime identifiers and metadata."""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator, StringConstraints


_HEX32 = r"[0-9a-f]{32}"

RunId = Annotated[str, StringConstraints(pattern=rf"^run-{_HEX32}$")]
TaskId = Annotated[str, StringConstraints(pattern=rf"^task-{_HEX32}$")]
TraceId = Annotated[str, StringConstraints(pattern=rf"^trace-{_HEX32}$")]
StepId = Annotated[str, StringConstraints(pattern=rf"^step-{_HEX32}$")]
ApprovalId = Annotated[str, StringConstraints(pattern=rf"^approval-{_HEX32}$")]
AuditId = Annotated[str, StringConstraints(pattern=rf"^audit-{_HEX32}$")]
ArtifactId = Annotated[str, StringConstraints(pattern=rf"^artifact-{_HEX32}$")]
ToolCallId = Annotated[str, StringConstraints(pattern=rf"^call-{_HEX32}$")]
PrincipalId = Annotated[str, StringConstraints(pattern=rf"^principal-{_HEX32}$")]
JavaTransientSnapshotId = Annotated[
    str,
    StringConstraints(pattern=r"^transient-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
]
KeyIdentifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    ),
]


def _validate_metadata_version(value: object) -> object:
    if not isinstance(value, str):
        return value
    normalized = value.lower()
    forbidden_prefixes = (
        "bearer ",
        "cohortcondition",
        "internalrecordid=",
        "sourcesampleid=",
        "payload:",
        "warning:",
        "jdbc:",
        "mysql://",
        "postgresql://",
        "ssh://",
        "s3://",
        "file://",
        "http://",
        "https://",
    )
    if any(normalized.startswith(prefix) for prefix in forbidden_prefixes):
        raise ValueError("metadata token contains forbidden runtime content")
    if not (normalized.startswith("v") and normalized[1:].isdigit()) and not any(
        separator in value for separator in "-:._"
    ):
        raise ValueError("metadata token must have a version-like structure")
    return value


_MetadataVersionShape = StringConstraints(
    min_length=1,
    max_length=256,
    pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$",
)

SchemaVersion = Annotated[
    str,
    BeforeValidator(_validate_metadata_version),
    _MetadataVersionShape,
]
ImportBatch = Annotated[str, BeforeValidator(_validate_metadata_version), _MetadataVersionShape]
DiseaseMappingVersion = Annotated[str, BeforeValidator(_validate_metadata_version), _MetadataVersionShape]
TaxonomyVersion = Annotated[str, BeforeValidator(_validate_metadata_version), _MetadataVersionShape]
FeatureVersion = Annotated[str, BeforeValidator(_validate_metadata_version), _MetadataVersionShape]
SourceBatch = Annotated[str, BeforeValidator(_validate_metadata_version), _MetadataVersionShape]
HashValue = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
ArtifactStorageRef = Annotated[str, StringConstraints(pattern=rf"^artifact://artifact-{_HEX32}$")]
