from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from .base import ClosedModel


CatalogIdentifier = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$", max_length=64),
]
CatalogTableName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$", max_length=64),
]
CatalogColumnName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$", max_length=64),
]


ScientificCapability = Literal["outcome", "covariate", "dimension", "stratifier"]


class SchemaFieldSemantics(ClosedModel):
    # Stable semantic ID used by QueryPlan/AnalysisPlan.  ``None`` is retained
    # for backwards-compatible reads of older catalog snapshots; new Java
    # catalogs must populate it and must not expose physical identifiers to a
    # Materializer.
    fieldId: Annotated[
        str | None,
        StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}\.[a-z][a-z0-9_]{1,63}$", max_length=128),
    ] = None
    name: CatalogColumnName
    dataType: Literal["string", "integer", "number", "boolean", "date", "json", "unknown"]
    nullable: bool
    semanticStatus: Literal["verified", "partially_verified", "unverified"]
    filterable: bool = False
    groupable: bool = False
    aggregatable: bool = False
    displayable: bool = False
    sensitive: bool = False
    # SQL aggregation capability is not scientific outcome semantics.  This
    # explicit list is the closed Catalog-to-Decision-State bridge.
    scientificCapabilities: list[ScientificCapability] = Field(default_factory=list, max_length=8)
    description: Annotated[str, StringConstraints(min_length=1, max_length=512)]

    @field_validator("scientificCapabilities")
    @classmethod
    def reject_duplicate_scientific_capabilities(
        cls, value: list[ScientificCapability]
    ) -> list[ScientificCapability]:
        if len(value) != len(set(value)):
            raise ValueError("scientific capabilities must be unique")
        return value

    @model_validator(mode="after")
    def validate_outcome_capability(self) -> "SchemaFieldSemantics":
        if "outcome" in self.scientificCapabilities and self.dataType not in {"integer", "number"}:
            raise ValueError("outcome capability requires a numeric catalog field")
        return self


class SchemaEntitySemantics(ClosedModel):
    entityId: Annotated[
        str | None,
        StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$", max_length=64),
    ] = None
    entityName: CatalogIdentifier
    sourceTable: CatalogTableName
    fields: list[SchemaFieldSemantics] = Field(min_length=1, max_length=128)
    primaryKeyFields: list[CatalogColumnName] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_primary_keys(self) -> "SchemaEntitySemantics":
        names = {field.name for field in self.fields}
        if not set(self.primaryKeyFields).issubset(names):
            raise ValueError("primary key field is not in the entity catalog")
        return self


class SchemaJoinSemantics(ClosedModel):
    relationId: Annotated[
        str | None,
        StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,63}$", max_length=64),
    ] = None
    leftEntity: CatalogIdentifier
    leftField: CatalogColumnName
    rightEntity: CatalogIdentifier
    rightField: CatalogColumnName
    relationshipStatus: Literal["verified", "partially_verified", "unverified"]
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many", "unknown"] = "unknown"
    description: Annotated[str, StringConstraints(min_length=1, max_length=512)]


class SchemaSemanticCatalog(ClosedModel):
    """Versioned, metadata-only query vocabulary supplied by Java.

    It describes what can be queried, not what the user must ask. It contains
    no sample values, raw disease labels, locator values, payloads or secrets.
    """

    schemaVersion: Annotated[
        str,
        StringConstraints(pattern=r"^schema-catalog-v[0-9]+$", max_length=64),
    ]
    source: Literal["java_schema_contract"]
    generatedAt: datetime
    entities: list[SchemaEntitySemantics] = Field(min_length=1, max_length=32)
    joins: list[SchemaJoinSemantics] = Field(default_factory=list, max_length=64)
    # Runtime-only technical bindings.  They are intentionally not ordinary
    # business fields and are omitted from the Materializer's selectable field
    # list.  Java may attach them to a raw projection when an analysis needs a
    # sample identity, while Policy/final-answer layers never see the value.
    internalAnalysisFields: list[
        Annotated[
            str,
            StringConstraints(
                pattern=r"^[a-z][a-z0-9_]{1,63}\.[a-z][a-z0-9_]{1,63}$",
                max_length=128,
            ),
        ]
    ] = Field(default_factory=list, max_length=16)
    queryRules: list[Literal[
        "select_or_with_only",
        "explicit_columns_only",
        "no_cross_database_reference",
        "bounded_limit_required",
        "java_final_validation",
    ]] = Field(min_length=1, max_length=8)

    @field_validator("generatedAt")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generatedAt must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_join_entities(self) -> "SchemaSemanticCatalog":
        names = {entity.entityName for entity in self.entities}
        if any(join.leftEntity not in names or join.rightEntity not in names for join in self.joins):
            raise ValueError("schema join references an unknown entity")
        return self
