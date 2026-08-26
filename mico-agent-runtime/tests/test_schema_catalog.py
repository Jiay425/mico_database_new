from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mico_agent_runtime.contracts.schema_catalog import (
    SchemaEntitySemantics,
    SchemaFieldSemantics,
    SchemaJoinSemantics,
    SchemaSemanticCatalog,
)
from mico_agent_runtime.contracts.tools import JavaQualitySummary, JavaToolResponse
from mico_agent_runtime.ports.schema_catalog import JavaSchemaCatalogPort
from mico_agent_runtime.ports.java_agent import JavaPortTransportError


def _catalog() -> SchemaSemanticCatalog:
    return SchemaSemanticCatalog(
        schemaVersion="schema-catalog-v1",
        source="java_schema_contract",
        generatedAt=datetime(2026, 8, 23, tzinfo=timezone.utc),
        entities=[
            SchemaEntitySemantics(
                entityName="sample_metadata",
                sourceTable="meta2db_sample_metadata",
                fields=[
                    SchemaFieldSemantics(
                        name="patient_id",
                        dataType="integer",
                        nullable=False,
                        semanticStatus="verified",
                        filterable=True,
                        groupable=True,
                        displayable=False,
                        sensitive=True,
                        description="Internal business record key; not a Subject ID",
                    )
                ],
                primaryKeyFields=["patient_id"],
            ),
            SchemaEntitySemantics(
                entityName="abundance_feature",
                sourceTable="microbe_abundance_standard",
                fields=[
                    SchemaFieldSemantics(
                        name="abundance_value",
                        dataType="number",
                        nullable=True,
                        semanticStatus="verified",
                        aggregatable=True,
                        displayable=True,
                        description="Stored abundance value",
                    )
                ],
            ),
        ],
        joins=[SchemaJoinSemantics(
            leftEntity="sample_metadata",
            leftField="patient_id",
            rightEntity="abundance_feature",
            rightField="patient_id",
            relationshipStatus="verified",
            description="Internal record relationship only",
        )],
        queryRules=[
            "select_or_with_only",
            "explicit_columns_only",
            "no_cross_database_reference",
            "bounded_limit_required",
            "java_final_validation",
        ],
    )


def test_schema_catalog_is_metadata_only_and_join_closed() -> None:
    catalog = _catalog()
    assert catalog.source == "java_schema_contract"
    assert catalog.entities[0].fields[0].sensitive is True
    assert catalog.joins[0].relationshipStatus == "verified"

    with pytest.raises(ValidationError):
        SchemaSemanticCatalog.model_validate({**catalog.model_dump(), "unknown": "field"})


def test_schema_catalog_rejects_unknown_join_entity() -> None:
    catalog = _catalog().model_dump()
    catalog["joins"][0]["rightEntity"] = "unknown_entity"
    with pytest.raises(ValidationError):
        SchemaSemanticCatalog.model_validate(catalog)


def test_java_schema_catalog_port_accepts_metadata_without_snapshot() -> None:
    catalog = _catalog()

    class FakeJava:
        def execute(self, call):
            return JavaToolResponse(
                toolCallId=call.toolCallId,
                runId=call.runId,
                status="COMPLETED",
                source="java_schema_contract",
                rowCount=2,
                schemaVersion="p1a-agent-tool-contract-v1",
                generatedAt=datetime(2026, 8, 23, tzinfo=timezone.utc),
                qualitySummary=JavaQualitySummary(),
                data=catalog.model_dump(mode="json"),
            )

    loaded = JavaSchemaCatalogPort(FakeJava()).load(
        run_id="run-catalog-00000000000000000000000000000001",
        task_id="task-catalog-00000000000000000000000000000001",
    )
    assert loaded.schemaVersion == "schema-catalog-v1"
    assert loaded.source == "java_schema_contract"


def test_java_schema_catalog_port_rejects_response_snapshot() -> None:
    catalog = _catalog()

    class FakeJava:
        def execute(self, call):
            from tests.conftest import completed_response
            response = completed_response(call)
            response.data = catalog.model_dump(mode="json")
            return response

    with pytest.raises(JavaPortTransportError) as error:
        JavaSchemaCatalogPort(FakeJava()).load(
            run_id="run-catalog-00000000000000000000000000000001",
            task_id="task-catalog-00000000000000000000000000000001",
        )
    assert "JAVA_SCHEMA_CATALOG_RESPONSE_INVALID" in str(error.value)
