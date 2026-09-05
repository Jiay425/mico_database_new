package com.database.mico_database.agent.readmodel;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SchemaSemanticCatalogReadModelTest {

    @Test
    void currentCatalogDeclaresScientificCapabilitiesSeparatelyFromSqlCapabilities() {
        SchemaSemanticCatalogReadModel catalog = SchemaSemanticCatalogReadModel.current();

        assertEquals(Arrays.asList("outcome"), field(catalog, "abundance", "abundance.value")
                .getScientificCapabilities());
        assertEquals(Arrays.asList("covariate", "stratifier"),
                field(catalog, "sample", "sample.age").getScientificCapabilities());
        assertEquals(Arrays.asList("dimension"),
                field(catalog, "metadata", "metadata.project").getScientificCapabilities());
        assertEquals(Arrays.asList("dimension", "stratifier"),
                field(catalog, "sample", "sample.disease").getScientificCapabilities());
        assertEquals(Arrays.asList("analysis.sample_key"), catalog.getInternalAnalysisFields());
    }

    private static SchemaSemanticCatalogReadModel.Field field(
            SchemaSemanticCatalogReadModel catalog, String entityId, String fieldId) {
        return catalog.getEntities().stream()
                .filter(entity -> entityId.equals(entity.getEntityId()))
                .flatMap(entity -> entity.getFields().stream())
                .filter(field -> fieldId.equals(field.getFieldId()))
                .findFirst()
                .orElseThrow(() -> new AssertionError("missing catalog field: " + fieldId));
    }
}
