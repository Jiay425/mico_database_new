package com.database.mico_database.agent.contract;

import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentDynamicReadQueryContractTest {

    private final AgentToolRequestValidator validator = new AgentToolRequestValidator();

    @Test
    void catalogExposesDynamicReadAndMetadataOnlySchemaBoundaries() {
        assertEquals(2, AgentToolCatalog.all().size());
        assertTrue(AgentToolCatalog.isRegistered("execute_read_query"));
        assertTrue(AgentToolCatalog.isRegistered("describe_read_schema"));
        assertFalse(AgentToolCatalog.isRegistered("build_cohort"));
    }

    @Test
    void schemaDescriptionAcceptsNoBusinessArguments() {
        AgentToolRequest request = new AgentToolRequest();
        request.setToolName("describe_read_schema");
        request.setRunId("run-schema");
        request.setToolCallId("call-schema");
        request.setRequesterId("internal-agent-runtime");
        request.setRequestedScopes(Collections.singleton(AgentScope.QUERY_READ.getWireName()));
        assertTrue(validator.validate(request).isValid());
    }

    @Test
    void validModelReadQueryPassesTheJavaContract() {
        AgentToolValidationResult result = validator.validate(request(
                "SELECT disease, COUNT(*) AS n FROM patients GROUP BY disease LIMIT 20", 20));
        assertTrue(result.isValid(), result.getErrors().toString());
    }

    @Test
    void writesCrossDatabaseReferencesAndUnknownArgumentsAreRejected() {
        assertFalse(validator.validate(request("DELETE FROM patients LIMIT 1", 1)).isValid());
        assertFalse(validator.validate(request("SELECT * FROM patient_data_manager.patients LIMIT 1", 1)).isValid());

        AgentToolRequest unknown = request("SELECT 1 LIMIT 1", 1);
        Map<String, Object> arguments = new LinkedHashMap<>(unknown.getArguments());
        arguments.put("table", "patients");
        unknown.setArguments(arguments);
        assertFalse(validator.validate(unknown).isValid());
    }

    private AgentToolRequest request(String sql, int limit) {
        AgentToolRequest request = new AgentToolRequest();
        request.setToolName("execute_read_query");
        request.setRunId("run-test");
        request.setToolCallId("call-test");
        request.setRequesterId("internal-agent-runtime");
        request.setRequestedScopes(Collections.singleton(AgentScope.QUERY_READ.getWireName()));
        Map<String, Object> arguments = new LinkedHashMap<>();
        arguments.put("sql", sql);
        arguments.put("limit", limit);
        request.setArguments(arguments);
        return request;
    }
}
