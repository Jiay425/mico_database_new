package com.database.mico_database.agent.contract;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * The executable Java catalog exposes one data boundary and one metadata-only
 * schema-description boundary. It does not encode diseases, cohorts or
 * analysis methods.
 */
public final class AgentToolCatalog {

    private static final Map<String, AgentToolDefinition> DEFINITIONS = buildDefinitions();

    private AgentToolCatalog() {
    }

    public static Collection<AgentToolDefinition> all() {
        return Collections.unmodifiableList(new ArrayList<>(DEFINITIONS.values()));
    }

    public static Optional<AgentToolDefinition> find(String toolName) {
        return Optional.ofNullable(DEFINITIONS.get(toolName));
    }

    public static boolean isRegistered(String toolName) {
        return toolName != null && DEFINITIONS.containsKey(toolName);
    }

    private static Map<String, AgentToolDefinition> buildDefinitions() {
        Map<String, AgentToolDefinition> definitions = new LinkedHashMap<>();
        Set<String> scopes = new LinkedHashSet<>();
        scopes.add(AgentScope.QUERY_READ.getWireName());
        Map<String, String> properties = new LinkedHashMap<>();
        properties.put("sql", "string");
        properties.put("queryPlan", "queryPlan");
        properties.put("limit", "integer");
        // Runtime-only technical projection.  It asks Java to return an
        // opaque run-scoped sample key; it is not a model-selected field.
        properties.put("includeAnalysisSampleKey", "boolean");
        definitions.put("execute_read_query", new AgentToolDefinition(
                AgentToolName.EXECUTE_READ_QUERY,
                "Execute one catalog-compiled read plan after Java policy validation.",
                scopes,
                AgentRiskLevel.HIGH,
                true,
                true,
                20_000,
                new AgentArgumentSchema(properties,
                        Collections.emptySet(), false)));
        definitions.put("describe_read_schema", new AgentToolDefinition(
                AgentToolName.DESCRIBE_READ_SCHEMA,
                "Return versioned, metadata-only semantics for the approved read surface.",
                scopes,
                AgentRiskLevel.LOW,
                true,
                false,
                1,
                new AgentArgumentSchema(Collections.emptyMap(),
                        Collections.emptySet(), false)));
        return Collections.unmodifiableMap(definitions);
    }
}
