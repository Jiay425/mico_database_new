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
 * The executable Java catalog contains one dynamic read boundary only.
 * Historical fixed-workflow enum values are not registered and therefore
 * cannot be called through the internal API.
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
        properties.put("limit", "integer");
        definitions.put("execute_read_query", new AgentToolDefinition(
                AgentToolName.EXECUTE_READ_QUERY,
                "Execute one model-proposed read query after Java policy validation.",
                scopes,
                AgentRiskLevel.HIGH,
                true,
                true,
                1000,
                new AgentArgumentSchema(properties,
                        Collections.singleton("sql"), false)));
        return Collections.unmodifiableMap(definitions);
    }
}
