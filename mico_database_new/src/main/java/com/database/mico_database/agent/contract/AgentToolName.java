package com.database.mico_database.agent.contract;

/** The only executable Java Agent tool in the dynamic core. */
public enum AgentToolName {
    EXECUTE_READ_QUERY("execute_read_query");

    private final String wireName;

    AgentToolName(String wireName) {
        this.wireName = wireName;
    }

    public String getWireName() {
        return wireName;
    }

    public static AgentToolName fromWireName(String value) {
        if (value == null) {
            return null;
        }
        for (AgentToolName toolName : values()) {
            if (toolName.wireName.equals(value)) {
                return toolName;
            }
        }
        return null;
    }
}
