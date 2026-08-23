package com.database.mico_database.agent.contract;

/** Scope names are part of the wire contract; presence is not authentication. */
public enum AgentScope {
    QUERY_READ("mico:query:read"),
    EVIDENCE_READ("mico:evidence:read");

    private final String wireName;

    AgentScope(String wireName) {
        this.wireName = wireName;
    }

    public String getWireName() {
        return wireName;
    }

    public static boolean isKnown(String value) {
        if (value == null) {
            return false;
        }
        for (AgentScope scope : values()) {
            if (scope.wireName.equals(value)) {
                return true;
            }
        }
        return false;
    }
}
