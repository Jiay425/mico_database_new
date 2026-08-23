package com.database.mico_database.agent.transport;

/** Internal marker for closed JSON/type failures. Its detail never reaches HTTP output. */
public final class AgentInternalToolFormatException extends RuntimeException {

    public AgentInternalToolFormatException(String reason) {
        super(reason);
    }
}
