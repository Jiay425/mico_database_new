package com.database.mico_database.agent.executor;

import com.database.mico_database.agent.contract.AgentToolRequest;
import com.database.mico_database.agent.contract.AgentToolResponse;

/** Narrow transport-facing port; implementations remain inside Java. */
public interface AgentToolExecutorPort {

    AgentToolResponse<?> execute(AgentToolRequest request);
}
