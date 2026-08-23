package com.database.mico_database.agent.contract;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Result of deterministic offline request validation. */
public final class AgentToolValidationResult {

    private final boolean valid;
    private final List<AgentToolError> errors;

    private AgentToolValidationResult(boolean valid, List<AgentToolError> errors) {
        this.valid = valid;
        this.errors = Collections.unmodifiableList(new ArrayList<>(errors));
    }

    public static AgentToolValidationResult valid() {
        return new AgentToolValidationResult(true, Collections.<AgentToolError>emptyList());
    }

    public static AgentToolValidationResult rejected(List<AgentToolError> errors) {
        return new AgentToolValidationResult(false, errors);
    }

    public boolean isValid() {
        return valid;
    }

    public List<AgentToolError> getErrors() {
        return errors;
    }
}
