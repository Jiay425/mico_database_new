package com.database.mico_database.agent.contract;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/** Small Java representation of the strict argument schema used by the catalog. */
public final class AgentArgumentSchema {

    private final Map<String, String> properties;
    private final Set<String> required;
    private final boolean additionalPropertiesAllowed;

    public AgentArgumentSchema(Map<String, String> properties,
                               Set<String> required,
                               boolean additionalPropertiesAllowed) {
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(properties));
        this.required = Collections.unmodifiableSet(new LinkedHashSet<>(required));
        this.additionalPropertiesAllowed = additionalPropertiesAllowed;
    }

    public Map<String, String> getProperties() {
        return properties;
    }

    public Set<String> getRequired() {
        return required;
    }

    public boolean isAdditionalPropertiesAllowed() {
        return additionalPropertiesAllowed;
    }

    public boolean supports(String property) {
        return properties.containsKey(property);
    }

    public boolean isRequired(String property) {
        return required.contains(property);
    }
}
