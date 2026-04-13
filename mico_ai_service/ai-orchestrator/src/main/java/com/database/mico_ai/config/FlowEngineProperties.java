package com.database.mico_ai.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.ArrayList;
import java.util.List;

@ConfigurationProperties(prefix = "app.harness.flow")
public class FlowEngineProperties {

    private boolean enabled = true;
    private boolean overrideSummary = true;
    private int maxIterations = 1;
    private List<FlowDefinition> definitions = new ArrayList<>();

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public boolean isOverrideSummary() {
        return overrideSummary;
    }

    public void setOverrideSummary(boolean overrideSummary) {
        this.overrideSummary = overrideSummary;
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    public List<FlowDefinition> getDefinitions() {
        return definitions;
    }

    public void setDefinitions(List<FlowDefinition> definitions) {
        this.definitions = definitions == null ? new ArrayList<>() : definitions;
    }

    public static class FlowDefinition {
        private String id;
        private String intent;
        private String description;
        private List<FlowStep> steps = new ArrayList<>();

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getIntent() {
            return intent;
        }

        public void setIntent(String intent) {
            this.intent = intent;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }

        public List<FlowStep> getSteps() {
            return steps;
        }

        public void setSteps(List<FlowStep> steps) {
            this.steps = steps == null ? new ArrayList<>() : steps;
        }
    }

    public static class FlowStep {
        private String key;
        private String role;
        private int sequence;
        private String promptTemplate;
        private String completionSignal;
        private boolean required = true;

        public String getKey() {
            return key;
        }

        public void setKey(String key) {
            this.key = key;
        }

        public String getRole() {
            return role;
        }

        public void setRole(String role) {
            this.role = role;
        }

        public int getSequence() {
            return sequence;
        }

        public void setSequence(int sequence) {
            this.sequence = sequence;
        }

        public String getPromptTemplate() {
            return promptTemplate;
        }

        public void setPromptTemplate(String promptTemplate) {
            this.promptTemplate = promptTemplate;
        }

        public String getCompletionSignal() {
            return completionSignal;
        }

        public void setCompletionSignal(String completionSignal) {
            this.completionSignal = completionSignal;
        }

        public boolean isRequired() {
            return required;
        }

        public void setRequired(boolean required) {
            this.required = required;
        }
    }
}