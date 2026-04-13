package com.database.mico_ai.service;

import com.database.mico_ai.config.FlowEngineProperties;
import com.database.mico_ai.dto.AiIntent;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

@Service
public class FlowDefinitionRegistryService {

    private final FlowEngineProperties properties;

    public FlowDefinitionRegistryService(FlowEngineProperties properties) {
        this.properties = properties;
    }

    public Optional<FlowEngineProperties.FlowDefinition> flowOf(AiIntent intent) {
        if (!properties.isEnabled() || intent == null) {
            return Optional.empty();
        }

        for (FlowEngineProperties.FlowDefinition definition : safeDefinitions()) {
            if (definition == null) {
                continue;
            }
            String configuredIntent = normalize(definition.getIntent());
            if (configuredIntent.equals(intent.name())) {
                FlowEngineProperties.FlowDefinition cloned = cloneDefinition(definition);
                cloned.getSteps().sort(Comparator.comparingInt(FlowEngineProperties.FlowStep::getSequence));
                return Optional.of(cloned);
            }
        }
        return Optional.empty();
    }

    public Map<String, Object> status() {
        Map<AiIntent, Integer> intentFlowCount = new EnumMap<>(AiIntent.class);
        for (AiIntent intent : AiIntent.values()) {
            intentFlowCount.put(intent, 0);
        }

        int total = 0;
        for (FlowEngineProperties.FlowDefinition definition : safeDefinitions()) {
            if (definition == null) {
                continue;
            }
            total++;
            AiIntent intent = parseIntent(definition.getIntent());
            intentFlowCount.put(intent, intentFlowCount.get(intent) + 1);
        }

        Map<String, Object> status = new LinkedHashMap<>();
        status.put("stagedFlowEnabled", properties.isEnabled());
        status.put("stagedFlowOverrideSummary", properties.isOverrideSummary());
        status.put("stagedFlowMaxIterations", properties.getMaxIterations());
        status.put("stagedFlowDefinitionCount", total);
        status.put("stagedFlowCountByIntent", intentFlowCount);
        return status;
    }

    private List<FlowEngineProperties.FlowDefinition> safeDefinitions() {
        List<FlowEngineProperties.FlowDefinition> definitions = properties.getDefinitions();
        return definitions == null ? List.of() : definitions;
    }

    private String normalize(String value) {
        return value == null ? "UNKNOWN" : value.trim().toUpperCase(Locale.ROOT);
    }

    private AiIntent parseIntent(String value) {
        String normalized = normalize(value);
        for (AiIntent intent : AiIntent.values()) {
            if (intent.name().equals(normalized)) {
                return intent;
            }
        }
        return AiIntent.UNKNOWN;
    }

    private FlowEngineProperties.FlowDefinition cloneDefinition(FlowEngineProperties.FlowDefinition source) {
        FlowEngineProperties.FlowDefinition target = new FlowEngineProperties.FlowDefinition();
        target.setId(source.getId());
        target.setIntent(source.getIntent());
        target.setDescription(source.getDescription());

        List<FlowEngineProperties.FlowStep> copied = new ArrayList<>();
        if (source.getSteps() != null) {
            for (FlowEngineProperties.FlowStep step : source.getSteps()) {
                if (step == null) {
                    continue;
                }
                FlowEngineProperties.FlowStep c = new FlowEngineProperties.FlowStep();
                c.setKey(step.getKey());
                c.setRole(step.getRole());
                c.setSequence(step.getSequence());
                c.setPromptTemplate(step.getPromptTemplate());
                c.setCompletionSignal(step.getCompletionSignal());
                c.setRequired(step.isRequired());
                copied.add(c);
            }
        }
        target.setSteps(copied);
        return target;
    }
}