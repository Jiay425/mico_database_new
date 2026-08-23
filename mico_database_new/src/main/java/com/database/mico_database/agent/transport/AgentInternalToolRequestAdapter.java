package com.database.mico_database.agent.transport;

import com.database.mico_database.agent.contract.AgentArgumentSchema;
import com.database.mico_database.agent.contract.AgentScope;
import com.database.mico_database.agent.contract.AgentToolCatalog;
import com.database.mico_database.agent.contract.AgentToolDefinition;
import com.database.mico_database.agent.contract.AgentToolRequest;
import com.database.mico_database.agent.contract.RecordProfileLocator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Converts closed JSON into the existing typed AgentToolRequest. It preserves
 * unknown argument names so the contract validator can reject them explicitly.
 */
public final class AgentInternalToolRequestAdapter {

    public static final String INTERNAL_REQUESTER_ID = "internal-agent-runtime";
    public static final Set<String> EFFECTIVE_SCOPES = Collections.unmodifiableSet(
            new LinkedHashSet<>(Arrays.asList(
                    AgentScope.QUERY_READ.getWireName())));

    private static final Set<String> TOP_LEVEL_FIELDS = new LinkedHashSet<>(Arrays.asList(
            "toolName", "runId", "toolCallId", "arguments"));

    private final ObjectMapper objectMapper;

    public AgentInternalToolRequestAdapter(ObjectMapper objectMapper) {
        if (objectMapper == null) {
            throw new IllegalArgumentException("objectMapper is required");
        }
        this.objectMapper = objectMapper;
    }

    public AgentToolRequest adapt(String rawJson) {
        AgentInternalToolRequestDto dto = parse(rawJson);
        AgentToolRequest request = new AgentToolRequest();
        request.setToolName(dto.getToolName());
        request.setRunId(dto.getRunId());
        request.setToolCallId(dto.getToolCallId());
        request.setRequesterId(INTERNAL_REQUESTER_ID);
        request.setDataContractVersion("v1");
        request.setRequestedScopes(EFFECTIVE_SCOPES);

        AgentToolDefinition definition = AgentToolCatalog.find(dto.getToolName()).orElse(null);
        Map<String, Object> arguments = new LinkedHashMap<>();
        for (Map.Entry<String, JsonNode> entry : dto.getArguments().entrySet()) {
            arguments.put(entry.getKey(), convertArgument(entry.getKey(), entry.getValue(), definition));
        }
        request.setArguments(arguments);
        return request;
    }

    private AgentInternalToolRequestDto parse(String rawJson) {
        if (rawJson == null || rawJson.trim().isEmpty()) {
            throw new AgentInternalToolFormatException("request body is empty");
        }
        try (JsonParser parser = objectMapper.getFactory().createParser(rawJson)) {
            JsonNode root = objectMapper.readTree(parser);
            if (root == null || !root.isObject() || parser.nextToken() != null) {
                throw new AgentInternalToolFormatException("request must be one JSON object");
            }
            Iterator<String> fields = root.fieldNames();
            while (fields.hasNext()) {
                if (!TOP_LEVEL_FIELDS.contains(fields.next())) {
                    throw new AgentInternalToolFormatException("unknown top-level field");
                }
            }

            String toolName = requiredText(root, "toolName");
            String runId = requiredText(root, "runId");
            String toolCallId = requiredText(root, "toolCallId");
            JsonNode argumentsNode = root.get("arguments");
            if (argumentsNode == null || !argumentsNode.isObject()) {
                throw new AgentInternalToolFormatException("arguments must be a JSON object");
            }

            Map<String, JsonNode> arguments = new LinkedHashMap<>();
            Iterator<Map.Entry<String, JsonNode>> entries = argumentsNode.fields();
            while (entries.hasNext()) {
                Map.Entry<String, JsonNode> entry = entries.next();
                arguments.put(entry.getKey(), entry.getValue());
            }
            return new AgentInternalToolRequestDto(toolName, runId, toolCallId, arguments);
        } catch (AgentInternalToolFormatException exception) {
            throw exception;
        } catch (IOException exception) {
            throw new AgentInternalToolFormatException("invalid JSON");
        }
    }

    private Object convertArgument(String name, JsonNode value, AgentToolDefinition definition) {
        String declaredType = null;
        if (definition != null) {
            AgentArgumentSchema schema = definition.getArgumentSchema();
            declaredType = schema.getProperties().get(name);
        }
        if ("string".equals(declaredType)) {
            return requiredArgumentText(value);
        }
        if ("integer".equals(declaredType)) {
            return requiredInteger(value);
        }
        if ("recordProfileLocator".equals(declaredType)) {
            return convertLocator(value);
        }
        // Unknown names remain in the contract request; the validator rejects them.
        return convertUntyped(value);
    }

    private RecordProfileLocator convertLocator(JsonNode value) {
        if (value == null || !value.isObject()) {
            throw new AgentInternalToolFormatException("recordProfileLocator must be an object");
        }
        Iterator<String> fields = value.fieldNames();
        while (fields.hasNext()) {
            String field = fields.next();
            if (!"internalRecordId".equals(field) && !"sourceSampleId".equals(field)) {
                throw new AgentInternalToolFormatException("recordProfileLocator has an unknown field");
            }
        }
        JsonNode idNode = value.get("internalRecordId");
        JsonNode sourceNode = value.get("sourceSampleId");
        if (idNode == null || !idNode.isIntegralNumber() || !idNode.canConvertToLong()) {
            throw new AgentInternalToolFormatException("internalRecordId must be a JSON integer");
        }
        if (sourceNode == null || !sourceNode.isTextual()) {
            throw new AgentInternalToolFormatException("sourceSampleId must be a JSON string");
        }
        return new RecordProfileLocator(idNode.longValue(), sourceNode.textValue());
    }

    private String requiredText(JsonNode object, String field) {
        JsonNode value = object.get(field);
        if (value == null || !value.isTextual()) {
            throw new AgentInternalToolFormatException("field must be a JSON string");
        }
        return value.textValue();
    }

    private String requiredArgumentText(JsonNode value) {
        if (value == null || !value.isTextual()) {
            throw new AgentInternalToolFormatException("argument must be a JSON string");
        }
        return value.textValue();
    }

    private Integer requiredInteger(JsonNode value) {
        if (value == null || !value.isIntegralNumber() || !value.canConvertToInt()) {
            throw new AgentInternalToolFormatException("limit must be a JSON integer");
        }
        return value.intValue();
    }

    private Object convertUntyped(JsonNode value) {
        if (value == null || value.isNull()) {
            return null;
        }
        if (value.isTextual()) {
            return value.textValue();
        }
        if (value.isBoolean()) {
            return value.booleanValue();
        }
        if (value.isIntegralNumber()) {
            return value.canConvertToInt() ? value.intValue() : value.longValue();
        }
        if (value.isFloatingPointNumber()) {
            BigDecimal decimal = value.decimalValue();
            return decimal;
        }
        return value;
    }
}
