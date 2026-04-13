package com.database.mico_ai.mcp;

import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.Map;

@Service
public class McpServerService {

    private static final String PROTOCOL_VERSION = "2025-11-25";

    private final McpToolService toolService;
    private final McpResourceService resourceService;
    private final McpPromptService promptService;

    public McpServerService(McpToolService toolService,
                            McpResourceService resourceService,
                            McpPromptService promptService) {
        this.toolService = toolService;
        this.resourceService = resourceService;
        this.promptService = promptService;
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> handle(Map<String, Object> request) {
        Object id = request.get("id");
        String method = stringValue(request.get("method"));
        Map<String, Object> params = request.get("params") instanceof Map<?, ?> raw
                ? (Map<String, Object>) raw
                : Map.of();

        try {
            return switch (method) {
                case "initialize" -> success(id, initializeResult());
                case "ping" -> success(id, Map.of());
                case "tools/list" -> success(id, Map.of("tools", toolService.listTools()));
                case "tools/call" -> success(id, toolCallResult(params));
                case "resources/list" -> success(id, Map.of("resources", resourceService.listResources()));
                case "resources/read" -> success(id, resourceService.readResource(required(params, "uri"), optional(params, "sessionId")));
                case "prompts/list" -> success(id, Map.of("prompts", promptService.listPrompts()));
                case "prompts/get" -> success(id, promptService.getPrompt(
                        required(params, "name"),
                        castMap(params.get("arguments")),
                        resolvePromptSessionId(params)
                ));
                case "notifications/initialized" -> notificationAck();
                default -> error(id, -32601, "Method not found", method);
            };
        } catch (IllegalArgumentException ex) {
            return error(id, -32602, ex.getMessage(), null);
        } catch (Exception ex) {
            return error(id, -32603, ex.getMessage(), null);
        }
    }

    private Map<String, Object> initializeResult() {
        Map<String, Object> capabilities = new LinkedHashMap<>();
        capabilities.put("tools", Map.of());
        capabilities.put("resources", Map.of("subscribe", false));
        capabilities.put("prompts", Map.of());

        Map<String, Object> serverInfo = new LinkedHashMap<>();
        serverInfo.put("name", "mico-ai-mcp-server");
        serverInfo.put("title", "Mico AI Harness MCP Server");
        serverInfo.put("version", "0.2.0");

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("protocolVersion", PROTOCOL_VERSION);
        result.put("capabilities", capabilities);
        result.put("serverInfo", serverInfo);
        result.put("instructions", "Use tools for business actions, resources for system-of-record documents and runtime facts, and prompts for reusable harness analysis templates. Pass sessionId whenever you want MCP activity correlated into a runtime trace.");
        return result;
    }

    private Map<String, Object> toolCallResult(Map<String, Object> params) {
        String name = required(params, "name");
        Map<String, Object> arguments = castMap(params.get("arguments"));
        String sessionId = optional(arguments, "sessionId");
        if (sessionId.isBlank()) {
            sessionId = optional(params, "sessionId");
        }
        Map<String, Object> toolResult = toolService.callTool(name, arguments, sessionId);
        return Map.of(
                "content", java.util.List.of(Map.of(
                        "type", "text",
                        "text", toJsonText(toolResult)
                )),
                "structuredContent", toolResult,
                "isError", Boolean.FALSE
        );
    }

    private String resolvePromptSessionId(Map<String, Object> params) {
        Map<String, Object> arguments = castMap(params.get("arguments"));
        String sessionId = optional(arguments, "sessionId");
        if (!sessionId.isBlank()) {
            return sessionId;
        }
        return optional(params, "sessionId");
    }

    private Map<String, Object> success(Object id, Map<String, Object> result) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("jsonrpc", "2.0");
        response.put("id", id);
        response.put("result", result);
        return response;
    }

    private Map<String, Object> notificationAck() {
        return Map.of("jsonrpc", "2.0", "result", Map.of());
    }

    private Map<String, Object> error(Object id, int code, String message, Object data) {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("code", code);
        error.put("message", message);
        if (data != null) {
            error.put("data", data);
        }
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("jsonrpc", "2.0");
        response.put("id", id);
        response.put("error", error);
        return response;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> castMap(Object value) {
        if (value instanceof Map<?, ?> map) {
            return (Map<String, Object>) map;
        }
        return Map.of();
    }

    private String required(Map<String, Object> params, String key) {
        String text = stringValue(params.get(key));
        if (text.isBlank()) {
            throw new IllegalArgumentException("Missing required parameter: " + key);
        }
        return text;
    }

    private String optional(Map<String, Object> params, String key) {
        return stringValue(params.get(key));
    }

    private String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value).trim();
    }

    private String toJsonText(Map<String, Object> value) {
        return value == null ? "{}" : value.toString();
    }
}