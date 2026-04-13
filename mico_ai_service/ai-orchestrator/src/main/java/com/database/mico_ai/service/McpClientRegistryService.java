package com.database.mico_ai.service;

import com.database.mico_ai.config.McpClientRegistryProperties;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

@Service
public class McpClientRegistryService {

    private final McpClientRegistryProperties properties;

    public McpClientRegistryService(McpClientRegistryProperties properties) {
        this.properties = properties;
    }

    public List<Map<String, Object>> listClients() {
        List<Map<String, Object>> clients = new ArrayList<>();
        for (McpClientRegistryProperties.McpClientDefinition definition : safeClients()) {
            if (definition == null) {
                continue;
            }
            clients.add(clientView(definition));
        }
        return clients;
    }

    public Optional<Map<String, Object>> getClient(String mcpId) {
        if (mcpId == null || mcpId.isBlank()) {
            return Optional.empty();
        }
        for (McpClientRegistryProperties.McpClientDefinition definition : safeClients()) {
            if (definition == null) {
                continue;
            }
            if (mcpId.equalsIgnoreCase(nullSafe(definition.getMcpId()))) {
                return Optional.of(clientView(definition));
            }
        }
        return Optional.empty();
    }

    public Map<String, Object> status() {
        int total = 0;
        int active = 0;
        Map<String, Integer> byTransport = new LinkedHashMap<>();

        for (McpClientRegistryProperties.McpClientDefinition definition : safeClients()) {
            if (definition == null) {
                continue;
            }
            total++;
            if (isActive(definition.getStatus())) {
                active++;
            }
            String transport = nullSafe(definition.getTransportType()).toLowerCase(Locale.ROOT);
            byTransport.put(transport, byTransport.getOrDefault(transport, 0) + 1);
        }

        Map<String, Object> status = new LinkedHashMap<>();
        status.put("mcpRegistryEnabled", properties.isEnabled());
        status.put("mcpRegistryClientCount", total);
        status.put("mcpRegistryActiveCount", active);
        status.put("mcpRegistryByTransport", byTransport);
        return status;
    }

    private List<McpClientRegistryProperties.McpClientDefinition> safeClients() {
        List<McpClientRegistryProperties.McpClientDefinition> clients = properties.getClients();
        return clients == null ? List.of() : clients;
    }

    private Map<String, Object> clientView(McpClientRegistryProperties.McpClientDefinition definition) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("mcpId", nullSafe(definition.getMcpId()));
        item.put("mcpName", nullSafe(definition.getMcpName()));
        item.put("transportType", nullSafe(definition.getTransportType()));
        item.put("endpoint", nullSafe(definition.getEndpoint()));
        item.put("command", nullSafe(definition.getCommand()));
        item.put("args", definition.getArgs() == null ? List.of() : definition.getArgs());
        item.put("env", definition.getEnv() == null ? Map.of() : definition.getEnv());
        item.put("requestTimeout", definition.getRequestTimeout() == null ? 60 : definition.getRequestTimeout());
        item.put("status", nullSafe(definition.getStatus()));
        item.put("active", isActive(definition.getStatus()));
        return item;
    }

    private boolean isActive(String status) {
        return "ACTIVE".equalsIgnoreCase(nullSafe(status)) || "ENABLED".equalsIgnoreCase(nullSafe(status));
    }

    private String nullSafe(String value) {
        return value == null ? "" : value.trim();
    }
}