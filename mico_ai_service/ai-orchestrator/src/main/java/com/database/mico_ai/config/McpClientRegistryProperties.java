package com.database.mico_ai.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@ConfigurationProperties(prefix = "app.harness.mcp-registry")
public class McpClientRegistryProperties {

    private boolean enabled = true;
    private List<McpClientDefinition> clients = new ArrayList<>();

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public List<McpClientDefinition> getClients() {
        return clients;
    }

    public void setClients(List<McpClientDefinition> clients) {
        this.clients = clients == null ? new ArrayList<>() : clients;
    }

    public static class McpClientDefinition {
        private String mcpId;
        private String mcpName;
        private String transportType;
        private String endpoint;
        private String command;
        private List<String> args = new ArrayList<>();
        private Map<String, String> env = new LinkedHashMap<>();
        private Integer requestTimeout = 60;
        private String status = "ACTIVE";

        public String getMcpId() {
            return mcpId;
        }

        public void setMcpId(String mcpId) {
            this.mcpId = mcpId;
        }

        public String getMcpName() {
            return mcpName;
        }

        public void setMcpName(String mcpName) {
            this.mcpName = mcpName;
        }

        public String getTransportType() {
            return transportType;
        }

        public void setTransportType(String transportType) {
            this.transportType = transportType;
        }

        public String getEndpoint() {
            return endpoint;
        }

        public void setEndpoint(String endpoint) {
            this.endpoint = endpoint;
        }

        public String getCommand() {
            return command;
        }

        public void setCommand(String command) {
            this.command = command;
        }

        public List<String> getArgs() {
            return args;
        }

        public void setArgs(List<String> args) {
            this.args = args == null ? new ArrayList<>() : args;
        }

        public Map<String, String> getEnv() {
            return env;
        }

        public void setEnv(Map<String, String> env) {
            this.env = env == null ? new LinkedHashMap<>() : env;
        }

        public Integer getRequestTimeout() {
            return requestTimeout;
        }

        public void setRequestTimeout(Integer requestTimeout) {
            this.requestTimeout = requestTimeout;
        }

        public String getStatus() {
            return status;
        }

        public void setStatus(String status) {
            this.status = status;
        }
    }
}