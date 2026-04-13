package com.database.mico_ai.controller;

import com.database.mico_ai.mcp.McpServerService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/mcp")
public class McpController {

    private final McpServerService mcpServerService;

    public McpController(McpServerService mcpServerService) {
        this.mcpServerService = mcpServerService;
    }

    @PostMapping
    public Map<String, Object> handle(@RequestBody Map<String, Object> request) {
        return mcpServerService.handle(request == null ? Map.of() : request);
    }
}