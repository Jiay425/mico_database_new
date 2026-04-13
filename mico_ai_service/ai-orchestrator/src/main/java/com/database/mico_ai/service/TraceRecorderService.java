package com.database.mico_ai.service;

import com.database.mico_ai.dto.AiChatResponse;
import com.database.mico_ai.dto.TaskPacket;
import org.springframework.stereotype.Service;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class TraceRecorderService {

    private static final int MAX_TRACES_PER_SESSION = 20;

    private final Map<String, Deque<Map<String, Object>>> tracesBySession = new ConcurrentHashMap<>();

    public Map<String, Object> start(TaskPacket taskPacket) {
        Map<String, Object> trace = new LinkedHashMap<>();
        trace.put("traceId", UUID.randomUUID().toString());
        trace.put("sessionId", taskPacket.sessionId());
        trace.put("taskPacket", taskPacket);
        trace.put("steps", new ArrayList<Map<String, Object>>());
        trace.put("mcpAccesses", new ArrayList<Map<String, Object>>());
        trace.put("mcpUsage", defaultMcpUsage());
        trace.put("startedAt", System.currentTimeMillis());
        return trace;
    }

    @SuppressWarnings("unchecked")
    public void recordStep(Map<String, Object> trace, String name, Map<String, Object> detail) {
        if (trace == null) {
            return;
        }
        Object rawSteps = trace.get("steps");
        if (!(rawSteps instanceof List<?> list)) {
            return;
        }
        Map<String, Object> step = new LinkedHashMap<>();
        step.put("name", name);
        step.put("detail", detail == null ? Map.of() : detail);
        step.put("timestamp", System.currentTimeMillis());
        ((List<Map<String, Object>>) list).add(step);
    }

    public void recordMcpToolCall(String sessionId, String toolName, Map<String, Object> detail) {
        recordMcpAccess(sessionId, "tool", "mcp_tool_call", toolName, detail);
    }

    public void recordMcpResourceRead(String sessionId, String uri, Map<String, Object> detail) {
        recordMcpAccess(sessionId, "resource", "mcp_resource_read", uri, detail);
    }

    public void recordMcpPromptGet(String sessionId, String promptName, Map<String, Object> detail) {
        recordMcpAccess(sessionId, "prompt", "mcp_prompt_get", promptName, detail);
    }

    public Map<String, Object> finish(Map<String, Object> trace, AiChatResponse response) {
        if (trace == null) {
            return Map.of();
        }

        trace.put("finishedAt", System.currentTimeMillis());
        trace.put("finalIntent", response == null ? "" : response.intent());
        trace.put("success", response != null && response.success());
        trace.put("review", buildReview(trace, response));

        String sessionId = String.valueOf(trace.getOrDefault("sessionId", "default"));
        storeTrace(sessionId, trace);
        return trace;
    }

    public Map<String, Object> latest(String sessionId) {
        Deque<Map<String, Object>> traces = tracesBySession.get(sessionId);
        if (traces == null || traces.isEmpty()) {
            return Map.of(
                    "sessionId", sessionId,
                    "count", 0,
                    "traces", List.of()
            );
        }
        return traces.peekFirst();
    }

    public Map<String, Object> list(String sessionId) {
        Deque<Map<String, Object>> traces = tracesBySession.get(sessionId);
        if (traces == null || traces.isEmpty()) {
            return Map.of(
                    "sessionId", sessionId,
                    "count", 0,
                    "traces", List.of()
            );
        }
        return Map.of(
                "sessionId", sessionId,
                "count", traces.size(),
                "traces", List.copyOf(traces)
        );
    }

    public Map<String, Object> status() {
        return Map.of(
                "traceSessions", tracesBySession.size(),
                "traceRetention", MAX_TRACES_PER_SESSION,
                "reviewLoopReady", true,
                "mcpTraceReady", true
        );
    }

    private void recordMcpAccess(String sessionId,
                                 String accessType,
                                 String stepName,
                                 String target,
                                 Map<String, Object> detail) {
        if (sessionId == null || sessionId.isBlank()) {
            return;
        }
        Map<String, Object> trace = latestTraceReference(sessionId);
        ensureMcpContainers(trace);

        Map<String, Object> access = new LinkedHashMap<>();
        access.put("type", accessType);
        access.put("target", target);
        access.put("detail", detail == null ? Map.of() : detail);
        access.put("timestamp", System.currentTimeMillis());

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> accesses = (List<Map<String, Object>>) trace.get("mcpAccesses");
        accesses.add(access);
        recordStep(trace, stepName, access);
        incrementMcpUsage(trace, accessType);
    }

    private Map<String, Object> latestTraceReference(String sessionId) {
        Deque<Map<String, Object>> traces = tracesBySession.computeIfAbsent(sessionId, key -> new ArrayDeque<>());
        if (traces.isEmpty()) {
            Map<String, Object> trace = new LinkedHashMap<>();
            trace.put("traceId", "mcp-" + UUID.randomUUID());
            trace.put("sessionId", sessionId);
            trace.put("steps", new ArrayList<Map<String, Object>>());
            trace.put("mcpAccesses", new ArrayList<Map<String, Object>>());
            trace.put("mcpUsage", defaultMcpUsage());
            trace.put("startedAt", System.currentTimeMillis());
            traces.addFirst(trace);
            return trace;
        }
        return traces.peekFirst();
    }

    private void ensureMcpContainers(Map<String, Object> trace) {
        if (!trace.containsKey("mcpAccesses")) {
            trace.put("mcpAccesses", new ArrayList<Map<String, Object>>());
        }
        if (!trace.containsKey("mcpUsage")) {
            trace.put("mcpUsage", defaultMcpUsage());
        }
    }

    @SuppressWarnings("unchecked")
    private void incrementMcpUsage(Map<String, Object> trace, String accessType) {
        Object rawUsage = trace.get("mcpUsage");
        if (!(rawUsage instanceof Map<?, ?> usage)) {
            return;
        }
        String key = switch (accessType) {
            case "tool" -> "tools";
            case "resource" -> "resources";
            case "prompt" -> "prompts";
            default -> "other";
        };
        Object current = usage.get(key);
        int count = current instanceof Number number ? number.intValue() : 0;
        ((Map<String, Object>) usage).put(key, count + 1);
    }

    private Map<String, Object> defaultMcpUsage() {
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("tools", 0);
        usage.put("resources", 0);
        usage.put("prompts", 0);
        return usage;
    }

    private void storeTrace(String sessionId, Map<String, Object> trace) {
        Deque<Map<String, Object>> traces = tracesBySession.computeIfAbsent(sessionId, key -> new ArrayDeque<>());
        traces.addFirst(deepCopy(trace));
        while (traces.size() > MAX_TRACES_PER_SESSION) {
            traces.removeLast();
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> buildReview(Map<String, Object> trace, AiChatResponse response) {
        Map<String, Object> review = new LinkedHashMap<>();
        Map<String, Object> metadata = response == null || response.metadata() == null
                ? Collections.emptyMap()
                : response.metadata();
        TaskPacket taskPacket = trace.get("taskPacket") instanceof TaskPacket packet ? packet : null;
        Map<String, Object> mcpUsage = trace.get("mcpUsage") instanceof Map<?, ?> usage
                ? (Map<String, Object>) usage
                : defaultMcpUsage();

        boolean usedKnowledge = metadata.containsKey("knowledgeHits");
        boolean usedWebEvidence = metadata.containsKey("webEvidenceHits");
        boolean usedPrediction = metadata.containsKey("predictionTop1Label");
        boolean usedHybrid = Boolean.TRUE.equals(metadata.get("hybridMode"));
        boolean usedMultiAgent = Boolean.TRUE.equals(metadata.get("multiAgentMode"));
        boolean usedMcpTools = intValue(mcpUsage.get("tools")) > 0;
        boolean usedMcpResources = intValue(mcpUsage.get("resources")) > 0;
        boolean usedMcpPrompts = intValue(mcpUsage.get("prompts")) > 0;

        List<String> missingChannels = new ArrayList<>();
        if (!usedKnowledge) {
            missingChannels.add("internal_knowledge");
        }
        if (!usedPrediction && taskPacket != null && "medium".equalsIgnoreCase(taskPacket.riskLevel())) {
            missingChannels.add("model_judgement");
        }

        review.put("usedKnowledge", usedKnowledge);
        review.put("usedWebEvidence", usedWebEvidence);
        review.put("usedPrediction", usedPrediction);
        review.put("usedHybrid", usedHybrid);
        review.put("usedMultiAgent", usedMultiAgent);
        review.put("usedMcpTools", usedMcpTools);
        review.put("usedMcpResources", usedMcpResources);
        review.put("usedMcpPrompts", usedMcpPrompts);
        review.put("mcpUsage", mcpUsage);
        review.put("missingChannels", missingChannels);
        review.put("needsHumanReview",
                taskPacket != null
                        && "high".equalsIgnoreCase(taskPacket.riskLevel())
                        && (!usedKnowledge || !usedPrediction));
        review.put("summaryQualityHint", buildSummaryHint(usedKnowledge, usedPrediction, usedWebEvidence, usedMcpResources, usedMcpPrompts));
        return review;
    }

    private String buildSummaryHint(boolean usedKnowledge,
                                    boolean usedPrediction,
                                    boolean usedWebEvidence,
                                    boolean usedMcpResources,
                                    boolean usedMcpPrompts) {
        if (usedKnowledge && usedPrediction && usedWebEvidence && usedMcpResources && usedMcpPrompts) {
            return "数据、模型、知识、外部证据以及 MCP 资源/提示词均已命中。";
        }
        if (usedKnowledge && usedPrediction) {
            return "已具备内部数据、模型判断和知识支撑，MCP 命中情况可继续增强。";
        }
        if (usedKnowledge) {
            return "已有内部知识支撑，但模型、外部证据或 MCP 连接层仍可继续增强。";
        }
        return "当前回答主要依赖业务工作流，建议继续补强知识、证据或 MCP 连接层命中。";
    }

    private int intValue(Object value) {
        return value instanceof Number number ? number.intValue() : 0;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> deepCopy(Map<String, Object> original) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : original.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof Map<?, ?> map) {
                copy.put(entry.getKey(), deepCopy((Map<String, Object>) map));
            } else if (value instanceof List<?> list) {
                copy.put(entry.getKey(), deepCopyList(list));
            } else {
                copy.put(entry.getKey(), value);
            }
        }
        return copy;
    }

    @SuppressWarnings("unchecked")
    private List<Object> deepCopyList(List<?> source) {
        List<Object> copy = new ArrayList<>();
        for (Object item : source) {
            if (item instanceof Map<?, ?> map) {
                copy.add(deepCopy((Map<String, Object>) map));
            } else if (item instanceof List<?> list) {
                copy.add(deepCopyList(list));
            } else {
                copy.add(item);
            }
        }
        return copy;
    }
}