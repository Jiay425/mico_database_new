package com.database.mico_database;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Static contract check for the browser-to-Java Agent entry points. */
class AgentPageContractTest {

    @Test
    void indexPageUsesJavaOnlyAgentRoutesAndSafeVisibleFields() throws Exception {
        String html = new String(Files.readAllBytes(Paths.get(
                "src/main/resources/templates/index.html")), StandardCharsets.UTF_8);

        assertTrue(html.contains("id='agent-research-run'"));
        assertTrue(html.contains("fetch('/agent/research'"));
        assertTrue(html.contains("statusNode.textContent"));
        assertTrue(html.contains("generatedAnalysis"));
        assertTrue(html.contains("科研支持，非临床诊断、治疗或因果结论"));

        assertFalse(html.contains("agent-cohort-run"));
        assertFalse(html.contains("agent-differential-run"));
        assertFalse(html.contains("/agent/cohort/feasibility"));
        assertFalse(html.contains("/agent/research/differential"));

        assertFalse(html.contains("patient_data_manager"));
        assertFalse(html.contains("fetch('http://"));
        assertFalse(html.contains("fetch(\"http://"));
        assertFalse(html.contains("MICO_RUNTIME_INTERNAL_TOKEN"));
        assertFalse(html.contains("Authorization"));
    }
}
