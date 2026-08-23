package com.database.mico_database;

import com.database.mico_database.agent.transport.AgentInternalToolController;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;
import org.springframework.http.MediaType;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.web.servlet.MockMvc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.MOCK)
@AutoConfigureMockMvc
@TestPropertySource(properties = "mico.agent.internal.enabled=false")
class MicoDatabaseApplicationTests {

    @Autowired
    private ApplicationContext applicationContext;

    @Autowired
    private MockMvc mockMvc;

    @Test
    void contextLoads() {
        assertNotNull(applicationContext);
    }

    @Test
    void internalToolControllerIsRegisteredExactlyOnce() {
        assertEquals(1, applicationContext
                .getBeansOfType(AgentInternalToolController.class).size());
    }

    @Test
    void disabledInternalEndpointPassesRealSecurityChainAndReturns503() throws Exception {
        mockMvc.perform(post(AgentInternalToolController.EXECUTE_PATH)
                        .contentType(MediaType.APPLICATION_JSON)
                .content("{\"toolName\":\"execute_read_query\","
                                + "\"runId\":\"spring-test\","
                                + "\"toolCallId\":\"spring-call\","
                                + "\"arguments\":{\"sql\":\"SELECT 1 LIMIT 1\"}}"))
                .andExpect(status().isServiceUnavailable())
                .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.code").value("INTERNAL_AGENT_DISABLED"));
    }

}
