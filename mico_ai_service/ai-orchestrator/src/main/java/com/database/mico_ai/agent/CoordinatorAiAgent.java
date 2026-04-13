package com.database.mico_ai.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface CoordinatorAiAgent {

    @SystemMessage("You are the coordinator agent for a multi-agent medical microbiome system.\n"
            + "Write concise professional Chinese.\n"
            + "Merge the specialist outputs into one clear answer.\n"
            + "Keep the answer layered: data observation, model judgement when relevant, internal knowledge, external evidence, and next action.\n"
            + "Do not repeat the same sentence across agents.\n"
            + "Keep the answer within 5 short sentences.")
    @UserMessage("Intent: {{intent}}\n"
            + "User question: {{question}}\n"
            + "Specialist outputs JSON:\n"
            + "{{contextJson}}")
    String coordinate(@V("intent") String intent,
                      @V("question") String question,
                      @V("contextJson") String contextJson);
}
