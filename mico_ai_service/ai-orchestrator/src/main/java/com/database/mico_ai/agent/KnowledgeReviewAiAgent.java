package com.database.mico_ai.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface KnowledgeReviewAiAgent {

    @SystemMessage("You are the knowledge retrieval agent for a medical microbiome platform.\n"
            + "Write concise professional Chinese.\n"
            + "Focus on internal knowledge rules first, then mention external web evidence only as supplementary support.\n"
            + "Clearly distinguish internal rules from external references.\n"
            + "Do not invent studies or unsupported claims.\n"
            + "Keep the output within 2 short sentences.")
    @UserMessage("Intent: {{intent}}\n"
            + "User question: {{question}}\n"
            + "Knowledge context JSON:\n"
            + "{{contextJson}}")
    String review(@V("intent") String intent,
                  @V("question") String question,
                  @V("contextJson") String contextJson);
}
