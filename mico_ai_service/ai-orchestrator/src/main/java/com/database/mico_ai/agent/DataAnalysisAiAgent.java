package com.database.mico_ai.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface DataAnalysisAiAgent {

    @SystemMessage("You are the data analysis agent for a medical microbiome platform.\n"
            + "Write concise professional Chinese.\n"
            + "Focus only on structured internal data observations.\n"
            + "Prefer sample structure, healthy-reference deviation, dashboard patterns, and measurable differences.\n"
            + "Do not explain model boundaries or cite external evidence.\n"
            + "Keep the output within 2 short sentences.")
    @UserMessage("Intent: {{intent}}\n"
            + "User question: {{question}}\n"
            + "Structured data JSON:\n"
            + "{{contextJson}}")
    String analyze(@V("intent") String intent,
                   @V("question") String question,
                   @V("contextJson") String contextJson);
}
