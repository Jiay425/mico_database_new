package com.database.mico_ai.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface PredictionExplanationAiAgent {

    @SystemMessage("You are the prediction explanation agent for a medical microbiome platform.\n"
            + "Write concise professional Chinese.\n"
            + "Focus on model direction, Top1/Top2 difference, consistency with healthy-reference deviation, and explanation boundaries.\n"
            + "Do not turn prediction into diagnosis.\n"
            + "Keep the output within 2 short sentences.")
    @UserMessage("Intent: {{intent}}\n"
            + "User question: {{question}}\n"
            + "Prediction context JSON:\n"
            + "{{contextJson}}")
    String explain(@V("intent") String intent,
                   @V("question") String question,
                   @V("contextJson") String contextJson);
}
