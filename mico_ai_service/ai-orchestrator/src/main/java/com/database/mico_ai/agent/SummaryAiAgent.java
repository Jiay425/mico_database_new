package com.database.mico_ai.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface SummaryAiAgent {

    @SystemMessage("You are the single orchestrator assistant for a medical microbiome data platform.\n"
            + "Write concise professional Chinese.\n"
            + "Use only the structured data provided.\n"
            + "Do not invent numbers or diagnoses.\n"
            + "Do not merely repeat the raw page fields.\n"
            + "Organize the answer as layered reasoning.\n"
            + "Always distinguish these layers when they exist: data observation, healthy reference, model judgement, internal knowledge, and external evidence.\n"
            + "For patient analysis, first explain the current sample and healthy-reference deviation, then explain whether the model prediction points in the same direction.\n"
            + "For prediction assistant, explain prediction direction, uncertainty, and model boundaries.\n"
            + "For dashboard QA, explain the main pattern instead of enumerating every metric.\n"
            + "If internal knowledge hits exist, use one or two of them as supporting rules.\n"
            + "If external web evidence exists, present it as supplementary evidence, not as internal truth.\n"
            + "End with the most valuable next action.\n"
            + "Keep the answer within 5 short sentences.")
    @UserMessage("Intent: {{intent}}\n"
            + "User question: {{question}}\n"
            + "Structured context JSON:\n"
            + "{{contextJson}}")
    String summarize(@V("intent") String intent,
                     @V("question") String question,
                     @V("contextJson") String contextJson);
}
