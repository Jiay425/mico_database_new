package com.database.mico_ai.config;

import com.database.mico_ai.agent.CoordinatorAiAgent;
import com.database.mico_ai.agent.DataAnalysisAiAgent;
import com.database.mico_ai.agent.KnowledgeReviewAiAgent;
import com.database.mico_ai.agent.PredictionExplanationAiAgent;
import com.database.mico_ai.agent.SummaryAiAgent;
import com.database.mico_ai.agent.TriageAiAgent;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.DisabledChatModel;
import dev.langchain4j.service.AiServices;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class LangChainConfiguration {

    @Bean
    @ConditionalOnMissingBean(ChatModel.class)
    public ChatModel openAiCompatibleChatModel(AiProviderProperties aiProviderProperties,
                                               OpenAiCompatibleProperties openAiCompatibleProperties) {
        if (!aiProviderProperties.isProviderEnabled() || !openAiCompatibleProperties.isConfigured()) {
            return new DisabledChatModel();
        }

        return OpenAiChatModel.builder()
                .apiKey(openAiCompatibleProperties.getApiKey())
                .baseUrl(openAiCompatibleProperties.getBaseUrl())
                .modelName(openAiCompatibleProperties.getModelName())
                .temperature(openAiCompatibleProperties.getTemperature())
                .logRequests(openAiCompatibleProperties.isLogRequests())
                .logResponses(openAiCompatibleProperties.isLogResponses())
                .build();
    }

    @Bean
    public TriageAiAgent triageAiAgent(ChatModel chatModel) {
        return AiServices.builder(TriageAiAgent.class)
                .chatModel(chatModel)
                .build();
    }

    @Bean
    public SummaryAiAgent summaryAiAgent(ChatModel chatModel) {
        return AiServices.builder(SummaryAiAgent.class)
                .chatModel(chatModel)
                .build();
    }

    @Bean
    public DataAnalysisAiAgent dataAnalysisAiAgent(ChatModel chatModel) {
        return AiServices.builder(DataAnalysisAiAgent.class)
                .chatModel(chatModel)
                .build();
    }

    @Bean
    public KnowledgeReviewAiAgent knowledgeReviewAiAgent(ChatModel chatModel) {
        return AiServices.builder(KnowledgeReviewAiAgent.class)
                .chatModel(chatModel)
                .build();
    }

    @Bean
    public PredictionExplanationAiAgent predictionExplanationAiAgent(ChatModel chatModel) {
        return AiServices.builder(PredictionExplanationAiAgent.class)
                .chatModel(chatModel)
                .build();
    }

    @Bean
    public CoordinatorAiAgent coordinatorAiAgent(ChatModel chatModel) {
        return AiServices.builder(CoordinatorAiAgent.class)
                .chatModel(chatModel)
                .build();
    }
}