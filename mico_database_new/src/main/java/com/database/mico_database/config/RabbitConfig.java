package com.database.mico_database.config;

import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.core.TopicExchange;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConditionalOnProperty(name = "app.rabbitmq.enabled", havingValue = "true")
public class RabbitConfig {

    public static final String ANALYSIS_QUEUE_NAME = "analysis_queue";
    public static final String UPDATE_EXCHANGE = "patient_exchange";
    public static final String CLEAN_CACHE_QUEUE = "clean_cache_queue";

    @Bean
    public Queue analysisQueue() {
        return new Queue(ANALYSIS_QUEUE_NAME, true);
    }

    @Bean
    public TopicExchange updateExchange() {
        return new TopicExchange(UPDATE_EXCHANGE);
    }

    @Bean
    public Queue cleanCacheQueue() {
        return new Queue(CLEAN_CACHE_QUEUE);
    }

    @Bean
    public Binding bindingCleanCache(Queue cleanCacheQueue, TopicExchange updateExchange) {
        return BindingBuilder.bind(cleanCacheQueue).to(updateExchange).with("patient.#");
    }
}
