package com.database.mico_database.service;

import com.database.mico_database.config.RabbitConfig;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.util.Set;

@Service
@Slf4j
@ConditionalOnProperty(name = "app.rabbitmq.enabled", havingValue = "true")
public class CacheCleanerConsumer {

    @Autowired
    private StringRedisTemplate redisTemplate;

    @RabbitListener(queues = RabbitConfig.CLEAN_CACHE_QUEUE)
    public void handleDataUpdate(String patientId) {
        log.info("[mq] received data update notice, id: {}", patientId);

        try {
            Set<String> keys = redisTemplate.keys("search:*");
            if (keys != null && !keys.isEmpty()) {
                redisTemplate.delete(keys);
                log.info("[redis] cleared {} cache keys", keys.size());
            }
        } catch (Exception e) {
            log.error("[redis] cache cleanup failed", e);
        }
    }
}
