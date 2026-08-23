package com.database.mico_database.agent.config;

import com.database.mico_database.agent.transport.AgentInternalToolController;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.web.SecurityFilterChain;

/** Lets the controller apply its own service-token policy without changing legacy UI security. */
@Configuration
public class AgentInternalSecurityConfiguration {

    @Bean
    @Order(1)
    public SecurityFilterChain agentInternalFilterChain(HttpSecurity http) throws Exception {
        http.antMatcher(AgentInternalToolController.EXECUTE_PATH)
                .authorizeRequests().anyRequest().permitAll()
                .and()
                .csrf().disable()
                .formLogin().disable()
                .httpBasic().disable();
        return http.build();
    }
}
