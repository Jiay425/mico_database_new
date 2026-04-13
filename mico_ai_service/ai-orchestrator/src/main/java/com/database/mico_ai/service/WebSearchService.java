package com.database.mico_ai.service;

import com.database.mico_ai.config.WebSearchProperties;
import com.database.mico_ai.dto.WebEvidenceHit;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.net.URI;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@Service
public class WebSearchService {

    private final RestClient restClient;
    private final WebSearchProperties properties;
    private final ObjectMapper objectMapper;

    public WebSearchService(RestClient.Builder builder,
                            WebSearchProperties properties,
                            ObjectMapper objectMapper) {
        this.properties = properties;
        this.objectMapper = objectMapper;
        this.restClient = builder
                .baseUrl(trimBaseUrl(properties.getBaseUrl()))
                .defaultHeader(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    public List<WebEvidenceHit> search(String query, Integer maxResults) {
        return search(query, maxResults, Map.of());
    }

    public List<WebEvidenceHit> search(String query, Integer maxResults, Map<String, Object> overrides) {
        if (!properties.isConfigured() || query == null || query.isBlank()) {
            return List.of();
        }

        int limit = maxResults == null || maxResults <= 0 ? properties.getMaxResults() : maxResults;
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("query", query.trim());
        payload.put("search_depth", properties.getSearchDepth());
        payload.put("topic", properties.getTopic());
        payload.put("max_results", limit);
        payload.put("include_answer", properties.isIncludeAnswer());
        payload.put("include_raw_content", false);
        payload.put("include_images", false);
        if (overrides != null && !overrides.isEmpty()) {
            payload.putAll(overrides);
        }

        try {
            String raw = restClient.post()
                    .uri(properties.getSearchPath())
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + properties.getApiKey())
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(payload)
                    .retrieve()
                    .body(String.class);
            return parseResults(raw);
        } catch (Exception ignored) {
            return List.of();
        }
    }

    public Map<String, Object> status() {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("webSearchEnabled", properties.isEnabled());
        status.put("webSearchProvider", properties.getProvider());
        status.put("webSearchConfigured", properties.isConfigured());
        status.put("webSearchBaseUrl", properties.getBaseUrl());
        return status;
    }

    @SuppressWarnings("unchecked")
    private List<WebEvidenceHit> parseResults(String raw) throws Exception {
        if (raw == null || raw.isBlank()) {
            return List.of();
        }
        String body = raw.trim();
        if (body.startsWith("<")) {
            return List.of();
        }
        Map<String, Object> parsed = objectMapper.readValue(body, Map.class);
        List<Map<String, Object>> results = (List<Map<String, Object>>) parsed.getOrDefault("results", List.of());
        List<WebEvidenceHit> hits = new ArrayList<>();
        for (Map<String, Object> item : results) {
            if (item == null || item.isEmpty()) {
                continue;
            }
            String url = safeText(item.get("url"));
            hits.add(new WebEvidenceHit(
                    safeText(item.get("title")),
                    url,
                    extractDomain(url),
                    safeText(item.get("content")),
                    preferNonBlank(safeText(item.get("published_date")), safeText(item.get("publishedDate")), safeText(item.get("date"))),
                    safeDouble(item.get("score"))
            ));
        }
        return hits;
    }

    private String extractDomain(String url) {
        if (url == null || url.isBlank()) {
            return "";
        }
        try {
            String host = URI.create(url).getHost();
            return host == null ? "" : host.toLowerCase(Locale.ROOT);
        } catch (Exception ex) {
            return "";
        }
    }

    private String trimBaseUrl(String value) {
        if (value == null || value.isBlank()) {
            return "https://api.tavily.com";
        }
        return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
    }

    private String safeText(Object value) {
        return value == null ? "" : String.valueOf(value).trim();
    }

    private String preferNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return "";
    }

    private double safeDouble(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        String text = safeText(value);
        if (text.isBlank()) {
            return 0D;
        }
        try {
            return Double.parseDouble(text);
        } catch (NumberFormatException ex) {
            return 0D;
        }
    }
}