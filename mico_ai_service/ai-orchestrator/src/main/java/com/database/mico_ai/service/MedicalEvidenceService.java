package com.database.mico_ai.service;

import com.database.mico_ai.config.WebSearchProperties;
import com.database.mico_ai.dto.MedicalEvidenceHit;
import com.database.mico_ai.dto.WebEvidenceHit;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@Service
public class MedicalEvidenceService {

    private final WebSearchService webSearchService;
    private final WebSearchProperties properties;
    private final LocalMedicalRagService localMedicalRagService;

    public MedicalEvidenceService(WebSearchService webSearchService,
                                  WebSearchProperties properties,
                                  LocalMedicalRagService localMedicalRagService) {
        this.webSearchService = webSearchService;
        this.properties = properties;
        this.localMedicalRagService = localMedicalRagService;
    }

    public List<MedicalEvidenceHit> searchMedicalEvidence(String query, Integer maxResults) {
        if (query == null || query.isBlank()) {
            return List.of();
        }
        int limit = normalizeLimit(maxResults);

        List<MedicalEvidenceHit> merged = new ArrayList<>();
        mergeUnique(merged, localMedicalRagService.search(query, limit), limit);
        if (merged.size() < limit) {
            mergeUnique(merged, localMedicalRagService.search(buildMedicalQuery(query), limit - merged.size()), limit);
        }

        if (merged.size() < limit && properties.isConfigured()) {
            int remaining = limit - merged.size();
            mergeUnique(merged, searchExternalMedicalEvidence(query, remaining), limit);
        }
        return merged;
    }

    public List<Map<String, Object>> listMedicalSources() {
        List<Map<String, Object>> sources = new ArrayList<>();
        sources.add(localMedicalRagService.sourceDescriptor());
        for (String domain : properties.getMedicalAllowedDomains()) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("domain", domain);
            item.put("source", sourceLabel(domain));
            item.put("evidenceType", evidenceType(domain));
            sources.add(item);
        }
        return sources;
    }

    public Map<String, Object> medicalEvidenceStatus() {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("medicalEvidenceReady", properties.isConfigured());
        status.put("medicalEvidenceProvider", properties.getProvider());
        status.put("medicalEvidenceAllowedSourceCount", properties.getMedicalAllowedDomains().size());
        status.put("medicalEvidenceAllowedDomains", properties.getMedicalAllowedDomains());
        status.put("medicalEvidenceSearchMode", "local-rag-first+web-search-fallback");
        status.putAll(localMedicalRagService.status());
        return status;
    }

    private List<MedicalEvidenceHit> searchExternalMedicalEvidence(String query, Integer maxResults) {
        String enrichedQuery = buildMedicalQuery(query);
        List<WebEvidenceHit> rawHits = webSearchService.search(enrichedQuery, maxResults, buildSearchOverrides(maxResults));
        List<MedicalEvidenceHit> hits = new ArrayList<>();
        for (WebEvidenceHit hit : rawHits) {
            if (!isAllowedDomain(hit.domain())) {
                continue;
            }
            String citation = buildExternalCitation(hit);
            hits.add(new MedicalEvidenceHit(
                    hit.title(),
                    hit.url(),
                    hit.domain(),
                    sourceLabel(hit.domain()),
                    citation.isBlank() ? hit.snippet() : citation + " | 证据片段: " + safeText(hit.snippet()),
                    hit.publishedDate(),
                    evidenceType(hit.domain()),
                    hit.score(),
                    "",
                    "",
                    "",
                    "",
                    citation
            ));
        }
        return hits;
    }

    private Map<String, Object> buildSearchOverrides(Integer maxResults) {
        Map<String, Object> overrides = new LinkedHashMap<>();
        overrides.put("topic", "general");
        overrides.put("search_depth", "advanced");
        overrides.put("max_results", normalizeLimit(maxResults));
        overrides.put("include_domains", properties.getMedicalAllowedDomains());
        overrides.put("include_answer", false);
        overrides.put("include_raw_content", false);
        overrides.put("include_images", false);
        return overrides;
    }

    private int normalizeLimit(Integer maxResults) {
        if (maxResults == null || maxResults <= 0) {
            return Math.min(properties.getMaxResults(), 6);
        }
        return Math.min(maxResults, 8);
    }

    private String buildMedicalQuery(String query) {
        return query.trim() + " microbiome disease clinical evidence";
    }

    private boolean isAllowedDomain(String domain) {
        if (domain == null || domain.isBlank()) {
            return false;
        }
        String lower = domain.toLowerCase(Locale.ROOT);
        for (String allowed : properties.getMedicalAllowedDomains()) {
            String normalized = allowed.toLowerCase(Locale.ROOT);
            if (lower.equals(normalized) || lower.endsWith("." + normalized)) {
                return true;
            }
        }
        return false;
    }

    private String sourceLabel(String domain) {
        String lower = domain == null ? "" : domain.toLowerCase(Locale.ROOT);
        if (lower.contains("pubmed")) {
            return "PubMed";
        }
        if (lower.contains("pmc")) {
            return "PubMed Central";
        }
        if (lower.endsWith("nih.gov") || lower.contains(".nih.gov")) {
            return "NIH";
        }
        if (lower.endsWith("who.int") || lower.contains(".who.int")) {
            return "WHO";
        }
        if (lower.endsWith("cdc.gov") || lower.contains(".cdc.gov")) {
            return "CDC";
        }
        if (lower.endsWith("mayoclinic.org")) {
            return "Mayo Clinic";
        }
        if (lower.endsWith("clevelandclinic.org")) {
            return "Cleveland Clinic";
        }
        if (lower.endsWith("hopkinsmedicine.org")) {
            return "Johns Hopkins Medicine";
        }
        return domain == null ? "Medical Source" : domain;
    }

    private String evidenceType(String domain) {
        String lower = domain == null ? "" : domain.toLowerCase(Locale.ROOT);
        if (lower.contains("pubmed")) {
            return "literature_index";
        }
        if (lower.contains("pmc")) {
            return "literature_fulltext";
        }
        if (lower.endsWith("nih.gov") || lower.contains(".nih.gov")) {
            return "government_research";
        }
        if (lower.endsWith("who.int") || lower.contains(".who.int")) {
            return "global_guidance";
        }
        if (lower.endsWith("cdc.gov") || lower.contains(".cdc.gov")) {
            return "public_health_guidance";
        }
        return "clinical_reference";
    }

    private void mergeUnique(List<MedicalEvidenceHit> target,
                             List<MedicalEvidenceHit> incoming,
                             int limit) {
        if (incoming == null || incoming.isEmpty() || target.size() >= limit) {
            return;
        }
        for (MedicalEvidenceHit hit : incoming) {
            if (target.size() >= limit) {
                break;
            }
            if (isDuplicate(target, hit)) {
                continue;
            }
            target.add(hit);
        }
    }

    private boolean isDuplicate(List<MedicalEvidenceHit> existing, MedicalEvidenceHit candidate) {
        String candidateKey = dedupeKey(candidate);
        for (MedicalEvidenceHit item : existing) {
            if (candidateKey.equals(dedupeKey(item))) {
                return true;
            }
        }
        return false;
    }

    private String dedupeKey(MedicalEvidenceHit hit) {
        if (hit.pmcid() != null && !hit.pmcid().isBlank()) {
            return "pmcid:" + hit.pmcid().toLowerCase(Locale.ROOT);
        }
        if (hit.url() != null && !hit.url().isBlank()) {
            return "url:" + hit.url().toLowerCase(Locale.ROOT);
        }
        return "title:" + safeText(hit.title()).toLowerCase(Locale.ROOT);
    }

    private String buildExternalCitation(WebEvidenceHit hit) {
        List<String> parts = new ArrayList<>();
        String source = sourceLabel(hit.domain());
        if (!source.isBlank()) {
            parts.add("Source:" + source);
        }
        String domain = safeText(hit.domain());
        if (!domain.isBlank()) {
            parts.add("Domain:" + domain);
        }
        String date = safeText(hit.publishedDate());
        if (!date.isBlank()) {
            parts.add("Date:" + date);
        }
        return String.join(" | ", parts);
    }

    private String safeText(String value) {
        return value == null ? "" : value.trim();
    }
}
