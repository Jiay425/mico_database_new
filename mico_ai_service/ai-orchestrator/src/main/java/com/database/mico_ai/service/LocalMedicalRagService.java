package com.database.mico_ai.service;

import com.database.mico_ai.dto.MedicalEvidenceHit;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

@Service
public class LocalMedicalRagService {

    private static final Pattern TOKEN_PATTERN = Pattern.compile("[A-Za-z][A-Za-z0-9_-]+|[\\u4e00-\\u9fff]+");
    private static final Set<String> EN_STOPWORDS = Set.of(
            "a", "an", "the", "and", "or", "to", "of", "in", "for", "on", "with", "by", "is", "are", "was", "were",
            "be", "as", "at", "from", "that", "this", "it", "its", "into", "their", "than", "then", "but", "if", "we",
            "our", "can", "could", "may", "might", "not", "no", "yes", "have", "has", "had", "also", "these", "those",
            "which", "who", "whom", "what", "when", "where", "why", "how", "such", "using", "used", "use", "between",
            "within", "without", "about", "after", "before", "during", "over", "under", "all", "any", "each", "other",
            "more", "most", "some", "many", "much", "via", "per", "both", "new", "one", "two", "three"
    );

    private final ObjectMapper objectMapper;
    private final boolean enabled;
    private final Path chunksPath;
    private final int defaultMaxResults;
    private final double minScore;

    private final List<LocalChunk> chunks = new ArrayList<>();
    private final Map<String, Integer> documentFrequency = new HashMap<>();
    private double averageDocLength = 0D;
    private boolean ready = false;
    private String lastError = "";

    public LocalMedicalRagService(ObjectMapper objectMapper,
                                  @Value("${app.medical-rag.enabled:true}") boolean enabled,
                                  @Value("${app.medical-rag.chunks-path:E:/DeskTop/java_code/mico_database_new/references/knowledge/medical/rag/medical_chunks.jsonl}") String chunksPath,
                                  @Value("${app.medical-rag.max-results:4}") int defaultMaxResults,
                                  @Value("${app.medical-rag.min-score:0.20}") double minScore) {
        this.objectMapper = objectMapper;
        this.enabled = enabled;
        this.chunksPath = Path.of(chunksPath);
        this.defaultMaxResults = defaultMaxResults;
        this.minScore = minScore;
    }

    @PostConstruct
    public void init() {
        loadChunks();
    }

    public List<MedicalEvidenceHit> search(String query, Integer maxResults) {
        if (!ready || query == null || query.isBlank()) {
            return List.of();
        }

        String expandedQuery = expandQueryForEnglishCorpus(query);
        List<String> queryTerms = tokenize(expandedQuery);
        if (queryTerms.isEmpty()) {
            return List.of();
        }

        int limit = normalizeLimit(maxResults);
        int docCount = chunks.size();
        if (docCount == 0 || averageDocLength <= 0) {
            return List.of();
        }

        List<SearchHit> candidates = new ArrayList<>();
        for (LocalChunk chunk : chunks) {
            double score = scoreBm25(queryTerms, chunk);
            if (score < minScore) {
                continue;
            }
            candidates.add(new SearchHit(chunk, score));
        }
        candidates.sort(Comparator.comparingDouble(SearchHit::score).reversed());

        List<MedicalEvidenceHit> hits = new ArrayList<>();
        for (int i = 0; i < Math.min(limit, candidates.size()); i++) {
            SearchHit hit = candidates.get(i);
            hits.add(toEvidenceHit(hit.chunk(), hit.score()));
        }
        return hits;
    }

    public Map<String, Object> status() {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("localMedicalRagEnabled", enabled);
        status.put("localMedicalRagReady", ready);
        status.put("localMedicalRagChunks", chunks.size());
        status.put("localMedicalRagAvgDocLength", averageDocLength);
        status.put("localMedicalRagPath", chunksPath.toString());
        status.put("localMedicalRagSearchMode", "bm25-local-fulltext");
        if (!lastError.isBlank()) {
            status.put("localMedicalRagLastError", lastError);
        }
        return status;
    }

    public Map<String, Object> sourceDescriptor() {
        Map<String, Object> source = new LinkedHashMap<>();
        source.put("domain", "local://medical-fulltext-rag");
        source.put("source", "Local Medical Fulltext RAG");
        source.put("evidenceType", "literature_fulltext_internal");
        source.put("ready", ready);
        source.put("chunkCount", chunks.size());
        return source;
    }

    private void loadChunks() {
        chunks.clear();
        documentFrequency.clear();
        averageDocLength = 0D;
        ready = false;
        lastError = "";

        if (!enabled) {
            lastError = "disabled";
            return;
        }
        if (!Files.exists(chunksPath)) {
            lastError = "chunks_path_not_found";
            return;
        }

        int docLengthTotal = 0;
        try (Stream<String> lines = Files.lines(chunksPath, StandardCharsets.UTF_8)) {
            lines.forEach(line -> {
                String raw = line == null ? "" : line.trim();
                if (raw.isBlank()) {
                    return;
                }
                try {
                    Map<String, Object> row = objectMapper.readValue(raw, new TypeReference<Map<String, Object>>() {});
                    String text = safeText(row.get("text"));
                    if (text.isBlank()) {
                        return;
                    }
                    List<String> tokens = tokenize(text);
                    if (tokens.isEmpty()) {
                        return;
                    }
                    Map<String, Integer> tf = new HashMap<>();
                    for (String token : tokens) {
                        tf.merge(token, 1, Integer::sum);
                    }
                    synchronized (this) {
                        LocalChunk chunk = new LocalChunk(
                                safeText(row.get("chunkId")),
                                safeText(row.get("pmcid")),
                                safeText(row.get("pmid")),
                                safeText(row.get("topic")),
                                safeText(row.get("title")),
                                safeText(row.get("year")),
                                safeText(row.get("section")),
                                safeText(row.get("sourceUrl")),
                                text,
                                tokens.size(),
                                tf
                        );
                        chunks.add(chunk);
                        for (String token : tf.keySet()) {
                            documentFrequency.merge(token, 1, Integer::sum);
                        }
                    }
                } catch (Exception ignored) {
                    // Skip malformed rows and continue loading.
                }
            });
        } catch (IOException ex) {
            lastError = "read_failed:" + ex.getClass().getSimpleName();
            return;
        }

        for (LocalChunk chunk : chunks) {
            docLengthTotal += chunk.docLength();
        }
        if (!chunks.isEmpty()) {
            averageDocLength = ((double) docLengthTotal) / chunks.size();
            ready = true;
        } else if (lastError.isBlank()) {
            lastError = "no_valid_chunks";
        }
    }

    private int normalizeLimit(Integer maxResults) {
        if (maxResults == null || maxResults <= 0) {
            return Math.max(1, Math.min(defaultMaxResults, 10));
        }
        return Math.max(1, Math.min(maxResults, 10));
    }

    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        String normalized = text == null ? "" : text.toLowerCase(Locale.ROOT);
        Matcher matcher = TOKEN_PATTERN.matcher(normalized);
        while (matcher.find()) {
            String token = matcher.group().trim();
            if (token.length() <= 1) {
                continue;
            }
            if (isAscii(token) && EN_STOPWORDS.contains(token)) {
                continue;
            }
            tokens.add(token);
        }
        return tokens;
    }

    /**
     * The full-text corpus is mostly English. For Chinese prompts we append
     * domain English retrieval hints so evidence can still be recalled.
     */
    private String expandQueryForEnglishCorpus(String query) {
        String lower = query == null ? "" : query.toLowerCase(Locale.ROOT);
        StringBuilder sb = new StringBuilder(query == null ? "" : query.trim());

        if (containsAny(lower, "健康组", "健康对照", "基线")) {
            sb.append(" healthy control baseline");
        }
        if (containsAny(lower, "差异", "偏离", "不同")) {
            sb.append(" differential abundance deviation dysbiosis");
        }
        if (containsAny(lower, "文献", "证据", "研究")) {
            sb.append(" clinical evidence literature study");
        }
        if (containsAny(lower, "样本", "患者")) {
            sb.append(" patient sample");
        }
        if (containsAny(lower, "微生物", "菌", "菌群")) {
            sb.append(" gut microbiome microbial taxa");
        }
        if (containsAny(lower, "预测", "模型")) {
            sb.append(" model interpretation prediction");
        }
        if (containsAny(lower, "糖尿病", "t2d")) {
            sb.append(" type 2 diabetes t2d");
        }
        if (containsAny(lower, "炎症性肠病", "ibd")) {
            sb.append(" inflammatory bowel disease ibd");
        }
        if (containsAny(lower, "结直肠癌", "crc")) {
            sb.append(" colorectal cancer crc");
        }
        if (containsAny(lower, "脂肪肝", "nafld")) {
            sb.append(" fatty liver nafld");
        }
        if (containsAny(lower, "肝硬化", "cirrhosis")) {
            sb.append(" cirrhosis");
        }
        if (containsAny(lower, "肥胖", "obesity")) {
            sb.append(" obesity");
        }
        if (containsAny(lower, "阿尔茨海默", "ad")) {
            sb.append(" alzheimer disease ad");
        }
        if (containsAny(lower, "多发性硬化", "ms")) {
            sb.append(" multiple sclerosis ms");
        }
        if (containsAny(lower, "冠心病", "cad")) {
            sb.append(" coronary artery disease cad");
        }

        if (!containsAsciiLetter(lower)) {
            sb.append(" gut microbiome healthy control dysbiosis differential abundance disease evidence");
        }
        return sb.toString().trim();
    }

    private boolean containsAny(String text, String... patterns) {
        if (text == null || text.isBlank()) {
            return false;
        }
        for (String p : patterns) {
            if (p != null && !p.isBlank() && text.contains(p)) {
                return true;
            }
        }
        return false;
    }

    private boolean containsAsciiLetter(String text) {
        if (text == null) {
            return false;
        }
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                return true;
            }
        }
        return false;
    }

    private boolean isAscii(String token) {
        for (int i = 0; i < token.length(); i++) {
            if (token.charAt(i) > 127) {
                return false;
            }
        }
        return true;
    }

    private double scoreBm25(List<String> queryTerms, LocalChunk chunk) {
        double k1 = 1.2;
        double b = 0.75;
        double score = 0D;

        for (String term : queryTerms) {
            Integer df = documentFrequency.get(term);
            if (df == null || df <= 0) {
                continue;
            }
            int tf = chunk.termFrequency().getOrDefault(term, 0);
            if (tf <= 0) {
                continue;
            }
            double idf = Math.log(1 + ((chunks.size() - df + 0.5D) / (df + 0.5D)));
            double denom = tf + k1 * (1 - b + b * chunk.docLength() / averageDocLength);
            score += idf * ((tf * (k1 + 1)) / denom);
        }

        if (chunk.topic() != null && !chunk.topic().isBlank()) {
            String q = String.join(" ", queryTerms);
            if (q.contains(chunk.topic().toLowerCase(Locale.ROOT))) {
                score += 0.20D;
            }
        }
        return score;
    }

    private MedicalEvidenceHit toEvidenceHit(LocalChunk chunk, double score) {
        String citation = buildCitation(chunk);
        String snippet = trimSnippet(chunk.text(), 320);
        String snippetWithCitation = citation.isBlank()
                ? snippet
                : citation + " | 证据片段: " + snippet;

        return new MedicalEvidenceHit(
                chunk.title(),
                chunk.sourceUrl(),
                extractDomain(chunk.sourceUrl()),
                "Local Medical Fulltext RAG",
                snippetWithCitation,
                chunk.year(),
                "literature_fulltext_internal",
                score,
                chunk.pmcid(),
                chunk.pmid(),
                chunk.section(),
                chunk.topic(),
                citation
        );
    }

    private String buildCitation(LocalChunk chunk) {
        List<String> parts = new ArrayList<>();
        if (!chunk.pmcid().isBlank()) {
            parts.add("PMCID:" + chunk.pmcid());
        }
        if (!chunk.pmid().isBlank()) {
            parts.add("PMID:" + chunk.pmid());
        }
        if (!chunk.section().isBlank()) {
            parts.add("Section:" + chunk.section());
        }
        if (!chunk.topic().isBlank()) {
            parts.add("Topic:" + chunk.topic());
        }
        return String.join(" | ", parts);
    }

    private String trimSnippet(String text, int maxChars) {
        if (text == null) {
            return "";
        }
        String oneLine = text.replace('\n', ' ').replace('\r', ' ').replaceAll("\\s{2,}", " ").trim();
        if (oneLine.length() <= maxChars) {
            return oneLine;
        }
        return oneLine.substring(0, maxChars) + "...";
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

    private String safeText(Object value) {
        return value == null ? "" : String.valueOf(value).trim();
    }

    private record LocalChunk(String chunkId,
                              String pmcid,
                              String pmid,
                              String topic,
                              String title,
                              String year,
                              String section,
                              String sourceUrl,
                              String text,
                              int docLength,
                              Map<String, Integer> termFrequency) {
    }

    private record SearchHit(LocalChunk chunk, double score) {
    }
}
