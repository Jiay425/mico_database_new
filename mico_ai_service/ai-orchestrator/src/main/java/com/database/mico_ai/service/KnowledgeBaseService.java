package com.database.mico_ai.service;

import com.database.mico_ai.config.KnowledgeBaseProperties;
import com.database.mico_ai.dto.KnowledgeHit;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Service
public class KnowledgeBaseService {

    private final KnowledgeBaseProperties properties;
    private final List<KnowledgeChunk> knowledgeChunks = new ArrayList<>();

    public KnowledgeBaseService(KnowledgeBaseProperties properties) {
        this.properties = properties;
    }

    @PostConstruct
    public void load() {
        reload();
    }

    public synchronized void reload() {
        knowledgeChunks.clear();
        if (!properties.isEnabled()) {
            return;
        }

        Path root = Paths.get(properties.getRootDir());
        if (!Files.exists(root)) {
            return;
        }

        try (Stream<Path> paths = Files.walk(root)) {
            paths.filter(Files::isRegularFile)
                    .filter(path -> path.toString().toLowerCase(Locale.ROOT).endsWith(".md"))
                    .sorted()
                    .forEach(path -> loadMarkdownFile(root, path));
        } catch (IOException ignored) {
        }
    }

    public List<KnowledgeHit> search(String query, Collection<String> preferredLibraries, int maxResults) {
        if (!properties.isEnabled() || knowledgeChunks.isEmpty()) {
            return List.of();
        }

        List<String> tokens = tokenizeQuery(query);
        if (tokens.isEmpty()) {
            return List.of();
        }

        Set<String> libraryHints = preferredLibraries == null
                ? Set.of()
                : preferredLibraries.stream()
                .filter(value -> value != null && !value.isBlank())
                .map(value -> value.trim().toLowerCase(Locale.ROOT))
                .collect(Collectors.toCollection(LinkedHashSet::new));

        List<SearchCandidate> candidates = new ArrayList<>();
        for (KnowledgeChunk chunk : knowledgeChunks) {
            double score = score(chunk, tokens, libraryHints);
            if (score > 0D) {
                candidates.add(new SearchCandidate(chunk, score));
            }
        }

        return candidates.stream()
                .sorted(Comparator.comparingDouble(SearchCandidate::score).reversed()
                        .thenComparing(candidate -> candidate.chunk().library())
                        .thenComparing(candidate -> candidate.chunk().title()))
                .limit(Math.max(1, maxResults))
                .map(candidate -> toHit(candidate.chunk(), candidate.score(), tokens))
                .collect(Collectors.toList());
    }

    public Map<String, Object> status() {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("knowledgeEnabled", properties.isEnabled());
        status.put("knowledgeRootDir", properties.getRootDir());
        status.put("knowledgeChunkCount", knowledgeChunks.size());
        status.put("knowledgeReady", properties.isEnabled() && !knowledgeChunks.isEmpty());
        return status;
    }

    private void loadMarkdownFile(Path root, Path file) {
        try {
            String markdown = Files.readString(file, StandardCharsets.UTF_8);
            if (markdown == null || markdown.isBlank()) {
                return;
            }

            String relativePath = root.relativize(file).toString().replace('\\', '/');
            String library = resolveLibrary(relativePath);
            String documentTitle = extractDocumentTitle(markdown, file);
            List<SectionChunk> sections = splitSections(documentTitle, markdown);

            int index = 0;
            for (SectionChunk section : sections) {
                String content = normalizeWhitespace(section.content());
                if (content.isBlank()) {
                    continue;
                }
                knowledgeChunks.add(new KnowledgeChunk(
                        relativePath + "#" + (++index),
                        library,
                        libraryLabel(library),
                        section.title(),
                        relativePath,
                        content,
                        content.toLowerCase(Locale.ROOT),
                        section.title().toLowerCase(Locale.ROOT)
                ));
            }
        } catch (IOException ignored) {
        }
    }

    private List<SectionChunk> splitSections(String documentTitle, String markdown) {
        List<SectionChunk> sections = new ArrayList<>();
        String currentTitle = documentTitle;
        StringBuilder currentContent = new StringBuilder();

        for (String rawLine : markdown.split("\\R")) {
            String line = rawLine == null ? "" : rawLine.trim();
            if (line.startsWith("# ")) {
                if (currentContent.length() == 0) {
                    currentTitle = line.substring(2).trim();
                }
                continue;
            }
            if (line.startsWith("## ") || line.startsWith("### ")) {
                flushSection(sections, currentTitle, currentContent);
                currentTitle = line.replaceFirst("^#{2,3}\\s+", "").trim();
                currentContent = new StringBuilder();
                continue;
            }
            if (!line.isBlank()) {
                currentContent.append(line).append('\n');
            }
        }

        flushSection(sections, currentTitle, currentContent);
        if (sections.isEmpty()) {
            sections.add(new SectionChunk(documentTitle, normalizeWhitespace(markdown)));
        }
        return sections;
    }

    private void flushSection(List<SectionChunk> sections, String title, StringBuilder content) {
        String normalized = normalizeWhitespace(content.toString());
        if (!normalized.isBlank()) {
            sections.add(new SectionChunk(title, normalized));
        }
    }

    private String extractDocumentTitle(String markdown, Path file) {
        for (String rawLine : markdown.split("\\R")) {
            String line = rawLine == null ? "" : rawLine.trim();
            if (line.startsWith("# ")) {
                return line.substring(2).trim();
            }
        }
        String name = file.getFileName().toString();
        return name.endsWith(".md") ? name.substring(0, name.length() - 3) : name;
    }

    private String resolveLibrary(String relativePath) {
        int slash = relativePath.indexOf('/');
        if (slash <= 0) {
            return "general";
        }
        return relativePath.substring(0, slash).toLowerCase(Locale.ROOT);
    }

    private double score(KnowledgeChunk chunk, List<String> tokens, Set<String> libraryHints) {
        double score = libraryHints.contains(chunk.library()) ? 0.35D : 0D;
        for (String token : tokens) {
            String normalizedToken = token.toLowerCase(Locale.ROOT);
            if (chunk.titleText().contains(normalizedToken)) {
                score += 1.8D;
            } else if (chunk.searchableText().contains(normalizedToken)) {
                score += 1.0D;
            }
        }
        return score;
    }

    private KnowledgeHit toHit(KnowledgeChunk chunk, double score, List<String> tokens) {
        return new KnowledgeHit(
                chunk.library(),
                chunk.libraryLabel(),
                chunk.title(),
                chunk.sourcePath(),
                buildSnippet(chunk.content(), tokens),
                Math.round(score * 100D) / 100D
        );
    }

    private String buildSnippet(String content, List<String> tokens) {
        String normalized = normalizeWhitespace(content);
        String lower = normalized.toLowerCase(Locale.ROOT);
        for (String token : tokens) {
            int index = lower.indexOf(token.toLowerCase(Locale.ROOT));
            if (index >= 0) {
                int start = Math.max(0, index - 28);
                int end = Math.min(normalized.length(), index + 88);
                String snippet = normalized.substring(start, end).trim();
                if (start > 0) {
                    snippet = "..." + snippet;
                }
                if (end < normalized.length()) {
                    snippet = snippet + "...";
                }
                return snippet;
            }
        }
        return normalized.length() > 120 ? normalized.substring(0, 120).trim() + "..." : normalized;
    }

    private List<String> tokenizeQuery(String query) {
        if (query == null || query.isBlank()) {
            return List.of();
        }
        String[] rawTokens = query.toLowerCase(Locale.ROOT)
                .replace('\n', ' ')
                .split("[\\s,，。；;、:：()（）\\[\\]{}<>|/\\\\]+");

        LinkedHashSet<String> tokens = new LinkedHashSet<>();
        for (String token : rawTokens) {
            if (token == null) {
                continue;
            }
            String cleaned = token.trim();
            if (cleaned.length() >= 2) {
                tokens.add(cleaned);
            }
        }
        return new ArrayList<>(tokens);
    }

    private String libraryLabel(String library) {
        return switch (library) {
            case "business" -> "业务库";
            case "medical" -> "医学库";
            case "model" -> "模型库";
            default -> "通用知识";
        };
    }

    private String normalizeWhitespace(String text) {
        if (text == null) {
            return "";
        }
        return text.replace('`', ' ')
                .replace("|", " ")
                .replace("- ", "")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private record SectionChunk(String title, String content) {
    }

    private record KnowledgeChunk(String id,
                                  String library,
                                  String libraryLabel,
                                  String title,
                                  String sourcePath,
                                  String content,
                                  String searchableText,
                                  String titleText) {
    }

    private record SearchCandidate(KnowledgeChunk chunk, double score) {
    }
}
