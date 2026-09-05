package com.database.mico_database.agent.contract;

import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Safety policy for model-proposed read SQL.
 *
 * <p>The SQL is intentionally dynamic, but it is still untrusted input.  This
 * policy is the Java execution boundary: one bounded SELECT/CTE query with a
 * model-chosen shape is accepted, the connection's business schema is the
 * data boundary, and mutating, cross-database, diagnostic or
 * resource-unbounded forms are rejected before a JDBC statement is prepared.</p>
 */
public final class DynamicReadQueryPolicy {

    public static final int MAX_SQL_LENGTH = 16_000;
    /** Typed sample-bounded reads may return one row per sample x feature. */
    public static final int MAX_QUERY_LIMIT = 20_000;
    public static final int MAX_QUERY_OFFSET = 1_000_000;

    private static final Pattern LIMIT = Pattern.compile(
            "(?is)\\blimit\\s+(\\d+)(?:\\s*,\\s*(\\d+)|\\s+offset\\s+(\\d+))?\\s*$");
    private static final Pattern DANGEROUS_WORD = Pattern.compile(
            "(?i)\\b(?:insert|update|delete|drop|alter|create|truncate|grant|revoke|call|load|outfile|dumpfile|set|use|show|describe|explain|handler|procedure|benchmark|sleep|load_file|into|for\\s+update|lock\\s+in\\s+share\\s+mode)\\b");
    private static final Pattern DISALLOWED_SCHEMA = Pattern.compile(
            "(?i)\\b(?:information_schema|performance_schema|mysql|sys|patient_data_manager)\\b");

    private DynamicReadQueryPolicy() {
    }

    public static Validation validate(String sql) {
        if (sql == null || sql.trim().isEmpty()) {
            return invalid("SQL is required");
        }
        if (sql.length() > MAX_SQL_LENGTH) {
            return invalid("SQL exceeds the maximum length");
        }
        if (containsControlCharacter(sql)) {
            return invalid("SQL contains a control character");
        }
        /*
         * Scan the SQL code rather than the quoted literals.  Domain values
         * can legitimately contain semicolons, question marks, @ signs or
         * comment-looking text; only the SQL syntax outside literals is a
         * policy concern.  The original SQL is still passed to the JDBC
         * PreparedStatement unchanged after this validation.
         */
        String code = withoutQuotedText(sql);
        if (code.indexOf(';') >= 0 || containsComment(code)) {
            return invalid("Only one uncommented SQL statement is supported");
        }

        String normalized = code.trim().toLowerCase(Locale.ROOT);
        if (!normalized.matches("(?s)^(select|with)\\b.*")) {
            return invalid("Only SELECT or WITH ... SELECT statements are supported");
        }
        if (!normalized.matches("(?s).*\\bselect\\b.*")) {
            return invalid("A read query must contain SELECT");
        }
        if (DANGEROUS_WORD.matcher(code).find() || DISALLOWED_SCHEMA.matcher(code).find()) {
            return invalid("The SQL contains a forbidden operation or schema reference");
        }
        if (code.indexOf('?') >= 0 || code.indexOf('@') >= 0) {
            return invalid("Unbound parameter markers are not accepted");
        }

        Matcher limitMatcher = LIMIT.matcher(sql.trim());
        if (!limitMatcher.find()) {
            return invalid("A bounded LIMIT is required");
        }
        long limit = Long.parseLong(limitMatcher.group(2) == null
                ? limitMatcher.group(1) : limitMatcher.group(2));
        if (limit < 1 || limit > MAX_QUERY_LIMIT) {
            return invalid("LIMIT is outside the approved range");
        }
        String offsetText = limitMatcher.group(3) != null
                ? limitMatcher.group(3) : limitMatcher.group(2) != null ? limitMatcher.group(1) : null;
        if (offsetText != null && Long.parseLong(offsetText) > MAX_QUERY_OFFSET) {
            return invalid("OFFSET is outside the approved range");
        }
        return new Validation(true, null);
    }

    private static boolean containsComment(String sql) {
        return sql.contains("--") || sql.contains("/*") || sql.contains("*/")
                || sql.indexOf('#') >= 0;
    }

    private static boolean containsControlCharacter(String value) {
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            if (character < 0x20 && character != '\n' && character != '\r' && character != '\t') {
                return true;
            }
            if (character == 0x7F) {
                return true;
            }
        }
        return false;
    }

    /** Removes quoted literals only for policy keyword scanning; the original SQL is executed unchanged. */
    private static String withoutQuotedText(String value) {
        StringBuilder result = new StringBuilder(value.length());
        char quote = 0;
        for (int index = 0; index < value.length(); index++) {
            char current = value.charAt(index);
            if (quote != 0) {
                if (current == '\\' && index + 1 < value.length()) {
                    result.append("  ");
                    index++;
                } else if (current == quote) {
                    if (index + 1 < value.length() && value.charAt(index + 1) == quote) {
                        result.append("  ");
                        index++;
                    } else {
                        quote = 0;
                        result.append(' ');
                    }
                } else {
                    result.append(' ');
                }
            } else if (current == '\'' || current == '"') {
                quote = current;
                result.append(' ');
            } else if (current == '`') {
                result.append(current);
            } else {
                result.append(current);
            }
        }
        return result.toString();
    }

    private static Validation invalid(String reason) {
        return new Validation(false, reason);
    }

    public static final class Validation {
        private final boolean valid;
        private final String reason;

        private Validation(boolean valid, String reason) {
            this.valid = valid;
            this.reason = reason;
        }

        public boolean isValid() {
            return valid;
        }

        public String getReason() {
            return reason;
        }
    }
}
