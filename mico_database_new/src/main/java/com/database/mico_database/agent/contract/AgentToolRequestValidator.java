package com.database.mico_database.agent.contract;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Deterministic policy and shape validation. It does not authenticate a caller,
 * access a database, or execute a query.
 */
public final class AgentToolRequestValidator {

    private static final Pattern RUN_ID = Pattern.compile("^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$");
    private static final int MAX_STRING_ARGUMENT_LENGTH = 4096;
    private static final Set<String> DANGEROUS_KEYS = new HashSet<>(Arrays.asList(
            "sql", "rawsql", "query", "where", "table", "tablename", "from", "join", "select",
            "insert", "update", "delete", "drop", "alter", "create", "truncate", "union",
            "database", "schema", "subjectid", "patientid"));
    private static final Pattern SQL_STATEMENT_START = Pattern.compile(
            "(?is)^\\s*(select|insert|update|delete|drop|alter|create|truncate)\\b");
    private static final Pattern STACKED_SQL_STATEMENT = Pattern.compile(
            "(?is);\\s*(select|insert|update|delete|drop|alter|create|truncate)\\b");

    public AgentToolValidationResult validate(AgentToolRequest request) {
        List<AgentToolError> errors = new ArrayList<>();
        if (request == null) {
            errors.add(error("INVALID_REQUEST", "Tool request is required", null, false,
                    "A tool call must be represented by the typed request envelope"));
            return AgentToolValidationResult.rejected(errors);
        }

        AgentToolDefinition definition = AgentToolCatalog.find(request.getToolName()).orElse(null);
        if (definition == null) {
            errors.add(error("UNKNOWN_TOOL", "Tool is not registered in the catalog", "toolName", false,
                    "Only the fixed P1-A Tool Catalog may be called"));
            return AgentToolValidationResult.rejected(errors);
        }

        validateEnvelope(request, errors);
        validateScopes(request, definition, errors);
        Map<String, Object> arguments = request.getArguments();
        if (arguments == null) {
            errors.add(error("MISSING_ARGUMENTS", "Arguments object is required", "arguments", false,
                    "The tool contract does not permit an absent argument object"));
            return AgentToolValidationResult.rejected(errors);
        }

        validateArgumentNames(request.getToolName(), arguments, definition, errors);
        validateArgumentValues(request.getToolName(), arguments, definition, errors);
        validateToolSpecificRules(request.getToolName(), arguments, errors);
        return errors.isEmpty() ? AgentToolValidationResult.valid() : AgentToolValidationResult.rejected(errors);
    }

    private void validateEnvelope(AgentToolRequest request, List<AgentToolError> errors) {
        if (isBlank(request.getRunId()) || !RUN_ID.matcher(request.getRunId()).matches()) {
            errors.add(error("INVALID_RUN_ID", "runId is empty or has an invalid format", "runId", false,
                    "runId must be a bounded opaque correlation identifier"));
        }
        if (isBlank(request.getToolCallId())) {
            errors.add(error("INVALID_TOOL_CALL_ID", "toolCallId is required", "toolCallId", false,
                    "Every call must be auditable by a non-empty call identifier"));
        }
        if (isBlank(request.getRequesterId())) {
            errors.add(error("INVALID_REQUESTER_ID", "requesterId is required", "requesterId", false,
                    "The policy layer requires a caller identity field; this is not authentication by itself"));
        }
        if (!AgentContractConstants.DATA_CONTRACT_VERSION.equals(request.getDataContractVersion())) {
            errors.add(error("INVALID_DATA_CONTRACT_VERSION", "Only data contract version v1 is supported",
                    "dataContractVersion", false, "The P1-A contract is versioned and fixed at v1"));
        }
    }

    private void validateScopes(AgentToolRequest request, AgentToolDefinition definition,
                                List<AgentToolError> errors) {
        Set<String> requestedScopes = request.getRequestedScopes();
        if (requestedScopes != null) {
            for (String requestedScope : requestedScopes) {
                if (!AgentScope.isKnown(requestedScope)) {
                    errors.add(error("UNKNOWN_SCOPE", "Scope is not registered", "requestedScopes", false,
                            "Scope names are closed and must come from the Java policy catalog"));
                }
            }
        }
        for (String requiredScope : definition.getRequiredScopes()) {
            if (requestedScopes == null || !requestedScopes.contains(requiredScope)) {
                errors.add(error("MISSING_SCOPE", "Required scope is missing", "requestedScopes", false,
                        "The request must carry the catalogued scope before policy evaluation can pass: "
                                + requiredScope));
            }
        }
    }

    private void validateArgumentNames(String toolName, Map<String, Object> arguments,
                                       AgentToolDefinition definition,
                                       List<AgentToolError> errors) {
        for (String name : arguments.keySet()) {
            if (name == null || isDangerousKey(name, toolName)) {
                errors.add(error("FORBIDDEN_QUERY_INPUT", "Query language or database selector input is forbidden",
                        "arguments", false, "P1-A accepts only catalogued structured arguments"));
            } else if (!definition.getArgumentSchema().supports(name)) {
                errors.add(error("UNKNOWN_ARGUMENT", "Argument is not allowed for this tool", name, false,
                        "The tool schema has additionalProperties=false"));
            }
            if ("patientId".equalsIgnoreCase(name) || "subjectId".equalsIgnoreCase(name)) {
                errors.add(error("PATIENT_ID_NOT_SUBJECT_ID",
                        "patientId cannot be interpreted as subjectId; use an explicitly verified subject link",
                        name, false, "patient_id is an internal business record identifier"));
            }
        }
    }

    private void validateArgumentValues(String toolName, Map<String, Object> arguments,
                                        AgentToolDefinition definition,
                                        List<AgentToolError> errors) {
        for (String required : definition.getArgumentSchema().getRequired()) {
            Object value = arguments.get(required);
            if (value == null || (value instanceof String && isBlank((String) value))) {
                errors.add(error("MISSING_ARGUMENT", "Required argument is missing or blank", required, false,
                        "The catalogued tool requires a precise structured locator or version"));
            }
        }
        for (Map.Entry<String, Object> entry : arguments.entrySet()) {
            String name = entry.getKey();
            Object value = entry.getValue();
            String type = definition.getArgumentSchema().getProperties().get(name);
            if (type == null) {
                continue;
            }
            if ("string".equals(type) && !(value instanceof String)) {
                errors.add(error("INVALID_ARGUMENT_TYPE", "Argument must be a string", name, false,
                        "The catalog declares a string field"));
            } else if ("integer".equals(type) && !isInteger(value)) {
                errors.add(error("INVALID_ARGUMENT_TYPE", "Argument must be an integer", name, false,
                        "The catalog declares a bounded integer field"));
            } else if ("recordProfileLocator".equals(type)) {
                validateRecordProfileLocator(value, name, errors);
            } else if ("queryPlan".equals(type) && !(value instanceof QueryPlan)) {
                errors.add(error("INVALID_ARGUMENT_TYPE", "queryPlan must be a typed QueryPlan object",
                        name, false, "Dynamic scientific reads require a catalog-oriented plan"));
            }
            if (value instanceof String && !(AgentToolName.EXECUTE_READ_QUERY.getWireName().equals(toolName)
                    && "sql".equals(name))) {
                validateStringValue((String) value, name, errors);
            }
            if ("limit".equals(name)) {
                validateLimit(value, definition.getMaxResultLimit(), errors);
            }
        }
        if (AgentToolName.EXECUTE_READ_QUERY.getWireName().equals(toolName)) {
            Object queryPlan = arguments.get("queryPlan");
            Object sql = arguments.get("sql");
            if (queryPlan != null && sql != null) {
                errors.add(error("AMBIGUOUS_QUERY_INPUT", "queryPlan and legacy sql cannot be supplied together",
                        "arguments", false, "A read call must choose one execution representation"));
            } else if (queryPlan instanceof QueryPlan) {
                try {
                    QueryPlan typedPlan = (QueryPlan) queryPlan;
                    if (arguments.get("limit") instanceof Number
                            && ((Number) arguments.get("limit")).intValue() != typedPlan.getLimit()) {
                        throw new IllegalArgumentException("query plan and request limits differ");
                    }
                    new QueryPlanCompiler().compile(typedPlan);
                } catch (IllegalArgumentException exception) {
                    errors.add(error("QUERY_PLAN_REJECTED", "The catalog query plan was rejected by Java policy",
                            "queryPlan", false, "Java owns relation, field, aggregation, and limit validation"));
                }
            }
            if (sql instanceof String) {
                DynamicReadQueryPolicy.Validation validation = DynamicReadQueryPolicy.validate((String) sql);
                if (!validation.isValid()) {
                    errors.add(error("DYNAMIC_QUERY_REJECTED",
                            "The model-proposed read query was rejected by Java policy",
                            "sql", false,
                            "Java permits only bounded read queries over the approved data surface"));
                }
            }
        }
    }

    private void validateRecordProfileLocator(Object value, String field, List<AgentToolError> errors) {
        if (!(value instanceof RecordProfileLocator)) {
            errors.add(error("INVALID_ARGUMENT_TYPE", "recordProfileLocator must be a typed locator object",
                    field, false, "Exact profile tools do not accept a concatenated sampleKey string"));
            return;
        }
        RecordProfileLocator locator = (RecordProfileLocator) value;
        if (locator.getInternalRecordId() == null || locator.getInternalRecordId() <= 0) {
            errors.add(error("INVALID_INTERNAL_RECORD_ID", "internalRecordId must be a positive integer",
                    field + ".internalRecordId", false,
                    "internalRecordId is the Mico internal business record key, not a Subject ID"));
        }
        String sourceSampleId = locator.getSourceSampleId();
        if (isBlank(sourceSampleId)) {
            errors.add(error("MISSING_SOURCE_SAMPLE_ID", "sourceSampleId must not be blank",
                    field + ".sourceSampleId", false,
                    "The exact locator requires the source sample namespace value"));
        } else {
            validateStringValue(sourceSampleId, field + ".sourceSampleId", errors);
        }
    }

    private void validateStringValue(String value, String field, List<AgentToolError> errors) {
        if (isBlank(value)) {
            errors.add(error("INVALID_ARGUMENT_VALUE", "String argument must not be blank", field, false,
                    "Domain locators and labels must remain explicit when present"));
        }
        if (value.length() > MAX_STRING_ARGUMENT_LENGTH) {
            errors.add(error("ARGUMENT_TOO_LONG", "String argument exceeds the allowed length", field, false,
                    "The contract bounds field size without interpreting domain text as SQL"));
        }
        if (containsControlCharacter(value)) {
            errors.add(error("INVALID_CONTROL_CHARACTER", "String argument contains a control character", field,
                    false, "Control characters are not accepted in structured tool arguments"));
        }
        if (containsObviousSqlStatement(value)) {
            errors.add(error("FORBIDDEN_QUERY_INPUT", "Argument begins with or stacks a SQL statement", field,
                    false, "Value-level screening is limited to obvious SQL statements; Java P1-B adapters must still use fixed parameterized queries"));
        }
    }

    private void validateToolSpecificRules(String toolName, Map<String, Object> arguments,
                                            List<AgentToolError> errors) {
        // The executable catalog currently has one dynamic read tool. Its SQL
        // policy is validated in validateArgumentValues above.
    }

    private void validateLimit(Object value, int maxResultLimit, List<AgentToolError> errors) {
        if (value == null) {
            return;
        }
        if (!isInteger(value)) {
            return;
        }
        long limit = ((Number) value).longValue();
        if (limit <= 0) {
            errors.add(error("INVALID_LIMIT", "limit must be greater than zero", "limit", false,
                    "Result bounds must be positive and explicit"));
        } else if (limit > maxResultLimit) {
            errors.add(error("LIMIT_EXCEEDED", "limit exceeds the catalogued tool maximum", "limit", false,
                    "The tool catalog bounds result volume to limit resource and disclosure risk"));
        }
    }

    private static boolean isInteger(Object value) {
        return value instanceof Byte || value instanceof Short || value instanceof Integer || value instanceof Long
                || value instanceof java.math.BigInteger;
    }

    private static boolean hasNonBlank(Map<String, Object> arguments, String key) {
        Object value = arguments.get(key);
        return value instanceof String && !isBlank((String) value);
    }

    private static boolean isDangerousKey(String key, String toolName) {
        if (AgentToolName.EXECUTE_READ_QUERY.getWireName().equals(toolName)
                && "sql".equalsIgnoreCase(key)) {
            return false;
        }
        return DANGEROUS_KEYS.contains(key.toLowerCase(Locale.ROOT));
    }

    private static boolean containsObviousSqlStatement(String value) {
        return SQL_STATEMENT_START.matcher(value).find() || STACKED_SQL_STATEMENT.matcher(value).find();
    }

    private static boolean containsControlCharacter(String value) {
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            if (character < 0x20 || character == 0x7F) {
                return true;
            }
        }
        return false;
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    private static AgentToolError error(String code, String message, String field,
                                        boolean retryable, String policyReason) {
        return AgentToolError.of(code, message, field, retryable, policyReason);
    }
}
