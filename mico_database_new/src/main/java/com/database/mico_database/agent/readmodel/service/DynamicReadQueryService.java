package com.database.mico_database.agent.readmodel.service;

import com.database.mico_database.agent.contract.DynamicReadQueryPolicy;
import com.database.mico_database.agent.contract.QueryPlan;
import com.database.mico_database.agent.contract.QueryPlanCompiler;
import com.database.mico_database.agent.readmodel.DynamicQueryReadModel;
import com.database.mico_database.agent.readmodel.OpaqueSampleKeyEncoder;
import com.database.mico_database.agent.readmodel.ReadModelResult;
import com.database.mico_database.agent.readmodel.ReadReceipt;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.sql.DataSource;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java-only execution boundary for model-proposed typed QueryPlans.  Python never gets
 * a DataSource and this service never exposes an arbitrary SQL interface to a
 * browser or to a persistence layer.
 */
@Service
public class DynamicReadQueryService {

    public static final String SOURCE = "java_agent_read_model";
    private static final String EXPECTED_CATALOG = "patient_data_manager";
    // A 20M-row abundance aggregation can legitimately exceed the former
    // ten-second cap over the SSH-backed local development database.  The
    // query is still a single Java-validated SELECT, read-only, and bounded
    // to at most 1,000 returned rows; this simply keeps the timeout finite
    // while allowing one controlled aggregate to complete.
    private static final int QUERY_TIMEOUT_SECONDS = 180;
    private static final int MAX_CELL_TEXT_LENGTH = 4096;

    private final DataSource dataSource;
    private final OpaqueSampleKeyEncoder opaqueSampleKeyEncoder;

    @Autowired
    public DynamicReadQueryService(DataSource dataSource) {
        this(dataSource, new OpaqueSampleKeyEncoder());
    }

    public DynamicReadQueryService(DataSource dataSource,
                                   OpaqueSampleKeyEncoder opaqueSampleKeyEncoder) {
        if (dataSource == null) {
            throw new IllegalArgumentException("dataSource is required");
        }
        if (opaqueSampleKeyEncoder == null) {
            throw new IllegalArgumentException("opaqueSampleKeyEncoder is required");
        }
        this.dataSource = dataSource;
        this.opaqueSampleKeyEncoder = opaqueSampleKeyEncoder;
    }

    public ReadModelResult<DynamicQueryReadModel> execute(String sql, int maxRows) {
        DynamicReadQueryPolicy.Validation validation = DynamicReadQueryPolicy.validate(sql);
        if (!validation.isValid()) {
            throw new IllegalArgumentException("dynamic read query rejected");
        }
        if (maxRows < 1 || maxRows > DynamicReadQueryPolicy.MAX_QUERY_LIMIT) {
            throw new IllegalArgumentException("dynamic read query row bound rejected");
        }

        return executeCompiled(sql, java.util.Collections.emptyList(), maxRows, null, false);
    }

    public ReadModelResult<DynamicQueryReadModel> execute(QueryPlan plan, int maxRows) {
        return execute(plan, maxRows, null, false);
    }

    /** Execute a typed plan with an optional Runtime-owned opaque sample key. */
    public ReadModelResult<DynamicQueryReadModel> execute(QueryPlan plan, int maxRows,
                                                           String runId,
                                                           boolean includeOpaqueSampleKey) {
        if (plan == null || maxRows < 1 || maxRows > DynamicReadQueryPolicy.MAX_QUERY_LIMIT
                || maxRows != plan.getLimit()) {
            throw new IllegalArgumentException("query plan row bound rejected");
        }
        if (includeOpaqueSampleKey && (runId == null || runId.trim().isEmpty())) {
            throw new IllegalArgumentException("runId is required for an opaque sample key");
        }
        QueryPlanCompiler.CompiledQuery compiled = new QueryPlanCompiler().compile(
                plan, includeOpaqueSampleKey);
        return executeCompiled(compiled.getSql(), compiled.getParameters(), maxRows,
                includeOpaqueSampleKey ? runId : null,
                compiled.includesOpaqueSampleKey());
    }

    private ReadModelResult<DynamicQueryReadModel> executeCompiled(
            String sql, List<QueryPlanCompiler.QueryParameter> parameters, int maxRows,
            String runId, boolean includesOpaqueSampleKey) {
        Instant generatedAt = Instant.now();
        try (Connection connection = dataSource.getConnection()) {
            String catalog = connection.getCatalog();
            if (!EXPECTED_CATALOG.equalsIgnoreCase(catalog)) {
                throw new IllegalArgumentException("dynamic read query catalog rejected");
            }
            connection.setReadOnly(true);
            try (PreparedStatement statement = connection.prepareStatement(sql)) {
                for (int index = 0; index < parameters.size(); index++) {
                    statement.setObject(index + 1, parameters.get(index).getValue());
                }
                statement.setMaxRows(maxRows);
                statement.setQueryTimeout(QUERY_TIMEOUT_SECONDS);
                try (ResultSet resultSet = statement.executeQuery()) {
                    DynamicQueryReadModel model = readRows(resultSet, maxRows, runId,
                            includesOpaqueSampleKey);
                    return new ReadModelResult<>(model, new ReadReceipt(
                            SOURCE, generatedAt, sha256(sql),
                            (long) model.getRows().size()));
                }
            }
        } catch (IllegalArgumentException exception) {
            throw exception;
        } catch (SQLException exception) {
            throw new IllegalStateException("dynamic read query execution failed");
        }
    }

    private DynamicQueryReadModel readRows(ResultSet resultSet, int maxRows,
                                           String runId, boolean includesOpaqueSampleKey)
            throws SQLException {
        ResultSetMetaData metadata = resultSet.getMetaData();
        List<String> columns = new ArrayList<>();
        int opaqueSourceIndex = -1;
        for (int index = 1; index <= metadata.getColumnCount(); index++) {
            String label = metadata.getColumnLabel(index);
            if (QueryPlanCompiler.OPAQUE_SAMPLE_KEY_SOURCE_ALIAS.equals(label)) {
                opaqueSourceIndex = index;
            } else {
                columns.add(label);
            }
        }
        if (includesOpaqueSampleKey && opaqueSourceIndex < 0) {
            throw new IllegalStateException("opaque sample key source column is missing");
        }
        if (includesOpaqueSampleKey) {
            columns.add(QueryPlanCompiler.OPAQUE_SAMPLE_KEY_RESULT_ALIAS);
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        while (resultSet.next() && rows.size() < maxRows) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (int index = 1; index <= metadata.getColumnCount(); index++) {
                if (index == opaqueSourceIndex) {
                    continue;
                }
                String label = metadata.getColumnLabel(index);
                row.put(label, safeScalar(resultSet.getObject(index)));
            }
            if (includesOpaqueSampleKey) {
                row.put(QueryPlanCompiler.OPAQUE_SAMPLE_KEY_RESULT_ALIAS,
                        opaqueSampleKeyEncoder.encode(runId, resultSet.getObject(opaqueSourceIndex)));
            }
            rows.add(row);
        }
        return new DynamicQueryReadModel(columns, rows);
    }

    private Object safeScalar(Object value) {
        if (value == null || value instanceof Number || value instanceof Boolean
                || value instanceof java.sql.Date || value instanceof java.sql.Timestamp
                || value instanceof java.util.Date) {
            return value;
        }
        if (value instanceof byte[]) {
            return "[binary-value-omitted]";
        }
        String text = String.valueOf(value);
        return text.length() > MAX_CELL_TEXT_LENGTH
                ? text.substring(0, MAX_CELL_TEXT_LENGTH) : text;
    }

    private String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder result = new StringBuilder("sha256:");
            for (byte item : digest) {
                result.append(String.format("%02x", item & 0xff));
            }
            return result.toString();
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("hash unavailable");
        }
    }
}
