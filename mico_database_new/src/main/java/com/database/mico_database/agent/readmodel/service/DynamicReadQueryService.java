package com.database.mico_database.agent.readmodel.service;

import com.database.mico_database.agent.contract.DynamicReadQueryPolicy;
import com.database.mico_database.agent.readmodel.DynamicQueryReadModel;
import com.database.mico_database.agent.readmodel.ReadModelResult;
import com.database.mico_database.agent.readmodel.ReadReceipt;
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
 * Java-only execution boundary for model-proposed read SQL.  Python never gets
 * a DataSource and this service never exposes an arbitrary SQL interface to a
 * browser or to a persistence layer.
 */
@Service
public class DynamicReadQueryService {

    public static final String SOURCE = "java_agent_read_model";
    private static final String EXPECTED_CATALOG = "patient_data_manager";
    private static final int QUERY_TIMEOUT_SECONDS = 10;
    private static final int MAX_CELL_TEXT_LENGTH = 4096;

    private final DataSource dataSource;

    public DynamicReadQueryService(DataSource dataSource) {
        if (dataSource == null) {
            throw new IllegalArgumentException("dataSource is required");
        }
        this.dataSource = dataSource;
    }

    public ReadModelResult<DynamicQueryReadModel> execute(String sql, int maxRows) {
        DynamicReadQueryPolicy.Validation validation = DynamicReadQueryPolicy.validate(sql);
        if (!validation.isValid()) {
            throw new IllegalArgumentException("dynamic read query rejected");
        }
        if (maxRows < 1 || maxRows > DynamicReadQueryPolicy.MAX_QUERY_LIMIT) {
            throw new IllegalArgumentException("dynamic read query row bound rejected");
        }

        Instant generatedAt = Instant.now();
        try (Connection connection = dataSource.getConnection()) {
            String catalog = connection.getCatalog();
            if (!EXPECTED_CATALOG.equalsIgnoreCase(catalog)) {
                throw new IllegalArgumentException("dynamic read query catalog rejected");
            }
            connection.setReadOnly(true);
            try (PreparedStatement statement = connection.prepareStatement(sql)) {
                statement.setMaxRows(maxRows);
                statement.setQueryTimeout(QUERY_TIMEOUT_SECONDS);
                try (ResultSet resultSet = statement.executeQuery()) {
                    DynamicQueryReadModel model = readRows(resultSet, maxRows);
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

    private DynamicQueryReadModel readRows(ResultSet resultSet, int maxRows) throws SQLException {
        ResultSetMetaData metadata = resultSet.getMetaData();
        List<String> columns = new ArrayList<>();
        for (int index = 1; index <= metadata.getColumnCount(); index++) {
            columns.add(metadata.getColumnLabel(index));
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        while (resultSet.next() && rows.size() < maxRows) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (int index = 1; index <= metadata.getColumnCount(); index++) {
                row.put(columns.get(index - 1), safeScalar(resultSet.getObject(index)));
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
