package com.database.mico_database.agent.readmodel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Bounded generic projection for a Java-validated, read-only analysis query. */
public final class DynamicQueryReadModel {

    private final List<String> columns;
    private final List<Map<String, Object>> rows;

    public DynamicQueryReadModel(List<String> columns, List<Map<String, Object>> rows) {
        this.columns = Collections.unmodifiableList(new ArrayList<>(columns));
        List<Map<String, Object>> copiedRows = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            copiedRows.add(Collections.unmodifiableMap(new LinkedHashMap<>(row)));
        }
        this.rows = Collections.unmodifiableList(copiedRows);
    }

    public List<String> getColumns() {
        return columns;
    }

    public List<Map<String, Object>> getRows() {
        return rows;
    }
}
