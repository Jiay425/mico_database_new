package com.database.mico_database.agent.readmodel;

import com.database.mico_database.agent.readmodel.service.DynamicReadQueryService;
import com.database.mico_database.agent.contract.QueryPlan;
import com.database.mico_database.agent.contract.QueryPlanCompiler;
import org.junit.jupiter.api.Test;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DynamicReadQueryServiceTest {

    @Test
    void executesOnlyAfterPolicyAndAppliesReadOnlyBoundaries() throws Exception {
        DataSource dataSource = mock(DataSource.class);
        Connection connection = mock(Connection.class);
        when(connection.getCatalog()).thenReturn("patient_data_manager");
        PreparedStatement statement = mock(PreparedStatement.class);
        ResultSet resultSet = mock(ResultSet.class);
        ResultSetMetaData metadata = mock(ResultSetMetaData.class);
        when(dataSource.getConnection()).thenReturn(connection);
        when(connection.getCatalog()).thenReturn("patient_data_manager");
        when(connection.prepareStatement("SELECT p.disease, COUNT(*) AS n FROM patients p GROUP BY p.disease LIMIT 10"))
                .thenReturn(statement);
        when(statement.executeQuery()).thenReturn(resultSet);
        when(resultSet.getMetaData()).thenReturn(metadata);
        when(metadata.getColumnCount()).thenReturn(1);
        when(metadata.getColumnLabel(1)).thenReturn("n");
        when(resultSet.next()).thenReturn(true, false);
        when(resultSet.getObject(1)).thenReturn(3L);

        DynamicReadQueryService service = new DynamicReadQueryService(dataSource);
        ReadModelResult<DynamicQueryReadModel> result = service.execute(
                "SELECT p.disease, COUNT(*) AS n FROM patients p GROUP BY p.disease LIMIT 10", 10);

        assertEquals(1, result.getData().getRows().size());
        assertEquals(3L, result.getData().getRows().get(0).get("n"));
        verify(connection).setReadOnly(true);
        verify(statement).setMaxRows(10);
        verify(statement).setQueryTimeout(180);
        verify(statement).executeQuery();
    }

    @Test
    void rejectedQueryDoesNotOpenAConnection() throws Exception {
        DataSource dataSource = mock(DataSource.class);
        DynamicReadQueryService service = new DynamicReadQueryService(dataSource);

        assertThrows(IllegalArgumentException.class,
                () -> service.execute("DELETE FROM patients LIMIT 1", 1));
        org.mockito.Mockito.verifyNoInteractions(dataSource);
    }

    @Test
    void returnsRunScopedOpaqueSampleKeyWithoutRawSourceIdentifier() throws Exception {
        DataSource dataSource = mock(DataSource.class);
        Connection connection = mock(Connection.class);
        PreparedStatement statement = mock(PreparedStatement.class);
        ResultSet resultSet = mock(ResultSet.class);
        ResultSetMetaData metadata = mock(ResultSetMetaData.class);
        when(dataSource.getConnection()).thenReturn(connection);
        when(connection.getCatalog()).thenReturn("patient_data_manager");

        QueryPlan plan = new QueryPlan();
        plan.setRootEntity("sample");
        plan.setRelationPath(java.util.Collections.singletonList("sample_to_abundance"));
        plan.setSelectFields(java.util.Arrays.asList(
                "sample.disease", "abundance.feature", "abundance.value"));
        plan.setLimit(5);
        QueryPlanCompiler.CompiledQuery compiled = new QueryPlanCompiler().compile(plan, true);
        when(connection.prepareStatement(compiled.getSql())).thenReturn(statement);
        when(statement.executeQuery()).thenReturn(resultSet);
        when(resultSet.getMetaData()).thenReturn(metadata);
        when(metadata.getColumnCount()).thenReturn(4);
        when(metadata.getColumnLabel(1)).thenReturn("a_sample_disease");
        when(metadata.getColumnLabel(2)).thenReturn("a_abundance_feature");
        when(metadata.getColumnLabel(3)).thenReturn("a_abundance_value");
        when(metadata.getColumnLabel(4)).thenReturn(QueryPlanCompiler.OPAQUE_SAMPLE_KEY_SOURCE_ALIAS);
        when(resultSet.next()).thenReturn(true, false);
        when(resultSet.getObject(1)).thenReturn("T2D");
        when(resultSet.getObject(2)).thenReturn("feature-a");
        when(resultSet.getObject(3)).thenReturn(0.5d);
        when(resultSet.getObject(4)).thenReturn("raw-sample-1");

        DynamicReadQueryService service = new DynamicReadQueryService(dataSource);
        DynamicQueryReadModel model = service.execute(plan, 5, "run-opaque-test", true)
                .getData();

        assertEquals(4, model.getColumns().size());
        assertFalse(model.getColumns().contains(QueryPlanCompiler.OPAQUE_SAMPLE_KEY_SOURCE_ALIAS));
        assertTrue(model.getColumns().contains(QueryPlanCompiler.OPAQUE_SAMPLE_KEY_RESULT_ALIAS));
        assertFalse(model.getRows().get(0).containsValue("raw-sample-1"));
        assertTrue(String.valueOf(model.getRows().get(0)
                .get(QueryPlanCompiler.OPAQUE_SAMPLE_KEY_RESULT_ALIAS)).matches("s_[0-9a-f]{48}"));
    }
}
