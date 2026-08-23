package com.database.mico_database.agent.readmodel;

import com.database.mico_database.agent.readmodel.service.DynamicReadQueryService;
import org.junit.jupiter.api.Test;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
        verify(statement).setQueryTimeout(10);
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
}
