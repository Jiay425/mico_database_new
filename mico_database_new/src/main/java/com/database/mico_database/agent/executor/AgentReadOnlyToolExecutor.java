package com.database.mico_database.agent.executor;

import com.database.mico_database.agent.contract.AgentContractConstants;
import com.database.mico_database.agent.contract.AgentToolError;
import com.database.mico_database.agent.contract.AgentToolName;
import com.database.mico_database.agent.contract.AgentToolRequest;
import com.database.mico_database.agent.contract.AgentToolRequestValidator;
import com.database.mico_database.agent.contract.AgentToolResponse;
import com.database.mico_database.agent.contract.AgentToolValidationResult;
import com.database.mico_database.agent.contract.DataSnapshot;
import com.database.mico_database.agent.contract.DynamicReadQueryPolicy;
import com.database.mico_database.agent.contract.QualitySummary;
import com.database.mico_database.agent.contract.QueryPlan;
import com.database.mico_database.agent.readmodel.DynamicQueryReadModel;
import com.database.mico_database.agent.readmodel.ReadModelResult;
import com.database.mico_database.agent.readmodel.ReadReceipt;
import com.database.mico_database.agent.readmodel.SchemaSemanticCatalogReadModel;
import com.database.mico_database.agent.readmodel.service.DynamicReadQueryService;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Java executable boundary: model-proposed typed read plans are compiled and
 * executed by Java. Schema description is metadata-only; no fixed cohort or
 * differential workflow is dispatched here.
 */
public final class AgentReadOnlyToolExecutor implements AgentToolExecutorPort {

    private final AgentToolRequestValidator validator;
    private final DynamicReadQueryService dynamicReadQueryService;

    public AgentReadOnlyToolExecutor(AgentToolRequestValidator validator,
                                     DynamicReadQueryService dynamicReadQueryService) {
        if (validator == null || dynamicReadQueryService == null) {
            throw new IllegalArgumentException("validator and dynamic read service are required");
        }
        this.validator = validator;
        this.dynamicReadQueryService = dynamicReadQueryService;
    }

    @Override
    public AgentToolResponse<?> execute(AgentToolRequest request) {
        AgentToolValidationResult validation = validator.validate(request);
        if (!validation.isValid()) {
            return rejected(request, validation.getErrors().isEmpty()
                    ? AgentToolError.of("INVALID_REQUEST", "Tool request was rejected", null, false,
                    "The request did not satisfy the closed Java tool contract")
                    : validation.getErrors().get(0));
        }
        if (AgentToolName.DESCRIBE_READ_SCHEMA.getWireName().equals(request.getToolName())) {
            SchemaSemanticCatalogReadModel catalog = SchemaSemanticCatalogReadModel.current();
            return AgentToolResponse.metadataCompleted(
                    request.getToolCallId(), request.getRunId(),
                    SchemaSemanticCatalogReadModel.SOURCE,
                    (long) catalog.getEntityCount(), catalog);
        }
        if (!AgentToolName.EXECUTE_READ_QUERY.getWireName().equals(request.getToolName())) {
            return AgentToolResponse.notImplemented(request.getToolCallId(), request.getRunId());
        }
        try {
            QueryPlan queryPlan = (QueryPlan) request.getArguments().get("queryPlan");
            String sql = (String) request.getArguments().get("sql");
            boolean includeOpaqueSampleKey = Boolean.TRUE.equals(
                    request.getArguments().get("includeAnalysisSampleKey"));
            if (includeOpaqueSampleKey && queryPlan == null) {
                throw new IllegalArgumentException("opaque sample key requires a typed QueryPlan");
            }
            Object rawLimit = request.getArguments().get("limit");
            int limit = rawLimit == null
                    ? queryPlan == null ? DynamicReadQueryPolicy.MAX_QUERY_LIMIT : queryPlan.getLimit()
                    : ((Number) rawLimit).intValue();
            ReadModelResult<DynamicQueryReadModel> result = queryPlan == null
                    ? dynamicReadQueryService.execute(sql, limit)
                    : dynamicReadQueryService.execute(queryPlan, limit, request.getRunId(),
                            includeOpaqueSampleKey);
            return completed(request, result);
        } catch (IllegalArgumentException exception) {
            return AgentToolResponse.rejected(request.getToolCallId(), request.getRunId(),
                    AgentToolError.of("READ_MODEL_ARGUMENT_REJECTED",
                            "The read-only operation rejected the supplied query boundary",
                            "arguments", false,
                            "Java validated the dynamic read request before execution"));
        } catch (RuntimeException exception) {
            return AgentToolResponse.failed(request.getToolCallId(), request.getRunId(),
                    AgentToolError.of("READ_MODEL_EXECUTION_FAILED",
                            "The read-only operation could not be completed", null, true,
                            "Internal read-model details are withheld from the tool response"));
        }
    }

    private AgentToolResponse<DynamicQueryReadModel> completed(
            AgentToolRequest request, ReadModelResult<DynamicQueryReadModel> result) {
        if (result == null || result.getReadReceipt() == null) {
            throw new IllegalArgumentException("read receipt is required");
        }
        ReadReceipt receipt = result.getReadReceipt();
        if (!ReadReceipt.TRANSIENT_PERSISTENCE.equals(receipt.getSnapshotPersistence())) {
            throw new IllegalArgumentException("only transient evidence is supported");
        }
        DataSnapshot snapshot = new DataSnapshot();
        snapshot.setDataSnapshotId("transient-" + UUID.randomUUID());
        snapshot.setDataSource(receipt.getSource());
        snapshot.setImportBatch(null);
        snapshot.setDiseaseMappingVersion(null);
        snapshot.setTaxonomyVersion(null);
        snapshot.setFeatureVersion(null);
        snapshot.setSourceBatch(null);
        snapshot.setCohortCondition("dynamic_read_query");
        snapshot.setQueryHash(receipt.getQueryHash());
        snapshot.setRowCount(receipt.getRowCount());
        snapshot.setGeneratedAt(receipt.getGeneratedAt());
        snapshot.setSnapshotPersistence(DataSnapshot.SNAPSHOT_PERSISTENCE_TRANSIENT);

        QualitySummary quality = quality(result.getData());
        AgentToolResponse<DynamicQueryReadModel> response = AgentToolResponse.completed(
                request.getToolCallId(), request.getRunId(), receipt.getSource(),
                receipt.getRowCount(), snapshot, quality, result.getData());
        response.setGeneratedAt(receipt.getGeneratedAt());
        return response;
    }

    private QualitySummary quality(DynamicQueryReadModel data) {
        Map<String, Long> missing = new LinkedHashMap<>();
        if (data == null) {
            missing.put("rows", 1L);
        } else {
            if (data.getColumns() == null || data.getColumns().isEmpty()) {
                missing.put("columns", 1L);
            }
            if (data.getRows() == null || data.getRows().isEmpty()) {
                missing.put("rows", 1L);
            }
        }
        QualitySummary quality = new QualitySummary();
        quality.setSubjectLinkStatus(AgentContractConstants.SUBJECT_LINK_STATUS_UNVERIFIED);
        quality.setMissingCounts(missing);
        List<String> warnings = new ArrayList<>();
        warnings.add("A typed QueryPlan was proposed by the model; Java compiled and executed the SQL");
        warnings.add("dynamic query output is bounded and transient; it is not a replayable snapshot");
        warnings.add("row counts do not establish independent Subject or patient counts");
        warnings.add("Python-generated analysis is bounded by the Runtime sandbox");
        quality.setWarnings(warnings);
        return quality;
    }

    private AgentToolResponse<Object> rejected(AgentToolRequest request, AgentToolError error) {
        return AgentToolResponse.rejected(request == null ? null : request.getToolCallId(),
                request == null ? null : request.getRunId(), error);
    }
}
