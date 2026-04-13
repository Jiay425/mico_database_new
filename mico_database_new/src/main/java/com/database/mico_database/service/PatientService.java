package com.database.mico_database.service;

import com.database.mico_database.mapper.PatientMapper;
import com.database.mico_database.model.CytokineData;
import com.database.mico_database.model.MicrobeAbundance;
import com.database.mico_database.model.Patient;
import com.database.mico_database.model.PredictionRequest;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.pagehelper.PageHelper;
import com.github.pagehelper.PageInfo;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
@Slf4j
public class PatientService {

    private static final String SEARCH_CACHE_VERSION = "v2-standard";

    @Autowired
    private PatientMapper patientMapper;

    @Autowired
    private StringRedisTemplate redisTemplate;

    @Autowired
    private ObjectMapper objectMapper;

    private List<MicrobeAbundance> loadStandardMicrobes(String patientId) {
        List<MicrobeAbundance> microbes = patientMapper.findStandardMicrobesByPatientId(patientId);
        if (microbes == null || microbes.isEmpty()) {
            log.warn("No standard abundance records found for patient_id={}", patientId);
            return Collections.emptyList();
        }
        return microbes;
    }

    private Patient getFullPatientData(String patientId) {
        Patient patient = patientMapper.findById(patientId);
        if (patient != null) {
            patient.setMicrobialAbundanceData(loadStandardMicrobes(patientId));
            patient.setCytokineData(patientMapper.findCytokinesByPatientId(patientId));
        }
        return patient;
    }

    public Patient getPatientDetail(String patientId) {
        return getFullPatientData(patientId);
    }

    public PageInfo<Patient> searchPatientsLite(String queryType, String queryValue, int pageNum, int pageSize) {
        PageHelper.startPage(pageNum, pageSize);
        List<Patient> list = patientMapper.searchPatients(queryType, queryValue);
        return new PageInfo<>(list);
    }

    public List<Map<String, Object>> listPatientSamples(String patientId) {
        return patientMapper.findStandardSampleSummariesByPatientId(patientId).stream()
                .map(item -> {
                    Map<String, Object> sample = new LinkedHashMap<>();
                    sample.put("patientId", String.valueOf(item.get("patient_id")));
                    sample.put("sampleId", item.get("sample_id"));
                    sample.put("sampleDate", item.get("sample_date"));
                    sample.put("featureCount", item.get("feature_count"));
                    sample.put("abundanceTotal", item.get("abundance_total"));
                    return sample;
                })
                .collect(Collectors.toList());
    }

    public Map<String, Object> getPatientSampleDetail(String patientId, String sampleId) {
        List<MicrobeAbundance> microbes = patientMapper.findStandardMicrobesByPatientIdAndSampleId(patientId, sampleId);
        if (microbes == null || microbes.isEmpty()) {
            return null;
        }

        Patient patient = patientMapper.findById(patientId);
        Map<String, Object> sample = new LinkedHashMap<>();
        sample.put("patientId", patientId);
        sample.put("sampleId", sampleId);
        sample.put("sampleDate", microbes.get(0).getSampleDate());
        sample.put("featureCount", microbes.size());
        sample.put("unit", microbes.get(0).getUnit());
        sample.put("patient", patient);
        return sample;
    }

    public List<MicrobeAbundance> getTopSampleFeatures(String patientId, String sampleId, int limit) {
        int safeLimit = limit <= 0 ? 20 : Math.min(limit, 100);
        return patientMapper.findTopStandardMicrobesByPatientIdAndSampleId(patientId, sampleId, safeLimit);
    }

    public PredictionRequest buildSamplePredictionRequest(String patientId, String sampleId) {
        List<MicrobeAbundance> microbes = patientMapper.findStandardMicrobesByPatientIdAndSampleId(patientId, sampleId);
        if (microbes == null || microbes.isEmpty()) {
            return null;
        }

        PredictionRequest request = new PredictionRequest();
        request.setSampleId(sampleId);
        request.setSource("standard_microbe_abundance");
        request.setRawData("");

        List<PredictionRequest.FeatureItem> features = microbes.stream()
                .map(microbe -> {
                    PredictionRequest.FeatureItem item = new PredictionRequest.FeatureItem();
                    item.setName(microbe.getMicrobeName());
                    item.setValue(microbe.getAbundanceValue());
                    item.setMeta(microbe.getUnit());
                    return item;
                })
                .collect(Collectors.toList());
        request.setFeatures(features);
        return request;
    }

    public Map<String, Object> buildHealthyReferenceComparison(String patientId, String sampleId, int limit) {
        Map<String, Object> sample = getPatientSampleDetail(patientId, sampleId);
        if (sample == null) {
            return null;
        }

        Patient patient = patientMapper.findById(patientId);
        String bodySite = patient == null ? null : patient.getBodySite();
        int safeLimit = limit <= 0 ? 8 : Math.min(limit, 30);

        Map<String, Object> scopedStats = patientMapper.countHealthyReferenceStats(bodySite, hasText(bodySite));
        long scopedSampleCount = safeLong(scopedStats == null ? null : scopedStats.get("sample_count"));
        boolean useBodySite = hasText(bodySite) && scopedSampleCount >= 20;

        Map<String, Object> referenceStats = useBodySite
                ? scopedStats
                : patientMapper.countHealthyReferenceStats(null, false);

        List<MicrobeAbundance> currentAllFeatures = patientMapper.findStandardMicrobesByPatientIdAndSampleId(patientId, sampleId);
        if (currentAllFeatures == null || currentAllFeatures.isEmpty()) {
            return null;
        }

        List<String> currentFeatureNames = currentAllFeatures.stream()
                .map(MicrobeAbundance::getMicrobeName)
                .filter(this::hasText)
                .collect(Collectors.toList());

        List<Map<String, Object>> healthyStatsRows = patientMapper.findHealthyMicrobeStats(bodySite, useBodySite);
        Map<String, Map<String, Object>> healthyStatsMap = healthyStatsRows.stream()
                .collect(Collectors.toMap(
                        item -> String.valueOf(item.get("microbe_name")),
                        item -> item,
                        (left, right) -> left,
                        LinkedHashMap::new
                ));

        double maxCurrentAbundance = currentAllFeatures.stream()
                .mapToDouble(feature -> feature.getAbundanceValue() == null ? 0D : feature.getAbundanceValue())
                .max()
                .orElse(0D);
        double epsilon = 1e-9D;

        List<Map<String, Object>> rankedComparisons = new ArrayList<>();
        for (MicrobeAbundance feature : currentAllFeatures) {
            String microbeName = feature.getMicrobeName();
            if (!hasText(microbeName)) {
                continue;
            }
            double currentAbundance = feature.getAbundanceValue() == null ? 0D : feature.getAbundanceValue();
            if (currentAbundance <= 0D) {
                continue;
            }

            Map<String, Object> healthyRow = healthyStatsMap.getOrDefault(microbeName, Collections.emptyMap());
            double healthyAverage = safeDouble(healthyRow.get("avg_abundance"));
            double healthyStd = safeDouble(healthyRow.get("std_abundance"));
            long sampleHits = safeLong(healthyRow.get("sample_hits"));
            double delta = currentAbundance - healthyAverage;
            double ratio = healthyAverage > 0D ? currentAbundance / healthyAverage : (currentAbundance > 0D ? Double.POSITIVE_INFINITY : 1D);
            double log2FoldChange = log2((currentAbundance + epsilon) / (healthyAverage + epsilon));
            double zScore = healthyStd > 0D ? (currentAbundance - healthyAverage) / healthyStd : 0D;
            double percentileInHealthy = percentileFromZScore(zScore);
            double percentileDeviation = Math.abs(percentileInHealthy - 50D) / 50D;
            double abundanceScore = maxCurrentAbundance > 0D ? currentAbundance / maxCurrentAbundance : 0D;
            double log2Score = Math.min(Math.abs(log2FoldChange) / 4D, 1D);
            double combinedScore = 0.40D * abundanceScore + 0.45D * log2Score + 0.15D * percentileDeviation;

            Map<String, Object> item = new LinkedHashMap<>();
            item.put("microbeName", microbeName);
            item.put("currentAbundance", currentAbundance);
            item.put("healthyAverageAbundance", healthyAverage);
            item.put("healthyStdAbundance", healthyStd);
            item.put("delta", delta);
            item.put("ratioToHealthy", ratio);
            item.put("log2FoldChange", log2FoldChange);
            item.put("zScore", zScore);
            item.put("percentileInHealthy", percentileInHealthy);
            item.put("percentileDeviation", percentileDeviation);
            item.put("sampleHitCount", sampleHits);
            item.put("combinedScore", combinedScore);
            item.put("direction", classifyDirectionByLog2(log2FoldChange, currentAbundance, healthyAverage));
            rankedComparisons.add(item);
        }

        rankedComparisons.sort(Comparator
                .comparingDouble((Map<String, Object> item) -> safeDouble(item.get("combinedScore"))).reversed()
                .thenComparing((Map<String, Object> item) -> Math.abs(safeDouble(item.get("log2FoldChange"))), Comparator.reverseOrder())
                .thenComparing((Map<String, Object> item) -> safeDouble(item.get("currentAbundance")), Comparator.reverseOrder()));

        List<Map<String, Object>> featureComparisons = rankedComparisons.stream()
                .limit(safeLimit)
                .collect(Collectors.toList());

        int spatialFeatureCount = Math.min(80, Math.max(40, safeLimit * 8));
        List<String> spatialFeatureNames = rankedComparisons.stream()
                .limit(spatialFeatureCount)
                .map(item -> String.valueOf(item.get("microbeName")))
                .collect(Collectors.toList());
        Map<String, Object> spatialDeviation = computeSpatialDeviation(
                currentAllFeatures,
                healthyStatsMap,
                patientMapper.findHealthyFeatureMatrix(bodySite, useBodySite, spatialFeatureNames),
                spatialFeatureNames
        );

        Set<String> currentDominantNames = currentAllFeatures.stream()
                .limit(120)
                .map(MicrobeAbundance::getMicrobeName)
                .filter(this::hasText)
                .collect(Collectors.toCollection(LinkedHashSet::new));

        List<Map<String, Object>> healthySignature = patientMapper.findHealthyTopMicrobes(bodySite, useBodySite, safeLimit + 4).stream()
                .filter(item -> !currentDominantNames.contains(String.valueOf(item.get("microbe_name"))))
                .limit(3)
                .collect(Collectors.toList());

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("patientId", patientId);
        response.put("sampleId", sampleId);
        response.put("bodySite", bodySite);
        response.put("referenceScope", useBodySite ? "same_body_site_healthy" : "all_healthy");
        response.put("healthyPatientCount", safeLong(referenceStats == null ? null : referenceStats.get("patient_count")));
        response.put("healthySampleCount", safeLong(referenceStats == null ? null : referenceStats.get("sample_count")));
        response.put("rankingStrategy", "combined_abundance_log2fc_percentile");
        response.put("comparedFeatureCount", rankedComparisons.size());
        response.put("featureComparisons", featureComparisons);
        response.put("spatialDeviation", spatialDeviation);
        response.put("healthySignatureFeatures", healthySignature);
        return response;
    }

    public Map<String, Object> search(String queryType, String queryValue) {
        Map<String, Object> results = new HashMap<>();

        switch (queryType) {
            case "id":
                Patient patient = getFullPatientData(queryValue);
                if (patient == null) {
                    results.put("error", "Patient not found for id: " + queryValue);
                }
                results.put("patient_data", patient);
                break;

            case "name":
                List<String> patientIdsByName = patientMapper.findIdsByName(queryValue);
                if (patientIdsByName.isEmpty()) {
                    results.put("error", "Patient not found for name: " + queryValue);
                } else if (patientIdsByName.size() == 1) {
                    results.put("patient_data", getFullPatientData(patientIdsByName.get(0)));
                } else {
                    List<Patient> fullPatientList = patientIdsByName.stream()
                            .map(this::getFullPatientData)
                            .collect(Collectors.toList());
                    results.put("patients_by_disease", fullPatientList);
                }
                break;

            case "disease":
                List<String> patientIdsByDisease = patientMapper.findIdsByDisease(queryValue);
                if (patientIdsByDisease.isEmpty()) {
                    results.put("error", "Patient not found for disease: " + queryValue);
                } else {
                    List<Patient> fullPatientList = patientIdsByDisease.stream()
                            .map(this::getFullPatientData)
                            .collect(Collectors.toList());
                    results.put("patients_by_disease", fullPatientList);
                }
                break;

            default:
                results.put("error", "Invalid query type");
                break;
        }

        return results;
    }

    public PageInfo<Patient> searchPatients(String queryType, String queryValue, int pageNum, int pageSize) {
        String cacheKey = "search:" + SEARCH_CACHE_VERSION + ":" + queryType + ":" + queryValue + ":" + pageNum + ":" + pageSize;

        try {
            String json = redisTemplate.opsForValue().get(cacheKey);
            if (json != null) {
                log.info("Redis cache hit: {}", cacheKey);
                return objectMapper.readValue(json, new TypeReference<PageInfo<Patient>>() {});
            }
        } catch (Exception e) {
            log.error("Redis read failed, fallback to database: {}", e.getMessage());
        }

        PageHelper.startPage(pageNum, pageSize);
        List<Patient> list = patientMapper.searchPatients(queryType, queryValue);

        for (Patient patient : list) {
            String linkKey = patient.getPatientId();
            if (linkKey != null) {
                patient.setMicrobialAbundanceData(loadStandardMicrobes(linkKey));
                patient.setCytokineData(patientMapper.findCytokinesByPatientId(linkKey));
            }
        }

        PageInfo<Patient> pageInfo = new PageInfo<>(list);

        try {
            String jsonResult = objectMapper.writeValueAsString(pageInfo);
            redisTemplate.opsForValue().set(cacheKey, jsonResult, 10, TimeUnit.MINUTES);
            log.info("Redis cache write: {}", cacheKey);
        } catch (Exception e) {
            log.error("Redis write failed: {}", e.getMessage());
        }

        return pageInfo;
    }

    private long safeLong(Object value) {
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        if (value == null) {
            return 0L;
        }
        try {
            return Long.parseLong(String.valueOf(value));
        } catch (NumberFormatException ex) {
            return 0L;
        }
    }

    private double safeDouble(Object value) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        if (value == null) {
            return 0D;
        }
        try {
            return Double.parseDouble(String.valueOf(value));
        } catch (NumberFormatException ex) {
            return 0D;
        }
    }

    private Map<String, Object> computeSpatialDeviation(List<MicrobeAbundance> currentFeatures,
                                                        Map<String, Map<String, Object>> healthyStatsMap,
                                                        List<Map<String, Object>> healthyMatrixRows,
                                                        List<String> featureNames) {
        Map<String, Object> result = new LinkedHashMap<>();
        if (featureNames == null || featureNames.isEmpty()) {
            result.put("featureSpaceSize", 0);
            result.put("healthySampleCountUsed", 0);
            result.put("currentDistance", 0D);
            result.put("healthyDistanceMean", 0D);
            result.put("healthyDistanceStd", 0D);
            result.put("distanceZScore", 0D);
            result.put("distancePercentile", 50D);
            result.put("deviationLevel", "unknown");
            return result;
        }

        Map<String, Double> currentVector = new LinkedHashMap<>();
        for (MicrobeAbundance feature : currentFeatures) {
            if (feature == null || !hasText(feature.getMicrobeName())) {
                continue;
            }
            if (!featureNames.contains(feature.getMicrobeName())) {
                continue;
            }
            currentVector.put(feature.getMicrobeName(), feature.getAbundanceValue() == null ? 0D : feature.getAbundanceValue());
        }

        Map<String, Double> centroid = new LinkedHashMap<>();
        for (String featureName : featureNames) {
            Map<String, Object> row = healthyStatsMap.getOrDefault(featureName, Collections.emptyMap());
            centroid.put(featureName, safeDouble(row.get("avg_abundance")));
        }

        Map<String, Map<String, Double>> healthySampleVectors = new LinkedHashMap<>();
        for (Map<String, Object> row : healthyMatrixRows) {
            String sampleKey = String.valueOf(row.get("sample_key"));
            String microbeName = String.valueOf(row.get("microbe_name"));
            if (!hasText(sampleKey) || !hasText(microbeName)) {
                continue;
            }
            double abundance = safeDouble(row.get("abundance_value"));
            healthySampleVectors
                    .computeIfAbsent(sampleKey, ignored -> new LinkedHashMap<>())
                    .put(microbeName, abundance);
        }

        double currentDistance = euclideanDistance(currentVector, centroid, featureNames);
        List<Double> healthyDistances = healthySampleVectors.values().stream()
                .map(vector -> euclideanDistance(vector, centroid, featureNames))
                .collect(Collectors.toList());

        double mean = mean(healthyDistances);
        double std = std(healthyDistances, mean);
        double zScore = std > 0D ? (currentDistance - mean) / std : 0D;
        double percentile = percentile(healthyDistances, currentDistance);

        result.put("featureSpaceSize", featureNames.size());
        result.put("healthySampleCountUsed", healthyDistances.size());
        result.put("currentDistance", currentDistance);
        result.put("healthyDistanceMean", mean);
        result.put("healthyDistanceStd", std);
        result.put("distanceZScore", zScore);
        result.put("distancePercentile", percentile);
        result.put("deviationLevel", classifySpatialDeviation(zScore, percentile));
        return result;
    }

    private double euclideanDistance(Map<String, Double> vector,
                                     Map<String, Double> centroid,
                                     List<String> featureNames) {
        double sum = 0D;
        int dims = 0;
        for (String featureName : featureNames) {
            double value = vector.getOrDefault(featureName, 0D);
            double center = centroid.getOrDefault(featureName, 0D);
            double delta = value - center;
            sum += delta * delta;
            dims++;
        }
        if (dims == 0) {
            return 0D;
        }
        return Math.sqrt(sum / dims);
    }

    private double mean(List<Double> values) {
        if (values == null || values.isEmpty()) {
            return 0D;
        }
        double sum = 0D;
        for (Double value : values) {
            sum += value == null ? 0D : value;
        }
        return sum / values.size();
    }

    private double std(List<Double> values, double mean) {
        if (values == null || values.size() < 2) {
            return 0D;
        }
        double sum = 0D;
        for (Double value : values) {
            double v = value == null ? 0D : value;
            double d = v - mean;
            sum += d * d;
        }
        return Math.sqrt(sum / values.size());
    }

    private double percentile(List<Double> values, double target) {
        if (values == null || values.isEmpty()) {
            return 50D;
        }
        int lessOrEqual = 0;
        for (Double value : values) {
            if ((value == null ? 0D : value) <= target) {
                lessOrEqual++;
            }
        }
        return (lessOrEqual * 100D) / values.size();
    }

    private String classifySpatialDeviation(double zScore, double percentile) {
        if (zScore >= 2D || percentile >= 97.5D) {
            return "high";
        }
        if (zScore >= 1D || percentile >= 85D) {
            return "moderate";
        }
        if (zScore <= -1D || percentile <= 35D) {
            return "close_to_healthy";
        }
        return "mild";
    }

    private double log2(double value) {
        return Math.log(value) / Math.log(2D);
    }

    private String classifyDirectionByLog2(double log2FoldChange, double currentAbundance, double healthyAverage) {
        if (healthyAverage <= 0D && currentAbundance > 0D) {
            return "higher";
        }
        if (currentAbundance <= 0D && healthyAverage > 0D) {
            return "lower";
        }
        if (log2FoldChange >= 0.58D) {
            return "higher";
        }
        if (log2FoldChange <= -0.58D) {
            return "lower";
        }
        return "similar";
    }

    private double percentileFromZScore(double zScore) {
        double clamped = Math.max(-8D, Math.min(8D, zScore));
        return 50D * (1D + erf(clamped / Math.sqrt(2D)));
    }

    private double erf(double x) {
        double sign = Math.signum(x);
        double absX = Math.abs(x);
        double a1 = 0.254829592D;
        double a2 = -0.284496736D;
        double a3 = 1.421413741D;
        double a4 = -1.453152027D;
        double a5 = 1.061405429D;
        double p = 0.3275911D;
        double t = 1D / (1D + p * absX);
        double y = 1D - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX));
        return sign * y;
    }

    private String classifyDirection(double currentAbundance, double healthyAverage) {
        if (healthyAverage <= 0D && currentAbundance > 0D) {
            return "higher";
        }
        if (currentAbundance <= 0D && healthyAverage > 0D) {
            return "lower";
        }
        if (healthyAverage <= 0D) {
            return "similar";
        }

        double ratio = currentAbundance / healthyAverage;
        if (ratio >= 1.5D) {
            return "higher";
        }
        if (ratio <= (1D / 1.5D)) {
            return "lower";
        }
        return "similar";
    }

    private boolean hasText(String value) {
        return value != null && !value.trim().isEmpty();
    }
}
