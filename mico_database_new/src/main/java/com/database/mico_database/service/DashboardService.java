package com.database.mico_database.service;

import com.database.mico_database.mapper.DashboardMapper;
import com.database.mico_database.model.AgeGenderCount;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class DashboardService {

    @Autowired
    private DashboardMapper dashboardMapper;

    @Cacheable(value = "dashboard_stats", key = "'homepage_data'")
    public Map<String, Object> getDashboardData() {
        return buildDashboardData();
    }

    public Map<String, Object> getDashboardDataDirect() {
        return buildDashboardData();
    }

    private Map<String, Object> getOverviewStats() {
        Map<String, Object> overviewStats = new HashMap<>();
        overviewStats.put("total_patients", dashboardMapper.getTotalPatients());

        Map<String, Object> mostCommonDisease = dashboardMapper.getMostCommonDisease();
        overviewStats.put("most_common_disease_name",
                mostCommonDisease != null ? mostCommonDisease.get("disease_name") : "N/A");
        overviewStats.put("most_common_disease_count",
                mostCommonDisease != null ? mostCommonDisease.get("patient_count") : 0);

        Map<String, Object> ageRange = dashboardMapper.getMinMaxPatientAge();
        overviewStats.put("min_age", ageRange != null ? ageRange.get("min_age") : "N/A");
        overviewStats.put("max_age", ageRange != null ? ageRange.get("max_age") : "N/A");

        return overviewStats;
    }

    private Map<String, Object> processAgeGenderData() {
        List<AgeGenderCount> rawData = dashboardMapper.getAgeGenderDistribution();

        List<String> ageBrackets = Arrays.asList("0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "60+");
        Map<String, Integer> maleMap = new LinkedHashMap<>();
        Map<String, Integer> femaleMap = new LinkedHashMap<>();

        for (String bracket : ageBrackets) {
            maleMap.put(bracket, 0);
            femaleMap.put(bracket, 0);
        }

        for (AgeGenderCount item : rawData) {
            if ("male".equalsIgnoreCase(item.getGender())) {
                maleMap.put(item.getAgeGroup(), item.getCount());
            } else if ("female".equalsIgnoreCase(item.getGender())) {
                femaleMap.put(item.getAgeGroup(), item.getCount());
            }
        }

        Map<String, Object> result = new HashMap<>();
        result.put("age_brackets", ageBrackets);
        result.put("male_counts", new ArrayList<>(maleMap.values()));
        result.put("female_counts", new ArrayList<>(femaleMap.values()));
        return result;
    }

    private Map<String, Object> buildDashboardData() {
        Map<String, Object> dashboardData = new HashMap<>();
        dashboardData.put("disease_data", dashboardMapper.getDiseaseDistribution());
        dashboardData.put("patient_origin_data", dashboardMapper.getPatientOriginDistribution());
        dashboardData.put("age_gender_data", processAgeGenderData());
        dashboardData.put("overview_stats", getOverviewStats());
        return dashboardData;
    }
}
