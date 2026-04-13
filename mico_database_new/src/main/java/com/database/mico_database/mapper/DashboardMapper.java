package com.database.mico_database.mapper;

import com.database.mico_database.model.AgeGenderCount;
import com.database.mico_database.model.CountryCount;
import com.database.mico_database.model.DiseaseCount;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;
import java.util.Map;

@Mapper
public interface DashboardMapper {

    List<DiseaseCount> getDiseaseDistribution();

    Integer getTotalPatients();

    Map<String, Object> getMostCommonDisease();

    Map<String, Object> getMinMaxPatientAge();

    List<AgeGenderCount> getAgeGenderDistribution();

    List<CountryCount> getPatientOriginDistribution();
}