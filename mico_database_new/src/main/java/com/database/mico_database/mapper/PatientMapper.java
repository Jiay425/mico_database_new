package com.database.mico_database.mapper;

import com.database.mico_database.model.CytokineData;
import com.database.mico_database.model.MicrobeAbundance;
import com.database.mico_database.model.Patient;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

@Mapper
public interface PatientMapper {

    Patient findById(String patientId);

    List<MicrobeAbundance> findMicrobesByPatientId(String patientId);

    List<MicrobeAbundance> findStandardMicrobesByPatientId(String patientId);

    List<MicrobeAbundance> findStandardMicrobesByPatientIdAndSampleId(@Param("patientId") String patientId,
                                                                      @Param("sampleId") String sampleId);

    List<MicrobeAbundance> findTopStandardMicrobesByPatientIdAndSampleId(@Param("patientId") String patientId,
                                                                         @Param("sampleId") String sampleId,
                                                                         @Param("limit") int limit);

    List<Map<String, Object>> findStandardSampleSummariesByPatientId(String patientId);

    Map<String, Object> countHealthyReferenceStats(@Param("bodySite") String bodySite,
                                                   @Param("useBodySite") boolean useBodySite);

    List<Map<String, Object>> findHealthyAverageForMicrobes(@Param("bodySite") String bodySite,
                                                            @Param("useBodySite") boolean useBodySite,
                                                            @Param("microbeNames") List<String> microbeNames);

    List<Map<String, Object>> findHealthyMicrobeStats(@Param("bodySite") String bodySite,
                                                      @Param("useBodySite") boolean useBodySite);

    List<Map<String, Object>> findHealthyFeatureMatrix(@Param("bodySite") String bodySite,
                                                       @Param("useBodySite") boolean useBodySite,
                                                       @Param("microbeNames") List<String> microbeNames);

    List<Map<String, Object>> findHealthyTopMicrobes(@Param("bodySite") String bodySite,
                                                     @Param("useBodySite") boolean useBodySite,
                                                     @Param("limit") int limit);

    List<CytokineData> findCytokinesByPatientId(String patientId);

    List<String> findIdsByName(String patientName);

    List<String> findIdsByDisease(String diseaseName);

    void updatePatientRisk(Patient patient);

    List<Patient> searchPatients(@Param("queryType") String queryType,
                                 @Param("queryValue") String queryValue);
}
