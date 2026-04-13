package com.database.mico_database.model;

import lombok.Data;

@Data
public class AgeGenderCount {
    private String ageGroup;
    private String gender;
    private Integer count;
}
