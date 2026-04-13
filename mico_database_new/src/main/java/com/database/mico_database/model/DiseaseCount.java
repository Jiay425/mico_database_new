package com.database.mico_database.model;

import lombok.Data;

@Data
public class DiseaseCount {
    private String disease;
    private Integer count;
    // 在 MyBatis 中，我们需要一个额外的字段来接收百分比计算（如果需要的话）
    // 为了与您之前的 Python 代码兼容，我们保留它
    private Integer percentage;
}
