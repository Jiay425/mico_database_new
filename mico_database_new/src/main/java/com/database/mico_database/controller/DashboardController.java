package com.database.mico_database.controller;

import com.database.mico_database.service.DashboardService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper; // 引入 ObjectMapper
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.Collections;
import java.util.Map;

@Controller
public class DashboardController {

    @Autowired
    private DashboardService dashboardService;

    // 注入 Jackson 的核心转换器
    @Autowired
    private ObjectMapper objectMapper;

    @GetMapping("/")
    public String getIndexPage(Model model) throws JsonProcessingException { // 声明可能抛出异常
        // 1. 获取业务数据 (不变)
        Map<String, Object> dashboardData = dashboardService.getDashboardData();

        // 2. 【关键修改】将 Java 对象手动转换为 JSON 字符串
        model.addAttribute("disease_data_json", objectMapper.writeValueAsString(dashboardData.get("disease_data")));
        model.addAttribute("patient_origin_data_json",
                objectMapper.writeValueAsString(dashboardData.get("patient_origin_data")));
        model.addAttribute("age_gender_data_json",
                objectMapper.writeValueAsString(dashboardData.get("age_gender_data")));
        model.addAttribute("overview_stats_json", objectMapper.writeValueAsString(dashboardData.get("overview_stats")));
        model.addAttribute("patientDataFromFlask_json", "{}"); // 保持兼容性

        // 3. 返回模板名 (不变)
        return "index";
    }
}