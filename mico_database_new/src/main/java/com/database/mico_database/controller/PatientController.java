package com.database.mico_database.controller;

import com.database.mico_database.config.RabbitConfig;
import com.database.mico_database.model.Patient;
import com.database.mico_database.service.PatientService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.pagehelper.PageInfo;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
@Slf4j
public class PatientController {

    @Autowired
    private PatientService patientService;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired(required = false)
    private RabbitTemplate rabbitTemplate;

    @Value("${app.rabbitmq.enabled:false}")
    private boolean rabbitmqEnabled;

    @GetMapping("/search_patient")
    public String showSearchPage(Model model) {
        model.addAttribute("currentPage", "search");
        return "search_input";
    }

    @PostMapping("/analyze")
    @ResponseBody
    public String analyzePatient(@RequestParam("patient_id") String patientId) {
        if (rabbitmqEnabled && rabbitTemplate != null) {
            rabbitTemplate.convertAndSend(RabbitConfig.ANALYSIS_QUEUE_NAME, patientId);
            log.info("[mq] analysis task submitted for patient {}", patientId);
            return "SUCCESS: 分析任务已提交，后台正在处理中，请稍后查看结果。";
        }

        log.info("[local] RabbitMQ disabled, skipping async analysis for patient {}", patientId);
        return "SUCCESS: 当前已关闭 RabbitMQ，本地模式下跳过异步分析队列。";
    }

    @RequestMapping("/search_patient/result")
    public String handleSearch(@RequestParam("query_type") String queryType,
                               @RequestParam("query_value") String queryValue,
                               @RequestParam(value = "page", defaultValue = "1") int page,
                               @RequestParam(value = "size", defaultValue = "10") int size,
                               Model model) throws JsonProcessingException {

        log.info("搜索请求: type={}, value={}, page={}, size={}", queryType, queryValue, page, size);

        PageInfo<Patient> pageInfo = patientService.searchPatients(queryType, queryValue, page, size);

        model.addAttribute("patients_by_disease", pageInfo.getList());
        model.addAttribute("current_page", pageInfo.getPageNum());
        model.addAttribute("total_pages", pageInfo.getPages());
        model.addAttribute("total_records", pageInfo.getTotal());
        model.addAttribute("query_type", queryType);
        model.addAttribute("query_value", queryValue);

        boolean isIdSearch = "id".equals(queryType);
        boolean isSingleResult = pageInfo.getList() != null && pageInfo.getList().size() == 1;

        if ((isIdSearch || isSingleResult) && !pageInfo.getList().isEmpty()) {
            Patient targetPatient = pageInfo.getList().get(0);
            model.addAttribute("patient_data", targetPatient);
            model.addAttribute("patient_data_json", objectMapper.writeValueAsString(targetPatient));
        } else {
            model.addAttribute("patient_data", null);
            model.addAttribute("patient_data_json", "{}");
        }

        model.addAttribute("patients_by_disease_json", "[]");
        if (pageInfo.getTotal() == 0) {
            model.addAttribute("error", "未找到符合条件的数据");
        }

        return "patient_results";
    }

    @GetMapping("/disease_prediction")
    public String showPredictionPage(Model model) {
        model.addAttribute("currentPage", "prediction");
        return "disease_prediction";
    }
}
