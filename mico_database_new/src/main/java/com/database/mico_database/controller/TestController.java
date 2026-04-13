package com.database.mico_database.controller;
import com.database.mico_database.mapper.PatientMapper;
import com.database.mico_database.mapper.UserMapper;
import com.database.mico_database.model.Patient;
import com.database.mico_database.model.SysUser;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import com.database.mico_database.common.Result; // 引入统一返回类

@RestController
public class TestController {

    @Autowired
    private PatientMapper patientMapper;
    @Autowired
    private UserMapper userMapper; // 注入 UserMapper
    @Autowired
    private PasswordEncoder passwordEncoder; // 注入加密器

    @GetMapping("/hello")
    public String sayHello() {
        return "Hello World! Spring Boot server is running successfully on port 5000.";
    }

    /**
     * 【关键修改】
     * 在 @GetMapping 中添加 produces = MediaType.APPLICATION_JSON_VALUE
     * 这会强制指定此方法的响应内容类型 (Content-Type) 为 "application/json"。
     * 这消除了任何模糊性，告诉 Spring Boot 必须使用能够生成 JSON 的转换器（也就是 Jackson）。
     */
    @GetMapping(value = "/test-patient", produces = MediaType.APPLICATION_JSON_VALUE)
    public Patient getPatientById(@RequestParam(value = "id", defaultValue = "P001") String id) {
        Patient patient = patientMapper.findById(id);

        if (patient == null) {
            return new Patient();
        }

        return patient;
    }


    @GetMapping("/test-init-user")
    public String initUser() {
        // 创建 Admin
        SysUser admin = new SysUser();
        admin.setUsername("admin");
        admin.setPassword(passwordEncoder.encode("123456")); // 加密
        admin.setRole("ROLE_ADMIN");
        try { userMapper.insert(admin); } catch (Exception e) {} // 忽略重复插入报错

        // 创建 Doctor
        SysUser doctor = new SysUser();
        doctor.setUsername("doctor");
        doctor.setPassword(passwordEncoder.encode("123456")); // 加密
        doctor.setRole("ROLE_DOCTOR");
        try { userMapper.insert(doctor); } catch (Exception e) {}

        return "账号初始化成功！admin/123456, doctor/123456";
    }

    // 1. 测试成功情况
    @GetMapping("/test-success")
    public Result<String> testSuccess() {
        // 以前是 return "Hello";
        // 现在是：
        return Result.success("Hello, Enterprise World!");
    }

    // 2. 测试失败情况 (模拟一个 Bug)
    @GetMapping("/test-fail")
    public Result<String> testFail() {
        int i = 10 / 0; // 这会抛出 ArithmeticException
        return Result.success("这行代码永远不会执行");
    }
}