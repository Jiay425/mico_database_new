package com.database.mico_database.common;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * 全局异常处理
 * 作用：捕获所有 Controller 抛出的异常，转为标准 JSON 返回
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    // 捕获所有未知的 Exception
    @ExceptionHandler(Exception.class)
    public Result<String> handleException(Exception e) {
        log.error("系统内部异常: ", e); // 打印错误堆栈到控制台（为了调试）
        return Result.failed("系统内部错误: " + e.getMessage()); // 给用户看优雅的提示
    }

    // 捕获算术异常（比如除以0）- 演示用
    @ExceptionHandler(ArithmeticException.class)
    public Result<String> handleArithmeticException(ArithmeticException e) {
        log.error("算术异常: ", e);
        return Result.failed("计算逻辑错误，请检查输入数据");
    }
}