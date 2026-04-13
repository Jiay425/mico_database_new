package com.database.mico_ai.mcp;

import com.database.mico_ai.service.TraceRecorderService;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class McpPromptService {

    private final TraceRecorderService traceRecorderService;

    public McpPromptService(TraceRecorderService traceRecorderService) {
        this.traceRecorderService = traceRecorderService;
    }

    public List<Map<String, Object>> listPrompts() {
        return List.of(
                prompt("patient_vs_healthy_analysis", "患者与健康组对照分析", "用于解读当前样本与健康组差异", List.of(
                        argument("patientId", "患者ID", true),
                        argument("sampleId", "样本ID", true),
                        argument("question", "补充问题", false),
                        argument("sessionId", "会话ID", false)
                )),
                prompt("prediction_consistency_explainer", "预测一致性解释", "用于解释预测方向与健康偏离是否一致", List.of(
                        argument("patientId", "患者ID", true),
                        argument("sampleId", "样本ID", true),
                        argument("sessionId", "会话ID", false)
                )),
                prompt("dashboard_briefing", "首页简报", "用于输出首页数据重点的结构化简报", List.of(
                        argument("question", "补充问题", false),
                        argument("sessionId", "会话ID", false)
                )),
                prompt("sample_abnormality_review", "样本异常回顾", "用于回顾当前样本的异常结构与解释边界", List.of(
                        argument("patientId", "患者ID", true),
                        argument("sampleId", "样本ID", true),
                        argument("sessionId", "会话ID", false)
                )),
                prompt("medical_evidence_briefing", "医学证据简报", "用于把外部医学证据整理成可解释的补充层", List.of(
                        argument("question", "医学问题", true),
                        argument("sessionId", "会话ID", false)
                )),
                prompt("task_packet_review", "Task Packet 审阅", "用于审阅本轮任务包是否具备正确上下文、工具与风险分级", List.of(
                        argument("sessionId", "会话ID", true),
                        argument("question", "用户问题", true),
                        argument("page", "页面上下文", false),
                        argument("patientId", "患者ID", false),
                        argument("sampleId", "样本ID", false)
                )),
                prompt("runtime_trace_review", "Runtime Trace 审阅", "用于审阅本轮 trace 是否完整覆盖工具、知识、证据与多智能体结论", List.of(
                        argument("sessionId", "会话ID", true),
                        argument("focus", "重点关注项", false)
                )),
                prompt("evidence_layer_audit", "证据分层审阅", "用于检查回答是否正确区分内部数据、模型判断、内部知识和外部证据", List.of(
                        argument("question", "用户问题", true),
                        argument("answer", "待审阅回答", true),
                        argument("sessionId", "会话ID", false)
                ))
        );
    }

    public Map<String, Object> getPrompt(String name, Map<String, Object> arguments, String sessionId) {
        Map<String, Object> args = arguments == null ? Map.of() : arguments;
        Map<String, Object> result = switch (name) {
            case "patient_vs_healthy_analysis" -> promptResult("患者与健康组对照分析", List.of(
                    message("user", text("请结合患者 " + required(args, "patientId") + " 的样本 " + required(args, "sampleId") + "，优先分析与健康组的偏离，再说明这些偏离与预测结果是否一致。"))
            ));
            case "prediction_consistency_explainer" -> promptResult("预测一致性解释", List.of(
                    message("user", text("请解释患者 " + required(args, "patientId") + " 的样本 " + required(args, "sampleId") + " 的预测方向、Top1/Top2 差异，以及与健康组偏离是否同向。"))
            ));
            case "dashboard_briefing" -> promptResult("首页简报", List.of(
                    message("user", text("请基于首页真实统计数据生成简洁简报，优先指出主要疾病、地区来源和群体结构重点。"))
            ));
            case "sample_abnormality_review" -> promptResult("样本异常回顾", List.of(
                    message("user", text("请回顾患者 " + required(args, "patientId") + " 的样本 " + required(args, "sampleId") + " 的主要异常特征，说明哪些现象值得重点关注，并明确解释边界。"))
            ));
            case "medical_evidence_briefing" -> promptResult("医学证据简报", List.of(
                    message("system", text("请只把外部医学证据作为补充层，不要覆盖内部数据、模型判断或诊断边界。")),
                    message("user", text("请为这个医学问题整理外部证据简报：" + required(args, "question") + "。优先引用权威来源，并标出来源类型。"))
            ));
            case "task_packet_review" -> promptResult("Task Packet 审阅", List.of(
                    message("system", text("请检查任务包是否具备正确上下文、工具、风险等级和执行策略，不要跳出 harness 边界。")),
                    message("user", text("请审阅本轮任务包。sessionId=" + required(args, "sessionId")
                            + " question=" + required(args, "question")
                            + optionalSegment(" page=", args.get("page"))
                            + optionalSegment(" patientId=", args.get("patientId"))
                            + optionalSegment(" sampleId=", args.get("sampleId"))
                            + "。请指出缺失上下文、风险分级是否合理、哪些工具必须可用。"))
            ));
            case "runtime_trace_review" -> promptResult("Runtime Trace 审阅", List.of(
                    message("system", text("请基于 harness trace 进行 review，重点看工具调用、证据命中和总控整合是否完整。")),
                    message("user", text("请审阅 sessionId=" + required(args, "sessionId")
                            + " 的最新 runtime trace"
                            + optionalSegment("，重点关注：", args.get("focus"))
                            + "。请指出缺失步骤、证据层空洞和是否需要人工复核。"))
            ));
            case "evidence_layer_audit" -> promptResult("证据分层审阅", List.of(
                    message("system", text("请只从证据分层角度审阅，不要重新发明事实。")),
                    message("user", text("用户问题是：" + required(args, "question") + "。待审阅回答是：" + required(args, "answer") + "。请检查它是否区分内部数据、模型判断、内部知识和外部证据。"))
            ));
            default -> throw new IllegalArgumentException("Prompt not found: " + name);
        };

        if (sessionId != null && !sessionId.isBlank()) {
            traceRecorderService.recordMcpPromptGet(sessionId, name, promptDetail(args));
        }
        return result;
    }

    private Map<String, Object> promptDetail(Map<String, Object> args) {
        Map<String, Object> detail = new LinkedHashMap<>();
        List<String> argumentKeys = new ArrayList<>();
        for (String key : args.keySet()) {
            if (!"sessionId".equals(key)) {
                argumentKeys.add(key);
            }
        }
        detail.put("argumentKeys", argumentKeys);
        return detail;
    }

    private Map<String, Object> prompt(String name, String title, String description, List<Map<String, Object>> arguments) {
        Map<String, Object> prompt = new LinkedHashMap<>();
        prompt.put("name", name);
        prompt.put("title", title);
        prompt.put("description", description);
        prompt.put("arguments", arguments);
        return prompt;
    }

    private Map<String, Object> argument(String name, String description, boolean required) {
        Map<String, Object> argument = new LinkedHashMap<>();
        argument.put("name", name);
        argument.put("description", description);
        argument.put("required", required);
        return argument;
    }

    private Map<String, Object> promptResult(String description, List<Map<String, Object>> messages) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("description", description);
        result.put("messages", messages);
        return result;
    }

    private Map<String, Object> message(String role, Map<String, Object> content) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", role);
        message.put("content", content);
        return message;
    }

    private Map<String, Object> text(String value) {
        return Map.of("type", "text", "text", value);
    }

    private String required(Map<String, Object> args, String key) {
        Object value = args.get(key);
        String text = value == null ? "" : String.valueOf(value).trim();
        if (text.isBlank()) {
            throw new IllegalArgumentException("Missing required argument: " + key);
        }
        return text;
    }

    private String optionalSegment(String prefix, Object value) {
        if (value == null) {
            return "";
        }
        String text = String.valueOf(value).trim();
        return text.isBlank() ? "" : prefix + text;
    }
}