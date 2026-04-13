# AGENT Runbook

## 1. 目标

本文件定义 agent 出错、接口失败、证据冲突和回答异常时的排查顺序。

## 2. 基础排查顺序

1. 先看主项目接口是否正常
2. 再看 `ai-orchestrator` 健康状态
3. 再看 Task Packet 是否正确
4. 再看 runtime trace
5. 最后看知识库和外部证据命中情况

## 3. 常见故障与处理

### 3.1 主项目 API 返回 HTML 或登录页
现象：
- `UnknownContentTypeException`
- JSON 解析报 `<html>`

处理：
- 检查主项目是否放行 `/api/**`
- 直接访问对应业务接口确认返回 JSON

### 3.2 预测结果异常或明显偏
处理：
- 确认是否使用标准丰度表
- 确认 `patientId + sampleId` 是否单一样本
- 确认模型标签映射与 Group 定义一致
- 查看 trace 中是否真正执行了 `run_prediction`

### 3.3 回答只复述页面
处理：
- 检查是否命中知识库
- 检查 `analysisLayers` 是否齐全
- 检查多智能体是否实际产出了 specialist insight
- 跑对应 eval 样例看是 must 没命中还是 forbidden 出现

### 3.4 外部证据缺失
处理：
- 查看 `/api/ai/health`
- 确认 `webSearchEnabled` 与 `webSearchConfigured`
- 确认本轮 Execution Policy 是否允许外部证据

### 3.5 MCP 能力可见但不会被用
处理：
- 检查 `tools/list`、`resources/list`、`prompts/list`
- 检查 Task Packet 的 availableTools
- 检查总控是否在 trace 中记录了 MCP 使用路径

## 4. 证据冲突处理原则

如果内部数据、模型判断、内部知识、外部证据出现冲突，处理优先级如下：
1. 内部业务数据优先
2. 模型判断只描述概率方向，不抢业务真相
3. 内部知识库用于解释规则与边界
4. 外部证据只做补充，不覆盖内部结论

## 5. 什么时候需要人工复核

以下场景应标记 `needsHumanReview=true`：
- 高风险问题涉及诊断、治疗、用药
- 高风险问题没有命中知识或模型层
- 结论之间明显冲突
- 预测与健康对照解释明显不一致

## 6. 回归检查清单

每次大改后至少检查：
- `/api/ai/health`
- `/api/ai/traces/{sessionId}/latest`
- 3 类 eval 套件
- 患者页一次真实问答
- 预测页一次真实问答
- 首页一次真实问答