# AGENT Architecture

## 1. 目标

本文件定义当前仓库作为 agent harness 的 system of record 时，主项目、`ai-orchestrator`、MCP、模型服务、知识库与评测体系的边界。

## 2. 系统分层

### 2.1 主业务系统 `mico_database_new`

职责：
- 页面展示与交互
- 患者、样本、首页统计等业务真相
- 与 MySQL / Redis 的业务访问
- 对外提供稳定 JSON API

不负责：
- LLM 推理与多智能体编排
- 外部证据搜索
- RAG 检索策略

### 2.2 AI 编排服务 `ai-orchestrator`

职责：
- 统一接收 AI 请求
- 生成 Task Packet
- 执行工作流、知识增强、混合编排、多智能体协调
- 输出分层结论
- 记录 Trace 与 Review
- 通过 MCP 暴露标准化能力

不负责：
- 直接访问业务数据库
- 绕开主系统接口读取业务数据

### 2.3 Python 模型服务

职责：
- 加载训练产物
- 对标准特征向量执行推理
- 返回预测概率、相关分析数据

边界：
- 模型输出是辅助判断，不是临床诊断
- 模型服务不解释业务规则，解释逻辑由 `ai-orchestrator` 负责

### 2.4 知识库

路径：`references/knowledge`

分为三类：
- `business`：字段说明、Group 映射、页面分析模板、工作流定义
- `medical`：疾病简介、健康组对照规则、常见微生物释义、解释边界
- `model`：标签定义、特征空间、预测边界、模型解释模板

原则：
- 内部知识库优先于外部网页证据
- 模型库只描述模型边界和解释口径，不伪装成医学事实

### 2.5 MCP Connector

职责：
- 统一对外暴露 `tools`、`resources`、`prompts`
- 让其他 agent、桌面端、评测器和自动化任务通过统一协议访问系统能力

当前定义：
- `tools`：动作与查询
- `resources`：系统事实与知识文档
- `prompts`：分析模板

## 3. 事实优先级

回答中应按以下优先级组织事实：
1. 内部业务数据
2. 模型判断
3. 内部知识库
4. 外部网页证据

任何情况下都不允许：
- 用外部网页覆盖内部业务真相
- 把模型判断写成确诊结论
- 把知识库规则写成当前患者的实时事实

## 4. Harness Runtime 核心对象

### 4.1 Task Packet

每次请求必须整理为统一任务包，至少包含：
- 用户问题
- 页面上下文
- `patientId` / `sampleId`
- 已解析意图
- 可用工具
- 风险等级
- Execution Policy
- Evidence Policy

### 4.2 Execution Policy

明确本轮是否允许：
- 只用内部知识
- 调外部证据
- 调预测
- 调 MCP tools/resources/prompts

### 4.3 Evidence Policy

明确输出必须区分：
- 内部数据
- 模型判断
- 内部知识
- 外部证据

## 5. Trace 与 Review Loop

每次运行都要记录：
- 命中的工具
- 读到的 resources
- 命中的知识片段
- 是否调用外部搜索
- 哪些 agent 参与了生成
- 总控如何整合结论

Review Loop 用于回答这些问题：
- 是数据层出错，还是知识层没命中
- 是模型解释不充分，还是外部证据不足
- 是否需要人工复核

## 6. 评测与回归

本仓库的 eval 样例位于：`references/harness-evals`

只要修改以下任一层，就应重新跑 eval：
- 工作流
- 知识增强
- 混合编排
- 多智能体协调
- MCP 工具 / 资源 / 提示词
- 知识库关键文档

## 7. 实施原则

- 主项目仍然是业务事实源
- `ai-orchestrator` 是 harness runtime
- MCP 是标准连接层，不是旁路系统
- 评测结果优先于“主观感觉变聪明了”