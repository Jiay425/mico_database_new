# AI Orchestrator

独立的 LangChain4j 编排服务。

## 当前能力

- `/api/ai/health`
- `/api/ai/chat`
- 对接主项目 `/api/dashboard/summary`
- 对接主项目患者、样本、Top 特征与预测接口
- 已接入 LangChain4j `AiService`
- 未配置可用模型时自动回退到规则模式

## 本地运行

先确认使用 JDK 17：

```powershell
$env:JAVA_HOME='D:\Java\jdk17'
$env:Path='D:\Java\jdk17\bin;' + $env:Path
```

再设置模型环境变量：

```powershell
$env:AI_OPENAI_API_KEY='your_api_key'
$env:AI_OPENAI_BASE_URL='https://yinli.one/v1'
$env:AI_OPENAI_MODEL='gpt-4o'
```

启动：

```powershell
mvn spring-boot:run
```

默认端口：`8088`

## 关键配置

```yaml
app:
  upstream:
    business-base-url: http://localhost:5000/api
  ai:
    provider-enabled: true
    openai:
      api-key: ${AI_OPENAI_API_KEY:}
      base-url: ${AI_OPENAI_BASE_URL:https://yinli.one/v1}
      model-name: ${AI_OPENAI_MODEL:gpt-4o}
```

## 说明

- 这个服务走 OpenAI 兼容接口
- 密钥不要写进仓库，统一通过环境变量注入
- 如果模型不可用，`/api/ai/chat` 仍会回退到规则模式
