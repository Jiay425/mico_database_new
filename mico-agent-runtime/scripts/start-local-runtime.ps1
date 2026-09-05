param(
    [int]$Port = 8000,
    [ValidateSet("deterministic", "sft")]
    [string]$PlannerMode = "deterministic"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Secrets remain only in the Windows user environment.  This launcher copies
# them into this one child process without printing, persisting, or writing
# them into the repository.
$requiredEnvironmentNames = @(
    "MICO_JAVA_AGENT_TOOL_BASE_URL",
    "MICO_AGENT_INTERNAL_TOKEN",
    "MICO_RUNTIME_INTERNAL_TOKEN",
    "MICO_RESEARCH_PLANNER_BASE_URL",
    "MICO_RESEARCH_PLANNER_MODEL",
    "MICO_RESEARCH_PLANNER_TOKEN",
    "MICO_GRAPH_RAG_GENERATOR_BASE_URL",
    "MICO_GRAPH_RAG_GENERATOR_MODEL",
    "MICO_GRAPH_RAG_GENERATOR_TOKEN",
    "MICO_GEMINI_EMBEDDING_ENABLED",
    "MICO_GEMINI_EMBEDDING_MODEL",
    "MICO_GEMINI_API_KEY",
    "MICO_KNOWLEDGE_RETRIEVAL_BACKEND",
    "MICO_KNOWLEDGE_VECTOR_ENABLED",
    "MICO_KNOWLEDGE_GRAPH_ENABLED",
    "MICO_KNOWLEDGE_VECTOR_DATABASE_URL",
    "MICO_KNOWLEDGE_NEO4J_URI",
    "MICO_KNOWLEDGE_NEO4J_USER",
    "MICO_KNOWLEDGE_NEO4J_PASSWORD",
    "MICO_KNOWLEDGE_GRAPH_VERSION",
    "MICO_LOCAL_KNOWLEDGE_INDEX_DIR",
    "MICO_SFT_POLICY_BASE_URL",
    "MICO_SFT_POLICY_MODEL"
)

foreach ($name in $requiredEnvironmentNames) {
    $value = [Environment]::GetEnvironmentVariable($name, "User")
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Missing Windows user environment variable: $name"
    }
    Set-Item -Path "Env:$name" -Value $value
}

# The loopback policy service may intentionally run without its own bearer
# token.  Copy it only when configured; never require or print it.
$policyToken = [Environment]::GetEnvironmentVariable("MICO_SFT_POLICY_TOKEN", "User")
if (-not [string]::IsNullOrWhiteSpace($policyToken)) {
    Set-Item -Path "Env:MICO_SFT_POLICY_TOKEN" -Value $policyToken
}

# The decision-policy mode is an explicit per-run choice.  It is intentionally
# not persisted in the user environment: deterministic mode remains the safe
# default, while Dynamic E2E runs can opt into the already deployed SFT/DPO
# policy endpoint through this launcher.
if ($PlannerMode -eq "sft") {
    $env:MICO_SFT_POLICY_ENABLED = "true"
}

$runtimeRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $runtimeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Runtime Python environment is unavailable"
}

Set-Location -LiteralPath $runtimeRoot
& $python -m uvicorn mico_agent_runtime.transport.app:create_app --factory --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
