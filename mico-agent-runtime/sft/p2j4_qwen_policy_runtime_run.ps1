param(
    [ValidateSet("Validate", "QwenBase", "QwenSft", "QwenDpo")]
    [string]$Mode = "Validate",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$runtimeRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $runtimeRoot ".venv\Scripts\python.exe"
$taskSet = Join-Path $runtimeRoot "evals\p2j4-policy-sensitive-runtime-v1\task-set.json"
$outputRoot = Join-Path $runtimeRoot "evals\p2j4-qwen-runtime-paired-v1"
$outputPath = if ($Mode -eq "QwenBase") {
    Join-Path $outputRoot "qwen-base.json"
} elseif ($Mode -eq "QwenDpo") {
    # The earlier v4 Runtime20 artifact is retained as a provenance-labeling
    # defect record.  Never overwrite it; this run is emitted only after
    # scientific_workflow correctly classifies `sft_policy` as model-origin.
    Join-Path $outputRoot "qwen-sft-v5-dpo-v4-controlled-runtime-r2-20260826.json"
} else {
    Join-Path $outputRoot "qwen-sft-v5.json"
}

if (-not (Test-Path -LiteralPath $pythonPath)) { throw "LOCAL_VENV_MISSING" }
if (-not (Test-Path -LiteralPath $taskSet)) { throw "POLICY_SENSITIVE_TASK_SET_MISSING" }

$userValues = [Environment]::GetEnvironmentVariables("User")
foreach ($entry in $userValues.GetEnumerator()) {
    if ([string]$entry.Key -like "MICO_*") {
        [Environment]::SetEnvironmentVariable([string]$entry.Key, [string]$entry.Value, "Process")
    }
}

$env:MICO_P2J4_REAL_RUNS = "true"
$env:MICO_P2J4_HARD_CONTROLLED_SCENARIOS = "false"
$env:MICO_JAVA_AGENT_TOOL_BASE_URL = "http://127.0.0.1:5000"
$env:MICO_KNOWLEDGE_RETRIEVAL_BACKEND = "database"
$env:MICO_KNOWLEDGE_VECTOR_ENABLED = "true"
$env:MICO_KNOWLEDGE_GRAPH_ENABLED = "true"
$env:MICO_KNOWLEDGE_GRAPH_VERSION = "fulltext-provenance-graphrag-v3"
$env:MICO_LOG_LEVEL = "WARNING"
$env:PYTHONPATH = $runtimeRoot
# Qwen chooses the next high-level Action.  The configured Research Planner
# must then materialize that exact action into question-specific SQL or a
# bounded analysis plan.  Do not clear these values and silently replace
# model-generated queries with catalog templates.
if (-not $env:MICO_RESEARCH_PLANNER_BASE_URL -or -not $env:MICO_RESEARCH_PLANNER_MODEL -or -not $env:MICO_RESEARCH_PLANNER_TOKEN) {
    throw "DYNAMIC_RESEARCH_PLANNER_CONFIG_REQUIRED"
}
$env:MICO_GRAPH_RAG_GENERATOR_BASE_URL = ""
$env:MICO_GRAPH_RAG_GENERATOR_MODEL = ""
$env:MICO_GRAPH_RAG_GENERATOR_TOKEN = ""

if ($Mode -eq "Validate") {
    & $pythonPath evals/p2j4_runner.py --validate --task-set $taskSet
    if ($LASTEXITCODE -ne 0) { throw "QWEN_PAIRED_TASK_SET_INVALID" }
    exit 0
}

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:19002/health" -TimeoutSec 10
} catch {
    throw "QWEN_POLICY_TUNNEL_UNHEALTHY"
}
if ($Mode -eq "QwenBase") {
    if ($health.model -ne "qwen3-8b-decision-base") { throw "QWEN_BASE_SERVER_MODEL_MISMATCH" }
    $env:MICO_SFT_POLICY_MODEL = "qwen3-8b-decision-base"
} elseif ($Mode -eq "QwenDpo") {
    if ($health.model -ne "qwen3-8b-decision-sft-v5-dpo-v4-controlled") { throw "QWEN_DPO_SERVER_MODEL_MISMATCH" }
    $env:MICO_SFT_POLICY_MODEL = "qwen3-8b-decision-sft-v5-dpo-v4-controlled"
} else {
    if ($health.model -ne "qwen3-8b-decision-sft-v5") { throw "QWEN_SFT_SERVER_MODEL_MISMATCH" }
    $env:MICO_SFT_POLICY_MODEL = "qwen3-8b-decision-sft-v5"
}
$env:MICO_SFT_POLICY_ENABLED = "true"
$env:MICO_SFT_POLICY_BASE_URL = "http://127.0.0.1:19002"
$env:MICO_SFT_POLICY_TOKEN = ""

if ((Test-Path -LiteralPath $outputPath) -and -not $Resume) {
    throw "QWEN_PAIRED_OUTPUT_EXISTS_USE_RESUME:$outputPath"
}
$arguments = @("evals/p2j4_runner.py", "--real", "--task-set", $taskSet, "--output", $outputPath)
if ($Resume) { $arguments += "--resume" }
& $pythonPath @arguments
if ($LASTEXITCODE -ne 0) { throw "QWEN_PAIRED_RUN_FAILED:${Mode}:$LASTEXITCODE" }

$outputBaseName = [System.IO.Path]::GetFileNameWithoutExtension($outputPath)
$bindingOutput = Join-Path $outputRoot ($outputBaseName + "-action-binding-audit.json")
& $pythonPath -m evals.p2j4_audit_execution_action_binding --input $outputPath --output $bindingOutput
if ($LASTEXITCODE -ne 0) { throw "QWEN_ACTION_BINDING_AUDIT_FAILED:${Mode}:$LASTEXITCODE" }
