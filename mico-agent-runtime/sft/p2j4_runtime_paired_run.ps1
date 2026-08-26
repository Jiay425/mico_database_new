param(
    [ValidateSet("Validate", "Base", "Sft", "Both")]
    [string]$Mode = "Validate",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$runtimeRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $runtimeRoot ".venv\Scripts\python.exe"
$taskSet = Join-Path $runtimeRoot "evals\p2j4-runtime-paired-v1\task-set.json"
$outputRoot = Join-Path $runtimeRoot "evals\p2j4-runtime-paired-v1"
$baseOutput = Join-Path $outputRoot "base.json"
$sftOutput = Join-Path $outputRoot "sft.json"

if (-not (Test-Path -LiteralPath $pythonPath)) { throw "LOCAL_VENV_MISSING" }
if (-not (Test-Path -LiteralPath $taskSet)) { throw "PAIRED_TASK_SET_MISSING" }

# Import the persisted user configuration into this process without printing
# any value.  The runtime runner itself never serializes secrets.
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

function Invoke-PairedMode([string]$Name, [string]$OutputPath) {
    if ((Test-Path -LiteralPath $OutputPath) -and -not $Resume) {
        throw "OUTPUT_EXISTS_USE_RESUME:$OutputPath"
    }

    if ($Name -eq "Base") {
        Remove-Item Env:MICO_SFT_POLICY_ENABLED -ErrorAction SilentlyContinue
        Remove-Item Env:MICO_SFT_POLICY_BASE_URL -ErrorAction SilentlyContinue
        Remove-Item Env:MICO_SFT_POLICY_MODEL -ErrorAction SilentlyContinue
        Remove-Item Env:MICO_SFT_POLICY_TOKEN -ErrorAction SilentlyContinue
    } else {
        $env:MICO_SFT_POLICY_ENABLED = "true"
        $env:MICO_SFT_POLICY_BASE_URL = "http://127.0.0.1:19002"
        $env:MICO_SFT_POLICY_MODEL = "qwen3-8b-decision-sft-v4"
        $env:MICO_SFT_POLICY_TOKEN = ""
    }

    $arguments = @(
        "evals/p2j4_runner.py", "--real",
        "--task-set", $taskSet,
        "--output", $OutputPath
    )
    if ($Resume) { $arguments += "--resume" }
    Write-Output ("PAIRED_MODE_START=" + $Name)
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw "PAIRED_MODE_FAILED:${Name}:$LASTEXITCODE" }
    Write-Output ("PAIRED_MODE_DONE=" + $Name)
}

switch ($Mode) {
    "Validate" {
        & $pythonPath evals/p2j4_runner.py --validate --task-set $taskSet
        if ($LASTEXITCODE -ne 0) { throw "PAIRED_TASK_SET_INVALID" }
    }
    "Base" { Invoke-PairedMode "Base" $baseOutput }
    "Sft" { Invoke-PairedMode "Sft" $sftOutput }
    "Both" {
        Invoke-PairedMode "Base" $baseOutput
        Invoke-PairedMode "Sft" $sftOutput
    }
}
