$ErrorActionPreference = "Stop"

$runtimeRoot = Split-Path -Parent $PSScriptRoot
$userValues = [Environment]::GetEnvironmentVariables("User")
foreach ($entry in $userValues.GetEnumerator()) {
    if ([string]$entry.Key -like "MICO_*") {
        [Environment]::SetEnvironmentVariable(
            [string]$entry.Key,
            [string]$entry.Value,
            "Process"
        )
    }
}

$env:MICO_JAVA_AGENT_TOOL_BASE_URL = "http://127.0.0.1:5000"
$env:MICO_RUNTIME_INTERNAL_TOKEN = [Environment]::GetEnvironmentVariable(
    "MICO_RUNTIME_INTERNAL_TOKEN", "User"
)
$env:MICO_SFT_POLICY_ENABLED = "true"
$env:MICO_SFT_POLICY_BASE_URL = "http://127.0.0.1:19002"
$env:MICO_SFT_POLICY_MODEL = "qwen3-8b-decision-sft-v4"
$env:MICO_SFT_POLICY_TOKEN = ""
$env:MICO_KNOWLEDGE_RETRIEVAL_BACKEND = "database"
$env:MICO_KNOWLEDGE_VECTOR_ENABLED = "true"
$env:MICO_KNOWLEDGE_GRAPH_ENABLED = "true"
$env:MICO_LOG_LEVEL = "WARNING"

$logDir = Join-Path $runtimeRoot "tmp-runtime-smoke"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdoutPath = Join-Path $logDir "runtime-8010.stdout.log"
$stderrPath = Join-Path $logDir "runtime-8010.stderr.log"
Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

$pythonPath = Join-Path $runtimeRoot ".venv\Scripts\python.exe"
$process = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList @(
        "-m", "uvicorn", "mico_agent_runtime.transport.app:create_app",
        "--factory", "--host", "127.0.0.1", "--port", "8010"
    ) `
    -WorkingDirectory $runtimeRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

Write-Output ("RUNTIME_PID=" + $process.Id)
Start-Sleep -Seconds 3
try {
    $openApi = Invoke-RestMethod -Uri "http://127.0.0.1:8010/openapi.json" -TimeoutSec 5
    Write-Output ("RUNTIME_READY paths=" + $openApi.paths.psobject.Properties.Count)
} catch {
    Write-Output ("RUNTIME_NOT_READY " + $_.Exception.Message)
    if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath -Tail 30
    }
    exit 1
}
