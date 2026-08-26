param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("data", "focused", "open")]
    [string]$CaseName,
    [int]$Port = 8010,
    [int]$Attempt = 1
)

$ErrorActionPreference = "Stop"
$runtimeRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $runtimeRoot "evals\p2j4-v4-runtime-smoke-20260825"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$token = [Environment]::GetEnvironmentVariable("MICO_RUNTIME_INTERNAL_TOKEN", "User")
if (-not $token -or -not $token.Trim()) {
    throw "MICO_RUNTIME_INTERNAL_TOKEN is not available"
}

$suffix = switch ($CaseName) {
    "data" { "data" }
    "focused" { "focused" }
    default { "open" }
}
$attemptToken = "{0:x32}" -f $Attempt
$now = [DateTime]::UtcNow.ToString("o")
if ($CaseName -eq "data") {
    $payload = @{
        dataContractVersion = "v1"
        runId = "run-smoke-v4-data-$attemptToken"
        taskId = "task-smoke-v4-data-$attemptToken"
        requesterId = "principal-smoke-v4-000000000000000000000000000001"
        traceId = "trace-smoke-v4-data-$attemptToken"
        question = "统计当前项目的样本覆盖记录"
        intent = "data_fact"
        requestedScopes = @("mico:query:read")
        allowedActions = @("inspect_cohort", "execute_read_query", "finish")
        maxActions = 3
        createdAt = $now
    }
} elseif ($CaseName -eq "focused") {
    $payload = @{
        dataContractVersion = "v1"
        runId = "run-smoke-v4-focused-$attemptToken"
        taskId = "task-smoke-v4-focused-$attemptToken"
        requesterId = "principal-smoke-v4-000000000000000000000000001"
        traceId = "trace-smoke-v4-focused-$attemptToken"
        question = "比较两组的微生物差异"
        intent = "focused_comparison"
        requestedScopes = @("mico:query:read", "mico:research:read")
        allowedActions = @(
            "inspect_cohort", "compare_groups", "analyze_projection",
            "retrieve_evidence", "finish"
        )
        maxActions = 5
        createdAt = $now
    }
} else {
    $payload = @{
        dataContractVersion = "v1"
        runId = "run-smoke-v4-open-$attemptToken"
        taskId = "task-smoke-v4-open-$attemptToken"
        requesterId = "principal-smoke-v4-000000000000000000000000001"
        traceId = "trace-smoke-v4-open-$attemptToken"
        question = "探索当前数据中值得进一步验证的微生态现象"
        intent = "scientific_exploration"
        requestedScopes = @("mico:query:read", "mico:research:read", "mico:evidence:read")
        allowedActions = @(
            "inspect_cohort", "compare_groups", "adjust_confounders",
            "cross_project_validate", "cross_disease_validate",
            "analyze_projection", "retrieve_evidence", "finish"
        )
        maxActions = 6
        createdAt = $now
    }
}

$uri = "http://127.0.0.1:$Port/internal/runtime/scientific-runs"
$body = $payload | ConvertTo-Json -Depth 10
$headers = @{ Authorization = "Bearer $token" }
$responsePath = Join-Path $outputDir "$suffix-attempt$Attempt-response.json"
$metaPath = Join-Path $outputDir "$suffix-attempt$Attempt-meta.json"
$started = Get-Date
try {
    $response = Invoke-WebRequest `
        -UseBasicParsing `
        -Uri $uri `
        -Method Post `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 240
    $response.Content | Set-Content -LiteralPath $responsePath -Encoding utf8
    [ordered]@{
        case = $CaseName
        statusCode = [int]$response.StatusCode
        elapsedSeconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
        responseFile = $responsePath
    } | ConvertTo-Json | Set-Content -LiteralPath $metaPath -Encoding utf8
    Get-Content -LiteralPath $responsePath -Raw
} catch {
    $errorBody = ""
    if ($_.Exception.Response) {
        $reader = [IO.StreamReader]::new($_.Exception.Response.GetResponseStream())
        $errorBody = $reader.ReadToEnd()
    }
    [ordered]@{
        case = $CaseName
        statusCode = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
        elapsedSeconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
        error = $_.Exception.Message
        errorBody = $errorBody
    } | ConvertTo-Json | Set-Content -LiteralPath $metaPath -Encoding utf8
    throw
}
