[CmdletBinding()]
param(
    [int]$JavaPort = 5000,
    [int]$RuntimePort = 8000,
    [switch]$SkipJavaBuild
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$javaRoot = Join-Path $root 'mico_database_new'
$runtimeRoot = Join-Path $root 'mico-agent-runtime'
$python = Join-Path $runtimeRoot '.venv\Scripts\python.exe'
$statePath = Join-Path ([System.IO.Path]::GetTempPath()) 'mico-agent-demo-state.json'

# A new PowerShell process does not inherit values that were injected into a
# previous terminal. Import the persisted user-level configuration without
# printing any value. Process-level values remain authoritative so a one-off
# local override still works.
$micoEnvironmentNames = @(
    'MICO_AGENT_INTERNAL_ENABLED',
    'MICO_AGENT_BFF_ENABLED',
    'MICO_JAVA_AGENT_TOOL_BASE_URL',
    'MICO_AGENT_RUNTIME_BASE_URL',
    'MICO_AGENT_RUNTIME_ENDPOINT_PATH',
    'MICO_AGENT_INTERNAL_TOKEN',
    'MICO_RUNTIME_INTERNAL_TOKEN',
    'MICO_RESEARCH_PLANNER_BASE_URL',
    'MICO_RESEARCH_PLANNER_MODEL',
    'MICO_RESEARCH_PLANNER_TOKEN',
    'MICO_GRAPH_RAG_GENERATOR_BASE_URL',
    'MICO_GRAPH_RAG_GENERATOR_MODEL',
    'MICO_GRAPH_RAG_GENERATOR_TOKEN',
    'MICO_GEMINI_EMBEDDING_ENABLED',
    'MICO_GEMINI_EMBEDDING_MODEL',
    'MICO_GEMINI_API_KEY',
    'MICO_KNOWLEDGE_RETRIEVAL_BACKEND',
    'MICO_KNOWLEDGE_VECTOR_ENABLED',
    'MICO_KNOWLEDGE_VECTOR_DATABASE_URL',
    'MICO_KNOWLEDGE_GRAPH_ENABLED',
    'MICO_KNOWLEDGE_NEO4J_URI',
    'MICO_KNOWLEDGE_NEO4J_USER',
    'MICO_KNOWLEDGE_NEO4J_PASSWORD',
    'MICO_KNOWLEDGE_GRAPH_VERSION',
    'MICO_LOCAL_KNOWLEDGE_INDEX_DIR'
)
foreach ($name in $micoEnvironmentNames) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, 'Process'))) {
        $userValue = [Environment]::GetEnvironmentVariable($name, 'User')
        if (-not [string]::IsNullOrWhiteSpace($userValue)) {
            Set-Item -Path ("Env:{0}" -f $name) -Value $userValue
        }
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project .venv is missing; install the declared runtime dependencies first.'
}
$required = @(
    'MICO_RUNTIME_INTERNAL_TOKEN',
    'MICO_AGENT_INTERNAL_TOKEN',
    'MICO_JAVA_AGENT_TOOL_BASE_URL',
    'MICO_AGENT_RUNTIME_BASE_URL'
)
foreach ($name in $required) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
        throw "Required private environment variable is missing: $name"
    }
}

if (Test-Path -LiteralPath $statePath) {
    throw 'A Mico demo state file already exists; run the matching stop script first.'
}

$runtimeLog = Join-Path ([System.IO.Path]::GetTempPath()) 'mico-agent-runtime-demo.log'
$runtimeErrorLog = Join-Path ([System.IO.Path]::GetTempPath()) 'mico-agent-runtime-demo.err.log'
$javaLog = Join-Path ([System.IO.Path]::GetTempPath()) 'mico-java-demo.log'
$javaErrorLog = Join-Path ([System.IO.Path]::GetTempPath()) 'mico-java-demo.err.log'
$runtime = Start-Process -FilePath $python -WorkingDirectory $runtimeRoot -ArgumentList @(
    '-m', 'uvicorn', 'mico_agent_runtime.transport.app:create_app', '--factory',
    '--host', '127.0.0.1', '--port', "$RuntimePort"
) -WindowStyle Hidden -RedirectStandardOutput $runtimeLog -RedirectStandardError $runtimeErrorLog -PassThru

$javaArgs = @('spring-boot:run', "-Dspring-boot.run.arguments=--server.port=$JavaPort")
if (-not $SkipJavaBuild) { $javaArgs = @('-q') + $javaArgs }
$java = Start-Process -FilePath 'mvn.cmd' -WorkingDirectory $javaRoot -ArgumentList $javaArgs -WindowStyle Hidden -RedirectStandardOutput $javaLog -RedirectStandardError $javaErrorLog -PassThru

[ordered]@{
    runtimePid = $runtime.Id
    javaPid = $java.Id
    runtimePort = $RuntimePort
    javaPort = $JavaPort
    statePath = $statePath
} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8

Write-Output "Mico demo processes started on loopback ports $RuntimePort and $JavaPort."
Write-Output 'Use stop-mico-agent-demo.ps1 to stop only the processes created by this script.'
