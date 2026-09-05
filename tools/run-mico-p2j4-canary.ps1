[CmdletBinding()]
param(
    [string[]]$CaseId = @(
        'p2j4-data-fact-001',
        'p2j4-focused-analysis-008',
        'p2j4-open-exploration-001'
    ),
    [switch]$Real,
    [switch]$All,
    [string]$OutputPath = '',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $root 'mico-agent-runtime'
$python = Join-Path $runtimeRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Project .venv is missing; install the declared runtime dependencies first.'
}

# Import persisted user-level configuration into this process without logging
# values. This also makes the command safe to run from a newly opened shell.
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

# Real execution is an explicit per-process choice and is never persisted in
# the Windows user environment.
if ($Real) {
    $env:MICO_P2J4_REAL_RUNS = 'true'
} else {
    Remove-Item Env:MICO_P2J4_REAL_RUNS -ErrorAction SilentlyContinue
}

if ($All) {
    # An empty case selection means the validated runner executes the full
    # task set. The default remains the three-case canary above.
    $CaseId = @()
}

$arguments = @('-m', 'evals.p2j4_runner')
if ($Real) {
    $arguments += '--real'
} else {
    $arguments += '--dry-run'
}
foreach ($id in $CaseId) {
    if (-not [string]::IsNullOrWhiteSpace($id)) {
        $arguments += @('--case-id', $id)
    }
}
if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
    if ([System.IO.Path]::IsPathRooted($OutputPath)) {
        $resolvedOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
    } else {
        $resolvedOutputPath = [System.IO.Path]::GetFullPath((Join-Path $root $OutputPath))
    }
    $outputDirectory = Split-Path -Parent $resolvedOutputPath
    if (-not (Test-Path -LiteralPath $outputDirectory)) {
        New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    }
    $arguments += @('--output', $resolvedOutputPath)
}
if ($Resume) {
    $arguments += '--resume'
}

Push-Location $runtimeRoot
try {
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
