[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

function Set-MicoUserValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Value
    )
    [Environment]::SetEnvironmentVariable($Name, $Value, 'User')
}

function Set-MicoUserSecretIfMissing {
    param([Parameter(Mandatory = $true)][string]$Name)

    $existing = [Environment]::GetEnvironmentVariable($Name, 'User')
    if (-not [string]::IsNullOrWhiteSpace($existing)) {
        Write-Output "PRESERVED_USER_SECRET $Name"
        return
    }

    $secureValue = Read-Host ("Enter {0} (hidden; stored only in Windows User environment)" -f $Name) -AsSecureString
    $pointer = [IntPtr]::Zero
    $plainValue = $null
    try {
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
        $plainValue = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
        if ([string]::IsNullOrWhiteSpace($plainValue)) {
            throw "A non-empty value is required for $Name."
        }
        [Environment]::SetEnvironmentVariable($Name, $plainValue, 'User')
        Write-Output "CONFIGURED_USER_SECRET $Name"
    }
    finally {
        if ($pointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        }
        if ($null -ne $secureValue) {
            $secureValue.Dispose()
        }
        $plainValue = $null
    }
}

# Non-sensitive configuration is persisted here so all future terminals and
# processes can load the same local topology. No secret value is embedded in
# this file.
$plainValues = [ordered]@{
    MICO_AGENT_INTERNAL_ENABLED = 'true'
    MICO_AGENT_BFF_ENABLED = 'true'
    MICO_JAVA_AGENT_TOOL_BASE_URL = 'http://127.0.0.1:5000'
    MICO_AGENT_RUNTIME_BASE_URL = 'http://127.0.0.1:8000'
    MICO_AGENT_RUNTIME_ENDPOINT_PATH = '/internal/runtime/scientific-runs'
    MICO_RESEARCH_PLANNER_BASE_URL = 'https://api.deepseek.com'
    MICO_GRAPH_RAG_GENERATOR_BASE_URL = 'https://api.deepseek.com'
    MICO_KNOWLEDGE_RETRIEVAL_BACKEND = 'database'
    MICO_KNOWLEDGE_VECTOR_ENABLED = 'true'
    MICO_KNOWLEDGE_GRAPH_ENABLED = 'true'
    MICO_KNOWLEDGE_NEO4J_URI = 'bolt://127.0.0.1:57687'
    MICO_KNOWLEDGE_NEO4J_USER = 'neo4j'
    MICO_KNOWLEDGE_GRAPH_VERSION = 'fulltext-provenance-graphrag-v3'
    MICO_LOCAL_KNOWLEDGE_INDEX_DIR = 'E:\DeskTop\java_project\mico_database_new\mico_database_new\references\knowledge\medical\rag'
    MICO_GEMINI_EMBEDDING_ENABLED = 'true'
    MICO_GEMINI_EMBEDDING_MODEL = 'gemini-embedding-2'
    MICO_RESEARCH_PLANNER_MODEL = 'deepseek-v4-flash'
    MICO_GRAPH_RAG_GENERATOR_MODEL = 'deepseek-v4-flash'
}
foreach ($entry in $plainValues.GetEnumerator()) {
    Set-MicoUserValue -Name $entry.Key -Value $entry.Value
}

foreach ($name in @(
    'MICO_AGENT_INTERNAL_TOKEN',
    'MICO_RUNTIME_INTERNAL_TOKEN',
    'MICO_KNOWLEDGE_VECTOR_DATABASE_URL',
    'MICO_KNOWLEDGE_NEO4J_PASSWORD',
    'MICO_RESEARCH_PLANNER_TOKEN',
    'MICO_GRAPH_RAG_GENERATOR_TOKEN',
    'MICO_GEMINI_API_KEY'
)) {
    Set-MicoUserSecretIfMissing -Name $name
}

Write-Output 'Mico user-level configuration is persisted.'
Write-Output 'P2-J4 real execution is intentionally not persisted; use run-mico-p2j4-canary.ps1 -Real for one session.'
Write-Output 'Start a new PowerShell/Codex process, then run tools/start-mico-agent-demo.ps1.'
