Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Keep the internal bearer token in the Windows user environment.  This local
# launcher loads it only into the Java child process and never writes or prints
# its value.  The Java bootstrap then owns creation of the SSH-backed MySQL
# tunnel before Spring initializes the datasource.
$internalToken = [Environment]::GetEnvironmentVariable(
    "MICO_AGENT_INTERNAL_TOKEN", "User"
)
if ([string]::IsNullOrWhiteSpace($internalToken)) {
    throw "Missing Windows user environment variable: MICO_AGENT_INTERNAL_TOKEN"
}

$env:MICO_AGENT_INTERNAL_TOKEN = $internalToken
$env:MICO_AGENT_INTERNAL_ENABLED = "true"
$env:MICO_AGENT_BFF_ENABLED = "true"

$projectRoot = Split-Path -Parent $PSScriptRoot
$maven = Join-Path $projectRoot "mvnw.cmd"
if (-not (Test-Path -LiteralPath $maven)) {
    throw "Maven wrapper is unavailable"
}

Set-Location -LiteralPath $projectRoot
& $maven spring-boot:run
exit $LASTEXITCODE
