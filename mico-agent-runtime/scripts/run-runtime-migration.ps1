[CmdletBinding()]
param(
    [string]$Python = '.\.venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($env:MICO_AGENT_RUNTIME_MIGRATION_DATABASE_URL)) {
    throw 'MIGRATION_DATABASE_URL_NOT_CONFIGURED'
}

# The migration-only Python validator rejects every URL except the dedicated
# mico_agent_runtime MySQL database. The URL itself is never printed.
& $Python -m alembic current
if ($LASTEXITCODE -ne 0) {
    throw 'ALEMBIC_CURRENT_FAILED'
}
& $Python -m alembic upgrade head
if ($LASTEXITCODE -ne 0) {
    throw 'ALEMBIC_UPGRADE_FAILED'
}
& $Python -m alembic current
if ($LASTEXITCODE -ne 0) {
    throw 'ALEMBIC_FINAL_CURRENT_FAILED'
}
