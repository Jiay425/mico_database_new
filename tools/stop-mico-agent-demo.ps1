[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$statePath = Join-Path ([System.IO.Path]::GetTempPath()) 'mico-agent-demo-state.json'
if (-not (Test-Path -LiteralPath $statePath)) {
    Write-Output 'No Mico demo state file found.'
    exit 0
}

$state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
foreach ($pidValue in @($state.runtimePid, $state.javaPid)) {
    if ($pidValue -and (Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue)) {
        & taskkill.exe /PID ([int]$pidValue) /T /F *> $null
    }
}

# Java owns the temporary SSH child in the normal demo path. If a forced stop
# bypassed its shutdown hook, remove only the exact loopback forwarding process
# created for this demo; never stop an unrelated listener.
$listeners = Get-NetTCPConnection -LocalPort 13306 -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction SilentlyContinue
    if ($process -and $process.Name -eq 'ssh.exe' -and $process.CommandLine -match '127\.0\.0\.1:13306:127\.0\.0\.1:3306') {
        & taskkill.exe /PID ([int]$listener.OwningProcess) /T /F *> $null
    }
}
Remove-Item -LiteralPath $statePath -Force
Write-Output 'Mico demo processes stopped; no demo-owned tunnel is retained.'
