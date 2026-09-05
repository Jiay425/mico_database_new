[CmdletBinding()]
param(
    [int]$LocalPort = 13306,
    [string]$RemoteHost = '10.31.2.52',
    [string]$RemoteUser = 'ljy'
)

$ErrorActionPreference = 'Stop'
$keyFile = Join-Path $env:USERPROFILE '.ssh\mico_remote_ed25519'

if (-not (Test-Path -LiteralPath $keyFile)) {
    throw "Remote database tunnel key not found: $keyFile"
}

$existing = Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Remote database tunnel is already listening on 127.0.0.1:$LocalPort."
    exit 0
}

$sshArgs = @(
    '-N',
    '-i', $keyFile,
    '-o', 'BatchMode=yes',
    '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=30',
    '-o', 'ServerAliveCountMax=3',
    '-o', 'StrictHostKeyChecking=accept-new',
    '-L', "127.0.0.1:${LocalPort}:127.0.0.1:3306",
    "${RemoteUser}@${RemoteHost}"
)

$process = Start-Process -FilePath 'ssh.exe' -ArgumentList $sshArgs -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 1

if (-not (Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue)) {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
    throw "Unable to create the remote MySQL tunnel on port $LocalPort."
}

Write-Host "Remote MySQL tunnel is ready: 127.0.0.1:${LocalPort} -> ${RemoteHost}:3306 (PID $($process.Id))."
