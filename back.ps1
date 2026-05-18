$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$BackendPort = 8000
$FrontendHost = '127.0.0.1'

function Import-LocalEnv([string]$Path) {
  if (-not (Test-Path $Path)) { return }
  foreach ($line in Get-Content $Path) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith('#') -or -not $trimmed.Contains('=')) { continue }
    $key, $value = $trimmed.Split('=', 2)
    if (-not [Environment]::GetEnvironmentVariable($key, 'Process')) {
      [Environment]::SetEnvironmentVariable($key, $value.Trim().Trim('"').Trim("'"), 'Process')
    }
  }
}

Import-LocalEnv (Join-Path $Root '.env.local')

$listenerProcessIds = Get-NetTCPConnection -LocalPort $BackendPort -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.OwningProcess -gt 0 } |
  Select-Object -ExpandProperty OwningProcess -Unique

foreach ($listenerProcessId in $listenerProcessIds) {
  Stop-Process -Id $listenerProcessId -Force
}

if (-not $env:RTD_DEVICE) { $env:RTD_DEVICE = 'cuda' }
if (-not $env:RTD_LOG_LEVEL) { $env:RTD_LOG_LEVEL = 'INFO' }

$python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Python venv not found at $python"
}

$backend = Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', "$BackendPort", '--no-access-log' -WorkingDirectory $Root -PassThru