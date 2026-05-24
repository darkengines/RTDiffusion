$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

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

if (-not $env:RTD_BACKEND_HOST) { $env:RTD_BACKEND_HOST = '127.0.0.1' }
if (-not $env:RTD_BACKEND_PORT) { $env:RTD_BACKEND_PORT = '8000' }
if (-not $env:RTD_FRONTEND_HOST) { $env:RTD_FRONTEND_HOST = '127.0.0.1' }
if (-not $env:RTD_FRONTEND_PORT) { $env:RTD_FRONTEND_PORT = '5173' }
if (-not $env:VITE_BACKEND_HOST) { $env:VITE_BACKEND_HOST = if ($env:RTD_BACKEND_HOST -eq '0.0.0.0') { '127.0.0.1' } else { $env:RTD_BACKEND_HOST } }
if (-not $env:VITE_BACKEND_PORT) { $env:VITE_BACKEND_PORT = $env:RTD_BACKEND_PORT }
if (-not $env:VITE_BACKEND_PROTOCOL) { $env:VITE_BACKEND_PROTOCOL = 'http' }

$BackendHost = $env:RTD_BACKEND_HOST
$BackendPort = [int]$env:RTD_BACKEND_PORT
$FrontendHost = $env:RTD_FRONTEND_HOST
$FrontendPort = [int]$env:RTD_FRONTEND_PORT

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

$backend = Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', "$BackendHost", '--port', "$BackendPort", '--no-access-log' -WorkingDirectory $Root -PassThru

try {
  Push-Location (Join-Path $Root 'frontend')
  npm run dev -- --host $FrontendHost --port $FrontendPort
}
finally {
  Pop-Location
  if ($backend -and -not $backend.HasExited) {
    Stop-Process -Id $backend.Id -Force
  }
}
