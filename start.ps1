$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$BackendPort = 8000
$FrontendHost = '127.0.0.1'

$listenerProcessIds = Get-NetTCPConnection -LocalPort $BackendPort -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.OwningProcess -gt 0 } |
  Select-Object -ExpandProperty OwningProcess -Unique

foreach ($listenerProcessId in $listenerProcessIds) {
  Stop-Process -Id $listenerProcessId -Force
}

$env:RTD_DEVICE = 'cuda'
$env:CUDA_VISIBLE_DEVICES = '0'
Remove-Item Env:\RTD_MODEL_ID -ErrorAction SilentlyContinue
Remove-Item Env:\RTD_MODEL_PATH -ErrorAction SilentlyContinue

$python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Python venv not found at $python"
}

$backend = Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', "$BackendPort" -WorkingDirectory $Root -PassThru

try {
  Push-Location (Join-Path $Root 'frontend')
  npm run dev -- --host $FrontendHost
}
finally {
  Pop-Location
  if ($backend -and -not $backend.HasExited) {
    Stop-Process -Id $backend.Id -Force
  }
}
