param([switch]$TRT)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "=== RTDiffusion Windows setup ===" -ForegroundColor Cyan

# -- Python venv --------------------------------------------------------------

$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$VenvPip    = Join-Path $Root '.venv\Scripts\pip.exe'

if (-not (Test-Path $VenvPython)) {
    Write-Host "> Creating Python venv at .venv ..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "> Venv already exists at .venv" -ForegroundColor Green
}

# -- PyTorch with CUDA --------------------------------------------------------
# Detects installed CUDA version; falls back to cu124 (CUDA 12.4, works on 12.x drivers).

$CudaVersion = 'cu124'
$_NvidiaSmiCmd = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
$NvidiaSmi = if ($_NvidiaSmiCmd) { $_NvidiaSmiCmd.Source } else { $null }
if ($NvidiaSmi) {
    $SmOutput = & $NvidiaSmi 2>$null | Select-String 'CUDA Version'
    if ($SmOutput -match 'CUDA Version:\s*(\d+)\.(\d+)') {
        $Major = [int]$Matches[1]
        $Minor = [int]$Matches[2]
        if     ($Major -ge 12) { $CudaVersion = 'cu124' }
        elseif ($Major -eq 11 -and $Minor -ge 8) { $CudaVersion = 'cu118' }
        else   { $CudaVersion = 'cpu' }
    }
}
Write-Host "> Installing PyTorch ($CudaVersion) - this may take a few minutes ..." -ForegroundColor Yellow
& $VenvPip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaVersion" --quiet

# -- Backend dependencies -----------------------------------------------------
Write-Host "> Installing backend[gpu] ..." -ForegroundColor Yellow
& $VenvPip install -e (Join-Path $Root 'backend[gpu]') --quiet

# -- TensorRT (optional) ------------------------------------------------------
# Pass -TRT to install TRT + ONNX extras. Requires CUDA 12.x drivers + NVIDIA GPU.
if ($TRT) {
    Write-Host "> Installing backend[tensorrt] (TRT + ONNX) ..." -ForegroundColor Yellow
    & $VenvPip install -e (Join-Path $Root 'backend[tensorrt]') --quiet
    Write-Host "> TRT extras installed. Enable with RTD_STREAM_TRT=1 in your .env" -ForegroundColor Green
}

# -- Frontend dependencies ----------------------------------------------------
$FrontendDir = Join-Path $Root 'frontend'
if (Test-Path (Join-Path $FrontendDir 'package.json')) {
    if (-not (Get-Command 'npm' -ErrorAction SilentlyContinue)) {
        Write-Warning "npm not found - skipping frontend install. Install Node.js from https://nodejs.org/"
    } else {
        Write-Host "> Installing frontend npm packages ..." -ForegroundColor Yellow
        Push-Location $FrontendDir
        npm install --prefer-offline | Out-Null
        Pop-Location
    }
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "  Run the app:     .\start.ps1"
Write-Host "  Docker backend:  .\start-backend-docker.ps1  (then open http://localhost:5173)"
Write-Host ""
Write-Host "Notes for Windows:"
Write-Host "  - torch.compile (Triton) is disabled automatically on Windows (RTD_STREAM_COMPILE=0)"
Write-Host "  - xFormers is NOT installed; SDP attention is used instead (similar performance)"
Write-Host "  - First StreamDiffusion session downloads TinyVAE + LCM LoRA from HuggingFace (~1 GB)"
Write-Host "  - Use the 'SDXL Turbo' preset in the StreamDiffusion panel for best results"
Write-Host "  - TRT acceleration: .\setup.ps1 -TRT  then set RTD_STREAM_TRT=1 in .env"
Write-Host "    First run exports UNet to ONNX + builds TRT engine (~5-10 min), cached after."
