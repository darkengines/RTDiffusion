@echo off
setlocal

cd /d "%~dp0"

set RTD_DEVICE=cuda
set CUDA_VISIBLE_DEVICES=0
set PYTHONUNBUFFERED=1
if "%RTD_LOG_LEVEL%"=="" set RTD_LOG_LEVEL=INFO
set RTD_MODEL_ID=
set RTD_MODEL_PATH=

for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do taskkill /F /PID %%a >nul 2>nul

if not exist ".venv\Scripts\python.exe" (
  echo Python venv not found at .venv\Scripts\python.exe
  exit /b 1
)

start "RTDiffusion Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000"

pushd frontend
npm run dev -- --host 127.0.0.1
set EXIT_CODE=%ERRORLEVEL%
popd

exit /b %EXIT_CODE%
