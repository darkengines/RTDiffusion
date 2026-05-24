@echo off
setlocal

cd /d "%~dp0"

if exist ".env.local" (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env.local") do if not "%%a"=="" if not defined %%a set "%%a=%%b"
)

if "%RTD_DEVICE%"=="" set RTD_DEVICE=cuda
set PYTHONUNBUFFERED=1
if "%RTD_LOG_LEVEL%"=="" set RTD_LOG_LEVEL=INFO
if "%RTD_BACKEND_HOST%"=="" set RTD_BACKEND_HOST=127.0.0.1
if "%RTD_BACKEND_PORT%"=="" set RTD_BACKEND_PORT=8000
if "%RTD_FRONTEND_HOST%"=="" set RTD_FRONTEND_HOST=127.0.0.1
if "%RTD_FRONTEND_PORT%"=="" set RTD_FRONTEND_PORT=5173
if "%VITE_BACKEND_HOST%"=="" set VITE_BACKEND_HOST=%RTD_BACKEND_HOST%
if "%VITE_BACKEND_HOST%"=="0.0.0.0" set VITE_BACKEND_HOST=127.0.0.1
if "%VITE_BACKEND_PORT%"=="" set VITE_BACKEND_PORT=%RTD_BACKEND_PORT%
if "%VITE_BACKEND_PROTOCOL%"=="" set VITE_BACKEND_PROTOCOL=http

for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":%RTD_BACKEND_PORT% .*LISTENING"') do taskkill /F /PID %%a >nul 2>nul

if not exist ".venv\Scripts\python.exe" (
  echo Python venv not found at .venv\Scripts\python.exe
  exit /b 1
)

start "RTDiffusion Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host %RTD_BACKEND_HOST% --port %RTD_BACKEND_PORT% --no-access-log"

pushd frontend
npm run dev -- --host %RTD_FRONTEND_HOST% --port %RTD_FRONTEND_PORT%
set EXIT_CODE=%ERRORLEVEL%
popd

exit /b %EXIT_CODE%
