#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

if [ -f "$ROOT/.env.local" ]; then
  set -a
  . "$ROOT/.env.local"
  set +a
fi

export RTD_DEVICE=${RTD_DEVICE:-cuda}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export RTD_LOG_LEVEL=${RTD_LOG_LEVEL:-INFO}
export RTD_BACKEND_HOST=${RTD_BACKEND_HOST:-127.0.0.1}
export RTD_BACKEND_PORT=${RTD_BACKEND_PORT:-8000}
export RTD_FRONTEND_HOST=${RTD_FRONTEND_HOST:-127.0.0.1}
export RTD_FRONTEND_PORT=${RTD_FRONTEND_PORT:-5173}
export VITE_BACKEND_HOST=${VITE_BACKEND_HOST:-$RTD_BACKEND_HOST}
if [ "$VITE_BACKEND_HOST" = "0.0.0.0" ]; then
  export VITE_BACKEND_HOST=127.0.0.1
fi
export VITE_BACKEND_PORT=${VITE_BACKEND_PORT:-$RTD_BACKEND_PORT}
export VITE_BACKEND_PROTOCOL=${VITE_BACKEND_PROTOCOL:-http}

PYTHON="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$ROOT/.venv/bin/python"
fi
if [ ! -x "$PYTHON" ]; then
  echo "Python venv not found in .venv" >&2
  exit 1
fi

"$PYTHON" -m uvicorn app.main:app --app-dir backend --host "$RTD_BACKEND_HOST" --port "$RTD_BACKEND_PORT" --no-access-log &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT/frontend"
npm run dev -- --host "$RTD_FRONTEND_HOST" --port "$RTD_FRONTEND_PORT"
