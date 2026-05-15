#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

export RTD_DEVICE=${RTD_DEVICE:-cuda}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset RTD_MODEL_ID || true
unset RTD_MODEL_PATH || true

PYTHON="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$ROOT/.venv/bin/python"
fi
if [ ! -x "$PYTHON" ]; then
  echo "Python venv not found in .venv" >&2
  exit 1
fi

"$PYTHON" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT/frontend"
npm run dev -- --host 127.0.0.1
