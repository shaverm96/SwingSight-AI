#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

cd "$SCRIPT_DIR"

if [ ! -x "$VENV_PYTHON" ]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required. Install it from https://www.python.org/downloads/"
    exit 1
  fi

  echo "Setting up SwingSight for the first time..."
  python3 -m venv "$SCRIPT_DIR/.venv"
  "$VENV_PYTHON" -m pip install --upgrade pip
  "$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

exec "$VENV_PYTHON" "$SCRIPT_DIR/src/run.py"
