#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

cd "$SCRIPT_DIR"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "SwingSight could not find requirements.txt."
  exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required. Install it from https://www.python.org/downloads/"
    exit 1
  fi

  echo "Setting up SwingSight for the first time..."
  python3 -m venv "$SCRIPT_DIR/.venv"
fi

echo "Checking required Python packages..."
"$VENV_PYTHON" -m pip install --disable-pip-version-check --upgrade pip
"$VENV_PYTHON" -m pip install --disable-pip-version-check -r "$REQUIREMENTS_FILE"

echo "Verifying club-number recognition packages..."
if ! "$VENV_PYTHON" -c "import rapidocr, onnxruntime" >/dev/null 2>&1; then
  echo "Repairing RapidOCR and ONNX Runtime..."
  "$VENV_PYTHON" -m pip install --disable-pip-version-check "rapidocr>=3.9.0,<4.0.0" "onnxruntime>=1.20.0"
fi

export SWINGSIGHT_OPEN_BROWSER=true
export SWINGSIGHT_DEBUG=false

exec "$VENV_PYTHON" "$SCRIPT_DIR/src/run.py"
