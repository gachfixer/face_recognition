#!/bin/sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/app.log"

mkdir -p "$LOG_DIR"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment not found. Creating .venv..."
    python3 -m venv "$VENV_DIR"
fi

. "$VENV_DIR/bin/activate"

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Installing/verifying requirements..."
    pip install -q -r "$SCRIPT_DIR/requirements.txt"
fi

echo "Logging to $LOG_FILE"
exec python app.py 2>&1 | tee -a "$LOG_FILE"
