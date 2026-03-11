#!/bin/sh

PID=$(pgrep -f "python app.py" 2>/dev/null)

if [ -z "$PID" ]; then
    echo "Application is not running."
    exit 0
fi

echo "Stopping application (PID $PID)..."
kill "$PID"
echo "Stopped."
