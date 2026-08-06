#!/bin/bash
# Sketch-to-Art: Real-time Drawing Relay
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Get local IP (works on macOS and Linux)
if command -v ipconfig &>/dev/null 2>&1; then
    IP=$(ipconfig getifaddr en0 2>/dev/null || echo "localhost")
else
    IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
fi

echo ""
echo "========================================"
echo "  Draw on your phone, see it on screen"
echo "========================================"
echo ""
echo "  Phone:  http://${IP}:8765"
echo "  Viewer: http://${IP}:8765/viewer"
echo ""
echo "========================================"
echo ""

cd backend
uvicorn app:app --host 0.0.0.0 --port 8765 --reload
