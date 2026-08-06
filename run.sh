#!/bin/bash
# Sketch-to-Art: Real-time AI Drawing App
# Run this script to start the server

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo ""
echo "============================================"
echo "  Sketch → Art: Real-time AI Drawing"
echo "============================================"
echo ""
echo "Starting server on http://0.0.0.0:8765"
echo ""
echo "→ Open this URL on your Surface Pro (same network):"
echo "  http://$(hostname -I | awk '{print $1}'):8765"
echo ""
echo "First run will download ~6GB of AI models."
echo "============================================"
echo ""

cd backend
uvicorn app:app --host 0.0.0.0 --port 8765 --reload
