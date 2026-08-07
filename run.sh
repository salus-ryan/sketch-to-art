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
    IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")
fi
if [ -z "$IP" ]; then
    IP=$(ifconfig 2>/dev/null | grep 'inet ' | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
fi
if [ -z "$IP" ]; then
    IP=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
if [ -z "$IP" ]; then
    IP="localhost"
fi

# Generate self-signed SSL cert if not present (needed for camera access over LAN)
CERT_DIR="$SCRIPT_DIR/certs"
if [ ! -f "$CERT_DIR/cert.pem" ] || [ ! -f "$CERT_DIR/key.pem" ]; then
    echo "Generating self-signed SSL certificate..."
    mkdir -p "$CERT_DIR"
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$CERT_DIR/key.pem" \
        -out "$CERT_DIR/cert.pem" \
        -days 365 \
        -subj "/CN=sketch-to-art" \
        -addext "subjectAltName=IP:${IP},IP:127.0.0.1,DNS:localhost,DNS:${IP}" \
        2>/dev/null
    echo "Certificate created at $CERT_DIR/"
    echo ""
    echo "NOTE: On first visit your browser will show a security warning."
    echo "      Click 'Advanced' → 'Proceed' to accept the self-signed cert."
    echo ""
fi

echo ""
echo "========================================"
echo "  Draw on your phone, see it on screen"
echo "========================================"
echo ""
echo "  Phone:  http://${IP}:8766"
echo "  Viewer: https://localhost:8765/viewer"
echo "          https://${IP}:8765/viewer"
echo ""
echo "  Heartbeat: https://localhost:8765/heartbeat"
echo ""
echo "========================================"
echo ""

cd backend

# Start HTTP server for phone (no SSL needed, phone only draws)
uvicorn app:app --host 0.0.0.0 --port 8766 --reload &
HTTP_PID=$!

# Start HTTPS server for viewer (SSL needed for camera access)
uvicorn app:app --host 0.0.0.0 --port 8765 \
    --ssl-certfile "$CERT_DIR/cert.pem" \
    --ssl-keyfile "$CERT_DIR/key.pem" &
HTTPS_PID=$!

# Wait for either to exit, kill both on interrupt
trap "kill $HTTP_PID $HTTPS_PID 2>/dev/null; exit" INT TERM
wait
