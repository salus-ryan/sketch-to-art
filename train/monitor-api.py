"""
Lightweight API that reads training logs and serves data to the frontend.
Run: python train/monitor-api.py
"""
import json
import re
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

LOG_FILES = {
    "baseline": "/tmp/train_baseline.log",
    "br6": "/tmp/train_braille6.log",
    "br8": "/tmp/train_braille8.log",
    "br6s": "/tmp/train_braille6s.log",
    "br10s": "/tmp/train_braille10s.log",
}

MODEL_META = {
    "baseline": {"label": "Baseline", "color": "#ef4444", "desc": "No pre-training", "patterns": "—"},
    "br6": {"label": "BR-6", "color": "#22c55e", "desc": "6-dot binary", "patterns": "2⁶ = 64"},
    "br8": {"label": "BR-8", "color": "#3b82f6", "desc": "8-dot binary", "patterns": "2⁸ = 256"},
    "br6s": {"label": "BR-6S", "color": "#a855f7", "desc": "6-dot signed", "patterns": "3⁶ = 729"},
    "br10s": {"label": "BR-10S", "color": "#f59e0b", "desc": "10-dot signed", "patterns": "3¹⁰ = 59,049"},
}


def parse_log(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"pretrain": [], "coco": [], "status": "waiting", "epoch": 0, "done": False}

    text = p.read_text()
    pretrain = []
    coco = []
    epoch_done_count = 0
    current_epoch = 0
    model_saved = "Model saved" in text or "Model size" in text

    for line in text.splitlines():
        # Pre-training steps
        m = re.search(r"step (\d+)/(\d+), Loss: ([\d.]+)", line)
        if m:
            pretrain.append({
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "loss": float(m.group(3)),
            })
            continue

        # COCO batches
        m = re.search(r"Epoch (\d+)/(\d+), Batch (\d+)/(\d+), Loss: ([\d.]+)", line)
        if m:
            current_epoch = int(m.group(1))
            coco.append({
                "epoch": int(m.group(1)),
                "total_epochs": int(m.group(2)),
                "batch": int(m.group(3)),
                "total_batches": int(m.group(4)),
                "loss": float(m.group(5)),
            })
            continue

        if "done" in line and "Epoch" in line:
            epoch_done_count += 1

    pretrain_done = "pre-training done" in text

    if model_saved:
        status = "complete"
    elif coco:
        status = "coco"
    elif pretrain_done:
        status = "loading_coco"
    elif pretrain:
        status = "pretrain"
    else:
        status = "starting"

    return {
        "pretrain": pretrain,
        "coco": coco,
        "status": status,
        "epoch": current_epoch,
        "epochs_done": epoch_done_count,
        "done": model_saved,
        "pretrain_done": pretrain_done,
    }


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/training":
            data = {}
            for key, logfile in LOG_FILES.items():
                parsed_log = parse_log(logfile)
                parsed_log["meta"] = MODEL_META[key]
                data[key] = parsed_log
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            return

        if parsed.path == "/" or parsed.path == "":
            self.path = "/train/monitor.html"

        # Serve static files from project root
        self.directory = str(Path(__file__).resolve().parent.parent)
        super().do_GET()

    def translate_path(self, path):
        root = Path(__file__).resolve().parent.parent
        rel = path.lstrip("/")
        return str(root / rel)

    def log_message(self, format, *args):
        pass  # quiet


if __name__ == "__main__":
    port = 8787
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Training monitor: http://localhost:{port}")
    server.serve_forever()
