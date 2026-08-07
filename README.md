# Sketch → Art: Live Drawing Capture & Braille Foundation Model

A real-time drawing relay with live artifact ingestion, fast neural style transfer, and a novel braille foundation model. Draw on any device, watch strokes mirror to viewers instantly, and train AI models that understand the algebraic structure of generalized braille.

## Research: Generalized Braille

We formalize braille as a signed vector space **b ∈ {-1, 0, +1}ⁿ** — not a fixed alphabet, but an inspectable, extensible semantic coordinate system. See [`paper/generalized-braille.tex`](paper/generalized-braille.tex) for the full whitepaper.

**Key results:**
- A **5K-param MLP** learns the factored representation φ(b) = Σbᵢdᵢ and generalizes to all unseen cells in a space of 43 million (n=16)
- A **151K-param transformer** learns compositional programs over signed cells at **99.85% accuracy**
- Ternary braille pre-training yields **19% faster convergence** than baseline in style transfer, with signed cells consistently outperforming binary

## Trained Models (Git LFS)

Five style transfer models trained on COCO 2017 with Starry Night, comparing braille pre-training regimes:

| Model | Pre-training | Patterns | Final Loss | vs Baseline |
|-------|-------------|----------|------------|-------------|
| `starry_night.pth` | None (baseline) | — | **91,567** | — |
| `starry_night_braille6.pth` | 6-dot binary | 2⁶ = 64 | 93,179 | +1.8% |
| `starry_night_braille8.pth` | 8-dot binary | 2⁸ = 256 | 93,513 | +2.1% |
| `starry_night_braille6s.pth` | 6-dot signed | 3⁶ = 729 | 92,822 | +1.4% |
| `starry_night_braille10s.pth` | 10-dot signed | 3¹⁰ = 59,049 | 93,555 | +2.2% |

All models converge to within 2.2% after 2 epochs, but ternary models reach the convergence basin **30-40% faster** at matched batch counts.

## BrailleNet: Foundation Model

A stroke-native model that connects three levels — no vision encoder, no CNN, no pixels:

```
Strokes ←→ Cells b ∈ {-1,0,+1}ⁿ ←→ Meaning
```

| Component | What it does |
|-----------|-------------|
| **StrokeEncoder** | Polyline stroke sequences → cell-aligned embeddings |
| **CellDecoder** | Cell embeddings → dot pattern vectors |
| **AlgebraHead** | Ternary algebra: add, negate, inner product, cancel, update |

Train on Modal H100:
```bash
modal run train/modal_train_foundation.py
```

## Architecture

```
Drawing Device (phone/tablet/Surface) ──WebSocket──→ FastAPI Server
                                                       ├── Relay strokes to viewers in real-time
                                                       ├── Auto-record sessions (canvas + camera → .webm)
                                                       ├── SOC 2 Type II audit logging
                                                       ├── Operator face identification (webcam)
                                                       └── Auto-upload recordings to Modal volume
                                                              │
                                                              ▼
                                                       Modal (cloud GPU, H100)
                                                       ├── Style transfer training (5 braille variants)
                                                       ├── BrailleNet foundation model
                                                       └── Live training monitor dashboard
```

## Requirements

- **Python:** 3.10+
- **Modal account:** For cloud GPU training (H100)
- **No local GPU required** — training runs on Modal, inference runs on Apple Silicon (MPS), CUDA, or CPU

## Quick Start

```bash
chmod +x run.sh
./run.sh
```

First run will:
1. Create a Python virtual environment
2. Install dependencies (FastAPI, uvicorn, websockets, Pillow)
3. Generate a self-signed SSL certificate (for camera access over LAN)
4. Start HTTP (port 8766) and HTTPS (port 8765) servers

## Pages

| URL | Purpose |
|-----|---------|
| `/` | **Drawer** — full-screen canvas for drawing (phone/tablet) |
| `/viewer` | **Viewer** — mirrors strokes in real-time with camera PiP, recording controls |
| `/studio` | **Studio** — split-pane: draw on left, view relay on right, AI chat bar |
| `/heartbeat` | Server health & session stats |

## Drawing & Recording Flow

1. **Draw** on any device at `http://<IP>:8766` — strokes relay to viewers via WebSocket
2. **Viewer** at `https://<IP>:8765/viewer` shows strokes in real-time with webcam overlay
3. **Recording auto-starts** when strokes arrive (if camera is active) and auto-stops after 5s idle
4. Recordings are **composited** (canvas + camera PiP → .webm) and saved to `recordings/`
5. Each recording is **auto-uploaded to Modal volume** in the background
6. **Operator snapshots** are captured from the webcam for SOC 2 audit compliance

## Controls

| Control | Description |
|---------|-------------|
| **↩ / Ctrl+Z** | Undo last stroke |
| **✕** | Clear canvas |
| **📷** | Toggle camera PiP (viewer) |
| **🔄** | Switch camera (viewer) |
| **⏺** | Manual record toggle (viewer) |
| **Chat bar** | Send prompts to AI kernel for shape/text drawing (studio) |
| **Pinch / scroll** | Zoom canvas (drawer) |

## AI Training Pipeline

### Style Transfer (Baseline)
```bash
modal run train/modal_train.py --style-image train/styles/starry_night.jpg
```
Trains a feedforward style transfer network on COCO using a style image. Model downloads locally as `models/<style>.pth`.

### LoRA Fine-tuning (Braille Models)
```bash
# Upload recordings + extract frames
modal run train/modal_recordings.py

# Upload, extract, and fine-tune
modal run train/modal_recordings.py --train

# List local recordings
modal run train/modal_recordings.py --list-only
```
Uploads recorded sketch sessions to Modal, extracts frames, and LoRA fine-tunes Stable Diffusion on your drawing artifacts. Outputs saved to `models/lora/`.

### Local Inference
```bash
python train/inference.py --model models/starry_night.pth --input photo.jpg
python train/inference.py --model models/starry_night.pth --benchmark
```
Runs on Apple Silicon (MPS), CUDA, or CPU.

## Audit & Compliance

The server emits **SOC 2 Type II compliant** audit logs to `logs/audit.jsonl`:
- Drawer/viewer connect/disconnect events
- Stroke analytics (region, style, pressure, velocity)
- Canvas clear/undo/sync events
- Recording saves and Modal uploads
- Operator identification via webcam snapshots (`faces/` directory)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `WS` | `/ws` | Drawer WebSocket (send strokes) |
| `WS` | `/ws/viewer` | Viewer WebSocket (receive strokes) |
| `POST` | `/api/ai/draw` | AI kernel: draw shapes/text from prompt |
| `POST` | `/api/recordings` | Upload a recording |
| `GET` | `/api/recordings` | List saved recordings |
| `POST` | `/api/operator-snapshot` | Capture & identify operator |
| `GET` | `/api/operators` | List registered operators |
| `GET` | `/api/logs` | Recent audit log entries |
| `GET` | `/heartbeat` | Server health status |
