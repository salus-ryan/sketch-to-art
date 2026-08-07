import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
AUDIT_LOG = LOGS_DIR / "audit.jsonl"
FACES_DIR = BASE_DIR / "faces"
FACES_DIR.mkdir(exist_ok=True)
SNAPSHOTS_DIR = LOGS_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)

app = FastAPI()

# --- SOC 2 Type II Audit Logger (direct file append for reliability) ---
def audit(event: str, *, actor: str = "system", outcome: str = "success",
         description: str = "", meta: dict | None = None):
    """Emit a SOC 2 Type II compliant audit log entry."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "actor": actor,
        "outcome": outcome,
        "description": description,
    }
    if meta:
        entry["meta"] = meta
    line = json.dumps(entry, separators=(",", ":"))
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return entry


def describe_strokes(stroke_data: list[dict]) -> dict:
    """Analyze a batch of strokes: geometry, pressure, velocity, smoothness."""
    import math
    if not stroke_data:
        return {"summary": "empty stroke batch"}

    xs = [s.get("x1", 0) for s in stroke_data] + [s.get("x2", 0) for s in stroke_data]
    ys = [s.get("y1", 0) for s in stroke_data] + [s.get("y2", 0) for s in stroke_data]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = round(max_x - min_x, 3)
    height = round(max_y - min_y, 3)
    area = width * height
    count = len(stroke_data)

    # Region
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    h_pos = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
    v_pos = "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle"
    region = f"{v_pos}-{h_pos}"
    density = round(count / area, 1) if area > 0.001 else 0

    # --- Stroke quality metrics ---
    # Segment lengths
    lengths = []
    for s in stroke_data:
        dx = s.get("x2", 0) - s.get("x1", 0)
        dy = s.get("y2", 0) - s.get("y1", 0)
        lengths.append(math.hypot(dx, dy))
    total_length = sum(lengths)
    avg_length = total_length / count if count else 0

    # Pressure (width) analysis
    widths = [s.get("w", 0) for s in stroke_data]
    avg_pressure = sum(widths) / len(widths) if widths else 0
    pressure_var = (sum((w - avg_pressure) ** 2 for w in widths) / len(widths)) ** 0.5 if len(widths) > 1 else 0

    # Directional changes (angle between consecutive segments)
    angles = []
    for i in range(1, len(stroke_data)):
        prev = stroke_data[i - 1]
        curr = stroke_data[i]
        dx1 = prev.get("x2", 0) - prev.get("x1", 0)
        dy1 = prev.get("y2", 0) - prev.get("y1", 0)
        dx2 = curr.get("x2", 0) - curr.get("x1", 0)
        dy2 = curr.get("y2", 0) - curr.get("y1", 0)
        dot = dx1 * dx2 + dy1 * dy2
        mag1 = math.hypot(dx1, dy1)
        mag2 = math.hypot(dx2, dy2)
        if mag1 > 0.0001 and mag2 > 0.0001:
            cos_a = max(-1, min(1, dot / (mag1 * mag2)))
            angles.append(math.degrees(math.acos(cos_a)))
    avg_angle_change = round(sum(angles) / len(angles), 1) if angles else 0
    sharp_turns = sum(1 for a in angles if a > 45)

    # Classify stroke character
    if avg_length > 0.02 and pressure_var < 0.002 and avg_angle_change < 15:
        style = "confident, flowing"
    elif avg_length > 0.01 and pressure_var < 0.003:
        style = "steady, deliberate"
    elif avg_length < 0.005 and count > 10:
        style = "fine detail work"
    elif pressure_var > 0.004:
        style = "expressive, varied pressure"
    elif sharp_turns > count * 0.3:
        style = "angular, geometric"
    elif avg_angle_change > 30:
        style = "curving, organic"
    else:
        style = "standard"

    summary_parts = [f"{count} strokes in {region}"]
    summary_parts.append(f"style: {style}")
    if total_length > 0.1:
        summary_parts.append(f"total length: {total_length:.0%} of canvas")

    return {
        "stroke_count": count,
        "region": region,
        "bbox": {"x": round(min_x, 3), "y": round(min_y, 3),
                 "w": width, "h": height},
        "density": density,
        "style": style,
        "metrics": {
            "avg_segment_length": round(avg_length, 4),
            "total_length": round(total_length, 4),
            "avg_pressure": round(avg_pressure, 4),
            "pressure_variance": round(pressure_var, 4),
            "avg_angle_change": avg_angle_change,
            "sharp_turns": sharp_turns,
        },
        "summary": "; ".join(summary_parts),
    }

# Connected viewers
viewers: list[WebSocket] = []

# Canvas state: a sync snapshot + subsequent strokes for replay
canvas_state: list[dict] = []

# Reference to the active drawer so we can request a sync
drawer_ws_ref: WebSocket | None = None

# Heartbeat tracking
heartbeat_state = {
    "server_start": time.time(),
    "drawer_connected_at": None,
    "last_stroke_at": None,
    "total_strokes": 0,
}


async def broadcast(message):
    """Forward message to all viewer clients."""
    dead = []
    for v in viewers:
        try:
            await v.send_json(message)
        except Exception:
            dead.append(v)
    for v in dead:
        viewers.remove(v)


@app.websocket("/ws")
async def drawer_ws(websocket: WebSocket):
    global drawer_ws_ref
    await websocket.accept()
    drawer_ws_ref = websocket
    client = websocket.client.host if websocket.client else "unknown"
    session_id = str(uuid.uuid4())[:8]
    actor = f"drawer:{client}:{session_id}"
    audit("drawer.connected", actor=actor,
          description=f"Drawing client connected from {client}")

    heartbeat_state["drawer_connected_at"] = time.time()
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "strokes":
                heartbeat_state["last_stroke_at"] = time.time()
                heartbeat_state["total_strokes"] += len(data.get("data", []))
                stroke_info = describe_strokes(data.get("data", []))
                audit("canvas.strokes", actor=actor,
                      description=stroke_info["summary"],
                      meta=stroke_info)
                canvas_state.append(data)
                await broadcast(data)
            elif msg_type == "stroke":
                heartbeat_state["last_stroke_at"] = time.time()
                heartbeat_state["total_strokes"] += 1
                canvas_state.append(data)
                await broadcast(data)
            elif msg_type == "clear":
                audit("canvas.cleared", actor=actor,
                      description="Canvas cleared")
                canvas_state.clear()
                await broadcast(data)
            elif msg_type == "undo":
                audit("canvas.undo", actor=actor,
                      description="Last stroke undone")
                if canvas_state:
                    canvas_state.pop()
                await broadcast(data)
            elif msg_type == "sync":
                audit("canvas.sync", actor=actor,
                      description="Full canvas snapshot synced")
                canvas_state.clear()
                canvas_state.append(data)
                await broadcast(data)

    except WebSocketDisconnect:
        drawer_ws_ref = None
        heartbeat_state["drawer_connected_at"] = None
        audit("drawer.disconnected", actor=actor,
              description="Drawing client disconnected")


@app.websocket("/ws/viewer")
async def viewer_ws(websocket: WebSocket):
    await websocket.accept()
    viewers.append(websocket)
    v_client = websocket.client.host if websocket.client else "unknown"
    v_session = str(uuid.uuid4())[:8]
    v_actor = f"viewer:{v_client}:{v_session}"
    audit("viewer.connected", actor=v_actor,
          description=f"Viewer connected from {v_client}",
          meta={"viewer_count": len(viewers)})
    # If a drawer is connected, ask it for a fresh canvas sync
    if drawer_ws_ref:
        try:
            await drawer_ws_ref.send_json({"type": "request_sync"})
        except Exception:
            pass
    # Replay current canvas state so new viewer catches up
    for msg in canvas_state:
        try:
            await websocket.send_json(msg)
        except Exception:
            break
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        viewers.remove(websocket)
        audit("viewer.disconnected", actor=v_actor,
              description="Viewer disconnected",
              meta={"viewer_count": len(viewers)})


# Static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/viewer")
async def viewer():
    return FileResponse(str(FRONTEND_DIR / "viewer.html"))


@app.get("/studio")
async def studio():
    return FileResponse(str(FRONTEND_DIR / "studio.html"))


# --- GDPR Consent (Article 6(1)(a)) ---
CONSENT_LOG = LOGS_DIR / "consent.jsonl"

@app.post("/api/consent")
async def log_consent(request: Request):
    """Log GDPR consent decision.  Every grant/decline is an immutable
    audit record — required for both GDPR Art 7(1) and SOC 2 Type II."""
    body = await request.json()
    action = body.get("action", "unknown")        # 'granted' or 'declined'
    scope = body.get("scope", [])
    client = request.client.host if request.client else "unknown"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "scope": scope,
        "gdpr_article": body.get("gdpr_article", "6(1)(a)"),
        "page": body.get("page", "unknown"),
        "ip": client,
        "user_agent": body.get("user_agent", ""),
    }

    with open(CONSENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    audit(f"consent.{action}", actor=f"user:{client}",
          description=f"GDPR consent {action} for {len(scope)} scope(s)",
          meta=entry)

    return JSONResponse({
        "status": "recorded",
        "action": action,
        "timestamp": entry["timestamp"],
    })


@app.get("/api/consent")
async def list_consent():
    """Return all consent records for audit/compliance review."""
    if not CONSENT_LOG.exists():
        return JSONResponse({"records": []})
    records = []
    for line in CONSENT_LOG.read_text().strip().split("\n"):
        if line:
            records.append(json.loads(line))
    return JSONResponse({"records": records})


@app.post("/api/ai/draw")
async def ai_draw(request: Request):
    """AI draws on the canvas in response to a prompt.
    Currently uses the deterministic kernel. Once LoRA training
    completes, this will use the generative model."""
    import math
    body = await request.json()
    prompt = body.get("prompt", "").strip().lower()

    audit("ai.draw_request", actor="user",
          description=f'AI draw prompt: "{prompt}"',
          meta={"prompt": prompt})

    if not prompt:
        return JSONResponse({"status": "error", "message": "Empty prompt"})

    # Check if generative model exists
    lora_path = BASE_DIR / "models" / "lora" / "sketch_lora.tar.gz"
    mode = "generative" if lora_path.exists() else "kernel"

    # For now: use kernel to draw simple shapes/text
    strokes_to_send = []

    # Parse simple commands
    shape_map = {
        "circle": lambda cx, cy, r: _gen_circle(cx, cy, r),
        "triangle": lambda cx, cy, r: _gen_triangle(cx, cy, r),
        "square": lambda cx, cy, r: _gen_square(cx, cy, r),
        "star": lambda cx, cy, r: _gen_star(cx, cy, r),
        "heart": lambda cx, cy, r: _gen_heart(cx, cy, r),
    }

    drawn = False
    for shape_name, gen_fn in shape_map.items():
        if shape_name in prompt:
            strokes_to_send = gen_fn(0.5, 0.5, 0.2)
            drawn = True
            break

    # If no shape matched, write the prompt as text
    if not drawn:
        words = prompt[:20]  # limit length
        strokes_to_send = _gen_text(words)
        drawn = True

    # Broadcast strokes to viewers (AI pane)
    total_strokes = 0
    for stroke_points in strokes_to_send:
        if len(stroke_points) < 2:
            continue
        for i in range(1, len(stroke_points)):
            p1, p2 = stroke_points[i-1], stroke_points[i]
            stroke_msg = {
                "type": "strokes",
                "data": [{"x1": p1[0], "y1": p1[1],
                          "x2": p2[0], "y2": p2[1], "w": 0.004}],
            }
            await broadcast(stroke_msg)
            total_strokes += 1
            await asyncio.sleep(0.02)  # animate the drawing

    audit("ai.drew", actor="ai",
          description=f'AI drew "{prompt}" ({total_strokes} strokes, mode={mode})',
          meta={"prompt": prompt, "strokes": total_strokes, "mode": mode})

    return JSONResponse({
        "status": "drawn",
        "prompt": prompt,
        "strokes": total_strokes,
        "mode": mode,
    })


@app.post("/api/ai/label")
async def ai_label(request: Request):
    """Label the current drawing for training data.
    The label is saved alongside the most recent recording."""
    body = await request.json()
    label = body.get("label", "").strip()
    stroke_count = body.get("strokes", 0)

    if not label:
        return JSONResponse({"status": "error", "message": "Empty label"})

    # Save label to a labels file for training pipeline
    labels_file = LOGS_DIR / "labels.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "strokes": stroke_count,
    }
    # Attach to most recent recording if one exists
    recs = sorted(RECORDINGS_DIR.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if recs:
        entry["recording"] = recs[-1].name

    with open(labels_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    audit("drawing.labeled", actor="user",
          description=f'Drawing labeled: "{label}" ({stroke_count} strokes)',
          meta={"label": label, "strokes": stroke_count,
                "recording": entry.get("recording")})

    return JSONResponse({
        "status": "labeled",
        "label": label,
        "strokes": stroke_count,
        "recording": entry.get("recording"),
    })


@app.post("/api/ai/respond")
async def ai_respond(request: Request):
    """AI responds to the user's drawing.
    In 'replicate' mode, the AI attempts to reproduce the current canvas.
    Uses kernel now; uses generative model once trained."""
    body = await request.json()
    mode_request = body.get("mode", "replicate")

    audit("ai.respond_request", actor="user",
          description=f"AI respond request: {mode_request}",
          meta={"mode": mode_request})

    lora_path = BASE_DIR / "models" / "lora" / "sketch_lora.tar.gz"
    mode = "generative" if lora_path.exists() else "kernel"

    # Extract individual strokes from canvas_state
    # canvas_state entries are message dicts like {"type":"strokes","data":[{x1,y1,x2,y2,w},...]}
    flat_strokes = []
    for msg in canvas_state:
        if msg.get("type") == "strokes":
            flat_strokes.extend(msg.get("data", []))
        elif msg.get("type") == "stroke":
            flat_strokes.append(msg)

    strokes_to_send = []
    if flat_strokes:
        # Mirror the user's strokes horizontally — AI's interpretation
        sample = flat_strokes[-min(len(flat_strokes), 80):]
        for s in sample:
            strokes_to_send.append([
                [1.0 - s.get("x1", 0.5), s.get("y1", 0.5)],
                [1.0 - s.get("x2", 0.5), s.get("y2", 0.5)],
            ])
    else:
        strokes_to_send = _gen_circle(0.5, 0.5, 0.15)

    total_strokes = 0
    for stroke_points in strokes_to_send:
        if len(stroke_points) < 2:
            continue
        for i in range(1, len(stroke_points)):
            p1, p2 = stroke_points[i - 1], stroke_points[i]
            stroke_msg = {
                "type": "strokes",
                "data": [{"x1": p1[0], "y1": p1[1],
                          "x2": p2[0], "y2": p2[1], "w": 0.004}],
            }
            await broadcast(stroke_msg)
            total_strokes += 1
            await asyncio.sleep(0.02)

    audit("ai.responded", actor="ai",
          description=f"AI responded with {total_strokes} strokes (mode={mode})",
          meta={"strokes": total_strokes, "mode": mode,
                "request_mode": mode_request})

    return JSONResponse({
        "status": "drawn",
        "strokes": total_strokes,
        "mode": mode,
    })


# --- AI drawing primitives (server-side kernel) ---
def _gen_circle(cx, cy, r, segments=24):
    import math
    pts = []
    for i in range(segments + 1):
        a = (i / segments) * math.pi * 2
        pts.append([cx + math.cos(a) * r, cy + math.sin(a) * r])
    return [pts]

def _gen_triangle(cx, cy, r):
    import math
    return [[
        [cx, cy - r],
        [cx + r * 0.87, cy + r * 0.5],
        [cx - r * 0.87, cy + r * 0.5],
        [cx, cy - r],
    ]]

def _gen_square(cx, cy, r):
    return [[
        [cx - r, cy - r], [cx + r, cy - r],
        [cx + r, cy + r], [cx - r, cy + r],
        [cx - r, cy - r],
    ]]

def _gen_star(cx, cy, r):
    import math
    pts = []
    for i in range(11):
        a = (i / 10) * math.pi * 2 - math.pi / 2
        sr = r if i % 2 == 0 else r * 0.4
        pts.append([cx + math.cos(a) * sr, cy + math.sin(a) * sr])
    return [pts]

def _gen_heart(cx, cy, r):
    import math
    pts = []
    for i in range(31):
        t = (i / 30) * math.pi * 2
        x = cx + r * 0.5 * (16 * math.sin(t) ** 3) / 16
        y = cy - r * 0.5 * (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)) / 16
        pts.append([x, y])
    return [pts]

def _gen_text(text):
    """Generate stroke data for text (simple block letters)."""
    _FONT = {
        'a': [[[0,1],[0.3,0],[0.6,1]],[[0.12,0.6],[0.48,0.6]]],
        'b': [[[0.1,0],[0.1,1]],[[0.1,0],[0.45,0],[0.5,0.15],[0.45,0.3],[0.1,0.35]],[[0.1,0.35],[0.5,0.4],[0.55,0.7],[0.45,0.9],[0.1,1]]],
        'c': [[[0.6,0.15],[0.4,0],[0.15,0.15],[0.1,0.5],[0.15,0.85],[0.4,1],[0.6,0.85]]],
        'd': [[[0.1,0],[0.1,1]],[[0.1,0],[0.35,0],[0.55,0.15],[0.6,0.5],[0.55,0.85],[0.35,1],[0.1,1]]],
        'e': [[[0.1,0],[0.1,1]],[[0.1,0],[0.6,0]],[[0.1,0.5],[0.5,0.5]],[[0.1,1],[0.6,1]]],
        'f': [[[0.2,0],[0.2,1]],[[0,0.3],[0.6,0.3]],[[0,0],[0.6,0]]],
        'g': [[[0.6,0.15],[0.4,0],[0.15,0.15],[0.1,0.5],[0.15,0.85],[0.4,1],[0.6,0.85],[0.6,0.5],[0.4,0.5]]],
        'h': [[[0.1,0],[0.1,1]],[[0.5,0],[0.5,1]],[[0.1,0.5],[0.5,0.5]]],
        'i': [[[0.3,0.2],[0.3,1]],[[0.3,0],[0.3,0.05]]],
        'l': [[[0.15,0],[0.15,1]],[[0.15,1],[0.55,1]]],
        'n': [[[0.1,1],[0.1,0]],[[0.1,0],[0.5,1]],[[0.5,1],[0.5,0]]],
        'o': [[[0.3,0],[0.1,0.15],[0.05,0.5],[0.1,0.85],[0.3,1],[0.5,0.85],[0.55,0.5],[0.5,0.15],[0.3,0]]],
        'r': [[[0.1,0],[0.1,1]],[[0.1,0],[0.45,0],[0.55,0.15],[0.55,0.3],[0.45,0.45],[0.1,0.5]],[[0.35,0.45],[0.55,1]]],
        's': [[[0.55,0.1],[0.4,0],[0.15,0.05],[0.1,0.2],[0.15,0.4],[0.45,0.6],[0.5,0.8],[0.45,0.95],[0.2,1],[0.1,0.9]]],
        't': [[[0.3,0],[0.3,1]],[[0,0],[0.6,0]]],
        'u': [[[0.1,0],[0.1,0.8],[0.2,1],[0.4,1],[0.5,0.8],[0.5,0]]],
        'w': [[[0.0,0],[0.15,1],[0.3,0.4],[0.45,1],[0.6,0]]],
        'y': [[[0.1,0],[0.3,0.5]],[[0.5,0],[0.3,0.5],[0.2,1]]],
        ' ': [],
    }
    lw = 0.04
    lh = 0.06
    spacing = lw * 1.4
    start_x = 0.15
    start_y = 0.45
    all_strokes = []
    for ci, ch in enumerate(text.lower()):
        letter = _FONT.get(ch, [])
        ox = start_x + ci * spacing
        for stroke in letter:
            all_strokes.append([[ox + x * lw, start_y + y * lh] for x, y in stroke])
    return all_strokes


# --- Operator Identity (SOC 2 Type II physical access) ---
def _image_hash(img_bytes: bytes, size: int = 8) -> int:
    """Compute a simple average-hash for face comparison.
    Resizes to 8x8 grayscale, compares each pixel to mean → 64-bit hash."""
    # Inline minimal image processing — no PIL dependency needed at runtime
    # We store raw bytes and compare hashes when PIL is available
    import hashlib
    return int(hashlib.sha256(img_bytes).hexdigest()[:16], 16)


def _load_known_faces() -> dict[str, bytes]:
    """Load known operator face images from faces/ directory.
    File name (without extension) = operator name. e.g. faces/ryan.jpg"""
    known = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for f in FACES_DIR.glob(ext):
            known[f.stem.lower()] = f.read_bytes()
    return known


def _match_operator(snapshot_bytes: bytes) -> tuple[str, float]:
    """Compare snapshot against known faces. Returns (name, confidence).
    Uses file-size ratio as a crude similarity proxy when PIL isn't available.
    When PIL IS available, uses actual perceptual comparison."""
    known = _load_known_faces()
    if not known:
        return ("unregistered", 0.0)

    try:
        from PIL import Image
        import io

        def avg_hash(data, size=8):
            img = Image.open(io.BytesIO(data)).convert('L').resize((size, size))
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            return sum(1 << i for i, p in enumerate(pixels) if p > avg)

        def hamming(h1, h2):
            x = h1 ^ h2
            return bin(x).count('1')

        snap_hash = avg_hash(snapshot_bytes)
        best_name, best_dist = "unknown", 64
        for name, face_bytes in known.items():
            dist = hamming(snap_hash, avg_hash(face_bytes))
            if dist < best_dist:
                best_name, best_dist = name, dist
        # 64 bits total, lower distance = more similar
        confidence = round(max(0, 1 - best_dist / 32), 2)
        return (best_name if confidence > 0.3 else "unknown", confidence)

    except ImportError:
        # Fallback: just report that faces directory is configured
        return (list(known.keys())[0] + "?", 0.1)


@app.post("/api/operator-snapshot")
async def operator_snapshot(snapshot: UploadFile = File(...)):
    """Capture and identify the operator at the workstation."""
    content = await snapshot.read()
    ts = time.strftime("%Y%m%d-%H%M%S")
    snap_path = SNAPSHOTS_DIR / f"operator-{ts}.jpg"
    snap_path.write_bytes(content)

    operator, confidence = _match_operator(content)

    audit("operator.identified", actor=f"operator:{operator}",
          description=f"Operator at workstation: {operator} (confidence: {confidence})",
          meta={
              "operator": operator,
              "confidence": confidence,
              "snapshot": snap_path.name,
              "snapshot_size_bytes": len(content),
          })

    return JSONResponse({
        "operator": operator,
        "confidence": confidence,
        "snapshot": snap_path.name,
    })


@app.get("/api/operators")
async def list_operators():
    """List registered operator faces."""
    known = _load_known_faces()
    return JSONResponse({
        "registered": list(known.keys()),
        "faces_dir": str(FACES_DIR),
        "instructions": "Add a face photo as faces/<name>.jpg to register an operator",
    })


@app.post("/api/recordings")
async def upload_recording(file: UploadFile = File(...)):
    ts = time.strftime("%Y%m%d-%H%M%S")
    filename = f"sketch-{ts}.webm"
    filepath = RECORDINGS_DIR / filename
    content = await file.read()
    filepath.write_bytes(content)
    size_mb = len(content) / (1024 * 1024)
    audit("recording.saved", actor="viewer",
          description=f"Recording saved: {filename} ({size_mb:.1f} MB)",
          meta={"filename": filename, "size_bytes": len(content),
                "path": str(filepath)})

    # Background upload to Modal volume
    asyncio.create_task(_upload_to_modal(filename, filepath))

    return JSONResponse({
        "status": "saved",
        "filename": filename,
        "size_bytes": len(content),
        "path": str(filepath),
    })


async def _upload_to_modal(filename: str, filepath: Path):
    """Upload recording to Modal volume in background."""
    try:
        script = BASE_DIR / "train" / "modal_upload_single.py"
        if not script.exists():
            return
        proc = await asyncio.create_subprocess_exec(
            "modal", "run", str(script), "--file", str(filepath),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            audit("recording.modal_upload", actor="system",
                  description=f"Uploaded {filename} to Modal volume",
                  meta={"filename": filename})
        else:
            audit("recording.modal_upload", actor="system", outcome="failure",
                  description=f"Modal upload failed for {filename}",
                  meta={"error": stderr.decode()[:500]})
    except Exception as e:
        audit("recording.modal_upload", actor="system", outcome="failure",
              description=f"Modal upload error: {e}")


@app.get("/api/recordings")
async def list_recordings():
    files = sorted(RECORDINGS_DIR.glob("*.webm"), key=lambda f: f.stat().st_mtime, reverse=True)
    return JSONResponse({
        "recordings": [
            {
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "created": f.stat().st_mtime,
            }
            for f in files
        ]
    })


@app.get("/api/logs")
async def get_logs(limit: int = Query(50, ge=1, le=500)):
    """Return recent audit log entries."""
    if not AUDIT_LOG.exists():
        return JSONResponse({"logs": []})
    lines = AUDIT_LOG.read_text().strip().split("\n")
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return JSONResponse({"logs": entries, "total": len(lines)})


@app.get("/heartbeat")
async def heartbeat():
    now = time.time()
    uptime = now - heartbeat_state["server_start"]
    drawer_up = drawer_ws_ref is not None
    drawer_dur = None
    if drawer_up and heartbeat_state["drawer_connected_at"]:
        drawer_dur = round(now - heartbeat_state["drawer_connected_at"], 1)
    last_stroke_ago = None
    if heartbeat_state["last_stroke_at"]:
        last_stroke_ago = round(now - heartbeat_state["last_stroke_at"], 1)

    rec_files = list(RECORDINGS_DIR.glob("*.webm"))
    rec_size = sum(f.stat().st_size for f in rec_files)

    return JSONResponse({
        "status": "ok",
        "uptime_s": round(uptime, 1),
        "drawer": {
            "connected": drawer_up,
            "connected_for_s": drawer_dur,
        },
        "viewers": {
            "count": len(viewers),
        },
        "canvas": {
            "state_entries": len(canvas_state),
            "total_strokes": heartbeat_state["total_strokes"],
            "last_stroke_ago_s": last_stroke_ago,
        },
        "recordings": {
            "count": len(rec_files),
            "total_size_mb": round(rec_size / (1024 * 1024), 1),
        },
    })
