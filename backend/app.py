from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI()

# Connected viewers
viewers: list[WebSocket] = []

# Canvas state: a sync snapshot + subsequent strokes for replay
canvas_state: list[dict] = []

# Reference to the active drawer so we can request a sync
drawer_ws_ref: WebSocket | None = None


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
    print("Drawer connected")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "strokes":
                canvas_state.append(data)
                await broadcast(data)
            elif msg_type == "stroke":
                canvas_state.append(data)
                await broadcast(data)
            elif msg_type == "clear":
                canvas_state.clear()
                await broadcast(data)
            elif msg_type == "undo":
                if canvas_state:
                    canvas_state.pop()
                await broadcast(data)
            elif msg_type == "sync":
                # A full canvas snapshot replaces all prior history
                canvas_state.clear()
                canvas_state.append(data)
                await broadcast(data)

    except WebSocketDisconnect:
        drawer_ws_ref = None
        print("Drawer disconnected")


@app.websocket("/ws/viewer")
async def viewer_ws(websocket: WebSocket):
    await websocket.accept()
    viewers.append(websocket)
    print(f"Viewer connected ({len(viewers)} total)")
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
        print(f"Viewer disconnected ({len(viewers)} total)")


# Static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/viewer")
async def viewer():
    return FileResponse(str(FRONTEND_DIR / "viewer.html"))
