from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI()

# Connected viewers
viewers: list[WebSocket] = []


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
    await websocket.accept()
    print("Drawer connected")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type in ("strokes", "stroke", "sync", "clear", "undo"):
                await broadcast(data)

    except WebSocketDisconnect:
        print("Drawer disconnected")


@app.websocket("/ws/viewer")
async def viewer_ws(websocket: WebSocket):
    await websocket.accept()
    viewers.append(websocket)
    print(f"Viewer connected ({len(viewers)} total)")
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
