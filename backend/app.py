import asyncio
import base64
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import LCMScheduler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Global references
pipeline = None
captioner = None
caption_processor = None

# Generation resolution — 512x512 fits easily in 8GB VRAM
GEN_SIZE = 512


def load_captioner():
    """Load BLIP for auto-detecting what the user is drawing."""
    global captioner, caption_processor
    print("Loading BLIP captioner...")
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    captioner = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16,
    ).to("cpu")
    print("Captioner ready!")


def load_pipeline():
    """Load SD 1.5 + ControlNet Scribble — fits in 8GB VRAM at 512x512."""
    print("Loading ControlNet Scribble model...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_scribble",
        torch_dtype=torch.float16,
    )

    print("Loading Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # LCM scheduler for fast inference
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Load LCM-LoRA for speed (4 steps instead of 20+)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

    pipe.to("cuda")

    print("Pipeline ready!")
    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    load_captioner()
    pipeline = load_pipeline()
    yield
    del pipeline
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


def auto_caption(image: Image.Image) -> str:
    """Use BLIP to detect what the sketch looks like and generate a prompt."""
    global captioner, caption_processor

    if captioner is None:
        return "a beautiful detailed realistic illustration, masterpiece"

    # Caption the sketch
    inputs = caption_processor(image, "a drawing of", return_tensors="pt").to("cpu", torch.float16)
    with torch.inference_mode():
        out = captioner.generate(**inputs, max_new_tokens=30)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)

    # Enhance caption into a good generation prompt
    prompt = f"{caption}, beautiful detailed realistic illustration, masterpiece, 4k, vivid colors, professional art"
    print(f"  Auto-prompt: {prompt}")
    return prompt


def generate_image(sketch_bytes: bytes, prompt: str, strength: float = 0.65) -> str:
    """Generate image from sketch using ControlNet. Auto-detects subject if prompt is '__auto__'."""
    global pipeline

    if pipeline is None:
        return ""

    # Load original image for captioning
    original = Image.open(io.BytesIO(sketch_bytes)).convert("RGB")

    # Auto-detect what they're drawing
    if prompt == "__auto__":
        prompt = auto_caption(original)

    # Preprocess for ControlNet — resize to 512x512, invert colors
    sketch = original.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)
    img_array = np.array(sketch)
    if img_array.mean() > 128:
        img_array = 255 - img_array
    sketch = Image.fromarray(img_array)

    # Clear VRAM cache before generation
    torch.cuda.empty_cache()

    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            image=sketch,
            num_inference_steps=4,
            guidance_scale=1.5,
            controlnet_conditioning_scale=strength,
            width=GEN_SIZE,
            height=GEN_SIZE,
        ).images[0]

    # Convert to base64
    buffer = io.BytesIO()
    result.save(buffer, format="WEBP", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Connected viewers (laptop screens showing the output)
viewers: list[WebSocket] = []


async def broadcast_to_viewers(message: dict):
    """Send a message to all connected viewer clients."""
    dead = []
    for v in viewers:
        try:
            await v.send_json(message)
        except Exception:
            dead.append(v)
    for v in dead:
        viewers.remove(v)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected (drawer)")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "stroke":
                # Forward raw stroke data to viewers in real-time
                await broadcast_to_viewers(data)

            elif data.get("type") == "sync":
                # Forward full canvas snapshot to viewers
                await broadcast_to_viewers(data)

            elif data.get("type") == "clear":
                await broadcast_to_viewers({"type": "clear"})

            elif data.get("type") == "generate":
                sketch_b64 = data.get("sketch", "")
                prompt = data.get("prompt", "__auto__")
                strength = data.get("strength", 0.65)

                if not sketch_b64:
                    continue

                sketch_bytes = base64.b64decode(sketch_b64)

                # Generate in thread pool to not block
                start = time.time()
                loop = asyncio.get_event_loop()
                result_b64 = await loop.run_in_executor(
                    None, generate_image, sketch_bytes, prompt, strength
                )
                elapsed = time.time() - start

                result_msg = {
                    "type": "result",
                    "image": result_b64,
                    "time": round(elapsed, 2),
                }

                # Send to drawer
                await websocket.send_json(result_msg)
                # Send to all viewers
                await broadcast_to_viewers(result_msg)

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print("Drawer disconnected")


@app.websocket("/ws/viewer")
async def viewer_ws(websocket: WebSocket):
    await websocket.accept()
    viewers.append(websocket)
    print(f"Viewer connected (total: {len(viewers)})")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        viewers.remove(websocket)
        print(f"Viewer disconnected (total: {len(viewers)})")


# Serve static frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# Viewer page — shows AI results full-screen on your laptop for streaming
@app.get("/viewer")
async def viewer():
    return FileResponse(str(FRONTEND_DIR / "viewer.html"))
