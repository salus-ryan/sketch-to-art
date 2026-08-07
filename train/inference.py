"""
Local inference for fast neural style transfer.

Runs on Apple Silicon (MPS), CUDA, or CPU.
Usage:
    python train/inference.py --model models/starry_night.pth --input photo.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

# Add parent so we can import model
sys.path.insert(0, str(Path(__file__).parent))
from model import TransformerNet


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path: str, device: torch.device) -> TransformerNet:
    model = TransformerNet()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def stylize(model: TransformerNet, image: Image.Image, device: torch.device,
            size: int = 512) -> Image.Image:
    """Run style transfer on a PIL image, return styled PIL image."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # Convert back to PIL
    output = output.squeeze(0).cpu().clamp(0, 1)
    return transforms.ToPILImage()(output)


def benchmark(model: TransformerNet, device: torch.device, size: int = 512,
              runs: int = 20):
    """Benchmark inference speed."""
    dummy = torch.randn(1, 3, size, size).to(device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model(dummy)
    if device.type == "mps":
        torch.mps.synchronize()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_ms = sum(times) / len(times) * 1000
    print(f"Avg inference: {avg_ms:.1f}ms ({size}x{size}) on {device}")
    return avg_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth model")
    parser.add_argument("--input", help="Input image path")
    parser.add_argument("--output", default="output.jpg", help="Output path")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.model, device)
    print(f"Model loaded: {args.model}")

    if args.benchmark:
        benchmark(model, device, args.size)

    if args.input:
        img = Image.open(args.input).convert("RGB")
        result = stylize(model, img, device, args.size)
        result.save(args.output)
        print(f"Saved: {args.output}")
