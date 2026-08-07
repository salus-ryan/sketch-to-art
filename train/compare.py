"""
Head-to-head comparison: baseline vs braille-pretrained style transfer.

Runs both models on the same set of test images and generates a side-by-side
comparison grid.

Usage:
    # After training both models:
    python train/compare.py \
        --baseline models/starry_night.pth \
        --braille  models/starry_night_braille.pth \
        --input    test_images/ \
        --output   comparison.jpg

    # Quick single-image comparison:
    python train/compare.py \
        --baseline models/starry_night.pth \
        --braille  models/starry_night_braille.pth \
        --input    photo.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from model import TransformerNet


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path, device):
    model = TransformerNet()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def stylize(model, img, device, size=512):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return transforms.ToPILImage()(out.squeeze(0).cpu().clamp(0, 1))


def add_label(img, text, position="bottom"):
    """Add a text label to an image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if position == "bottom":
        x = (img.width - tw) // 2
        y = img.height - th - 10
    else:
        x = (img.width - tw) // 2
        y = 8

    # Background
    draw.rectangle([x - 4, y - 2, x + tw + 4, y + th + 2], fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill="white", font=font)
    return img


def compare_single(baseline_model, braille_model, img_path, device, size=512):
    """Compare both models on a single image. Returns (original, baseline, braille)."""
    img = Image.open(img_path).convert("RGB")

    t0 = time.perf_counter()
    baseline_out = stylize(baseline_model, img, device, size)
    baseline_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    braille_out = stylize(braille_model, img, device, size)
    braille_ms = (time.perf_counter() - t0) * 1000

    # Resize original to match
    original = img.resize((size, size))

    print(f"  {Path(img_path).name}: baseline={baseline_ms:.0f}ms, braille={braille_ms:.0f}ms")
    return original, baseline_out, braille_out


def build_grid(rows, size=512):
    """Build a comparison grid: [Original | Baseline | Braille] per row."""
    cols = 3
    padding = 4
    header = 40

    grid_w = cols * size + (cols + 1) * padding
    grid_h = header + len(rows) * size + (len(rows) + 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        font = ImageFont.load_default()

    headers = ["Original", "Baseline", "Braille-pretrained"]
    for ci, label in enumerate(headers):
        x = padding + ci * (size + padding)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (size - tw) // 2, 10), label, fill="white", font=font)

    for ri, (original, baseline, braille) in enumerate(rows):
        y = header + padding + ri * (size + padding)
        for ci, img in enumerate([original, baseline, braille]):
            x = padding + ci * (size + padding)
            grid.paste(img.resize((size, size)), (x, y))

    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs braille style transfer")
    parser.add_argument("--baseline", required=True, help="Path to baseline .pth model")
    parser.add_argument("--braille", required=True, help="Path to braille-pretrained .pth model")
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument("--output", default="comparison.jpg", help="Output comparison image")
    parser.add_argument("--size", type=int, default=512, help="Image size")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    baseline_model = load_model(args.baseline, device)
    braille_model = load_model(args.braille, device)
    print(f"Loaded baseline: {args.baseline}")
    print(f"Loaded braille:  {args.braille}")

    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
    else:
        images = [input_path]

    print(f"\nComparing on {len(images)} image(s)...\n")

    rows = []
    for img_path in images[:8]:  # Max 8 for grid
        row = compare_single(baseline_model, braille_model, img_path, device, args.size)
        rows.append(row)

    grid = build_grid(rows, args.size)
    grid.save(args.output, quality=92)
    print(f"\nComparison saved: {args.output} ({grid.width}x{grid.height})")
