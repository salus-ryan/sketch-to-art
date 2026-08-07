"""
QuickDraw → BrailleNet data pipeline.

Downloads Google QuickDraw stroke data (.ndjson) and converts to
BrailleNet training format:
  - Points: (T, 2) normalized 0-1, with SEP tokens (-1,-1) between strokes
  - Labels: category string
  - Raster: 256x256 rendered sketch (for target image generation)

Usage:
    python train/quickdraw_data.py --categories 50 --samples-per-cat 10000 --output data/quickdraw
    python train/quickdraw_data.py --categories 50 --samples-per-cat 10000 --output data/quickdraw --modal
"""

import argparse
import io
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Top 50 QuickDraw categories (diverse, recognizable)
DEFAULT_CATEGORIES = [
    "cat", "dog", "fish", "bird", "horse", "butterfly", "spider",
    "tree", "flower", "mushroom", "cactus",
    "house", "castle", "church", "bridge",
    "car", "bicycle", "bus", "airplane", "helicopter", "boat",
    "face", "hand", "eye", "skull",
    "guitar", "piano", "trumpet",
    "apple", "banana", "pizza", "cake", "ice cream",
    "sun", "moon", "star", "cloud", "rainbow", "lightning",
    "sword", "crown", "key", "clock", "book",
    "circle", "square", "triangle", "hexagon",
    "smiley face", "heart",
]

QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"

# Special tokens for BrailleNet
SEP_TOKEN = np.array([-1.0, -1.0])
PAD_TOKEN = np.array([-2.0, -2.0])


def download_category(category: str, max_samples: int = 10000) -> list[dict]:
    """Download simplified .ndjson for a category from Google Cloud."""
    url = f"{QUICKDRAW_BASE_URL}/{category.replace(' ', '%20')}.ndjson"
    print(f"  Downloading: {category} ...", end=" ", flush=True)

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            samples = []
            for line in resp:
                if len(samples) >= max_samples:
                    break
                obj = json.loads(line.decode("utf-8"))
                if obj.get("recognized", False):
                    samples.append(obj)
            print(f"{len(samples)} samples")
            return samples
    except Exception as e:
        print(f"FAILED: {e}")
        return []


def quickdraw_to_points(drawing: list[list[list[int]]], max_points: int = 512) -> np.ndarray:
    """Convert QuickDraw drawing format to normalized point sequence.

    QuickDraw format: [[[x, x, ...], [y, y, ...]], [[x, x, ...], [y, y, ...]], ...]
    Output: (T, 2) array with points normalized to [0, 1], SEP tokens between strokes.
    """
    all_points = []

    for stroke_idx, stroke in enumerate(drawing):
        xs, ys = stroke[0], stroke[1]
        for i in range(len(xs)):
            all_points.append([xs[i], ys[i]])
        # Add SEP between strokes (not after last)
        if stroke_idx < len(drawing) - 1:
            all_points.append(SEP_TOKEN.tolist())

    if not all_points:
        return np.zeros((1, 2))

    points = np.array(all_points, dtype=np.float32)

    # Normalize real points (not SEP tokens) to [0, 1]
    real_mask = ~((points[:, 0] == -1) & (points[:, 1] == -1))
    real_pts = points[real_mask]

    if len(real_pts) == 0:
        return np.zeros((1, 2))

    min_x, min_y = real_pts[:, 0].min(), real_pts[:, 1].min()
    max_x, max_y = real_pts[:, 0].max(), real_pts[:, 1].max()

    # Avoid division by zero
    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    # Normalize preserving aspect ratio
    scale = max(range_x, range_y)
    offset_x = (scale - range_x) / 2
    offset_y = (scale - range_y) / 2

    points[real_mask, 0] = (real_pts[:, 0] - min_x + offset_x) / scale
    points[real_mask, 1] = (real_pts[:, 1] - min_y + offset_y) / scale

    # Truncate to max_points
    if len(points) > max_points:
        points = points[:max_points]

    return points


def render_sketch(points: np.ndarray, size: int = 256) -> Image.Image:
    """Render a point sequence to a PIL Image (white bg, black strokes)."""
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    # Draw strokes between non-SEP consecutive points
    margin = int(size * 0.05)
    draw_size = size - 2 * margin

    i = 0
    while i < len(points) - 1:
        # Skip SEP tokens
        if points[i][0] == -1 and points[i][1] == -1:
            i += 1
            continue
        if points[i + 1][0] == -1 and points[i + 1][1] == -1:
            i += 1
            continue

        x1 = int(points[i][0] * draw_size + margin)
        y1 = int(points[i][1] * draw_size + margin)
        x2 = int(points[i + 1][0] * draw_size + margin)
        y2 = int(points[i + 1][1] * draw_size + margin)

        draw.line([(x1, y1), (x2, y2)], fill=0, width=2)
        i += 1

    return img


def process_category(category: str, samples: list[dict], output_dir: Path,
                     max_points: int = 512, render_size: int = 256):
    """Process all samples for a category and save as binary files."""
    cat_dir = output_dir / "strokes" / category.replace(" ", "_")
    cat_dir.mkdir(parents=True, exist_ok=True)

    raster_dir = output_dir / "rasters" / category.replace(" ", "_")
    raster_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, sample in enumerate(samples):
        drawing = sample.get("drawing", [])
        if not drawing:
            continue

        points = quickdraw_to_points(drawing, max_points=max_points)
        if len(points) < 3:
            continue

        # Save stroke data as .npz
        np.savez_compressed(
            cat_dir / f"{idx:05d}.npz",
            points=points,
            category=category,
        )

        # Render and save raster
        img = render_sketch(points, size=render_size)
        img.save(raster_dir / f"{idx:05d}.png")

        count += 1

    return count


def build_manifest(output_dir: Path):
    """Build a manifest file listing all processed samples."""
    strokes_dir = output_dir / "strokes"
    manifest = []

    for cat_dir in sorted(strokes_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name.replace("_", " ")
        for npz_file in sorted(cat_dir.glob("*.npz")):
            manifest.append({
                "category": category,
                "strokes": str(npz_file.relative_to(output_dir)),
                "raster": str(Path("rasters") / cat_dir.name / (npz_file.stem + ".png")),
            })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"\nManifest: {len(manifest)} samples → {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download & process QuickDraw data for BrailleNet")
    parser.add_argument("--output", type=str, default="data/quickdraw",
                        help="Output directory")
    parser.add_argument("--categories", type=int, default=50,
                        help="Number of categories to download")
    parser.add_argument("--samples-per-cat", type=int, default=10000,
                        help="Max samples per category")
    parser.add_argument("--max-points", type=int, default=512,
                        help="Max points per stroke sequence")
    parser.add_argument("--render-size", type=int, default=256,
                        help="Raster render size")
    parser.add_argument("--modal", action="store_true",
                        help="Upload to Modal volume after processing")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = DEFAULT_CATEGORIES[:args.categories]
    print(f"Processing {len(categories)} categories, {args.samples_per_cat} samples each")
    print(f"Output: {output_dir}\n")

    total = 0
    for cat in categories:
        samples = download_category(cat, max_samples=args.samples_per_cat)
        if samples:
            count = process_category(
                cat, samples, output_dir,
                max_points=args.max_points,
                render_size=args.render_size,
            )
            total += count
            print(f"    → {count} processed")

    print(f"\n{'='*50}")
    print(f"Total: {total} samples across {len(categories)} categories")

    build_manifest(output_dir)

    if args.modal:
        upload_to_modal(output_dir)


def upload_to_modal(output_dir: Path):
    """Upload processed data to Modal volume."""
    print("\nUploading to Modal volume...")
    import subprocess
    # Create a simple upload script
    script = f"""
import modal
vol = modal.Volume.from_name("braillenet-data", create_if_missing=True)
import os
for root, dirs, files in os.walk("{output_dir}"):
    for f in files:
        local = os.path.join(root, f)
        remote = "/" + os.path.relpath(local, "{output_dir}")
        vol.upload_file(local, remote)
vol.commit()
print("Upload complete!")
"""
    script_path = output_dir / "_upload.py"
    script_path.write_text(script)
    subprocess.run(["modal", "run", str(script_path)], check=True)
    script_path.unlink()


if __name__ == "__main__":
    main()
