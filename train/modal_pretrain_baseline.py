"""
Baseline: pixel-based encoder pre-training on QuickDraw rasters.

This is the control experiment — a CNN/ViT encoder that sees rasterized
sketches instead of raw strokes. Same data, same categories, same compute.
Comparison validates the paper's claim that stroke-native pre-training
converges faster and produces better representations.

Usage:
    modal run train/modal_pretrain_baseline.py
"""

import modal

app = modal.App("pretrain-baseline-pixel")
data_volume = modal.Volume.from_name("braillenet-data", create_if_missing=True)
model_volume = modal.Volume.from_name("braillenet-models", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.3.1", "numpy",
)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": data_volume, "/models": model_volume},
    timeout=14400,
)
def pretrain_baseline(
    num_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
    d_model: int = 256,
    img_size: int = 64,
    mask_ratio: float = 0.75,
):
    import sys
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pathlib import Path
    from torch.utils.data import DataLoader, Dataset

    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name()})", flush=True)
    print(f"\n=== BASELINE: Pixel-based encoder (control experiment) ===", flush=True)

    # ===================================================================
    # DATASET — rasterize strokes on the fly (no pre-rendered PNGs needed)
    # ===================================================================

    class QuickDrawRasterDataset(Dataset):
        """Load consolidated strokes, rasterize to images on the fly."""

        def __init__(self, data_dir: str, img_size: int = 64):
            self.img_size = img_size

            consolidated = Path(data_dir) / "quickdraw_consolidated.npz"
            if not consolidated.exists():
                raise FileNotFoundError(f"No data at {consolidated}")

            print(f"Loading consolidated dataset...", flush=True)
            t0 = time.time()
            data = np.load(consolidated)
            self.raw_points = data["points"]   # (N, 512, 2)
            self.lengths = data["lengths"]      # (N,)
            self.categories = torch.from_numpy(data["categories"].astype(np.int64))
            cat_names = data["cat_names"]
            self.num_categories = len(cat_names)
            print(f"Dataset: {len(self.raw_points)} samples, {self.num_categories} cats, "
                  f"loaded in {time.time()-t0:.1f}s", flush=True)

        def __len__(self):
            return len(self.raw_points)

        def _rasterize(self, points, length):
            """Render stroke points to a grayscale image."""
            img = np.ones((self.img_size, self.img_size), dtype=np.float32)
            pts = points[:length]
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                # Skip SEP tokens
                if x0 < 0 or x1 < 0:
                    continue
                # Scale to image coordinates
                ix0 = int(x0 * (self.img_size - 1))
                iy0 = int(y0 * (self.img_size - 1))
                ix1 = int(x1 * (self.img_size - 1))
                iy1 = int(y1 * (self.img_size - 1))
                # Bresenham-lite: just draw endpoints + midpoint
                for px, py in [(ix0, iy0), (ix1, iy1),
                               ((ix0+ix1)//2, (iy0+iy1)//2)]:
                    if 0 <= px < self.img_size and 0 <= py < self.img_size:
                        img[py, px] = 0.0
            return img

        def __getitem__(self, idx):
            img = self._rasterize(self.raw_points[idx], self.lengths[idx])
            return {
                "image": torch.from_numpy(img).unsqueeze(0),  # (1, H, W)
                "category": self.categories[idx],
            }

    # ===================================================================
    # MODEL — CNN encoder with MAE-style pre-training
    # ===================================================================

    class CNNEncoder(nn.Module):
        """Simple CNN encoder for sketch images."""

        def __init__(self, d_model=256, img_size=64):
            super().__init__()
            # 4 conv blocks: 64→64, 64→32, 32→16, 16→8
            self.conv = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(),
                nn.MaxPool2d(2),  # /2

                nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.GELU(),
                nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.GELU(),
                nn.MaxPool2d(2),  # /4

                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU(),
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU(),
                nn.MaxPool2d(2),  # /8

                nn.Conv2d(256, d_model, 3, 1, 1), nn.BatchNorm2d(d_model), nn.GELU(),
                nn.AdaptiveAvgPool2d(4),  # → (d_model, 4, 4)
            )
            self.d_model = d_model

        def forward(self, x):
            # x: (B, 1, H, W) → (B, d_model, 4, 4)
            return self.conv(x)

    class PixelBaseline(nn.Module):
        """Pixel encoder + category classification + reconstruction."""

        def __init__(self, d_model=256, num_categories=48, img_size=64):
            super().__init__()
            self.encoder = CNNEncoder(d_model, img_size)
            # Category head: pool → classify
            self.category_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(d_model * 4 * 4, d_model),
                nn.GELU(),
                nn.Linear(d_model, num_categories),
            )
            # Reconstruction head: upsample back to image
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(d_model, 128, 4, 2, 1), nn.GELU(),  # 4→8
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),       # 8→16
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),        # 16→32
                nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),      # 32→64
            )

        def forward(self, images):
            features = self.encoder(images)
            cat_pred = self.category_head(features)
            recon = self.decoder(features)
            return cat_pred, recon, features

    # ===================================================================
    # TRAINING
    # ===================================================================

    print(f"\nConfig: d_model={d_model}, img_size={img_size}", flush=True)
    print(f"  batch_size={batch_size}, lr={lr}, epochs={num_epochs}", flush=True)

    dataset = QuickDrawRasterDataset("/data", img_size=img_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True,
    )

    model = PixelBaseline(
        d_model=d_model, num_categories=dataset.num_categories, img_size=img_size,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(loader),
    )

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_cat_loss = 0
        epoch_recon_loss = 0
        epoch_cat_acc = 0
        num_batches = 0

        for batch in loader:
            images = batch["image"].to(device)
            categories = batch["category"].to(device)

            # Add noise for denoising objective
            noise = torch.randn_like(images) * 0.1
            noisy_images = (images + noise).clamp(0, 1)

            cat_pred, recon, _ = model(noisy_images)

            # Losses
            cat_loss = F.cross_entropy(cat_pred, categories)
            recon_loss = F.mse_loss(recon, images)
            loss = cat_loss + recon_loss * 10.0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_cat_loss += cat_loss.item()
            epoch_recon_loss += recon_loss.item()
            with torch.no_grad():
                epoch_cat_acc += (cat_pred.argmax(1) == categories).float().mean().item()
            num_batches += 1

            if num_batches % 100 == 0:
                print(f"  [{epoch+1}/{num_epochs}] batch {num_batches}/{len(loader)} "
                      f"loss={loss.item():.4f} cat_acc={(cat_pred.argmax(1) == categories).float().mean().item():.1%} "
                      f"recon={recon_loss.item():.4f}", flush=True)

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_cat = epoch_cat_loss / num_batches
        avg_recon = epoch_recon_loss / num_batches
        avg_cat_acc = epoch_cat_acc / num_batches
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs} ({elapsed/60:.1f}min) | "
              f"loss={avg_loss:.4f} cat={avg_cat:.4f}(acc={avg_cat_acc:.1%}) "
              f"recon={avg_recon:.4f}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = "/models/baseline_pixel_encoder.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.encoder.state_dict(),
                "loss": best_loss,
                "config": {
                    "d_model": d_model,
                    "img_size": img_size,
                    "num_categories": dataset.num_categories,
                    "param_count": param_count,
                },
            }, save_path)
            print(f"  → Saved best: {save_path}", flush=True)

    total_time = time.time() - start_time
    final_path = "/models/baseline_pixel_final.pth"
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.encoder.state_dict(),
        "full_model_state_dict": model.state_dict(),
        "loss": avg_loss,
        "training_time_s": total_time,
        "config": {
            "d_model": d_model,
            "img_size": img_size,
            "num_categories": dataset.num_categories,
            "param_count": param_count,
        },
    }, final_path)
    model_volume.commit()

    print(f"\nBaseline training complete in {total_time/3600:.2f} hours", flush=True)
    print(f"Best loss: {best_loss:.4f}", flush=True)
    return {"best_loss": best_loss, "training_time_h": total_time / 3600}


@app.local_entrypoint()
def main():
    result = pretrain_baseline.remote()
    print(f"\nBaseline result: {result}")
